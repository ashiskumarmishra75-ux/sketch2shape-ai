r"""Connects every stage into one programmatic pipeline.

    sketch --> [CV preprocessing] --structural map--> [ControlNet + SD] --> candidates
          \--> [VLM understanding] --prompt-------/            |
                                                   rank by structural fidelity (rembg masks)
                                                               v
                       [background removal / framing] --> [TripoSR | InstantMesh]
                                                               v
                                          [mesh validation + clean-up] --> GLB / OBJ

The pipeline is split into three resumable steps so the UI can show intermediate
results and let the user regenerate images or rebuild the 3D model from a different
candidate without redoing earlier work:
    analyze()      -> stages 2-3
    generate()     -> stage 4 (+ ranking)
    reconstruct()  -> stages 5-7
Models are injected (generator / reconstructor), so each component can be swapped.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from computer_vision.edge_detection import fidelity_overlay, structural_fidelity
from computer_vision.image_preparation import prepare_for_reconstruction, segment_object
from computer_vision.preprocessing import dominant_colour_name, preprocess_sketch
from config.settings import Settings, get_settings
from generative_ai.prompt_builder import build_prompts
from generative_ai.sketch_understanding import parse_user_text, understand_sketch
from pipeline.job_manager import Job, JobManager
from reconstruction.mesh_processing import process_mesh
from utils.file_handler import ImageLike, save_image
from utils.gpu import free_memory, memory_snapshot, peak_vram_gb_and_reset, vram_gb
from utils.logging_utils import get_logger

log = get_logger("orchestrator")
StatusFn = Optional[Callable[[str], None]]
LOW_FIDELITY = 0.35


class Sketch2ShapePipeline:
    def __init__(self, settings: Optional[Settings] = None, generator=None, reconstructor=None,
                 understand_fn: Callable = understand_sketch, segment_fn: Callable = segment_object):
        self.s = settings or get_settings()
        self.jobs = JobManager(self.s.outputs_dir, self.s.keep_jobs)
        self._generator = generator
        self._reconstructor = reconstructor
        self.understand_fn = understand_fn
        self.segment_fn = segment_fn

    # -------------------------------------------------- lazily created model holders (loaded once)
    @property
    def generator(self):
        if self._generator is None:
            from generative_ai.image_generation import ControlNetGenerator
            self._generator = ControlNetGenerator(self.s)
        return self._generator

    @property
    def reconstructor(self):
        if self._reconstructor is None:
            from reconstruction.image_to_3d import get_backend
            self._reconstructor = get_backend(self.s)
        return self._reconstructor

    @staticmethod
    def _say(job: Job, status: StatusFn):
        def _f(msg: str) -> None:
            log.info("[%s] %s   [%s]", job.id, msg, memory_snapshot())
            if status:
                status(msg)
        return _f

    def _low_vram(self) -> bool:
        if self.s.offload_mode == "always":
            return True
        if self.s.offload_mode == "never":
            return False
        return vram_gb() < self.s.low_vram_threshold_gb

    # ================================================================ stage 2 + 3
    def analyze(self, image: ImageLike, user_text: str = "", *, category: str = "", material: str = "",
                color: str = "", style: str = "", control_type: Optional[str] = None,
                job: Optional[Job] = None, status: StatusFn = None) -> Job:
        job = job or self.jobs.create()
        control_type = control_type or self.s.control_type
        job.inputs.update(user_text=user_text, category=category, material=material, color=color, style=style,
                          control_type=control_type)
        say = self._say(job, status)

        say("Stage 2/7 - computer-vision preprocessing")
        with job.timer.stage("1_cv_preprocessing"):
            prep = preprocess_sketch(image, self.s.work_resolution, control_type)
        job.preprocess = prep
        job.files.update({k: str(v) for k, v in prep.save(job.dir).items()})
        for w in prep.warnings:
            job.warn(w)
        job.note(f"Preprocessing done ({prep.characteristics.input_type}): " + "; ".join(prep.operations))

        say("Stage 3/7 - understanding the sketch")
        with job.timer.stage("2_sketch_understanding"):
            desc = self.understand_fn(Image.fromarray(prep.clean_sketch), user_text, category=category,
                                      material=material, color=color, style=style, binary=prep.binary,
                                      mask=prep.mask, settings=self.s)
        job.description = desc
        for n in desc.notes:
            if "user asked for" in n or "Low-confidence" in n:
                job.warn(n)
        user_colour = parse_user_text(user_text or "", category, material, color, style)["color"]
        if prep.characteristics.input_type == "photo" and not user_colour:
            picked = dominant_colour_name(prep.normalized, prep.mask)
            if picked:
                desc.color = picked
                desc.notes.append(f"Colour '{picked}' taken from your picture")
        job.prompts = build_prompts(desc, self.s.gen_family)
        job.note(f"Sketch understood as '{desc.object}' via {desc.source}")
        job.save_report()
        return job

    # ================================================================ stage 4
    def generate(self, job: Job, *, num_candidates: Optional[int] = None, seed: int = 42,
                 steps: Optional[int] = None, guidance: Optional[float] = None,
                 controlnet_scale: Optional[float] = None, positive_override: str = "",
                 status: StatusFn = None, progress: Optional[Callable[[int, int], None]] = None) -> Job:
        if job.preprocess is None:
            raise RuntimeError("analyze() must run before generate()")
        say = self._say(job, status)
        n = num_candidates or self.s.num_candidates
        prompts = job.prompts
        if positive_override.strip():
            prompts.positive = positive_override.strip()
        if self._reconstructor is not None and self._low_vram():
            self._reconstructor.park()

        control_end = None
        init_image = None
        if job.preprocess.characteristics and job.preprocess.characteristics.input_type == "photo":
            # Edges of a colour picture are far denser than a sketch's; full-strength control copies its flat
            # illustration style. Loosen it so the model can render a proper 3D-looking object.
            if controlnet_scale is None or abs(controlnet_scale - self.s.controlnet_scale) < 1e-6:
                controlnet_scale = 0.7
            control_end = 0.7
            init_image = Image.fromarray(job.preprocess.normalized)
            job.note(f"Colour-picture input: image-to-image from your picture (strength {self.s.img2img_strength}), "
                     "ControlNet 0.7 released at 70 % of the steps")
        say(f"Stage 4/7 - generating {n} sketch-guided image(s) with ControlNet")
        control = Image.fromarray(job.preprocess.control_image)
        with job.timer.stage("3_image_generation"):
            cands = self.generator.generate(control, prompts.positive, prompts.negative, num_candidates=n, seed=seed,
                                            steps=steps, guidance=guidance, controlnet_scale=controlnet_scale,
                                            control_end=control_end, control_type=job.inputs.get("control_type"),
                                            progress=progress, init_image=init_image,
                                            strength=self.s.img2img_strength)
        peak = peak_vram_gb_and_reset()
        job.inputs.update(seed=seed, num_candidates=n, steps=steps or self.s.num_inference_steps,
                          guidance=guidance or self.s.guidance_scale,
                          controlnet_scale=controlnet_scale or self.s.controlnet_scale)

        say("Ranking candidates by structural fidelity to the sketch")
        with job.timer.stage("4_candidate_ranking"):
            round_id = len(job.candidates)
            for c in cands:
                rgb = np.array(c.image.convert("RGB"))
                mask, method = self.segment_fn(rgb, "auto", self.s.rembg_model)
                metrics = structural_fidelity(job.preprocess.binary, job.preprocess.mask, rgb, mask)
                idx = len(job.candidates)
                path = save_image(rgb, job.dir / f"08_candidate_{idx}_seed{c.seed}.png")
                ov = save_image(fidelity_overlay(job.preprocess.binary, rgb), job.dir / f"09_overlay_{idx}.png")
                job.candidates.append({"index": idx, "seed": c.seed, "round": round_id, "path": str(path),
                                       "overlay": str(ov), "segmentation": method, "image": rgb, "mask": mask,
                                       **metrics})
        new = [c for c in job.candidates if c["round"] == round_id]
        best = max(new, key=lambda c: c["fidelity_score"])
        job.selected = best["index"]
        if best["fidelity_score"] < LOW_FIDELITY:
            job.warn(f"Best candidate only scores {best['fidelity_score']:.2f} structural fidelity - the image may "
                     "have drifted from your sketch. Try 'Regenerate' with a higher ControlNet strength or a clearer prompt.")
        job.note(f"Generated {len(new)} candidate(s); auto-selected #{best['index']} "
                 f"(fidelity {best['fidelity_score']:.2f}); peak VRAM {peak} GB")
        job.save_report()
        return job

    # ================================================================ stages 5-7
    def reconstruct(self, job: Job, candidate_index: Optional[int] = None, status: StatusFn = None) -> Job:
        if not job.candidates:
            raise RuntimeError("generate() must run before reconstruct()")
        say = self._say(job, status)
        idx = job.selected if candidate_index is None else int(candidate_index)
        cand = next((c for c in job.candidates if c["index"] == idx), None)
        if cand is None:
            raise ValueError(f"Candidate {idx} does not exist")
        job.selected = idx

        say("Stage 5/7 - background removal and framing for 3D")
        with job.timer.stage("5_image_preparation"):
            prepared = prepare_for_reconstruction(cand["image"], foreground_ratio=self.s.triposr_foreground_ratio,
                                                  size=512, rembg_model=self.s.rembg_model,
                                                  precomputed_mask=cand["mask"])
        prepared.method = cand.get("segmentation", prepared.method)
        job.prepared = prepared
        job.files.update({k: str(v) for k, v in prepared.save(job.dir).items()})
        for w in prepared.warnings:
            job.warn(w)

        if self._generator is not None and self._low_vram():
            self._generator.park()
            free_memory()

        say(f"Stage 6/7 - image-to-3D reconstruction ({self.reconstructor.name})")
        with job.timer.stage("6_reconstruction"):
            if self.reconstructor.name == "instantmesh":
                recon = self.reconstructor.reconstruct(prepared.recon_rgba, job.dir, status=say)
            else:
                recon = self.reconstructor.reconstruct(prepared.recon_rgb, job.dir, status=say)
        job.recon = recon
        job.files["raw_mesh"] = str(recon.raw_path)
        peak = peak_vram_gb_and_reset()

        say("Stage 7/7 - mesh validation, clean-up and export")
        with job.timer.stage("7_mesh_processing"):
            processed = process_mesh(recon.mesh, job.dir / "export", to_y_up=recon.to_y_up,
                                     remove_floaters=self.s.remove_floaters, floater_ratio=self.s.floater_face_ratio,
                                     export_obj=self.s.export_obj, smooth_iterations=self.s.smooth_iterations,
                                     color_reference=prepared.recon_rgba, enhance_colors=self.s.enhance_colors)
        job.mesh = processed
        job.files["glb"] = str(processed.glb_path)
        if processed.obj_path:
            job.files["obj"] = str(processed.obj_path)
        job.note(f"3D model exported: {processed.report_after['num_vertices']} vertices, "
                 f"{processed.report_after['num_faces']} faces ({processed.report_after.get('visual_kind')} colour); "
                 f"peak VRAM {peak} GB")
        job.save_report()
        return job

    # ================================================================ one-shot
    def run(self, image: ImageLike, user_text: str = "", status: StatusFn = None, seed: int = 42,
            num_candidates: Optional[int] = None, **kwargs) -> Job:
        job = self.analyze(image, user_text, status=status, **kwargs)
        try:
            self.generate(job, num_candidates=num_candidates, seed=seed, status=status)
            self.reconstruct(job, status=status)
        except Exception as exc:
            job.error = f"{exc.__class__.__name__}: {exc}"
            job.save_report()
            raise
        return job
