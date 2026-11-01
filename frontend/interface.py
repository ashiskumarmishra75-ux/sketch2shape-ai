"""Gradio user interface (Stage 1 inputs + Stage 8 viewer/export).

Layout
  A  Input       - drawing canvas (brush / eraser / brush size / undo / clear / download) or upload,
                   optional description + structured fields, advanced generation controls
  B  Analysis    - original, cleaned, binary, Canny, contours, mask, ControlNet map + AI description
  C  Images      - candidate gallery with structural-fidelity scores; select -> rebuild 3D; regenerate
  D  Progress    - live stage status + progress bar; errors shown in red, never a fake model
  E  Output      - interactive 3D viewer (Gradio Model3D + Three.js viewer), GLB/OBJ/report downloads,
                   real measured stage timings
"""
from __future__ import annotations

import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import numpy as np

from computer_vision.image_preparation import rembg_available
from config.settings import get_settings
from frontend.viewer import iframe, write_viewer
from generative_ai.image_generation import GenerationError
from pipeline.job_manager import Job
from pipeline.orchestrator import Sketch2ShapePipeline
from reconstruction.image_to_3d import ReconstructionError
from reconstruction.mesh_processing import MeshValidationError
from utils.file_handler import InputValidationError, validate_upload_file
from utils.gpu import hardware_report
from utils.logging_utils import get_logger

log = get_logger("ui")
S = get_settings()
SAMPLES_DIR = S.project_root / "samples"
USER_ERRORS = (InputValidationError, GenerationError, ReconstructionError, MeshValidationError, ValueError)

CSS = """
#s2s-header h1 { margin-bottom: 0; letter-spacing: -0.02em; }
#s2s-header p { margin-top: 4px; opacity: .8; }
.s2s-steps { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 2px; }
.s2s-steps span { font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid var(--border-color-primary); }
#s2s-status { min-height: 46px; }
.s2s-error { color:#b42318; background:rgba(180,35,24,.08); border-left:3px solid #b42318; padding:8px 12px; border-radius:6px; }
.s2s-ok { color:#05603a; }
#s2s-generate { min-height: 52px; font-size: 1.05em; }
"""

HEADER = """
<div id="s2s-header">
<h1>✏️ → 🧊 Sketch2Shape AI</h1>
<p>Draw or upload a rough sketch, optionally describe it, and get a textured, exportable 3D model.</p>
<div class="s2s-steps"><span>1 · OpenCV conditioning</span><span>2 · VLM sketch understanding</span>
<span>3 · Stable Diffusion + ControlNet</span><span>4 · rembg segmentation</span><span>5 · TripoSR image→3D</span>
<span>6 · trimesh validation</span><span>7 · GLB / OBJ export</span></div>
</div>
"""

pipe = Sketch2ShapePipeline(S)


# ----------------------------------------------------------------------------- helpers


def _status(msg: str, kind: str = "info") -> str:
    if kind == "error":
        return f'<div class="s2s-error"><b>⚠️ {msg}</b></div>'
    if kind == "ok":
        return f'<div class="s2s-ok"><b>✅ {msg}</b></div>'
    return f"⏳ **{msg}**"


def _get_input(src: str, drawing: Any, upload: Any):
    def drawn():
        if drawing is None:
            return None
        comp = drawing.get("composite") if isinstance(drawing, dict) else drawing
        if comp is None:
            return None
        arr = np.asarray(comp)
        if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].max() == 0:
            return None  # untouched transparent canvas
        return arr

    def uploaded():
        if upload is None:
            return None
        if isinstance(upload, (str, Path)):          # plain file path (API / examples)
            validate_upload_file(upload, S.max_upload_mb)
            return upload
        if isinstance(upload, dict) and upload.get("background") is None:
            return None
        comp = upload.get("composite") if isinstance(upload, dict) else upload
        if comp is None:
            return None
        arr = np.asarray(comp)
        if arr.ndim == 3 and arr.shape[2] == 4 and arr[..., 3].max() == 0:
            return None
        if max(arr.shape[:2]) > S.max_input_side:
            raise InputValidationError(f"Image is {arr.shape[1]}x{arr.shape[0]} px; the maximum is {S.max_input_side} px.")
        return arr

    order = ["upload", "draw"] if src == "upload" else ["draw", "upload"]
    for which in order:
        if which == "upload":
            img = uploaded()
            if img is not None:
                return img, "upload"
        if which == "draw":
            arr = drawn()
            if arr is not None:
                return arr, "canvas"
    raise InputValidationError("Please draw a sketch on the canvas or upload a sketch image first.")


def _analysis_gallery(job: Job) -> List:
    f = job.files
    items = [("original", "Original input"), ("normalized", "Cropped & normalised"), ("gray", "Cleaned grayscale"),
             ("binary", "Binary sketch"), ("edges", "Canny edges"), ("contours", "Contours (red = outer)"),
             ("mask", "Foreground mask"), ("control", "ControlNet conditioning map")]
    return [(f[k], cap) for k, cap in items if k in f]


def _ops_md(job: Job) -> str:
    p = job.preprocess
    lines = [f"**Detected input type:** `{p.characteristics.input_type}`", "", "**Operations applied (adaptive):**"]
    lines += [f"{i}. {op}" for i, op in enumerate(p.operations, 1)]
    if job.prompts:
        lines += ["", f"**Positive prompt:** {job.prompts.positive}", "", f"**Negative prompt:** {job.prompts.negative}"]
    return "\n".join(lines)


def _cand_gallery(job: Job) -> List:
    out = []
    for c in job.candidates:
        mark = "★ " if c["index"] == job.selected else ""
        out.append((c["path"], f"{mark}#{c['index']} · seed {c['seed']} · fidelity {c['fidelity_score']:.2f}"))
    return out


def _cand_md(job: Job) -> str:
    rows = ["| # | seed | fidelity | silhouette IoU | stroke recall | edge precision |", "|---|---|---|---|---|---|"]
    for c in job.candidates:
        sel = " ★" if c["index"] == job.selected else ""
        rows.append(f"| {c['index']}{sel} | {c['seed']} | **{c['fidelity_score']:.2f}** | {c['silhouette_iou']:.2f} | "
                    f"{c['stroke_recall']:.2f} | {c['edge_precision']:.2f} |")
    rows.append("\n<sub>Fidelity is a structural diagnostic (edge/silhouette agreement with your sketch), "
                "not a semantic score. Click an image, then **Rebuild 3D from selected**.</sub>")
    return "\n".join(rows)


def _timings_md(job: Job) -> str:
    t = job.timer.as_dict()
    rows = ["| stage | seconds |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in t.items() if k != "total"]
    rows.append(f"| **total** | **{t['total']:.2f}** |")
    if job.mesh is not None:
        r = job.mesh.report_after
        rows += ["", "| mesh check | value |", "|---|---|",
                 f"| vertices / faces | {r['num_vertices']:,} / {r['num_faces']:,} |",
                 f"| bounding box (normalised) | {r['bbox_extents']} |",
                 f"| flatness (min/max extent) | {r['flatness']} |",
                 f"| watertight | {r['is_watertight']} |",
                 f"| winding consistent | {r['is_winding_consistent']} |",
                 f"| connected components | {r['connected_components']} |",
                 f"| colour / material | {r.get('visual_kind')} |",
                 f"| GLB re-load check | {'passed' if job.mesh.export_check.get('readable') else 'FAILED'} "
                 f"({job.mesh.export_check.get('file_size_kb')} KB) |",
                 f"| reconstruction backend | {job.recon.backend} ({'; '.join(job.recon.notes)}) |"]
        if job.mesh.actions:
            rows += ["", "**Clean-up:** " + "; ".join(job.mesh.actions)]
    return "\n".join(rows)


def _final_outputs(job: Job) -> Dict:
    glb = Path(job.files["glb"])
    stats = {"vertices": job.mesh.report_after["num_vertices"], "faces": job.mesh.report_after["num_faces"],
             "visual": job.mesh.report_after.get("visual_kind"), "backend": job.recon.backend}
    viewer_path = write_viewer(glb, job.dir / "export" / "viewer.html",
                               f"Sketch2Shape · {job.description.object}", stats)
    files = [str(glb), str(viewer_path), job.files.get("report", "")]
    if job.files.get("obj"):
        obj_dir = Path(job.files["obj"]).parent
        files.insert(1, shutil.make_archive(str(obj_dir), "zip", obj_dir))
    return {"glb": str(glb), "viewer": iframe(viewer_path.read_text(encoding="utf-8")),
            "files": [f for f in files if f]}


def _err(exc: Exception) -> str:
    if isinstance(exc, USER_ERRORS):
        return str(exc)
    log.error("Unexpected error:\n%s", traceback.format_exc())
    return f"Unexpected {exc.__class__.__name__}: {exc} (full traceback in the server log)"


def _flush_warnings(job: Job, shown: set) -> None:
    for w in job.warnings:
        if w not in shown:
            gr.Warning(w)
            shown.add(w)


# ----------------------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="stone",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"])
    with gr.Blocks(title="Sketch2Shape AI", theme=theme, css=CSS) as demo:
        job_state = gr.State(None)
        src_state = gr.State("draw")
        sel_state = gr.State(None)
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):
            # ============================================================ A: input
            with gr.Column(scale=5, min_width=360):
                with gr.Tabs() as in_tabs:
                    with gr.Tab("✏️ Draw", id="draw_tab") as draw_tab:
                        canvas = gr.ImageEditor(
                            label="Sketch canvas - brush, eraser, size slider, undo & clear in the toolbar",
                            type="numpy", image_mode="RGBA", sources=(), transforms=(), layers=False,
                            canvas_size=(512, 512), fixed_canvas=True, format="png", height=520,
                            brush=gr.Brush(colors=["#111111"], color_mode="fixed", default_size=3),
                            eraser=gr.Eraser(default_size=14), show_download_button=True)
                    with gr.Tab("📤 Upload", id="upload_tab") as upload_tab:
                        upload = gr.ImageEditor(
                            label="Sketch image (PNG / JPG / JPEG) - crop ✂ or erase background clutter before generating",
                            type="numpy", image_mode="RGBA", sources=("upload", "clipboard"), transforms=("crop",),
                            layers=False, brush=False, eraser=gr.Eraser(default_size=24), format="png", height=460,
                            show_download_button=True)
                text = gr.Textbox(label="Describe the object (optional)", lines=2,
                                  placeholder="e.g. A modern wooden chair with four legs, a curved backrest, dark brown polished wood")
                with gr.Accordion("Object details (optional)", open=False):
                    with gr.Row():
                        category = gr.Textbox(label="Category", placeholder="chair")
                        material = gr.Textbox(label="Material", placeholder="oak wood")
                    with gr.Row():
                        color = gr.Textbox(label="Colour", placeholder="dark brown")
                        style = gr.Textbox(label="Style", placeholder="mid-century modern")
                with gr.Accordion("Advanced generation settings", open=False):
                    control_type = gr.Radio(["scribble", "lineart", "canny"], value=S.control_type,
                                            label="ControlNet conditioning",
                                            info="scribble = tolerant to rough strokes · lineart = clean drawings · canny = photos")
                    with gr.Row():
                        n_cand = gr.Slider(1, 4, value=S.num_candidates, step=1, label="Candidates")
                        steps = gr.Slider(10, 50, value=S.num_inference_steps, step=1, label="Diffusion steps")
                    with gr.Row():
                        cfg = gr.Slider(2.0, 12.0, value=S.guidance_scale, step=0.5, label="Text guidance (CFG)")
                        cn_scale = gr.Slider(0.3, 1.6, value=S.controlnet_scale, step=0.05,
                                             label="ControlNet strength (sketch fidelity)")
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        randomize = gr.Checkbox(value=True, label="Random seed each run")
                with gr.Row():
                    gr.ClearButton(value="Clear", components=[canvas, upload, text])
                    run_btn = gr.Button("Generate 3D model", variant="primary", elem_id="s2s-generate", scale=3)
                samples = sorted(SAMPLES_DIR.glob("*.png"))
                if samples:
                    gr.Examples(examples=[[str(p)] for p in samples], inputs=[upload],
                                label="Test sketches (click to load into Upload)")

            # ============================================================ B-E: outputs
            with gr.Column(scale=7, min_width=420):
                status = gr.Markdown("Ready. Draw or upload a sketch, then press **Generate 3D model**.",
                                     elem_id="s2s-status")
                with gr.Tabs() as out_tabs:
                    with gr.Tab("🧊 3D model", id="model_tab"):
                        with gr.Tabs():
                            with gr.Tab("Three.js viewer"):
                                viewer = gr.HTML("<div style='height:520px;display:flex;align-items:center;"
                                                 "justify-content:center;opacity:.6'>The 3D model will appear here.</div>")
                            with gr.Tab("Model3D viewer"):
                                model3d = gr.Model3D(label="Drag to rotate · scroll to zoom", height=520,
                                                     clear_color=(0.96, 0.95, 0.92, 1.0))
                        files = gr.File(label="Downloads: model.glb · OBJ (zip) · standalone viewer.html · report.json",
                                        file_count="multiple", interactive=False)
                        timings = gr.Markdown()
                    with gr.Tab("🔍 Sketch analysis", id="analysis_tab"):
                        analysis = gr.Gallery(label="Computer-vision intermediates", columns=4, height=420,
                                              format="png", object_fit="contain")
                        description = gr.JSON(label="AI sketch understanding")
                        ops = gr.Markdown()
                    with gr.Tab("🎨 Generated images", id="images_tab"):
                        cands = gr.Gallery(label="ControlNet candidates (★ = used for 3D)", columns=4, height=300,
                                           format="png", object_fit="contain", allow_preview=True)
                        cand_md = gr.Markdown()
                        prompt_edit = gr.Textbox(label="Prompt used (edit, then Regenerate)", lines=3)
                        with gr.Row():
                            rebuild_btn = gr.Button("Rebuild 3D from selected image")
                            regen_btn = gr.Button("Regenerate images + 3D (new seed)", variant="secondary")
                with gr.Accordion("System / hardware", open=False):
                    gr.JSON(value={**hardware_report(), "recon_backend": S.recon_backend,
                                   "diffusion": f"{S.base_model_id()} + {S.controlnet_id()}",
                                   "rembg_available": rembg_available()})

        all_out = [status, analysis, description, ops, cands, cand_md, prompt_edit, model3d, viewer, files,
                   timings, job_state, seed, out_tabs]

        # ------------------------------------------------------------ handlers
        def _stages_4_to_7(job: Job, n, seed_val, steps_v, cfg_v, cn_v, prompt_override, progress, shown):
            def prog(done, total):
                progress(0.25 + 0.45 * done / max(total, 1), desc=f"Diffusion step {done}/{total}")
            pipe.generate(job, num_candidates=int(n), seed=int(seed_val), steps=int(steps_v), guidance=float(cfg_v),
                          controlnet_scale=float(cn_v), positive_override=prompt_override or "", progress=prog)
            _flush_warnings(job, shown)
            yield {status: _status("Images generated - building the 3D model…"), cands: _cand_gallery(job),
                   cand_md: _cand_md(job), prompt_edit: job.prompts.positive, out_tabs: gr.Tabs(selected="images_tab")}
            progress(0.75, desc="Background removal + TripoSR reconstruction")
            pipe.reconstruct(job, status=lambda m: progress(0.8, desc=m))
            _flush_warnings(job, shown)
            fin = _final_outputs(job)
            progress(1.0, desc="Done")
            yield {status: _status(f"Done in {job.timer.total:.1f}s - {job.mesh.report_after['num_faces']:,} faces, "
                                   f"{job.mesh.report_after.get('visual_kind')} colours. Inspect it below or download the GLB.", "ok"),
                   cands: _cand_gallery(job), cand_md: _cand_md(job), model3d: fin["glb"], viewer: fin["viewer"],
                   files: fin["files"], timings: _timings_md(job), out_tabs: gr.Tabs(selected="model_tab")}

        def run_full(src, drawing, up, txt, cat, mat, col, sty, ctype, n, seed_val, rnd, steps_v, cfg_v, cn_v,
                     progress=gr.Progress()):
            shown: set = set()
            job = None
            try:
                img, used = _get_input(src, drawing, up)
                seed_val = random.randint(0, 2 ** 31 - 1) if rnd else int(seed_val or 0)
                progress(0.05, desc="Preprocessing sketch")
                yield {status: _status("Analysing your sketch (OpenCV + VLM)…"), seed: seed_val, model3d: None,
                       files: None, cands: None, cand_md: "", timings: ""}
                job = pipe.analyze(img, txt or "", category=cat or "", material=mat or "", color=col or "",
                                   style=sty or "", control_type=ctype)
                job.inputs["source"] = used
                _flush_warnings(job, shown)
                yield {status: _status("Sketch analysed - generating images with ControlNet…"), job_state: job.id,
                       analysis: _analysis_gallery(job), description: job.description.to_dict(), ops: _ops_md(job),
                       prompt_edit: job.prompts.positive}
                yield from _stages_4_to_7(job, n, seed_val, steps_v, cfg_v, cn_v, "", progress, shown)
            except Exception as exc:
                if job is not None:
                    job.error = _err(exc)
                    job.save_report()
                yield {status: _status(_err(exc), "error")}

        def rebuild(job_id, sel, progress=gr.Progress()):
            job = pipe.jobs.get(job_id)
            if job is None or not job.candidates:
                yield {status: _status("Generate a model first, then pick a candidate image.", "error")}
                return
            shown: set = set()
            try:
                idx = job.candidates[int(sel)]["index"] if sel is not None else job.selected
                progress(0.3, desc=f"Rebuilding from candidate #{idx}")
                yield {status: _status(f"Rebuilding the 3D model from candidate #{idx}…")}
                pipe.reconstruct(job, candidate_index=idx, status=lambda m: progress(0.6, desc=m))
                _flush_warnings(job, shown)
                fin = _final_outputs(job)
                yield {status: _status(f"Rebuilt from candidate #{idx}.", "ok"), cands: _cand_gallery(job),
                       cand_md: _cand_md(job), model3d: fin["glb"], viewer: fin["viewer"], files: fin["files"],
                       timings: _timings_md(job), out_tabs: gr.Tabs(selected="model_tab")}
            except Exception as exc:
                yield {status: _status(_err(exc), "error")}

        def regenerate(job_id, n, steps_v, cfg_v, cn_v, prompt_override, progress=gr.Progress()):
            job = pipe.jobs.get(job_id)
            if job is None:
                yield {status: _status("Generate a model first.", "error")}
                return
            new_seed = random.randint(0, 2 ** 31 - 1)
            try:
                yield {status: _status(f"Regenerating with seed {new_seed}…"), seed: new_seed}
                yield from _stages_4_to_7(job, n, new_seed, steps_v, cfg_v, cn_v, prompt_override, progress, set())
            except Exception as exc:
                yield {status: _status(_err(exc), "error")}

        def on_select(evt: gr.SelectData):
            return evt.index

        draw_tab.select(lambda: "draw", outputs=src_state)
        upload_tab.select(lambda: "upload", outputs=src_state)
        def _on_upload(v):
            has = v is not None and (not isinstance(v, dict) or v.get("background") is not None)
            return ("upload", gr.Tabs(selected="upload_tab")) if has else ("draw", gr.Tabs())

        upload.upload(_on_upload, inputs=upload, outputs=[src_state, in_tabs])
        upload.change(_on_upload, inputs=upload, outputs=[src_state, in_tabs])
        cands.select(on_select, outputs=sel_state)
        run_btn.click(run_full, inputs=[src_state, canvas, upload, text, category, material, color, style,
                                        control_type, n_cand, seed, randomize, steps, cfg, cn_scale],
                      outputs=all_out, concurrency_limit=1)
        rebuild_btn.click(rebuild, inputs=[job_state, sel_state], outputs=all_out, concurrency_limit=1)
        regen_btn.click(regenerate, inputs=[job_state, n_cand, steps, cfg, cn_scale, prompt_edit],
                        outputs=all_out, concurrency_limit=1)
    return demo
