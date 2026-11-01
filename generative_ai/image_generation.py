"""Stage 4 - Sketch-guided image generation with Stable Diffusion + ControlNet (Hugging Face Diffusers).

Inputs:  the OpenCV structural map (ControlNet condition) + positive/negative prompt
Output:  N candidate product-style images, spatially aligned with the sketch.

Default: DreamShaper-8 (an SD-1.5 fine-tune with clean product renders) + ControlNet v1.1
scribble, fp16, DPM++ 2M Karras, 25 steps at 512 px. `S2S_GEN_FAMILY=sdxl` switches to
SDXL + xinsir ControlNet (better images, needs ~12 GB+ VRAM, 1024 px).

The weights are loaded once and cached in-process. On GPUs below the low-VRAM threshold the
pipeline uses diffusers' model-CPU-offload so it can share the GPU with the 3D model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from utils.gpu import free_memory, is_oom_error, resolve_device, vram_gb
from utils.logging_utils import get_logger

log = get_logger("image_generation")
ProgressFn = Optional[Callable[[int, int], None]]


class GenerationError(RuntimeError):
    """User-facing diffusion failure (missing weights, OOM after retries, ...)."""


@dataclass
class Candidate:
    image: Image.Image
    seed: int
    index: int


class ControlNetGenerator:
    def __init__(self, settings):
        self.s = settings
        self.device = resolve_device(settings.device)
        self.pipe = None
        self.img2img_pipe = None
        self.loaded_key: Optional[tuple] = None
        self.offloaded = False

    # ------------------------------------------------------------------ loading
    def _use_offload(self) -> bool:
        if self.device != "cuda":
            return False
        if self.s.offload_mode == "always":
            return True
        if self.s.offload_mode == "never":
            return False
        return vram_gb() < self.s.low_vram_threshold_gb

    def load(self, control_type: Optional[str] = None):
        control_type = control_type or self.s.control_type
        key = (self.s.gen_family, self.s.base_model_id(), self.s.controlnet_id(control_type))
        if self.pipe is not None and self.loaded_key == key:
            return self.pipe
        self.unload()
        try:
            import torch
            from diffusers import ControlNetModel, DPMSolverMultistepScheduler
        except ImportError as exc:
            raise GenerationError(f"Diffusers/PyTorch not installed: {exc}. Run `pip install -r requirements.txt`.") from exc

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        t0 = time.perf_counter()
        log.info("Loading %s + ControlNet %s (%s, %s)", key[1], key[2], self.device, dtype)
        try:
            controlnet = ControlNetModel.from_pretrained(key[2], torch_dtype=dtype)
            if self.s.gen_family == "sdxl":
                from diffusers import AutoencoderKL, StableDiffusionXLControlNetPipeline
                vae = AutoencoderKL.from_pretrained(self.s.sdxl_vae_id, torch_dtype=dtype)
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    key[1], controlnet=controlnet, vae=vae, torch_dtype=dtype)
            else:
                from diffusers import StableDiffusionControlNetPipeline
                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    key[1], controlnet=controlnet, torch_dtype=dtype,
                    safety_checker=None, requires_safety_checker=False)
        except OSError as exc:
            raise GenerationError(
                f"Could not load diffusion weights ({exc.__class__.__name__}). Check internet access to "
                f"huggingface.co or pre-download with `python scripts/download_models.py`. Detail: {exc}") from exc

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        pipe.set_progress_bar_config(disable=True)
        if self._use_offload():
            pipe.enable_model_cpu_offload()
            self.offloaded = True
            log.info("Low-VRAM mode: model CPU offload enabled")
        else:
            pipe.to(self.device)
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()  # decode candidates one at a time (lower peak VRAM)
        self.pipe, self.loaded_key = pipe, key
        log.info("Diffusion pipeline ready in %.1fs", time.perf_counter() - t0)
        return pipe

    def _img2img(self):
        """ControlNet image-to-image pipeline sharing the loaded weights (no extra VRAM)."""
        if self.img2img_pipe is None:
            if self.s.gen_family == "sdxl":
                from diffusers import StableDiffusionXLControlNetImg2ImgPipeline as Cls
            else:
                from diffusers import StableDiffusionControlNetImg2ImgPipeline as Cls
            self.img2img_pipe = Cls.from_pipe(self.pipe)
            self.img2img_pipe.set_progress_bar_config(disable=True)
            if self.offloaded:
                self.img2img_pipe.enable_model_cpu_offload()
        return self.img2img_pipe

    def unload(self) -> None:
        self.img2img_pipe = None
        if self.pipe is not None:
            del self.pipe
            self.pipe, self.loaded_key = None, None
            free_memory()

    def park(self) -> None:
        """Move weights to CPU RAM so the 3D model can use the GPU (no reload from disk next time)."""
        if self.pipe is not None and not self.offloaded and self.device == "cuda":
            self.pipe.to("cpu")
            free_memory()

    def unpark(self) -> None:
        if self.pipe is not None and not self.offloaded and self.device == "cuda":
            self.pipe.to("cuda")

    # ------------------------------------------------------------------ inference
    def generate(self, control_image: Image.Image, positive: str, negative: str, *, num_candidates: int = 2,
                 seed: int = 0, steps: Optional[int] = None, guidance: Optional[float] = None,
                 controlnet_scale: Optional[float] = None, control_end: Optional[float] = None,
                 control_type: Optional[str] = None, progress: ProgressFn = None,
                 init_image: Optional[Image.Image] = None, strength: float = 0.75) -> List[Candidate]:
        """init_image: optional colour picture to start from (image-to-image). Its colours and shading carry
        into the result, which ControlNet alone discards; `strength` = how much the image may change (0-1)."""
        import torch
        self.load(control_type)
        self.unpark()
        pipe = self._img2img() if init_image is not None else self.pipe
        res = self.s.gen_resolution if self.s.gen_family == "sd15" else max(self.s.gen_resolution, 1024)
        cond = control_image.convert("RGB").resize((res, res), Image.BICUBIC)
        steps = steps or self.s.num_inference_steps
        kwargs = dict(
            prompt=positive, negative_prompt=negative, height=res, width=res,
            num_inference_steps=steps, guidance_scale=guidance if guidance is not None else self.s.guidance_scale,
            controlnet_conditioning_scale=controlnet_scale if controlnet_scale is not None else self.s.controlnet_scale,
            control_guidance_end=control_end if control_end is not None else self.s.control_guidance_end,
        )

        seeds = [int(seed) + i for i in range(max(1, num_candidates))]
        # a batched call fires the callback once per step for the whole batch
        if init_image is not None:
            kwargs.update(image=init_image.convert("RGB").resize((res, res), Image.BICUBIC), control_image=cond,
                          strength=float(strength))
            steps_run = max(1, int(steps * float(strength)))   # img2img only runs the last `strength` share
            log.info("Image-to-image from the coloured input (strength %.2f)", strength)
        else:
            kwargs.update(image=cond)
            steps_run = steps
        done = {"n": 0, "total": steps_run}

        t_start = time.perf_counter()

        def _cb(_pipe, _step, _t, cb_kwargs):
            done["n"] += 1
            if done["n"] % 5 == 0 or done["n"] == done["total"]:
                log.info("Diffusion step %d/%d (%.1fs)", done["n"], done["total"], time.perf_counter() - t_start)
            if progress:
                progress(min(done["n"], done["total"]), done["total"])
            return cb_kwargs

        def _run(batch_seeds: List[int]) -> List[Image.Image]:
            gens = [torch.Generator("cpu").manual_seed(s) for s in batch_seeds]
            with torch.inference_mode():
                out = pipe(**kwargs, num_images_per_prompt=len(batch_seeds), generator=gens, callback_on_step_end=_cb)
            return list(out.images)

        images: List[Image.Image] = []
        try:
            try:
                images = _run(seeds)
            except Exception as exc:
                if not is_oom_error(exc) or len(seeds) == 1:
                    raise
                log.warning("OOM with batch %d - retrying one image at a time", len(seeds))
                free_memory()
                done.update(n=0, total=steps_run * len(seeds))  # now one image per call
                for s in seeds:
                    images += _run([s])
        except Exception as exc:
            free_memory()
            if is_oom_error(exc):
                raise GenerationError("GPU ran out of memory during image generation. Lower S2S_NUM_CANDIDATES, "
                                      "set S2S_OFFLOAD=always or use S2S_GEN_FAMILY=sd15.") from exc
            raise GenerationError(f"Image generation failed: {exc}") from exc

        cands = []
        for i, (img, s) in enumerate(zip(images, seeds)):
            if np.asarray(img).mean() < 3:
                log.warning("Candidate %d is black - skipped", i)
                continue
            cands.append(Candidate(image=img, seed=s, index=i))
        if not cands:
            raise GenerationError("The diffusion model returned only empty images. Try another seed.")
        return cands
