"""Stage 6 - Image-to-3D reconstruction.

Primary backend: **TripoSR** (Stability AI x Tripo, MIT licence, `stabilityai/TripoSR`).
  * single image -> triplane NeRF -> marching cubes mesh with per-vertex colour
  * optional UV texture baking (xatlas + moderngl) via TripoSR's own `bake_texture`
  * ~6 GB VRAM at the default marching-cubes resolution (per TripoSR README), seconds on a GPU
  * its only heavy deps (transformers ViT, omegaconf, einops) coexist with our diffusers stack

Optional backend: **InstantMesh** (TencentARC, Apache-2.0). Higher-quality textured meshes, but it
pins diffusers==0.20.2 / transformers==4.34.1 and needs nvdiffrast, which conflict with this
project's diffusers version. It is therefore run as a *subprocess in its own virtualenv*
(S2S_INSTANTMESH_PYTHON) through its official `run.py` - never imported into this process.

Both backends return a `ReconResult` whose `to_y_up` matrix converts the model's native
frame to glTF's Y-up convention; mesh_processing applies it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from utils.gpu import free_memory, is_oom_error, resolve_device
from utils.logging_utils import get_logger

log = get_logger("image_to_3d")
StatusFn = Optional[Callable[[str], None]]


class ReconstructionError(RuntimeError):
    """User-facing reconstruction failure."""


@dataclass
class ReconResult:
    mesh: object                         # trimesh.Trimesh (native frame of the model)
    raw_path: Path                       # untouched model output, preserved for inspection
    backend: str
    textured: bool                       # True = UV texture map, False = per-vertex colour
    to_y_up: np.ndarray = field(default_factory=lambda: np.eye(4))
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _rot(angle: float, axis) -> np.ndarray:
    import trimesh
    return trimesh.transformations.rotation_matrix(angle, axis)


# =========================================================================== TripoSR


class TripoSRBackend:
    name = "triposr"

    def __init__(self, settings):
        self.s = settings
        self.device = resolve_device(settings.device)
        self.model = None
        self.mc_impl = "unknown"

    def _import_tsr(self):
        repo = Path(self.s.triposr_repo_dir)
        if not (repo / "tsr" / "system.py").exists():
            raise ReconstructionError(
                f"TripoSR source not found at {repo}. Run `bash scripts/setup.sh` (or "
                f"`git clone https://github.com/VAST-AI-Research/TripoSR {repo}`).")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from reconstruction.mc_shim import ensure_marching_cubes
        self.mc_impl = ensure_marching_cubes()
        from tsr.system import TSR  # noqa: E402
        return TSR

    def load(self):
        if self.model is not None:
            return self.model
        TSR = self._import_tsr()
        t0 = time.perf_counter()
        src = self.s.triposr_model_id
        log.info("Loading TripoSR weights from %s (marching cubes: %s)", src, self.mc_impl)
        try:
            model = TSR.from_pretrained(src, config_name="config.yaml", weight_name="model.ckpt")
        except Exception as exc:
            raise ReconstructionError(
                f"Could not load TripoSR weights from '{src}': {exc}. Check access to huggingface.co or "
                f"point S2S_TRIPOSR_MODEL to a local folder with config.yaml + model.ckpt.") from exc
        model.renderer.set_chunk_size(self.s.triposr_chunk_size)
        model.to(self.device)
        model.eval()
        self.model = model
        log.info("TripoSR ready in %.1fs on %s", time.perf_counter() - t0, self.device)
        return model

    def park(self) -> None:
        if self.model is not None and self.device == "cuda":
            self.model.to("cpu")
            free_memory()

    def _bake(self, mesh, scene_code):
        """UV-unwrap + bake the triplane colour field into a texture (TripoSR's own implementation)."""
        import trimesh
        from tsr.bake_texture import bake_texture  # needs xatlas + moderngl (OpenGL context)
        out = bake_texture(mesh, self.model, scene_code, self.s.texture_resolution)
        tex = Image.fromarray((np.clip(out["colors"], 0, 1) * 255).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
        verts = mesh.vertices[out["vmapping"]]
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9)
        visual = trimesh.visual.TextureVisuals(uv=out["uvs"], material=material)
        return trimesh.Trimesh(vertices=verts, faces=out["indices"].astype(np.int64), visual=visual, process=False)

    def reconstruct(self, recon_rgb: np.ndarray, out_dir: Path, status: StatusFn = None) -> ReconResult:
        import torch
        model = self.load()
        if self.device == "cuda":
            model.to("cuda")
        out_dir = Path(out_dir)
        image = Image.fromarray(recon_rgb[..., :3])
        timings: Dict[str, float] = {}
        notes: List[str] = [f"marching cubes: {self.mc_impl}"]
        mc_res = self.s.triposr_mc_resolution
        bake = self.s.bake_texture

        for attempt in range(3):
            try:
                if status:
                    status(f"TripoSR: encoding image -> triplane (attempt {attempt + 1})")
                t0 = time.perf_counter()
                with torch.no_grad():
                    scene_codes = model([image], device=self.device)
                timings["triposr_forward"] = round(time.perf_counter() - t0, 3)
                if status:
                    status(f"TripoSR: extracting mesh (marching cubes {mc_res}^3)")
                t0 = time.perf_counter()
                mesh = model.extract_mesh(scene_codes, True, resolution=mc_res)[0]
                timings["triposr_mesh_extraction"] = round(time.perf_counter() - t0, 3)
                break
            except Exception as exc:
                if is_oom_error(exc) and attempt < 2:
                    free_memory()
                    mc_res = max(128, int(mc_res * 0.75))
                    model.renderer.set_chunk_size(max(1024, self.s.triposr_chunk_size // (2 ** (attempt + 1))))
                    notes.append(f"OOM -> retry with marching-cubes resolution {mc_res}")
                    log.warning("TripoSR OOM, retrying at mc resolution %d", mc_res)
                    continue
                free_memory()
                if is_oom_error(exc):
                    raise ReconstructionError("GPU ran out of memory in TripoSR even at reduced resolution. "
                                              "Set S2S_OFFLOAD=always or S2S_MC_RES=160.") from exc
                raise ReconstructionError(f"TripoSR inference failed: {exc}") from exc

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ReconstructionError("TripoSR produced an empty mesh (the object may be too thin or the "
                                      "background removal failed). Try regenerating the image.")

        textured = False
        if bake:
            try:
                if status:
                    status("TripoSR: baking UV texture")
                t0 = time.perf_counter()
                mesh = self._bake(mesh, scene_codes[0])
                textured = True
                timings["texture_bake"] = round(time.perf_counter() - t0, 3)
            except Exception as exc:  # headless servers often lack an OpenGL context for moderngl
                notes.append(f"texture baking unavailable ({exc.__class__.__name__}: {exc}); kept vertex colours")
                log.warning("Texture bake failed, keeping vertex colours: %s", exc)

        raw_path = out_dir / "20_raw_reconstruction.glb"
        mesh.export(raw_path)
        # TripoSR's own viewer transform (tsr.utils.to_gradio_3d_orientation): Rx(-90deg) then Ry(+90deg)
        to_y_up = _rot(np.pi / 2, [0, 1, 0]) @ _rot(-np.pi / 2, [1, 0, 0])
        notes.append(f"marching-cubes resolution {mc_res}")
        return ReconResult(mesh=mesh, raw_path=raw_path, backend=self.name, textured=textured,
                           to_y_up=to_y_up, timings=timings, notes=notes)


# =========================================================================== InstantMesh


class InstantMeshBackend:
    """Runs the official InstantMesh `run.py` in a separate Python environment."""

    name = "instantmesh"

    def __init__(self, settings):
        self.s = settings

    def load(self):
        repo = Path(self.s.instantmesh_repo_dir)
        if not (repo / "run.py").exists():
            raise ReconstructionError(f"InstantMesh not found at {repo}. See README > Optional InstantMesh backend.")
        return None

    def park(self) -> None:  # separate process: nothing resident here
        return None

    def reconstruct(self, recon_rgba: np.ndarray, out_dir: Path, status: StatusFn = None, seed: int = 42) -> ReconResult:
        import trimesh
        self.load()
        repo = Path(self.s.instantmesh_repo_dir).resolve()
        out_dir = Path(out_dir).resolve()
        work = out_dir / "instantmesh_run"
        work.mkdir(parents=True, exist_ok=True)
        inp = work / "input.png"
        Image.fromarray(recon_rgba).save(inp)
        config = self.s.instantmesh_config
        cmd = [self.s.instantmesh_python, "run.py", config, str(inp), "--output_path", str(work),
               "--no_rembg", "--export_texmap", "--seed", str(seed)]
        if status:
            status("InstantMesh: multi-view diffusion + sparse-view reconstruction (separate process)")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=self.s.instantmesh_timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ReconstructionError(f"InstantMesh timed out after {self.s.instantmesh_timeout_s}s") from exc
        (work / "instantmesh_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            raise ReconstructionError("InstantMesh failed:\n" + "\n".join(tail))
        name = Path(config).stem
        obj = work / name / "meshes" / "input.obj"
        if not obj.exists():
            raise ReconstructionError(f"InstantMesh finished but {obj} was not written.")
        raw_dir = out_dir / "20_raw_reconstruction_instantmesh"
        shutil.copytree(obj.parent, raw_dir, dirs_exist_ok=True)
        mesh = trimesh.load(raw_dir / "input.obj", force="mesh", process=False)
        # InstantMesh's own GLB export uses vertices[:, [1, 2, 0]] -> equivalent rotation matrix:
        to_y_up = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
        textured = isinstance(mesh.visual, trimesh.visual.TextureVisuals)
        return ReconResult(mesh=mesh, raw_path=raw_dir / "input.obj", backend=self.name, textured=textured,
                           to_y_up=to_y_up, timings={"instantmesh_total": round(time.perf_counter() - t0, 3)},
                           notes=["ran in separate venv via official run.py"])


def get_backend(settings, name: Optional[str] = None):
    name = (name or settings.recon_backend).lower()
    if name == "triposr":
        return TripoSRBackend(settings)
    if name == "instantmesh":
        return InstantMeshBackend(settings)
    raise ValueError(f"Unknown reconstruction backend '{name}' (use 'triposr' or 'instantmesh')")
