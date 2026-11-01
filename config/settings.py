"""Central configuration for Sketch2Shape AI.

Every tunable value can be overridden with an environment variable (or a `.env`
file in the project root). Nothing in the pipeline hard-codes a model id, path
or inference setting - they all come from here.
"""
from __future__ import annotations

import os

# Colab pre-installs TensorFlow and JAX; transformers/diffusers import them if present, which costs
# >1 GB of RAM in this process for nothing. Must run before any transformers/diffusers import.
for _k, _v in {"USE_TF": "0", "USE_FLAX": "0", "USE_JAX": "0", "TF_CPP_MIN_LOG_LEVEL": "3",
               "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(_k, _v)
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a hard dependency on python-dotenv)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].split("\t#", 1)[0]  # strip inline comments
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)  # real env vars win over .env


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---------------------------------------------------------------- paths
    project_root: Path = PROJECT_ROOT
    outputs_dir: Path = field(default_factory=lambda: Path(_env("S2S_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))))
    uploads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    third_party_dir: Path = field(default_factory=lambda: Path(_env("S2S_THIRD_PARTY_DIR", str(PROJECT_ROOT / "third_party"))))

    # ---------------------------------------------------------------- device
    device: str = field(default_factory=lambda: _env("S2S_DEVICE", "auto"))  # auto | cuda | cpu
    # "auto" -> offload models between stages when VRAM < low_vram_threshold_gb
    offload_mode: str = field(default_factory=lambda: _env("S2S_OFFLOAD", "auto"))  # auto | always | never
    low_vram_threshold_gb: float = field(default_factory=lambda: _env_float("S2S_LOW_VRAM_GB", 14.0))

    # ---------------------------------------------------------------- input validation
    max_upload_mb: float = field(default_factory=lambda: _env_float("S2S_MAX_UPLOAD_MB", 10.0))
    min_input_side: int = field(default_factory=lambda: _env_int("S2S_MIN_INPUT_SIDE", 64))
    max_input_side: int = field(default_factory=lambda: _env_int("S2S_MAX_INPUT_SIDE", 4096))
    work_resolution: int = field(default_factory=lambda: _env_int("S2S_WORK_RES", 512))

    # ---------------------------------------------------------------- sketch understanding (VLM)
    # auto -> anthropic if ANTHROPIC_API_KEY, else openai-compatible if OPENAI_API_KEY,
    #         else local CLIP zero-shot + keyword parsing (always works offline)
    vlm_backend: str = field(default_factory=lambda: _env("S2S_VLM_BACKEND", "auto"))  # auto|anthropic|openai|clip|none
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    clip_model_id: str = field(default_factory=lambda: _env("S2S_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    vlm_timeout_s: float = field(default_factory=lambda: _env_float("S2S_VLM_TIMEOUT", 45.0))

    # ---------------------------------------------------------------- diffusion + ControlNet
    gen_family: str = field(default_factory=lambda: _env("S2S_GEN_FAMILY", "sd15"))  # sd15 | sdxl
    sd15_model_id: str = field(default_factory=lambda: _env("S2S_SD15_MODEL", "Lykon/dreamshaper-8"))
    sd15_controlnets: dict = field(default_factory=lambda: {
        "scribble": _env("S2S_CN_SCRIBBLE", "lllyasviel/control_v11p_sd15_scribble"),
        "lineart": _env("S2S_CN_LINEART", "lllyasviel/control_v11p_sd15_lineart"),
        "canny": _env("S2S_CN_CANNY", "lllyasviel/control_v11p_sd15_canny"),
    })
    sdxl_model_id: str = field(default_factory=lambda: _env("S2S_SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"))
    sdxl_controlnets: dict = field(default_factory=lambda: {
        "scribble": _env("S2S_SDXL_CN_SCRIBBLE", "xinsir/controlnet-scribble-sdxl-1.0"),
        # no official SDXL lineart ControlNet from xinsir; scribble handles clean line art well
        "lineart": _env("S2S_SDXL_CN_LINEART", "xinsir/controlnet-scribble-sdxl-1.0"),
        "canny": _env("S2S_SDXL_CN_CANNY", "xinsir/controlnet-canny-sdxl-1.0"),
    })
    sdxl_vae_id: str = field(default_factory=lambda: _env("S2S_SDXL_VAE", "madebyollin/sdxl-vae-fp16-fix"))
    control_type: str = field(default_factory=lambda: _env("S2S_CONTROL_TYPE", "scribble"))  # scribble|lineart|canny
    num_candidates: int = field(default_factory=lambda: _env_int("S2S_NUM_CANDIDATES", 2))
    num_inference_steps: int = field(default_factory=lambda: _env_int("S2S_STEPS", 25))
    guidance_scale: float = field(default_factory=lambda: _env_float("S2S_CFG", 7.0))
    controlnet_scale: float = field(default_factory=lambda: _env_float("S2S_CN_SCALE", 0.9))
    control_guidance_end: float = field(default_factory=lambda: _env_float("S2S_CN_END", 0.85))
    img2img_strength: float = field(default_factory=lambda: _env_float("S2S_IMG2IMG_STRENGTH", 0.75))
    gen_resolution: int = field(default_factory=lambda: _env_int("S2S_GEN_RES", 512))  # 1024 recommended for sdxl

    # ---------------------------------------------------------------- background removal
    rembg_model: str = field(default_factory=lambda: _env("S2S_REMBG_MODEL", "isnet-general-use"))

    # ---------------------------------------------------------------- image-to-3D
    recon_backend: str = field(default_factory=lambda: _env("S2S_RECON_BACKEND", "triposr"))  # triposr | instantmesh
    triposr_repo_dir: Path = field(default_factory=lambda: Path(_env("S2S_TRIPOSR_DIR", str(PROJECT_ROOT / "third_party" / "TripoSR"))))
    triposr_model_id: str = field(default_factory=lambda: _env("S2S_TRIPOSR_MODEL", "stabilityai/TripoSR"))
    triposr_mc_resolution: int = field(default_factory=lambda: _env_int("S2S_MC_RES", 256))
    triposr_chunk_size: int = field(default_factory=lambda: _env_int("S2S_CHUNK", 8192))
    triposr_foreground_ratio: float = field(default_factory=lambda: _env_float("S2S_FG_RATIO", 0.85))
    bake_texture: bool = field(default_factory=lambda: _env_bool("S2S_BAKE_TEXTURE", False))
    texture_resolution: int = field(default_factory=lambda: _env_int("S2S_TEX_RES", 1024))
    # InstantMesh runs in its OWN virtualenv (it pins diffusers==0.20.2 / transformers==4.34.1)
    instantmesh_repo_dir: Path = field(default_factory=lambda: Path(_env("S2S_INSTANTMESH_DIR", str(PROJECT_ROOT / "third_party" / "InstantMesh"))))
    instantmesh_python: str = field(default_factory=lambda: _env("S2S_INSTANTMESH_PYTHON", "python"))
    instantmesh_config: str = field(default_factory=lambda: _env("S2S_INSTANTMESH_CONFIG", "configs/instant-mesh-large.yaml"))
    instantmesh_timeout_s: int = field(default_factory=lambda: _env_int("S2S_INSTANTMESH_TIMEOUT", 900))

    # ---------------------------------------------------------------- mesh post-processing
    remove_floaters: bool = field(default_factory=lambda: _env_bool("S2S_REMOVE_FLOATERS", True))
    floater_face_ratio: float = field(default_factory=lambda: _env_float("S2S_FLOATER_RATIO", 0.02))
    export_obj: bool = field(default_factory=lambda: _env_bool("S2S_EXPORT_OBJ", True))
    smooth_iterations: int = field(default_factory=lambda: _env_int("S2S_SMOOTH_ITERS", 8))
    enhance_colors: bool = field(default_factory=lambda: _env_bool("S2S_ENHANCE_COLORS", True))

    # ---------------------------------------------------------------- server
    server_name: str = field(default_factory=lambda: _env("S2S_HOST", "0.0.0.0"))
    server_port: int = field(default_factory=lambda: _env_int("S2S_PORT", 7860))
    share: bool = field(default_factory=lambda: _env_bool("S2S_SHARE", False))
    keep_jobs: int = field(default_factory=lambda: _env_int("S2S_KEEP_JOBS", 50))

    def controlnet_id(self, control_type: str | None = None) -> str:
        ct = (control_type or self.control_type).lower()
        table = self.sdxl_controlnets if self.gen_family == "sdxl" else self.sd15_controlnets
        if ct not in table:
            raise ValueError(f"Unknown control type '{ct}'. Choose one of {sorted(table)}")
        return table[ct]

    def base_model_id(self) -> str:
        return self.sdxl_model_id if self.gen_family == "sdxl" else self.sd15_model_id

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
        _SETTINGS.outputs_dir.mkdir(parents=True, exist_ok=True)
    return _SETTINGS
