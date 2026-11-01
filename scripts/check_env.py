"""Inspect hardware + dependencies and say which configuration will run here.

    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from utils.gpu import hardware_report  # noqa: E402

PACKAGES = ["torch", "diffusers", "transformers", "accelerate", "gradio", "trimesh", "rembg", "onnxruntime",
            "cv2", "omegaconf", "einops", "torchmcubes", "mcubes", "skimage", "xatlas", "moderngl"]

# Rough budgets (GB) used to recommend a configuration; measured on your GPU via scripts/benchmark.py
VRAM_NEEDS = {"sd15 + ControlNet (fp16)": 4.5, "TripoSR (mc 256)": 6.0, "sdxl + ControlNet (fp16)": 11.0}


def main() -> None:
    s = get_settings()
    hw = hardware_report()
    print("=== Hardware ===")
    print(json.dumps(hw, indent=2))
    print("\n=== Packages ===")
    for p in PACKAGES:
        try:
            m = importlib.import_module(p)
            print(f"  [ok]      {p:13s} {getattr(m, '__version__', '')}")
        except Exception as exc:
            print(f"  [missing] {p:13s} ({exc.__class__.__name__})")
    print("\n=== TripoSR source ===")
    ok = (s.triposr_repo_dir / "tsr" / "system.py").exists()
    print(f"  {'[ok]' if ok else '[missing]'} {s.triposr_repo_dir}" + ("" if ok else "  -> run bash scripts/setup.sh"))

    print("\n=== Recommendation ===")
    vram = hw.get("vram_gb") or 0
    if not hw.get("cuda_available"):
        print("  No CUDA GPU: the pipeline runs on CPU but diffusion takes several minutes per image.\n"
              "  Use the Colab notebook (free T4 GPU): notebooks/Sketch2Shape_Colab.ipynb")
    elif vram < 8:
        print(f"  {vram} GB VRAM: use S2S_GEN_FAMILY=sd15 S2S_OFFLOAD=always S2S_NUM_CANDIDATES=1 S2S_MC_RES=192")
    elif vram < 14:
        print(f"  {vram} GB VRAM: default config (sd15 + TripoSR, models parked on CPU between stages).")
    else:
        print(f"  {vram} GB VRAM: everything stays on GPU; S2S_GEN_FAMILY=sdxl is an option for better images.")
    for k, v in VRAM_NEEDS.items():
        print(f"    ~{v:4.1f} GB  {k}")


if __name__ == "__main__":
    main()
