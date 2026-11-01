"""Pre-download every model the pipeline uses, so the first request (and the demo) is fast.

    python scripts/download_models.py            # models for the current .env configuration
    python scripts/download_models.py --all      # also the alternative ControlNets / SDXL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402

# diffusers-format folders only (skips single-file checkpoints and the unused safety checker)
SD_PATTERNS = ["model_index.json", "*/config.json", "*/*.json", "*/*.txt", "*/*.model",
               "*/diffusion_pytorch_model.safetensors", "*/model.safetensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    s = get_settings()
    jobs = []
    if s.gen_family == "sdxl" or args.all:
        jobs += [(s.sdxl_model_id, SD_PATTERNS, ["*.fp16.*", "*refiner*", "*.onnx*", "*openvino*"]),
                 (s.sdxl_vae_id, ["config.json", "diffusion_pytorch_model.safetensors"], None)]
        jobs += [(m, ["config.json", "diffusion_pytorch_model.safetensors"], None) for m in set(s.sdxl_controlnets.values())]
    if s.gen_family == "sd15" or args.all:
        jobs.append((s.sd15_model_id, SD_PATTERNS, ["*.fp16.*", "*.ckpt", "*.onnx*", "*non-ema*", "*ema-only*", "safety_checker/*"]))
        cns = set(s.sd15_controlnets.values()) if args.all else {s.controlnet_id()}
        for cn in cns:
            jobs.append((cn, ["config.json", "diffusion_pytorch_model.safetensors"], None))
    if s.recon_backend == "triposr":
        jobs.append((s.triposr_model_id, ["config.yaml", "model.ckpt"], None))
    for repo, allow, ignore in jobs:
        print(f"-> {repo}")
        try:
            snapshot_download(repo, allow_patterns=allow, ignore_patterns=ignore)
        except Exception as exc:
            print(f"   !! failed: {exc}")
    # TripoSR's image tokenizer fetches the DINO ViT config at construction time
    hf_hub_download("facebook/dino-vitb16", "config.json")
    if s.vlm_backend in ("auto", "clip"):
        print(f"-> {s.clip_model_id} (offline sketch-understanding fallback)")
        snapshot_download(s.clip_model_id, allow_patterns=["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"])
    try:
        from rembg import new_session
        print(f"-> rembg model {s.rembg_model}")
        new_session(s.rembg_model)
    except Exception as exc:
        print(f"   !! rembg model download failed: {exc}")
    print("All downloads finished.")


if __name__ == "__main__":
    main()
