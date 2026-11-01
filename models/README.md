# Model weights

Weights are **not** stored in this repository. They are downloaded from the Hugging Face Hub on
first use (or ahead of time with `python scripts/download_models.py`) into the standard cache
(`~/.cache/huggingface`, override with `HF_HOME`). rembg stores its ONNX model in `~/.u2net`.

| Stage | Model (default) | Hub id | Licence | Approx. download |
|---|---|---|---|---|
| Sketch → image | DreamShaper 8 (SD-1.5 fine-tune) | `Lykon/dreamshaper-8` | CreativeML OpenRAIL-M | ~2–4 GB |
| Structural control | ControlNet v1.1 scribble | `lllyasviel/control_v11p_sd15_scribble` | OpenRAIL | ~1.4 GB |
| (alt.) | ControlNet v1.1 lineart / canny | `lllyasviel/control_v11p_sd15_lineart`, `..._canny` | OpenRAIL | ~1.4 GB each |
| Image → 3D | TripoSR | `stabilityai/TripoSR` | MIT | ~1.7 GB |
| TripoSR backbone config | DINO ViT-B/16 (config only) | `facebook/dino-vitb16` | Apache-2.0 | < 1 MB |
| Offline sketch understanding | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | MIT | ~600 MB |
| Background removal | IS-Net general | rembg `isnet-general-use` | Apache-2.0 (DIS) | ~170 MB |
| Optional (SDXL mode) | SDXL base + xinsir scribble ControlNet + fp16 VAE | `stabilityai/stable-diffusion-xl-base-1.0`, `xinsir/controlnet-scribble-sdxl-1.0`, `madebyollin/sdxl-vae-fp16-fix` | OpenRAIL++ / Apache-2.0 / MIT | ~10 GB |

Download sizes are approximate; check each model card for the authoritative licence text.

To run fully offline, download once, then set `HF_HUB_OFFLINE=1`. To use local copies, point the
`S2S_*_MODEL` variables in `.env` at the local folders.
