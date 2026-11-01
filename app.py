"""Sketch2Shape AI - application entry point.

    python app.py                 # http://localhost:7860
    python app.py --share         # public Gradio link (Colab / remote GPU)
    python app.py --preload       # load all models at start-up (no first-request delay in demos)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch2Shape AI web app")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--preload", action="store_true", help="load diffusion + 3D models before serving")
    args = parser.parse_args()

    from config.settings import get_settings
    from utils.gpu import hardware_report
    from utils.logging_utils import get_logger

    log = get_logger("app")
    s = get_settings()
    log.info("Hardware: %s", hardware_report())

    from frontend.interface import build_ui, pipe

    if args.preload or os.environ.get("S2S_PRELOAD", "").lower() in {"1", "true", "yes"}:
        log.info("Preloading models…")
        pipe.generator.load()
        if pipe._low_vram():
            pipe.generator.park()
        pipe.reconstructor.load()
        log.info("Models ready.")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(server_name=args.host or s.server_name, server_port=args.port or s.server_port,
                share=args.share or s.share, allowed_paths=[str(s.outputs_dir), str(ROOT / "samples")],
                show_error=True)


if __name__ == "__main__":
    main()
