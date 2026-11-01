"""Headless end-to-end run: sketch image in -> GLB/OBJ + report out.

    python scripts/run_cli.py samples/02_chair.png --prompt "modern oak chair" --candidates 2 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from pipeline.orchestrator import Sketch2ShapePipeline  # noqa: E402
from utils.file_handler import validate_upload_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Sketch2Shape CLI")
    ap.add_argument("image")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--category", default="")
    ap.add_argument("--material", default="")
    ap.add_argument("--color", default="")
    ap.add_argument("--style", default="")
    ap.add_argument("--control", default=None, choices=["scribble", "lineart", "canny"])
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s = get_settings()
    validate_upload_file(args.image, s.max_upload_mb)
    pipe = Sketch2ShapePipeline(s)
    job = pipe.run(args.image, args.prompt, status=print, seed=args.seed, num_candidates=args.candidates,
                   category=args.category, material=args.material, color=args.color, style=args.style,
                   control_type=args.control)
    from frontend.viewer import write_viewer
    viewer = write_viewer(Path(job.files["glb"]), job.dir / "export" / "viewer.html")
    print("\n=== Done ===")
    print("GLB    :", job.files["glb"])
    print("OBJ    :", job.files.get("obj"))
    print("Viewer :", viewer)
    print("Report :", job.files["report"])
    print("Timings:", json.dumps(job.timer.as_dict(), indent=2))
    for w in job.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    main()
