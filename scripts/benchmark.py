"""Run the full pipeline on every test sketch and record REAL measurements.

    python scripts/benchmark.py                # all samples/*.png
    python scripts/benchmark.py --warmup       # one extra unrecorded run first (excludes model loading)

Writes outputs/benchmark/results.json and outputs/benchmark/results.md (paste the table into the
README "Measured results" section). Nothing is estimated: every number comes from this run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from pipeline.orchestrator import Sketch2ShapePipeline  # noqa: E402
from utils.gpu import hardware_report  # noqa: E402

PROMPTS = {  # optional descriptions per sample (empty = sketch only, tests the no-prompt path)
    "01_cup": "",
    "02_chair": "a modern wooden chair, dark brown walnut",
    "03_table_lamp": "a table lamp with a white fabric shade and brass base",
    "04_car_rough": "",
    "05_noisy_photo_vase": "a ceramic vase, glossy blue glaze",
    "06_house_chalkboard": "a small cartoon house with a red roof",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "samples"))
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s = get_settings()
    pipe = Sketch2ShapePipeline(s)
    out = s.outputs_dir / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    samples = sorted(Path(args.samples).glob("*.png"))
    hw = hardware_report()

    t0 = time.perf_counter()
    if args.warmup and samples:
        print("Warm-up run (loads models, not recorded)…")
        pipe.run(samples[0], "", seed=args.seed, num_candidates=1)
    load_s = round(time.perf_counter() - t0, 1)

    rows = []
    for p in samples:
        prompt = PROMPTS.get(p.stem, "")
        print(f"\n### {p.name}  prompt={prompt!r}")
        rec = {"sample": p.name, "prompt": prompt}
        try:
            job = pipe.run(p, prompt, status=print, seed=args.seed)
            fin = job.mesh.report_after
            best = next(c for c in job.candidates if c["index"] == job.selected)
            rec.update(ok=True, object=job.description.object, vlm=job.description.source,
                       input_type=job.preprocess.characteristics.input_type, timings=job.timer.as_dict(),
                       fidelity=best["fidelity_score"], silhouette_iou=best["silhouette_iou"],
                       vertices=fin["num_vertices"], faces=fin["num_faces"], watertight=fin["is_watertight"],
                       flatness=fin["flatness"], visual=fin.get("visual_kind"),
                       glb_reload_ok=job.mesh.export_check.get("readable"), job_dir=str(job.dir))
            dst = out / p.stem
            dst.mkdir(exist_ok=True)
            shutil.copy(job.files["glb"], dst / "model.glb")
            shutil.copy(best["path"], dst / "generated.png")
            shutil.copy(job.files["original"], dst / "input.png")
            shutil.copy(job.files["control"], dst / "control_map.png")
        except Exception as exc:
            rec.update(ok=False, error=f"{exc.__class__.__name__}: {exc}", trace=traceback.format_exc()[-1500:])
            print("FAILED:", rec["error"])
        rows.append(rec)

    result = {"hardware": hw, "settings": s.as_dict(), "warmup_load_seconds": load_s if args.warmup else None,
              "runs": rows}
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))

    md = [f"Hardware: {hw.get('gpu', 'CPU')} | VRAM {hw.get('vram_gb')} GB | torch {hw.get('torch')} | "
          f"CUDA {hw.get('cuda_version')} | RAM {hw.get('ram_gb')} GB",
          f"Config: {s.base_model_id()} + {s.controlnet_id()} | {s.num_candidates} candidates x "
          f"{s.num_inference_steps} steps @ {s.gen_resolution}px | {s.recon_backend} mc={s.triposr_mc_resolution}", "",
          "| sample | input type | object (VLM) | CV | understand | diffusion | rank | prep | 3D | mesh | total s | fidelity | faces | GLB ok |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("ok"):
            md.append(f"| {r['sample']} | FAILED: {r.get('error', '')[:80]} |" + " |" * 12)
            continue
        t = r["timings"]
        g = lambda k: f"{t.get(k, 0):.1f}"  # noqa: E731
        md.append(f"| {r['sample']} | {r['input_type']} | {r['object']} | {g('1_cv_preprocessing')} | "
                  f"{g('2_sketch_understanding')} | {g('3_image_generation')} | {g('4_candidate_ranking')} | "
                  f"{g('5_image_preparation')} | {g('6_reconstruction')} | {g('7_mesh_processing')} | "
                  f"**{t['total']:.1f}** | {r['fidelity']:.2f} | {r['faces']:,} | {r['glb_reload_ok']} |")
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nSaved {out/'results.md'} and {out/'results.json'}")


if __name__ == "__main__":
    main()
