"""Build the final submission package from a real benchmark run.

    python scripts/benchmark.py --warmup          # first: real measurements on the GPU
    python scripts/finalize_submission.py         # then: this script

What it does
  1. Copies outputs/benchmark/<sample>/ (input, control map, generated image, model.glb) into
     outputs/submission/  - the "output artefact" (kept in git, unlike other outputs).
  2. Writes the measured results table (outputs/benchmark/results.md) into README.md between the
     BENCHMARK_RESULTS markers, plus a gallery of example input/output files.
  3. Re-opens every submitted GLB and checks it is a readable, non-empty 3D mesh.
  4. Creates Sketch2Shape_submission.zip next to the project folder: source code, docs, samples and
     outputs/submission - without model weights, caches, third_party or temporary job folders.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reconstruction.mesh_processing import verify_export  # noqa: E402

BENCH = ROOT / "outputs" / "benchmark"
SUB = ROOT / "outputs" / "submission"
START, END = "<!-- BENCHMARK_RESULTS_START -->", "<!-- BENCHMARK_RESULTS_END -->"
EXCLUDE_DIRS = {"third_party", "__pycache__", ".pytest_cache", ".git", ".venv", "venv", ".gradio"}
EXCLUDE_PATHS = {"outputs/jobs", "outputs/benchmark"}


def copy_artefacts() -> list[dict]:
    if not (BENCH / "results.json").exists():
        raise SystemExit("No benchmark found. Run `python scripts/benchmark.py --warmup` on the GPU first.")
    shutil.rmtree(SUB, ignore_errors=True)
    SUB.mkdir(parents=True)
    rows = []
    for d in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        if not (d / "model.glb").exists():
            continue
        shutil.copytree(d, SUB / d.name)
        check = verify_export(SUB / d.name / "model.glb")
        rows.append({"sample": d.name, **check})
    for f in ("results.md", "results.json"):
        shutil.copy(BENCH / f, SUB / f)
    if not rows:
        raise SystemExit("Benchmark folder has no model.glb files - did every run fail? See results.md.")
    (SUB / "glb_checks.json").write_text(json.dumps(rows, indent=2))
    return rows


def update_readme(rows: list[dict]) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README.md is missing the BENCHMARK_RESULTS markers.")
    table = (BENCH / "results.md").read_text(encoding="utf-8").strip()
    gallery = ["", "**Example inputs and outputs** (files in `outputs/submission/<sample>/`):", "",
               "| Sample | Input | ControlNet map | Generated image | 3D model | GLB re-open check |",
               "|---|---|---|---|---|---|"]
    for r in rows:
        s = r["sample"]
        ok = "✅ {:,} faces, {}".format(r.get("num_faces", 0), r.get("visual_kind")) if r.get("readable") \
            and r.get("non_empty") else "❌"
        gallery.append(f"| {s} | ![]({'outputs/submission/' + s + '/input.png'}) | "
                       f"![]({'outputs/submission/' + s + '/control_map.png'}) | "
                       f"![]({'outputs/submission/' + s + '/generated.png'}) | "
                       f"[model.glb](outputs/submission/{s}/model.glb) | {ok} |")
    block = f"{START}\n{table}\n" + "\n".join(gallery) + f"\n{END}"
    text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    readme.write_text(text, encoding="utf-8")


def build_zip() -> Path:
    out = ROOT.parent / "Sketch2Shape_submission.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if any(str(rel).startswith(x) for x in EXCLUDE_PATHS) or p.name == ".env":
                continue
            z.write(p, Path("Sketch2Shape") / rel)
    return out


def main() -> None:
    rows = copy_artefacts()
    update_readme(rows)
    bad = [r["sample"] for r in rows if not (r.get("readable") and r.get("non_empty"))]
    z = build_zip()
    print(f"Artefacts: {len(rows)} GLB files in {SUB} ({'all valid' if not bad else 'INVALID: ' + ', '.join(bad)})")
    print("README.md updated with the measured results table and example gallery.")
    print(f"Submission zip: {z} ({z.stat().st_size / 1024 ** 2:.1f} MB)")


if __name__ == "__main__":
    main()
