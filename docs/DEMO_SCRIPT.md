# Final demonstration — 5-minute script

**Before judges arrive (10 min):** `python app.py --preload` (or Colab notebook cell 7) so model loading is not on
the clock. Keep `outputs/benchmark/results.md` open in a tab. Have one backup GLB from the benchmark ready.

## 0:00 – Problem in one sentence
"A sketch fixes structure but not depth, materials or hidden sides. We let the **sketch control structure** and
**AI fill in semantics, appearance and hidden geometry**, and we prove at every stage that the result still matches
the drawing."

## 0:30 – Live run with the judges' sketch
1. Ask a judge to draw on the canvas (or photograph a paper sketch and upload it — the noisy-photo path is handled).
2. Leave the prompt **empty** first to show text is optional. Press **Generate 3D model**.
3. While it runs, open **Sketch analysis**: point at *detected input type* and the list of operations applied —
   "we only apply what this input needs" — then the Canny, contour and ControlNet maps.
4. **Generated images** tab: two candidates with fidelity scores; explain the score (stroke recall + silhouette IoU,
   computable because ControlNet is pixel-aligned). Star = auto-selected.
5. **3D model** tab: rotate, zoom, reset, wireframe, unlit colours. Read out the mesh table: vertices/faces,
   flatness (proves it is not a plane), watertight as measured, GLB re-load check passed, per-stage timings.
6. Download `model.glb` and drop it into any viewer (e.g. Windows 3D Viewer / Blender / gltf-viewer) to show it opens.

## 3:00 – Control and recovery
* Add a prompt ("dark walnut, mid-century") → **Regenerate**: material changes, structure stays.
* Click the other candidate → **Rebuild 3D from selected** (only stages 5–7 rerun — seconds).
* Show a warning case: draw two separate objects → multi-object warning; or an open outline → convex-hull warning.

## 4:00 – Key decisions (be ready for questions)
| Question | Answer |
|---|---|
| Why TripoSR over InstantMesh/Unique3D? | MIT, ~6 GB VRAM, seconds per mesh, deps compatible with modern diffusers. InstantMesh pins diffusers 0.20.2 / transformers 4.34.1 + nvdiffrast, so we run it as an optional backend in its own venv instead of breaking stage 4. |
| Why scribble ControlNet by default? | Trained on rough, thick strokes — matches hand sketches; lineart/canny selectable for clean drawings/photos. |
| How do you stop the text overriding the sketch? | Structure only enters through the ControlNet map; VLM prompt says "describe only what is drawn"; fidelity ranking + drift warning. |
| What if the VLM/API is down? | Falls back to Ollama/OpenAI-compatible, then offline CLIP, then keywords — the pipeline never blocks on it. |
| GPU memory? | Sequential execution; models parked in CPU RAM between stages below 14 GB; OOM retries with smaller batch / marching-cubes resolution. |
| torchmcubes won't compile? | Built-in PyMCubes/scikit-image shim with the same vertex convention, unit-tested on an analytic sphere. |
| Is it watertight / printable? | We report what trimesh measures; we never claim print-readiness (out of scope). |

## Fallback plan
If the venue GPU fails: run the Colab notebook (public `gradio.live` link) and show `outputs/benchmark/` results.
