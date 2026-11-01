"""Integration tests of the orchestrator wiring.

The diffusion model and the image-to-3D network need a GPU and multi-GB weights, so here they
are replaced by small TEST DOUBLES that honour the same interfaces. The real models are
exercised by `scripts/benchmark.py` / `scripts/run_cli.py` on a GPU machine. The application
itself never uses these doubles.
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

from computer_vision.image_preparation import segment_object
from config.settings import Settings
from generative_ai.image_generation import Candidate, GenerationError
from generative_ai.prompt_builder import build_prompts
from generative_ai.sketch_understanding import SketchDescription, parse_user_text, understand_sketch
from pipeline.orchestrator import Sketch2ShapePipeline
from reconstruction.image_to_3d import ReconResult


class FakeControlNet:
    """Paints the ControlNet condition's filled silhouette as a shaded object (aligned like ControlNet)."""
    name = "fake"

    def __init__(self, fail=False):
        self.fail, self.calls, self.parked = fail, [], 0

    def generate(self, control, positive, negative, num_candidates=2, seed=0, progress=None, **kw):
        import cv2
        self.calls.append(dict(positive=positive, negative=negative, n=num_candidates, seed=seed, **kw))
        if self.fail:
            raise GenerationError("simulated CUDA out of memory")
        c = np.array(control)[..., 0]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(c, cv2.MORPH_CLOSE, k)
        ff = closed.copy()
        cv2.floodFill(ff, np.zeros((c.shape[0] + 2, c.shape[1] + 2), np.uint8), (0, 0), 255)
        fill = closed | cv2.bitwise_not(ff)
        out = []
        for i in range(num_candidates):
            img = np.full((*c.shape, 3), 245, np.uint8)
            shift = 0 if i == 0 else 60 * i          # candidate 0 aligned, others drift
            m = np.roll(fill, shift, axis=1)
            img[m > 0] = (120 + 20 * i, 80, 40)
            if progress:
                progress(i + 1, num_candidates)
            out.append(Candidate(Image.fromarray(img), seed + i, i))
        return out

    def park(self):
        self.parked += 1


class FakeImageTo3D:
    name = "fake3d"

    def __init__(self):
        self.inputs = []

    def reconstruct(self, recon_rgb, out_dir, status=None):
        self.inputs.append(recon_rgb)
        m = trimesh.creation.capsule(height=0.6, radius=0.25)
        col = recon_rgb[recon_rgb.shape[0] // 2, recon_rgb.shape[1] // 2]
        m.visual.vertex_colors = np.tile(np.r_[col, 255], (len(m.vertices), 1)).astype(np.uint8)
        raw = out_dir / "20_raw_reconstruction.glb"
        m.export(raw)
        return ReconResult(mesh=m, raw_path=raw, backend=self.name, textured=False, timings={"fake": 0.0})

    def park(self):
        pass


@pytest.fixture
def pipeline(tmp_path):
    s = Settings(outputs_dir=tmp_path, vlm_backend="none", num_candidates=2)
    return Sketch2ShapePipeline(s, generator=FakeControlNet(), reconstructor=FakeImageTo3D(),
                                segment_fn=lambda rgb, method, model: segment_object(rgb, "opencv"))


def test_end_to_end_wiring(pipeline, sample_paths):
    chair = [p for p in sample_paths if "chair" in p.stem][0]
    job = pipeline.run(chair, "a modern walnut chair, dark brown", seed=7)
    # stage outputs exist on disk
    for key in ("original", "gray", "binary", "edges", "contours", "mask", "control", "generated",
                "rgba", "recon_rgb", "raw_mesh", "glb", "report"):
        assert key in job.files, key
    # user text reached the generator prompt
    call = pipeline.generator.calls[0]
    assert "chair" in call["positive"] and "walnut" in call["positive"] and call["seed"] == 7
    # the aligned candidate wins the structural-fidelity ranking
    assert job.selected == 0
    assert job.candidates[0]["fidelity_score"] > job.candidates[1]["fidelity_score"]
    # 3D model got the prepared (gray background) image
    assert pipeline.reconstructor.inputs[0].shape == (512, 512, 3)
    # exported, re-loaded and reported
    assert job.mesh.export_check["readable"] and job.mesh.report_after["num_faces"] > 0
    rep = job.report()
    for k in ("preprocessing", "sketch_understanding", "prompts", "candidates", "reconstruction", "mesh"):
        assert k in rep
    assert set(job.timer.timings) >= {"1_cv_preprocessing", "3_image_generation", "6_reconstruction", "7_mesh_processing"}


def test_sketch_only_no_prompt(pipeline, sample_paths):
    job = pipeline.run(sample_paths[0], "")
    assert job.mesh is not None and job.prompts.positive


def test_rebuild_from_selected_candidate(pipeline, sample_paths):
    job = pipeline.run(sample_paths[0], "a mug")
    pipeline.reconstruct(job, candidate_index=1)
    assert job.selected == 1 and job.mesh is not None


def test_regenerate_adds_a_round(pipeline, sample_paths):
    job = pipeline.run(sample_paths[0], "a mug", seed=1)
    pipeline.generate(job, seed=99, num_candidates=1)
    assert len(job.candidates) == 3 and job.candidates[-1]["round"] != job.candidates[0]["round"]
    assert job.selected == 2   # best of the new round is selected


def test_generation_failure_is_reported_not_faked(tmp_path, sample_paths):
    s = Settings(outputs_dir=tmp_path, vlm_backend="none")
    rec = FakeImageTo3D()
    p = Sketch2ShapePipeline(s, generator=FakeControlNet(fail=True), reconstructor=rec)
    with pytest.raises(GenerationError):
        p.run(sample_paths[0], "")
    job = next(iter(p.jobs._jobs.values()))
    assert job.error and "out of memory" in job.error
    assert rec.inputs == [] and "glb" not in job.files   # no placeholder model produced


def test_user_text_overrides_and_keyword_parsing():
    u = parse_user_text("A modern wooden chair with four legs, dark brown polished wood")
    assert u["object"] == "chair" and "wood" in u["material"] and u["color"] == "dark brown" and u["style"] == "modern"
    explicit = parse_user_text("whatever", category="lamp", material="brass")
    assert explicit["object"] == "lamp" and explicit["material"] == "brass"


def test_understanding_offline_uses_cv_cues(sample_paths):
    from computer_vision.preprocessing import preprocess_sketch
    r = preprocess_sketch(sample_paths[0])
    d = understand_sketch(Image.fromarray(r.clean_sketch), "a ceramic mug", binary=r.binary, mask=r.mask,
                          settings=Settings(vlm_backend="none"))
    assert d.object == "mug" and d.material == "ceramic" and d.geometry_description


def test_prompt_builder_limits_and_negatives():
    d = SketchDescription(object="chair", material="oak", color="dark brown", style="modern",
                          components=["four legs", "seat", "backrest"], view="three-quarter view",
                          user_text=" ".join(["very"] * 80))
    p = build_prompts(d, "sd15")
    assert p.positive.startswith("modern dark brown oak chair")
    assert len(p.positive.split()) <= 70
    assert "multiple objects" in p.negative and "cropped" in p.negative


def test_vlm_drawing_style_is_not_passed_to_the_prompt(monkeypatch):
    import generative_ai.sketch_understanding as su
    monkeypatch.setattr(su, "query_openai_compatible", lambda *a, **k: {
        "object": "chair", "confidence": 0.95, "style": "sketch", "color": "black and white", "material": "wood"})
    d = understand_sketch(Image.new("L", (64, 64), 255), "", settings=Settings(vlm_backend="openai"))
    assert d.object == "chair" and d.material == "wood"
    assert d.style == "" and d.color == ""
    assert "sketch" not in build_prompts(d).positive.split(",")[0]


def test_colour_illustration_gets_softer_control_and_warning(tmp_path):
    import cv2
    img = np.full((512, 512, 3), 250, np.uint8)
    cv2.circle(img, (256, 260), 190, (180, 225, 245), -1)          # backdrop disc like clip-art
    cv2.rectangle(img, (140, 250), (370, 420), (255, 255, 255), -1)
    cv2.rectangle(img, (140, 250), (370, 420), (40, 40, 70), 3)
    cv2.fillPoly(img, [np.array([[120, 255], [255, 150], [390, 255]])], (150, 70, 40))
    p = Sketch2ShapePipeline(Settings(outputs_dir=tmp_path, vlm_backend="none"), generator=FakeControlNet(),
                             reconstructor=FakeImageTo3D(),
                             segment_fn=lambda rgb, m, r: segment_object(rgb, "opencv"))
    job = p.run(img, "house", num_candidates=1)
    assert job.preprocess.characteristics.input_type == "photo"
    assert any("colour picture" in w for w in job.warnings)
    call = p.generator.calls[-1]
    assert call["controlnet_scale"] == 0.7 and call["control_end"] == 0.7
    assert job.prompts.positive.count("house") == 1


def test_colour_picture_uses_img2img_and_its_colour(tmp_path):
    import cv2
    img = np.full((512, 512, 3), 245, np.uint8)
    cv2.ellipse(img, (256, 290), (150, 170), 0, 0, 360, (245, 205, 90), -1)   # yellow body
    cv2.ellipse(img, (256, 290), (150, 170), 0, 0, 360, (20, 20, 20), 4)      # black outline
    cv2.ellipse(img, (256, 250), (60, 30), 0, 0, 360, (250, 225, 170), -1)    # pale beak
    p = Sketch2ShapePipeline(Settings(outputs_dir=tmp_path, vlm_backend="none"), generator=FakeControlNet(),
                             reconstructor=FakeImageTo3D(),
                             segment_fn=lambda rgb, m, r: segment_object(rgb, "opencv"))
    job = p.run(img, "duck", num_candidates=1)
    assert job.description.color == "yellow"                      # read from the picture
    assert "yellow duck" in job.prompts.positive
    call = p.generator.calls[-1]
    assert call["init_image"] is not None and call["strength"] == 0.75


def test_dominant_colour_names():
    from computer_vision.preprocessing import dominant_colour_name
    for rgb, name in [((245, 205, 60), "yellow"), ((200, 40, 35), "red"), ((40, 90, 200), "blue"),
                      ((250, 250, 250), "white"), ((60, 160, 70), "green")]:
        img = np.zeros((64, 64, 3), np.uint8)
        img[:] = rgb
        assert dominant_colour_name(img) == name, (rgb, dominant_colour_name(img))
