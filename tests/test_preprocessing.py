import cv2
import numpy as np
import pytest

from computer_vision.edge_detection import extract_edges
from computer_vision.preprocessing import (generate_structural_map, preprocess_sketch, segment_foreground)
from utils.file_handler import InputValidationError, validate_upload_file


def test_all_samples_produce_every_representation(sample_paths, tmp_path):
    for p in sample_paths:
        r = preprocess_sketch(p, 512, "scribble")
        assert r.normalized.shape == (512, 512, 3)
        for arr in (r.gray, r.clean_sketch, r.binary, r.edges, r.mask):
            assert arr.shape == (512, 512) and arr.dtype == np.uint8
        assert r.control_image.shape == (512, 512, 3)
        assert r.binary.max() == 255 and 0.002 < r.binary.mean() / 255 < 0.45
        assert r.edges.max() == 255
        assert len(r.contours) > 0 and r.mask.max() == 255
        files = r.save(tmp_path / p.stem)
        assert all(f.exists() for f in files.values())


def test_input_type_detection(sample_paths):
    kinds = {p.stem: preprocess_sketch(p).characteristics.input_type for p in sample_paths}
    assert kinds["01_cup"] == "clean_sketch"
    assert kinds["05_noisy_photo_vase"] in ("paper_photo", "noisy_sketch")
    assert kinds["06_house_chalkboard"] == "dark_background"


def test_operations_are_adaptive(sample_paths):
    clean = preprocess_sketch(sample_paths[0])
    noisy = preprocess_sketch([p for p in sample_paths if "noisy" in p.stem][0])
    assert not any("median filter" in o for o in clean.operations)
    assert any("median filter" in o for o in noisy.operations)
    assert any("illumination" in o or "median" in o for o in noisy.operations)


def test_noise_is_removed_from_noisy_photo(sample_paths):
    r = preprocess_sketch([p for p in sample_paths if "noisy" in p.stem][0])
    n, *_ = cv2.connectedComponentsWithStats(r.binary)
    assert n - 1 <= 15, f"{n - 1} components left - specks not removed"


def test_original_is_preserved(rect_sketch):
    before = rect_sketch.copy()
    r = preprocess_sketch(rect_sketch)
    assert np.array_equal(r.original, before)


def test_aspect_ratio_preserved(rect_sketch):
    r = preprocess_sketch(rect_sketch)
    ys, xs = np.where(r.binary > 0)
    ratio = (xs.max() - xs.min()) / (ys.max() - ys.min())
    assert abs(ratio - 400 / 160) < 0.15


def test_transparent_canvas_input_like_gradio():
    rgba = np.zeros((512, 512, 4), np.uint8)
    cv2.circle(rgba, (256, 256), 150, (0, 0, 0, 255), 4)
    r = preprocess_sketch(rgba)
    assert r.characteristics.input_type == "clean_sketch"
    assert r.stats["closed_outline"] == 1.0


@pytest.mark.parametrize("img", [
    np.full((300, 300, 3), 255, np.uint8),          # blank white
    np.zeros((300, 300, 4), np.uint8),              # untouched transparent canvas
    np.full((20, 20, 3), 255, np.uint8),            # too small image
])
def test_empty_or_invalid_inputs_raise(img):
    with pytest.raises(InputValidationError):
        preprocess_sketch(img)


def test_extremely_small_sketch_raises():
    img = np.full((600, 600, 3), 255, np.uint8)
    cv2.line(img, (300, 300), (306, 304), (0, 0, 0), 2)
    with pytest.raises(InputValidationError):
        preprocess_sketch(img)


def test_open_outline_falls_back_to_hull():
    b = np.zeros((512, 512), np.uint8)
    cv2.line(b, (100, 400), (256, 100), 255, 3)
    cv2.line(b, (256, 100), (412, 400), 255, 3)   # open "^" shape
    mask, stats, warnings = segment_foreground(b)
    assert stats["closed_outline"] == 0.0 and warnings
    assert mask.mean() > b.mean()


def test_multiple_objects_warning():
    b = np.zeros((512, 512), np.uint8)
    cv2.circle(b, (130, 256), 80, 255, 3)
    cv2.circle(b, (380, 256), 80, 255, 3)
    _, stats, warnings = segment_foreground(b)
    assert stats["num_objects"] == 2.0 and any("separate objects" in w for w in warnings)


def test_structural_maps_differ():
    b = np.zeros((512, 512), np.uint8)
    cv2.rectangle(b, (100, 100), (400, 400), 255, 2)
    e = extract_edges(255 - b)
    maps = {t: generate_structural_map(b, e, t) for t in ("scribble", "lineart", "canny")}
    assert maps["scribble"][..., 0].sum() > maps["lineart"][..., 0].sum()
    assert all(m.shape == (512, 512, 3) for m in maps.values())


def test_upload_validation(tmp_path):
    bad = tmp_path / "sketch.gif"
    bad.write_bytes(b"GIF89a")
    with pytest.raises(InputValidationError):
        validate_upload_file(bad, 10)
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    with pytest.raises(InputValidationError):
        validate_upload_file(big, 1)
