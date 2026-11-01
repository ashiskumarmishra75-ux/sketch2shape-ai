import cv2
import numpy as np
import pytest

from computer_vision.edge_detection import structural_fidelity
from computer_vision.image_preparation import (composite_on_gray, prepare_for_reconstruction,
                                               remove_background_opencv, segment_object)


def product_shot(shift=0):
    """Synthetic 'generated' image: shaded object on a plain off-white studio background."""
    img = np.full((512, 512, 3), 246, np.uint8)
    mask = np.zeros((512, 512), np.uint8)
    cv2.ellipse(mask, (256 + shift, 280), (120, 170), 0, 0, 360, 255, -1)
    yy, xx = np.mgrid[0:512, 0:512]
    shade = np.clip(1.15 - ((xx - 220 - shift) ** 2 + (yy - 220) ** 2) / 60000, 0.35, 1.0)
    obj = (np.array([150, 90, 50])[None, None] * shade[..., None]).astype(np.uint8)
    img[mask > 0] = obj[mask > 0]
    return img, mask


def iou(a, b):
    a, b = a > 127, b > 127
    return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def test_opencv_background_removal_is_accurate():
    img, gt = product_shot()
    m = remove_background_opencv(img)
    assert iou(m, gt) > 0.95


def test_segment_object_explicit_opencv():
    img, gt = product_shot()
    m, method = segment_object(img, "opencv")
    assert method.startswith("opencv") and iou(m, gt) > 0.95


def test_prepare_for_reconstruction_framing():
    img, gt = product_shot(shift=90)
    prep = prepare_for_reconstruction(img, precomputed_mask=gt, foreground_ratio=0.85, size=512)
    assert prep.recon_rgb.shape == (512, 512, 3) and prep.recon_rgba.shape == (512, 512, 4)
    a = prep.recon_rgba[..., 3]
    ys, xs = np.where(a > 0)
    assert abs((ys.max() - ys.min()) / 512 - 0.85) < 0.03          # longest side fills 85 %
    assert abs((xs.min() + xs.max()) / 2 - 256) < 4                  # centred
    assert np.allclose(prep.recon_rgb[0, 0], 128, atol=1)           # TripoSR gray background
    assert np.array_equal(prep.generated, img)                       # original kept


def test_empty_mask_is_rejected():
    img, _ = product_shot()
    with pytest.raises(ValueError):
        prepare_for_reconstruction(img, precomputed_mask=np.zeros((512, 512), np.uint8))


def test_border_touch_warning():
    img = np.full((512, 512, 3), 246, np.uint8)
    m = np.zeros((512, 512), np.uint8)
    m[0:300, 0:300] = 255
    img[m > 0] = (40, 40, 160)
    prep = prepare_for_reconstruction(img, precomputed_mask=m)
    assert any("border" in w for w in prep.warnings)


def test_composite_on_gray():
    rgba = np.zeros((4, 4, 4), np.uint8)
    assert np.all(composite_on_gray(rgba) == 128)


def test_structural_fidelity_ranks_aligned_candidate_higher():
    img, gt = product_shot()
    sketch = np.zeros((512, 512), np.uint8)
    cv2.ellipse(sketch, (256, 280), (120, 170), 0, 0, 360, 255, 3)
    good = structural_fidelity(sketch, gt, img, gt)
    shifted_img, shifted_mask = product_shot(shift=140)
    bad = structural_fidelity(sketch, gt, shifted_img, shifted_mask)
    assert good["fidelity_score"] > 0.8
    assert good["fidelity_score"] > bad["fidelity_score"] + 0.3
