"""Edge extraction and the structural-fidelity diagnostic.

ControlNet keeps the generated image spatially aligned with its conditioning
map, so the sketch and every generated candidate share one 2D frame. That lets
us measure, per candidate:

* stroke recall    - share of sketch stroke pixels that have a generated edge nearby
* edge precision   - share of generated edges that lie near a sketch stroke
* silhouette IoU   - overlap of the filled sketch silhouette and the generated object mask

These are *structural* diagnostics used to rank candidates and to warn when a
generation drifted from the drawing. They are not a measure of semantic
similarity (a correct chair with a different texture scores the same).
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def extract_edges(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds derived from the image median (robust across inputs)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    v = float(np.median(blurred))
    lo, hi = int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v))
    if hi - lo < 30:  # nearly-binary image: median-based thresholds collapse
        lo, hi = 50, 150
    return cv2.Canny(blurred, lo, hi, L2gradient=True)


def mask_boundary(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    m = (mask > 127).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    return cv2.subtract(cv2.dilate(m, k, iterations=thickness), m)


def _near(target: np.ndarray, reference: np.ndarray, tol: int) -> float:
    """Fraction of target pixels within `tol` px of any reference pixel."""
    t = target > 0
    if t.sum() == 0:
        return 0.0
    ref = (reference > 0).astype(np.uint8)
    if ref.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(1 - ref, cv2.DIST_L2, 3)
    return float((dist[t] <= tol).mean())


def structural_fidelity(sketch_binary: np.ndarray, sketch_mask: np.ndarray, generated_rgb: np.ndarray,
                        generated_mask: Optional[np.ndarray] = None, tol: Optional[int] = None) -> Dict[str, float]:
    h, w = sketch_binary.shape[:2]
    if generated_rgb.shape[:2] != (h, w):
        generated_rgb = cv2.resize(generated_rgb, (w, h), interpolation=cv2.INTER_AREA)
    if generated_mask is not None and generated_mask.shape[:2] != (h, w):
        generated_mask = cv2.resize(generated_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    tol = tol or max(4, w // 80)

    gen_edges = extract_edges(generated_rgb)
    if generated_mask is not None:
        # ignore background clutter; add the silhouette outline (often the strongest structure)
        roi = cv2.dilate((generated_mask > 127).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        gen_edges = cv2.bitwise_and(gen_edges, roi) | mask_boundary(generated_mask)

    recall = _near(sketch_binary, gen_edges, tol)
    precision = _near(gen_edges, sketch_binary, tol)
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)

    iou = 0.0
    if generated_mask is not None:
        a, b = sketch_mask > 127, generated_mask > 127
        union = np.logical_or(a, b).sum()
        iou = float(np.logical_and(a, b).sum() / union) if union else 0.0

    score = 0.45 * iou + 0.40 * recall + 0.15 * precision if generated_mask is not None else f1
    return {
        "stroke_recall": round(recall, 3),
        "edge_precision": round(precision, 3),
        "edge_f1": round(f1, 3),
        "silhouette_iou": round(iou, 3),
        "fidelity_score": round(float(score), 3),
    }


def fidelity_overlay(sketch_binary: np.ndarray, generated_rgb: np.ndarray) -> np.ndarray:
    """Visual check: generated image with the sketch strokes drawn on top in red."""
    h, w = sketch_binary.shape[:2]
    img = cv2.resize(generated_rgb, (w, h)) if generated_rgb.shape[:2] != (h, w) else generated_rgb.copy()
    img = (img.astype(np.float32) * 0.7 + 255 * 0.3).astype(np.uint8)
    img[sketch_binary > 0] = (230, 30, 30)
    return img
