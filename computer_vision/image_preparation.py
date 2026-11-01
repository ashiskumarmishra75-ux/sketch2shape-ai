"""Stage 5 - Segmentation and preparation of the generated image for image-to-3D.

Steps: background removal (rembg; OpenCV flood-fill + GrabCut fallback) ->
mask clean-up -> alpha matte -> crop to object -> pad to square -> resize to the
reconstruction model's input size -> composite on the background the model was
trained with (TripoSR: 50 % gray).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from utils.file_handler import save_image
from utils.logging_utils import get_logger

log = get_logger("image_preparation")
_REMBG_SESSIONS: Dict[str, object] = {}


@dataclass
class PreparedImage:
    generated: np.ndarray            # RGB, untouched generated image
    rgba: np.ndarray                 # RGBA, background removed, same frame as `generated`
    mask: np.ndarray                 # uint8 0/255, same frame as `generated`
    recon_rgba: np.ndarray           # RGBA, centred + padded + resized (for models that take alpha)
    recon_rgb: np.ndarray            # RGB composited on gray (TripoSR input)
    method: str = "rembg"
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    def save(self, out_dir: Path) -> Dict[str, Path]:
        out_dir = Path(out_dir)
        return {
            "generated": save_image(self.generated, out_dir / "10_generated.png"),
            "rgba": save_image(self.rgba, out_dir / "11_background_removed.png"),
            "mask": save_image(self.mask, out_dir / "12_alpha_mask.png"),
            "recon_rgba": save_image(self.recon_rgba, out_dir / "13_recon_input_rgba.png"),
            "recon_rgb": save_image(self.recon_rgb, out_dir / "13_recon_input.png"),
        }


# --------------------------------------------------------------------------- background removal


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except Exception:
        return False


def remove_background_rembg(rgb: np.ndarray, model_name: str = "isnet-general-use") -> np.ndarray:
    from rembg import new_session, remove
    if model_name not in _REMBG_SESSIONS:
        log.info("Loading rembg session '%s' (downloads once, then cached)", model_name)
        _REMBG_SESSIONS[model_name] = new_session(model_name)
    out = remove(Image.fromarray(rgb), session=_REMBG_SESSIONS[model_name], post_process_mask=True)
    return np.array(out.convert("RGBA"))[..., 3]


def remove_background_opencv(rgb: np.ndarray) -> np.ndarray:
    """Fallback for plain studio backgrounds: flood-fill from the border, refine with GrabCut."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[:4].reshape(-1, 3), lab[-4:].reshape(-1, 3),
                             lab[:, :4].reshape(-1, 3), lab[:, -4:].reshape(-1, 3)])
    bg_color = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg_color, axis=2)
    similar = (dist < 12).astype(np.uint8) * 255
    # background = pixels similar to the border colour AND connected to the border
    ff = similar.copy()
    seed_mask = np.zeros((h + 2, w + 2), np.uint8)
    bg = np.zeros((h, w), np.uint8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:
        if ff[y, x] == 255:
            tmp = ff.copy()
            cv2.floodFill(tmp, seed_mask, (x, y), 128)
            bg |= (tmp == 128).astype(np.uint8)
    fg = (bg == 0).astype(np.uint8)
    if fg.sum() < 0.002 * h * w or fg.sum() > 0.98 * h * w:
        return fg * 255
    gc_mask = np.where(fg > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    core = cv2.erode(fg, np.ones((9, 9), np.uint8))
    gc_mask[core > 0] = cv2.GC_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        fg = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        pass
    return fg * 255


def clean_mask(mask: np.ndarray, min_rel_area: float = 0.03) -> np.ndarray:
    """Binarise, drop small blobs, smooth the edge slightly. Real openings are NOT filled."""
    m = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= min_rel_area * areas.max()
        m = keep[labels].astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m * 255


def segment_object(rgb: np.ndarray, method: str = "auto", rembg_model: str = "isnet-general-use") -> tuple[np.ndarray, str]:
    if method in ("auto", "rembg") and rembg_available():
        try:
            return clean_mask(remove_background_rembg(rgb, rembg_model)), "rembg:" + rembg_model
        except Exception as exc:  # model download failure etc.
            if method == "rembg":
                raise
            log.warning("rembg failed (%s) - using OpenCV fallback", exc)
    return clean_mask(remove_background_opencv(rgb)), "opencv-floodfill-grabcut"


# --------------------------------------------------------------------------- framing


def center_and_pad(rgba: np.ndarray, foreground_ratio: float = 0.85, size: int = 512) -> np.ndarray:
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        raise ValueError("Empty foreground mask")
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape[:2]
    side = int(max(ch, cw) / foreground_ratio)
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC)


def composite_on_gray(rgba: np.ndarray, value: float = 0.5) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb * a + (1.0 - a) * value
    return (out * 255.0).round().astype(np.uint8)


def prepare_for_reconstruction(generated_rgb: np.ndarray, method: str = "auto", foreground_ratio: float = 0.85,
                               size: int = 512, rembg_model: str = "isnet-general-use",
                               precomputed_mask: Optional[np.ndarray] = None) -> PreparedImage:
    rgb = np.ascontiguousarray(generated_rgb[..., :3])
    if precomputed_mask is not None:
        mask, used = precomputed_mask, "precomputed"
    else:
        mask, used = segment_object(rgb, method, rembg_model)
    warnings: List[str] = []
    h, w = mask.shape
    area = float((mask > 0).mean())
    if area < 0.01:
        raise ValueError("Background removal found no object in the generated image. Try regenerating.")
    if area > 0.95:
        warnings.append("Background removal kept almost the whole image - the object may not be isolated.")
    ys, xs = np.where(mask > 0)
    touches = int(ys.min() <= 1) + int(xs.min() <= 1) + int(ys.max() >= h - 2) + int(xs.max() >= w - 2)
    if touches >= 2:
        warnings.append("The object touches the image border and may be cropped; consider regenerating.")
    soft = cv2.GaussianBlur(mask, (3, 3), 0)
    rgba = np.dstack([rgb, soft]).astype(np.uint8)
    framed = center_and_pad(rgba, foreground_ratio, size)
    return PreparedImage(
        generated=rgb, rgba=rgba, mask=mask, recon_rgba=framed, recon_rgb=composite_on_gray(framed),
        method=used, warnings=warnings, stats={"foreground_area": round(area, 4), "border_touches": float(touches)},
    )
