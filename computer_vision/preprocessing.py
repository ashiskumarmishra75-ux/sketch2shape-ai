"""Stage 2 - Computer-vision preprocessing and sketch conditioning.

The module inspects the input first (clean digital canvas? photo of paper with
shadows? noisy scan? white-on-dark? colour photo?) and then applies only the
operations that input needs. Every operation that was applied is recorded in
`PreprocessResult.operations`, so the UI and the report show exactly what was done.

All structural maps use the convention *strokes = 255 (white) on 0 (black)*,
which is also what ControlNet scribble / lineart / canny models expect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from computer_vision.edge_detection import extract_edges
from utils.file_handler import InputValidationError, ImageLike, load_image, save_image, to_rgb_on_white

# --------------------------------------------------------------------------- data


@dataclass
class InputCharacteristics:
    input_type: str                 # clean_sketch | paper_photo | noisy_sketch | dark_background | photo
    background_brightness: float
    background_unevenness: float    # std of the low-frequency background (lighting gradient)
    colorfulness: float
    speckle_ratio: float            # share of tiny connected components (salt noise)
    ink_ratio: float
    colour_fraction: float = 0.0    # share of saturated pixels (filled colour regions => photo/illustration)


@dataclass
class PreprocessResult:
    original: np.ndarray            # RGB uint8, untouched (only transparency flattened)
    normalized: np.ndarray          # RGB, cropped around the sketch, padded square, work resolution
    gray: np.ndarray                # cleaned grayscale (denoised / contrast-enhanced as needed)
    clean_sketch: np.ndarray        # dark strokes on white, after thresholding + morphology
    binary: np.ndarray              # strokes = 255
    edges: np.ndarray               # Canny edge map, edges = 255
    contour_vis: np.ndarray         # RGB visualisation of the extracted contours
    mask: np.ndarray                # filled foreground silhouette, 255 = object
    control_image: np.ndarray       # RGB conditioning image for ControlNet
    contours: List[np.ndarray] = field(default_factory=list)
    characteristics: Optional[InputCharacteristics] = None
    operations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    crop_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1 in original pixels

    def save(self, out_dir: Path) -> Dict[str, Path]:
        out_dir = Path(out_dir)
        files = {
            "original": save_image(self.original, out_dir / "00_original.png"),
            "normalized": save_image(self.normalized, out_dir / "01_normalized.png"),
            "gray": save_image(self.gray, out_dir / "02_clean_gray.png"),
            "clean_sketch": save_image(self.clean_sketch, out_dir / "02b_clean_sketch.png"),
            "binary": save_image(self.binary, out_dir / "03_binary.png"),
            "edges": save_image(self.edges, out_dir / "04_canny_edges.png"),
            "contours": save_image(self.contour_vis, out_dir / "05_contours.png"),
            "mask": save_image(self.mask, out_dir / "06_foreground_mask.png"),
            "control": save_image(self.control_image, out_dir / "07_control_map.png"),
        }
        return files


# --------------------------------------------------------------------------- helpers


def _border_pixels(gray: np.ndarray, width: int = 8) -> np.ndarray:
    w = max(1, min(width, gray.shape[0] // 4, gray.shape[1] // 4))
    return np.concatenate([gray[:w].ravel(), gray[-w:].ravel(), gray[:, :w].ravel(), gray[:, -w:].ravel()])


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric."""
    r, g, b = [c.astype(np.float32) for c in cv2.split(rgb)]
    rg, yb = np.abs(r - g), np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _estimate_background(gray: np.ndarray) -> np.ndarray:
    """Low-frequency paper/lighting estimate (strokes removed by a large median)."""
    h, w = gray.shape
    scale = 256.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    k = max(3, (min(small.shape) // 6) | 1)
    bg_small = cv2.medianBlur(cv2.dilate(small, np.ones((3, 3), np.uint8)), k)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def _speckle_ratio(binary: np.ndarray, max_area: int = 4) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float((areas <= max_area).sum() / len(areas))


def drop_specks(binary: np.ndarray, min_area: int, min_extent: int) -> np.ndarray:
    """Remove compact specks but keep thin, elongated stroke fragments.

    A component survives if it is big enough (area) OR long enough (bounding-box extent),
    so a short dash of a pencil line is kept while a dot of paper grain is removed.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (stats[:, cv2.CC_STAT_AREA] >= min_area) | (extent >= min_extent)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _quick_ink_mask(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # if Otsu picked nearly everything (blank image), treat as empty
    if b.mean() / 255.0 > 0.6:
        b[:] = 0
    return b


# --------------------------------------------------------------------------- public API


def analyze_input(rgb: np.ndarray) -> InputCharacteristics:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = _border_pixels(gray)
    bg_brightness = float(np.median(border))
    bg = _estimate_background(gray if bg_brightness > 110 else 255 - gray)
    unevenness = float(bg.std())
    colorful = _colorfulness(rgb)
    ink = _quick_ink_mask(gray if bg_brightness > 110 else 255 - gray)
    speckle = _speckle_ratio(ink)
    ink_ratio = float(ink.mean() / 255.0)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colour_fraction = float(((hsv[..., 1] > 50) & (hsv[..., 2] > 40)).mean())  # saturated, non-black pixels

    if bg_brightness < 110:
        kind = "dark_background"
    elif colour_fraction > 0.10 or (colorful > 35 and ink_ratio > 0.15):
        kind = "photo"                      # colour photo / illustration: filled colour regions, not strokes
    elif unevenness > 10:
        kind = "paper_photo"                # phone photo of paper: shadows / gradients
    elif speckle > 0.35 or float(np.std(border)) > 12:
        kind = "noisy_sketch"
    else:
        kind = "clean_sketch"
    return InputCharacteristics(kind, bg_brightness, unevenness, colorful, speckle, ink_ratio, colour_fraction)


def normalize_canvas(rgb: np.ndarray, gray_for_ink: np.ndarray, size: int, margin: float = 0.08
                     ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop to the drawn content, pad to a square (aspect ratio preserved) and resize."""
    h, w = gray_for_ink.shape
    ink = _quick_ink_mask(gray_for_ink)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink = drop_specks(ink, min_area=int(h * w * 0.0004), min_extent=max(8, int(max(h, w) * 0.03)))
    pts = cv2.findNonZero(ink)
    if pts is None:
        raise InputValidationError("The sketch looks empty - please draw or upload an outline of an object.")
    x, y, bw, bh = cv2.boundingRect(pts)
    if max(bw, bh) < 16:
        raise InputValidationError("The sketch is extremely small. Please draw the object larger.")
    pad = int(max(bw, bh) * margin)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    bg_color = np.median(_border_pixels(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))
    fill = int(bg_color)
    canvas = np.full((side, side, 3), fill, np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp), (x0, y0, x1, y1)


def remove_noise(gray: np.ndarray, kind: str = "gaussian") -> np.ndarray:
    if kind == "median":
        return cv2.medianBlur(gray, 3)
    if kind == "bilateral":
        return cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by the estimated paper background -> removes shadows / vignetting."""
    bg = _estimate_background(gray).astype(np.float32) + 1.0
    return np.clip(gray.astype(np.float32) / bg * 255.0, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "adaptive":
        block = max(15, (gray.shape[0] // 20) | 1)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def clean_strokes(binary: np.ndarray, open_noise: bool, min_component_area: int) -> np.ndarray:
    """Join broken strokes (small closing) and drop isolated specks, keeping thin details."""
    out = binary
    if open_noise:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    min_extent = max(6, int(binary.shape[0] * (0.035 if open_noise else 0.015)))
    return drop_specks(out, min_component_area, min_extent)


def extract_contours(binary: np.ndarray, min_length: float = 12.0) -> Tuple[List[np.ndarray], np.ndarray]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.arcLength(c, False) >= min_length]
    vis = np.full((*binary.shape, 3), 255, np.uint8)
    outer = [c for i, c in enumerate(contours) if hierarchy is not None and hierarchy[0][i][3] == -1
             and cv2.arcLength(c, False) >= min_length]
    cv2.drawContours(vis, kept, -1, (40, 110, 255), 1, cv2.LINE_AA)
    cv2.drawContours(vis, outer, -1, (230, 60, 60), 2, cv2.LINE_AA)
    return kept, vis


def segment_foreground(binary: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
    """Filled silhouette of the drawn object(s).

    Strokes are closed with a kernel proportional to the image size, holes are
    filled by flood-filling the background from the border. If the outline is
    not closed, the filled area barely exceeds the stroke area - then we fall
    back to the convex hull of the strokes and warn the user.
    """
    h, w = binary.shape
    warnings: List[str] = []
    pts = cv2.findNonZero(binary)
    hull_area = float(cv2.contourArea(cv2.convexHull(pts))) if pts is not None else 0.0
    stroke_area = float((binary > 0).sum())

    def fill_with(k: int) -> np.ndarray:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        flood = closed.copy()
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        return closed | cv2.bitwise_not(flood)

    # grow the closing kernel until gaps in the outline are bridged (fill covers enough of the hull)
    filled = None
    for frac in (0.025, 0.045, 0.07):
        k = max(5, int(min(h, w) * frac) | 1)
        filled = fill_with(k)
        if float((filled > 0).sum()) >= 0.45 * hull_area:
            break
    fill_area = float((filled > 0).sum())
    closed_outline = fill_area > 1.6 * max(stroke_area, 1.0) and fill_area >= 0.3 * hull_area
    if not closed_outline:
        filled = np.zeros_like(binary)
        if pts is not None:
            cv2.fillPoly(filled, [cv2.convexHull(pts)], 255)
        warnings.append("The outline does not look closed - using the convex hull of your strokes as the silhouette. "
                        "The generator will still follow your strokes.")

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    significant = int((areas >= 0.05 * areas.max()).sum()) if areas.size else 0
    if significant > 1:
        warnings.append(f"Detected {significant} separate objects. Sketch2Shape works best with ONE object; "
                        "all parts will be passed to the generator.")
    stats_out = {
        "fill_ratio": round(fill_area / (h * w), 4),
        "closed_outline": float(closed_outline),
        "num_objects": float(significant),
    }
    return filled, stats_out, warnings


def generate_structural_map(binary: np.ndarray, edges: np.ndarray, control_type: str = "scribble") -> np.ndarray:
    """Builds the ControlNet conditioning image (white structure on black, RGB)."""
    size = binary.shape[0]
    if control_type == "canny":
        m = edges
    elif control_type == "lineart":
        m = cv2.GaussianBlur(binary, (3, 3), 0)  # anti-aliased, original stroke width
    else:  # scribble: bolder strokes, tolerant to wobbly lines
        t = max(2, size // 170)
        m = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t)))
    return cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)


def _main_object_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Salient-object mask for colour inputs (rembg). Returns None when rembg is unavailable or unsure."""
    try:
        from computer_vision.image_preparation import rembg_available, remove_background_rembg
        if not rembg_available():
            return None
        m = (remove_background_rembg(rgb) > 127).astype(np.uint8) * 255
    except Exception:
        return None
    area = m.mean() / 255.0
    return m if 0.03 < area < 0.97 else None


_HUES = [(10, "red"), (22, "orange"), (35, "yellow"), (85, "green"), (100, "teal"), (130, "blue"),
         (150, "purple"), (170, "pink"), (180, "red")]   # OpenCV hue scale 0-180, upper bounds


def dominant_colour_name(rgb: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Plain-English main colour of the object (e.g. 'yellow', 'white') from a colour picture.

    Ignores dark outline pixels; returns '' when the object has no clear dominant colour."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sel = (mask > 0) if mask is not None and mask.any() else np.ones(rgb.shape[:2], bool)
    sel &= hsv[..., 2] > 70                       # drop black outlines / shadows
    if sel.sum() < 200:
        return ""
    h, sat, val = hsv[..., 0][sel], hsv[..., 1][sel], hsv[..., 2][sel]
    coloured = sat > 60
    if coloured.mean() < 0.25:
        if np.median(val) > 200:
            return "white"
        return "gray" if np.median(val) > 90 else ""
    counts = {}
    for hue in h[coloured]:
        name = next(n for upper, n in _HUES if hue < upper)
        counts[name] = counts.get(name, 0) + 1
    name, n = max(counts.items(), key=lambda kv: kv[1])
    if name == "orange" and np.median(val[coloured]) < 150:
        name = "brown"
    return name if n / coloured.sum() > 0.4 else ""


def preprocess_sketch(image: ImageLike, size: int = 512, control_type: str = "scribble") -> PreprocessResult:
    """Full adaptive preprocessing pipeline. Raises InputValidationError for unusable inputs."""
    pil = load_image(image)
    if min(pil.size) < 32:
        raise InputValidationError(f"Image is too small ({pil.size[0]}x{pil.size[1]} px). Minimum is 64 px.")
    original = np.array(to_rgb_on_white(pil))
    if original.std() < 2.0:
        raise InputValidationError("The image is blank - please draw or upload a sketch.")

    ops: List[str] = []
    warnings: List[str] = []
    ch = analyze_input(original)
    ops.append(f"analyse input -> {ch.input_type}")

    gray_full = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    ops.append("grayscale conversion")
    if ch.input_type == "dark_background":
        gray_full = 255 - gray_full
        ops.append("invert (light strokes on dark background)")
    if ch.input_type == "paper_photo":
        gray_full = flatten_illumination(gray_full)
        ops.append("illumination flattening (background division)")

    normalized, crop_box = normalize_canvas(original, gray_full, size)
    gray, _ = normalize_canvas(np.dstack([gray_full] * 3), gray_full, size)
    gray = gray[..., 0]
    ops.append(f"crop to content + pad to square + resize to {size}px (aspect preserved)")

    if ch.input_type in ("noisy_sketch", "paper_photo"):
        gray = remove_noise(gray, "median")
        ops.append("median filter (salt/pepper noise)")
    elif ch.input_type == "photo":
        gray = remove_noise(gray, "bilateral")
        ops.append("bilateral filter (edge-preserving)")
    elif ch.input_type == "clean_sketch":
        ops.append("no denoising needed (clean canvas)")
    else:
        gray = remove_noise(gray, "gaussian")
        ops.append("gaussian filter")

    if float(gray.std()) < 40 or ch.input_type in ("paper_photo", "photo"):
        gray = enhance_contrast(gray)
        ops.append("CLAHE contrast enhancement")

    edges = extract_edges(gray)
    ops.append("Canny edge detection (median-adaptive thresholds)")

    if ch.input_type == "photo":
        binary = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        ops.append("binary structure from Canny edges (photo input)")
        obj_mask = _main_object_mask(normalized)
        if obj_mask is not None:
            keep = cv2.dilate(obj_mask, np.ones((7, 7), np.uint8))
            binary = cv2.bitwise_and(binary, keep)
            edges = cv2.bitwise_and(edges, keep)
            ops.append("kept only edges of the main object (rembg mask) - background clutter removed")
        warnings.append("This looks like a colour picture rather than a sketch. Background shapes (circles, ground, "
                        "trees) get modelled too - use the eraser/crop in the Upload box to keep just one object.")
    elif ch.input_type == "paper_photo" or ch.background_unevenness > 6:
        binary = binarize(gray, "adaptive")
        ops.append("adaptive Gaussian threshold")
    else:
        binary = binarize(gray, "otsu")
        ops.append("Otsu binary threshold")

    noisy = ch.input_type in ("noisy_sketch", "paper_photo")
    min_area = max(4, int(size * size * (0.0004 if noisy else 0.00004)))
    binary = clean_strokes(binary, open_noise=noisy, min_component_area=min_area)
    ops.append("morphology: close broken strokes, remove specks")

    ink_ratio = float(binary.mean() / 255.0)
    if ink_ratio < 0.002:
        raise InputValidationError("Almost no strokes were detected after cleaning. Draw darker / thicker lines.")
    if ink_ratio > 0.45:
        warnings.append("The sketch is very dense - a cleaner outline usually gives a better 3D shape.")

    contours, contour_vis = extract_contours(binary)
    ops.append(f"contour extraction ({len(contours)} contours)")
    mask, seg_stats, seg_warn = segment_foreground(binary)
    warnings += seg_warn
    ops.append("foreground segmentation (closing + hole filling)")

    control = generate_structural_map(binary, edges, control_type)
    ops.append(f"structural map for ControlNet ({control_type})")

    stats = {"ink_ratio": round(ink_ratio, 4), "num_contours": float(len(contours)), **seg_stats}
    return PreprocessResult(
        original=original, normalized=normalized, gray=gray, clean_sketch=255 - binary, binary=binary, edges=edges,
        contour_vis=contour_vis, mask=mask, control_image=control, contours=contours,
        characteristics=ch, operations=ops, warnings=warnings, stats=stats, crop_box=crop_box,
    )


def to_pil(arr: np.ndarray) -> Image.Image:
    return load_image(arr)
