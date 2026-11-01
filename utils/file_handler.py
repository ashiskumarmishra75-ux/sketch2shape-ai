"""Image I/O and upload validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ImageLike = Union[str, Path, Image.Image, np.ndarray]


class InputValidationError(ValueError):
    """Raised for user-facing input problems (bad file, empty sketch, ...)."""


def validate_upload_file(path: Union[str, Path], max_mb: float) -> None:
    p = Path(path)
    if not p.exists():
        raise InputValidationError(f"File not found: {p.name}")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InputValidationError(
            f"Unsupported file type '{p.suffix}'. Please upload PNG, JPG or JPEG.")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise InputValidationError(f"File is {size_mb:.1f} MB; the limit is {max_mb:.0f} MB.")


def load_image(image: ImageLike) -> Image.Image:
    """Load any supported image representation as a PIL image (mode preserved)."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, "L")
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return Image.fromarray(arr, "RGBA" if arr.shape[2] == 4 else "RGB")
        raise InputValidationError(f"Unsupported array shape {arr.shape}")
    try:
        img = Image.open(Path(image))
        img.load()
        return img
    except (UnidentifiedImageError, OSError) as exc:
        raise InputValidationError(f"Could not read the image: {exc}") from exc


def to_rgb_on_white(img: Image.Image) -> Image.Image:
    """Flatten transparency onto white. Canvas drawings arrive as RGBA with a transparent background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def save_image(img: Union[Image.Image, np.ndarray], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, np.ndarray):
        img = load_image(img)
    img.save(path)
    return path


def save_json(data: Any, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
