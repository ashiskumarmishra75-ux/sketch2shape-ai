"""Builds the positive / negative text prompts for ControlNet from the sketch description.

SD-1.5's CLIP text encoder reads at most 77 tokens, so the prompt is assembled in
priority order (object identity -> user's own words -> reconstruction-friendly
framing -> visible components) and trimmed at a word budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from generative_ai.sketch_understanding import SketchDescription

# Ordered by importance: CLIP reads only ~77 tokens, so anything past that is silently dropped.
BASE_NEGATIVE = [
    "multiple objects", "duplicate", "cropped", "out of frame", "flat illustration", "vector art", "clipart",
    "deformed", "distorted geometry", "missing parts",
    "extra parts", "blurry", "lowres", "cluttered background", "floor", "harsh shadows",
    # TripoSR bakes the image's lighting into the model's colours: avoid dark, muddy, grimy renders
    "underexposed", "dim lighting", "dull colors", "desaturated", "muddy colors", "dirty", "rotten",
    "text", "watermark", "people", "hands", "pencil sketch", "lineart", "monochrome", "low contrast", "noisy",
    "jpeg artifacts",
]
FRAMING = ["single complete object", "centered", "entire object visible", "isolated on a plain white background",
           "studio product photo", "sharp focus", "highly detailed"]
# Even, bright light gives the 3D model clean albedo-like colours instead of baked-in dark shadows.
# A 3D-render look gives TripoSR shading/depth cues; flat vector art gives it none.
LOOK = "3d render, soft volumetric shading, vibrant natural colors, bright soft diffuse lighting"


@dataclass
class Prompts:
    positive: str
    negative: str


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" ,.")
    return text


def _limit_words(parts: List[str], max_words: int) -> str:
    out, count = [], 0
    for p in parts:
        p = _clean(p)
        if not p:
            continue
        n = len(p.split())
        if count + n > max_words:
            if count == 0:  # always keep the first chunk, trimmed
                out.append(" ".join(p.split()[:max_words]))
            break
        out.append(p)
        count += n
    return ", ".join(out)


def build_prompts(desc: SketchDescription, family: str = "sd15", extra_negative: str = "") -> Prompts:
    obj = desc.object if desc.object and desc.object != "object" else "object"
    core = " ".join(x for x in [desc.style, desc.color, desc.material, obj] if x)
    if desc.material and desc.material not in core:
        core += f" made of {desc.material}"
    view = desc.view if desc.view and desc.view != "as drawn" else ""

    user_words = _clean(desc.user_text)
    if user_words.lower() in core.lower():          # "house" typed as the whole description
        user_words = ""
    if len(user_words.split()) > 28:
        user_words = " ".join(user_words.split()[:28])

    tex = f"clean realistic {desc.material} texture" if desc.material else "clean realistic materials"
    parts = [core, user_words, *FRAMING[:4], LOOK, tex, view, *desc.components[:5], desc.geometry_description,
             *FRAMING[4:]]
    max_words = 60 if family == "sd15" else 90
    positive = _limit_words(parts, max_words)

    neg = list(BASE_NEGATIVE)
    style = (desc.style or "").lower()
    if any(s in style for s in ("cartoon", "low poly", "toy")):
        neg = [n for n in neg if n not in ("flat illustration", "vector art", "clipart")]
    if extra_negative.strip():
        neg.insert(0, extra_negative.strip())
    return Prompts(positive=positive, negative=", ".join(neg))
