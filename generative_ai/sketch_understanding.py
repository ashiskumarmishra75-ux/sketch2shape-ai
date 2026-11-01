"""Stage 3 - Semantic sketch understanding.

Produces a structured `SketchDescription` (object, components, material, colour,
style, geometry, proportions, connections, ambiguities, view).

Backends, tried in order when S2S_VLM_BACKEND=auto:
  1. Anthropic Claude (vision)        - needs ANTHROPIC_API_KEY
  2. Any OpenAI-compatible VLM API     - needs OPENAI_API_KEY (+ OPENAI_BASE_URL); works with
                                         OpenAI, Gemini's OpenAI endpoint, Groq, or a LOCAL Ollama
                                         server (e.g. qwen2.5vl / llava) for a fully offline VLM
  3. Local CLIP zero-shot classifier   - no key, runs on CPU; gives the object category only
  4. Keyword parser                    - always available

Rules that keep the sketch in charge:
  * The VLM is told to describe only what is drawn and to list ambiguities instead of inventing.
  * Structure for generation comes from the OpenCV maps, never from this description.
  * The user's explicit words (object / material / colour / style) override VLM guesses.
  * Hidden dimensions are never requested.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from utils.logging_utils import get_logger

log = get_logger("sketch_understanding")

CATEGORIES = [
    "chair", "armchair", "stool", "table", "desk", "sofa", "bed", "shelf", "cabinet", "lamp",
    "cup", "mug", "bottle", "vase", "bowl", "teapot", "kettle", "jar", "bucket", "box",
    "car", "truck", "bus", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "rocket", "train",
    "house", "tower", "bridge", "tree", "flower pot", "mushroom",
    "robot", "teddy bear", "toy", "duck", "cat", "dog", "fish", "bird",
    "sword", "hammer", "axe", "guitar", "headphones", "camera", "phone", "laptop",
    "shoe", "hat", "backpack", "umbrella", "clock", "glasses", "key", "spoon", "fork", "knife",
    "pen", "light bulb", "candle", "trophy", "crown", "ring", "chess piece", "apple", "ice cream",
]
MATERIALS = ["wood", "wooden", "oak", "walnut", "metal", "metallic", "steel", "aluminium", "aluminum", "iron", "brass",
             "copper", "gold", "silver", "chrome", "plastic", "glass", "ceramic", "porcelain", "clay", "stone",
             "marble", "concrete", "leather", "fabric", "velvet", "cotton", "wool", "rubber", "paper", "bamboo",
             "rattan", "wicker", "crystal", "carbon fiber", "ice", "plush"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "dark brown", "light brown",
          "black", "white", "gray", "grey", "beige", "cream", "teal", "cyan", "turquoise", "navy", "maroon",
          "gold", "golden", "silver", "olive", "mint", "pastel", "matte black", "transparent"]
STYLES = ["modern", "minimalist", "vintage", "retro", "rustic", "industrial", "scandinavian", "futuristic", "sci-fi",
          "classic", "victorian", "art deco", "mid-century", "cartoon", "cute", "low poly", "realistic", "steampunk",
          "luxury", "ergonomic", "sporty", "toy-like"]

_DRAWING_WORDS = ("sketch", "drawing", "drawn", "line art", "lineart", "doodle", "pencil", "outline",
                  "black and white", "black-and-white", "monochrome", "ink", "graphite", "charcoal")

VLM_INSTRUCTIONS = """You are the sketch-understanding module of a sketch-to-3D pipeline.
Look at the hand-drawn sketch and return ONLY a JSON object with these keys:
{
  "object": "single short noun for the object that is drawn",
  "confidence": 0.0-1.0,
  "components": ["visible parts, with counts if visible, e.g. 'four legs'"],
  "geometry_description": "one sentence on the overall shape as drawn",
  "proportions": "relative proportions as drawn, e.g. 'taller than wide, seat at half height'",
  "connections": ["which parts connect, e.g. 'legs attach to seat corners'"],
  "ambiguities": ["parts that are unclear, hidden or not drawn"],
  "material": "material if the user text states it, else a plausible material for this object",
  "color": "colour of the real object if the user text states it, else a plausible colour (not the ink colour)",
  "style": "style of the real object if the user text states it, else a plausible product style (never 'sketch' or 'drawing')",
  "view": "the viewpoint of the drawing: 'front view', 'side view', 'three-quarter view', 'top view' or 'perspective view'"
}
Rules: describe ONLY what is actually drawn; do not add parts that are not in the sketch; do not guess
exact dimensions; if unsure, say so in "ambiguities". The user's text, if any, describes their intent
for the object category, material, colour and style - respect it, but structure must follow the drawing."""


@dataclass
class SketchDescription:
    object: str = "object"
    confidence: float = 0.0
    components: List[str] = field(default_factory=list)
    geometry_description: str = ""
    proportions: str = ""
    connections: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    material: str = ""
    color: str = ""
    style: str = ""
    view: str = ""
    source: str = "keywords"
    user_text: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- user-text parsing


def _find_terms(text: str, vocab: List[str]) -> List[str]:
    t = f" {text.lower()} "
    found = [v for v in sorted(vocab, key=len, reverse=True) if re.search(rf"(?<![a-z]){re.escape(v)}(?![a-z])", t)]
    # drop terms contained in longer matches ("brown" inside "dark brown")
    return [f for f in found if not any(f != g and f in g for g in found)]


def parse_user_text(text: str, category: str = "", material: str = "", color: str = "", style: str = "") -> Dict[str, str]:
    """Explicit form fields win; otherwise pull keywords from the free-text description."""
    text = text or ""
    out = {"object": category.strip(), "material": material.strip(), "color": color.strip(), "style": style.strip()}
    if not out["object"]:
        objs = _find_terms(text, CATEGORIES)
        if objs:
            out["object"] = objs[0]
    if not out["material"]:
        m = _find_terms(text, MATERIALS)
        out["material"] = ", ".join(m[:2])
    if not out["color"]:
        c = _find_terms(text, COLORS)
        out["color"] = ", ".join(c[:2])
    if not out["style"]:
        s = _find_terms(text, STYLES)
        out["style"] = ", ".join(s[:2])
    return out


def geometry_cues(binary: Optional[np.ndarray], mask: Optional[np.ndarray]) -> Dict[str, str]:
    """Purely measured cues from the CV stage (used by the offline fallback)."""
    if mask is None or mask.sum() == 0:
        return {}
    ys, xs = np.where(mask > 0)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    ar = h / max(w, 1)
    shape = "a tall, narrow" if ar > 1.6 else "a wide, low" if ar < 0.62 else "a compact, roughly square"
    fill = float((mask > 0).sum()) / float(h * w)
    solidity = "solid, closed silhouette" if fill > 0.75 else "open structure with gaps" if fill < 0.45 else "moderately open silhouette"
    return {"geometry_description": f"{shape} object with a {solidity} (as drawn)",
            "proportions": f"height/width ratio about {ar:.2f}"}


# --------------------------------------------------------------------------- backends


def _png_b64(img: Image.Image, max_side: int = 768) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("VLM reply contained no JSON object")
    return json.loads(match.group(0))


def _user_block(user_text: str) -> str:
    return f'User text: "{user_text.strip()}"' if user_text.strip() else "User text: (none provided)"


def query_anthropic(img: Image.Image, user_text: str, model: str, timeout: float) -> Dict:
    import requests
    key = os.environ["ANTHROPIC_API_KEY"]
    payload = {
        "model": model,
        "max_tokens": 700,
        "system": VLM_INSTRUCTIONS,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img)}},
            {"type": "text", "text": _user_block(user_text) + "\nReturn the JSON now."},
        ]}],
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):  # only needed for keys not scoped to a workspace
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    r = requests.post("https://api.anthropic.com/v1/messages", json=payload, timeout=timeout, headers=headers)
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    return _extract_json(text)


def query_openai_compatible(img: Image.Image, user_text: str, model: str, base_url: str, timeout: float) -> Dict:
    import requests
    key = os.environ.get("OPENAI_API_KEY", "ollama")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": VLM_INSTRUCTIONS},
            {"role": "user", "content": [
                {"type": "text", "text": _user_block(user_text) + "\nReturn the JSON now."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(img)}},
            ]},
        ],
    }
    try:
        r = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        return _extract_json(r.json()["choices"][0]["message"]["content"])
    finally:
        _unload_local_vlm(base_url, model)


def _unload_local_vlm(base_url: str, model: str) -> None:
    """A local Ollama server keeps the VLM in GPU memory for minutes by default. Sketch2Shape needs that
    memory for diffusion + TripoSR, so ask Ollama to unload it now (keep_alive=0). No-op for hosted APIs."""
    if os.environ.get("S2S_VLM_UNLOAD", "1") == "0":
        return
    if ":11434" not in base_url and "ollama" not in base_url.lower():
        return
    import requests
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        requests.post(root + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=15)
        log.info("Unloaded local VLM '%s' to free GPU memory", model)
    except Exception as exc:  # never fail the pipeline over this
        log.warning("Could not unload local VLM: %s", exc)


class ClipClassifier:
    """Zero-shot category classifier (openai/clip-vit-base-patch32, ~600 MB, CPU-friendly)."""

    _instance: Optional["ClipClassifier"] = None

    def __init__(self, model_id: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        prompts = [f"a simple line drawing sketch of a {c}" for c in CATEGORIES]
        with torch.no_grad():
            t = self.processor(text=prompts, return_tensors="pt", padding=True)
            feats = self.model.get_text_features(**t)
        self.text_feats = feats / feats.norm(dim=-1, keepdim=True)

    @classmethod
    def get(cls, model_id: str) -> "ClipClassifier":
        if cls._instance is None:
            cls._instance = cls(model_id)
        return cls._instance

    def classify(self, img: Image.Image, top_k: int = 3) -> List[tuple]:
        with self.torch.no_grad():
            px = self.processor(images=img.convert("RGB"), return_tensors="pt")
            f = self.model.get_image_features(**px)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ self.text_feats.T).softmax(dim=-1)[0]
        vals, idx = probs.topk(top_k)
        return [(CATEGORIES[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist())]


# --------------------------------------------------------------------------- main entry


def _choose_backends(pref: str) -> List[str]:
    if pref != "auto":
        return [pref]
    order = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        order.append("anthropic")
    base = os.environ.get("OPENAI_BASE_URL", "")
    if os.environ.get("OPENAI_API_KEY") or (base and "api.openai.com" not in base):  # key, or a local server
        order.append("openai")
    order.append("clip")
    return order


def understand_sketch(sketch_img: Image.Image, user_text: str = "", *, category: str = "", material: str = "",
                      color: str = "", style: str = "", binary: Optional[np.ndarray] = None,
                      mask: Optional[np.ndarray] = None, settings=None) -> SketchDescription:
    from config.settings import get_settings
    s = settings or get_settings()
    user = parse_user_text(user_text, category, material, color, style)
    desc = SketchDescription(user_text=user_text or "")
    raw: Dict = {}

    for backend in _choose_backends(s.vlm_backend):
        try:
            if backend == "anthropic":
                raw = query_anthropic(sketch_img, user_text, s.anthropic_model, s.vlm_timeout_s)
                desc.source = f"anthropic:{s.anthropic_model}"
            elif backend == "openai":
                raw = query_openai_compatible(sketch_img, user_text, s.openai_model, s.openai_base_url, s.vlm_timeout_s)
                desc.source = f"openai-compatible:{s.openai_model}"
            elif backend == "clip":
                top = ClipClassifier.get(s.clip_model_id).classify(sketch_img)
                raw = {"object": top[0][0], "confidence": round(top[0][1], 3)}
                desc.notes.append("CLIP top-3: " + ", ".join(f"{c} ({p:.2f})" for c, p in top))
                desc.source = f"clip-zero-shot:{s.clip_model_id}"
            elif backend == "none":
                desc.source = "keywords"
            break
        except Exception as exc:  # network, quota, missing deps... -> next backend
            log.warning("Sketch understanding backend '%s' failed: %s", backend, exc)
            desc.notes.append(f"{backend} unavailable ({exc.__class__.__name__})")

    # ---- merge: VLM fields -----------------------------------------------
    for key in ("object", "geometry_description", "proportions", "material", "color", "style", "view"):
        if isinstance(raw.get(key), str) and raw[key].strip():
            setattr(desc, key, raw[key].strip())
    for key in ("components", "connections", "ambiguities"):
        if isinstance(raw.get(key), list):
            setattr(desc, key, [str(x) for x in raw[key] if str(x).strip()][:8])
    try:
        desc.confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        desc.confidence = 0.0

    # VLMs often describe the *drawing* ("style: sketch", "color: black and white"). Passing that to the
    # diffusion prompt would push it towards a pencil-sketch render, so drop such appearance values.
    for key in ("style", "material", "color"):
        val = getattr(desc, key)
        if val and any(w in val.lower() for w in _DRAWING_WORDS):
            desc.notes.append(f"Ignored VLM {key} '{val}' (describes the drawing, not the object)")
            setattr(desc, key, "")

    # a low-confidence zero-shot guess could steer the diffusion model to the wrong object
    if desc.source.startswith("clip") and desc.confidence < 0.2 and not user["object"]:
        desc.notes.append(f"Low-confidence guess '{desc.object}' ignored - add the object name in the prompt "
                          "for better results.")
        desc.object = "object"

    # ---- measured geometry cues fill gaps (never override the VLM) ---------
    for k, v in geometry_cues(binary, mask).items():
        if not getattr(desc, k):
            setattr(desc, k, v)

    # ---- the user's explicit intent wins for semantics & appearance --------
    if user["object"]:
        if desc.object not in ("", "object") and user["object"].lower() not in desc.object.lower() \
                and desc.object.lower() not in user["object"].lower():
            desc.notes.append(f"Sketch looked like '{desc.object}', but the user asked for '{user['object']}' - "
                              "using the user's category; structure still follows the sketch.")
        desc.object = user["object"]
    for key in ("material", "color", "style"):
        if user[key]:
            setattr(desc, key, user[key])
    if not desc.view:
        desc.view = "as drawn"
    return desc
