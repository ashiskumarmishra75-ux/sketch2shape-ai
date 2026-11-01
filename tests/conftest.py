import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "samples"


@pytest.fixture
def sample_paths():
    paths = sorted(SAMPLES.glob("*.png"))
    assert len(paths) >= 5, "run `python scripts/make_samples.py` first"
    return paths


@pytest.fixture
def rect_sketch():
    """A simple closed rectangle drawn on white paper (RGB)."""
    import cv2
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 120), (500, 280), (20, 20, 20), 3)
    return img
