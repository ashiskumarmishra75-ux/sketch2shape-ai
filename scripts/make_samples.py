"""Generates the test sketches in samples/ procedurally (jittered, hand-drawn-looking strokes).

They were NOT used to tune the pipeline; they cover the five categories required by the
problem statement plus a dark-background case:
  01 simple object (cup) · 02 furniture (chair) · 03 multi-component (lamp)
  04 rough / incomplete outline (car) · 05 photo of paper with noisy background (vase)
  06 white chalk on a dark board (house)

    python scripts/make_samples.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples"
rng = np.random.default_rng(7)


def stroke(img, pts, color=(25, 25, 25), width=3, jitter=1.6, gap=None):
    """Draw a wobbly polyline through pts; `gap=(a,b)` leaves a hole (fraction of the stroke)."""
    pts = np.asarray(pts, np.float32)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 4))
        for t in np.linspace(0, 1, n, endpoint=False):
            dense.append(a + (b - a) * t)
    dense.append(pts[-1])
    dense = np.array(dense)
    noise = np.cumsum(rng.normal(0, jitter * 0.35, dense.shape), axis=0)
    noise -= np.linspace(0, 1, len(dense))[:, None] * noise[-1]  # keep ends near targets
    dense = dense + noise + rng.normal(0, jitter * 0.25, dense.shape)
    n = len(dense)
    for i in range(n - 1):
        if gap and gap[0] * n <= i <= gap[1] * n:
            continue
        w = max(1, int(round(width + rng.normal(0, 0.4))))
        cv2.line(img, tuple(np.round(dense[i]).astype(int)), tuple(np.round(dense[i + 1]).astype(int)), color, w, cv2.LINE_AA)


def ellipse_pts(cx, cy, rx, ry, a0=0, a1=360, n=60):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def canvas(h=768, w=768, value=255):
    return np.full((h, w, 3), value, np.uint8)


def cup():
    im = canvas()
    stroke(im, ellipse_pts(360, 230, 150, 45))
    stroke(im, [(210, 230), (225, 560)])
    stroke(im, [(510, 230), (495, 560)])
    stroke(im, ellipse_pts(360, 560, 135, 40, 0, 180))
    stroke(im, ellipse_pts(540, 380, 85, 105, -80, 85), width=4)
    stroke(im, ellipse_pts(530, 380, 50, 65, -75, 80), width=3)
    return im


def chair():
    im = canvas()
    s = [(230, 430), (470, 470), (560, 400), (330, 370)]          # seat (three-quarter view)
    stroke(im, s + [s[0]])
    stroke(im, [(230, 430), (230, 450), (470, 492), (470, 470)])  # seat thickness
    stroke(im, [(470, 492), (560, 420), (560, 400)])
    for (x, y), L in [((240, 448), 230), ((462, 490), 230), ((552, 418), 210), ((338, 384), 170)]:
        stroke(im, [(x, y), (x + rng.normal(0, 2), y + L)], width=4)
        stroke(im, [(x + 14, y + 4), (x + 14 + rng.normal(0, 2), y + L)], width=3)
    b = [(330, 370), (318, 120), (548, 150), (560, 400)]           # backrest
    stroke(im, b)
    for t in (0.28, 0.5, 0.72):
        x0 = 330 + (560 - 330) * t
        stroke(im, [(x0, 368 + 10 * t), (x0 - 6, 140 + 10 * t)], width=2)
    return im


def lamp():
    im = canvas()
    stroke(im, [(270, 330), (330, 140), (450, 140), (510, 330)])   # shade
    stroke(im, ellipse_pts(390, 330, 120, 28, 0, 180))
    stroke(im, ellipse_pts(390, 140, 60, 14))
    stroke(im, [(380, 345), (378, 600)], width=4)                    # stem
    stroke(im, [(402, 345), (404, 600)], width=4)
    stroke(im, ellipse_pts(390, 470, 30, 22), width=3)               # knob
    stroke(im, ellipse_pts(390, 610, 130, 32))                        # base
    stroke(im, ellipse_pts(390, 640, 130, 32, 0, 180))
    stroke(im, [(260, 610), (260, 640)]); stroke(im, [(520, 610), (520, 640)])
    return im


def car_rough():
    im = canvas()
    body = [(90, 470), (100, 400), (220, 380), (300, 300), (500, 295), (590, 380), (690, 400), (700, 470)]
    stroke(im, body, width=3, jitter=2.6, gap=(0.42, 0.47))         # deliberately broken outline
    stroke(im, [(700, 470), (630, 470)], jitter=2.4)
    stroke(im, [(510, 470), (280, 470)], jitter=2.4, gap=(0.55, 0.68))
    stroke(im, [(170, 470), (90, 470)], jitter=2.4)
    stroke(im, ellipse_pts(225, 475, 58, 58, -20, 300), width=4, jitter=2.0)   # open wheel circle
    stroke(im, ellipse_pts(570, 475, 58, 58), width=4, jitter=2.0)
    stroke(im, [(320, 312), (300, 375), (400, 378), (402, 310)], width=2, jitter=2.0)
    stroke(im, [(425, 310), (425, 378), (560, 380), (500, 310)], width=2, jitter=2.0)
    return im


def noisy_vase():
    h, w = 900, 760
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    paper = 228 - 45 * (xx / w) * (yy / h) - 18 * np.exp(-(((xx - 600) / 250) ** 2 + ((yy - 120) / 200) ** 2))
    paper = paper[..., None] * np.array([1.0, 0.985, 0.95])
    paper += rng.normal(0, 7, (h, w, 1))
    im = np.clip(paper, 0, 255).astype(np.uint8)
    ink = (120, 60, 30)  # blue ballpoint (BGR->RGB handled by caller via cv2 order irrelevance)
    left = [(300, 180), (305, 260), (240, 380), (225, 500), (250, 620), (300, 700)]
    right = [(440, 180), (435, 260), (500, 380), (515, 500), (490, 620), (440, 700)]
    stroke(im, left, ink, 3); stroke(im, right, ink, 3)
    stroke(im, ellipse_pts(370, 180, 70, 16), ink, 3)
    stroke(im, ellipse_pts(370, 700, 70, 16, 0, 180), ink, 3)
    stroke(im, ellipse_pts(370, 430, 135, 25, 10, 170), ink, 2)
    for _ in range(900):  # specks / paper grain
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.circle(im, (x, y), int(rng.integers(1, 3)), tuple(int(v) for v in rng.integers(90, 170, 3)), -1)
    shadow = (np.clip((xx - 520) / 240, 0, 1) * 55)[..., None]
    im = np.clip(im.astype(np.float32) - shadow, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def chalk_house():
    im = canvas(value=0)
    im[:] = (40, 58, 34)
    c = (235, 240, 235)
    stroke(im, [(200, 350), (200, 620), (560, 620), (560, 350)], c, 4)
    stroke(im, [(170, 360), (380, 170), (590, 360), (170, 360)], c, 4)
    stroke(im, [(340, 620), (340, 480), (420, 480), (420, 620)], c, 3)
    stroke(im, [(240, 410), (310, 410), (310, 470), (240, 470), (240, 410)], c, 3)
    stroke(im, [(460, 410), (530, 410), (530, 470), (460, 470), (460, 410)], c, 3)
    stroke(im, [(470, 250), (470, 180), (510, 180), (510, 285)], c, 3)
    return im


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, fn in [("01_cup", cup), ("02_chair", chair), ("03_table_lamp", lamp), ("04_car_rough", car_rough),
                     ("05_noisy_photo_vase", noisy_vase), ("06_house_chalkboard", chalk_house)]:
        cv2.imwrite(str(OUT / f"{name}.png"), fn())
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
