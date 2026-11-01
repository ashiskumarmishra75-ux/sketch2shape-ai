"""Logging + stage timing helpers."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator

_CONFIGURED = False


def get_logger(name: str = "sketch2shape") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        root = logging.getLogger("sketch2shape")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    if not name.startswith("sketch2shape"):
        name = f"sketch2shape.{name}"
    return logging.getLogger(name)


class StageTimer:
    """Records wall-clock time per pipeline stage (real measurements, in seconds)."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - start, 3)

    @property
    def total(self) -> float:
        return round(sum(self.timings.values()), 3)

    def as_dict(self) -> Dict[str, float]:
        return {**self.timings, "total": self.total}
