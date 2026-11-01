"""Drop-in replacement for `torchmcubes.marching_cubes`, used ONLY when torchmcubes is not installed.

TripoSR imports `from torchmcubes import marching_cubes`. torchmcubes must be compiled from
source (CUDA toolkit + C++ compiler) which is the #1 TripoSR install failure on Windows and on
some cloud images. This shim provides the same call signature using PyMCubes (preferred) or
scikit-image.

Coordinate convention: torchmcubes returns vertices ordered (dim2, dim1, dim0) - TripoSR
reverses them with `v_pos[..., [2, 1, 0]]`. PyMCubes / scikit-image return (dim0, dim1, dim2),
so we reverse here to stay equivalent. Face winding may differ; reconstruction/mesh_processing.py
detects inverted meshes (negative signed volume) and flips them.
"""
from __future__ import annotations

import sys
import types

import numpy as np


def marching_cubes_numpy(vol: np.ndarray, threshold: float):
    """Returns (verts[N,3] float32 in torchmcubes order (dim2, dim1, dim0), faces[M,3] int64)."""
    vol = np.asarray(vol, dtype=np.float32)
    if not (vol.min() < threshold < vol.max()):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    try:
        import mcubes  # PyMCubes
        verts, faces = mcubes.marching_cubes(vol, float(threshold))
    except ImportError:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(vol, level=float(threshold))
    verts = np.ascontiguousarray(np.asarray(verts)[:, ::-1], dtype=np.float32)
    return verts, np.ascontiguousarray(faces, dtype=np.int64)


def marching_cubes(volume, threshold: float):
    """torchmcubes-compatible signature: (torch volume, float) -> (verts tensor, faces tensor)."""
    import torch

    verts, faces = marching_cubes_numpy(volume.detach().float().cpu().numpy(), threshold)
    return torch.from_numpy(verts), torch.from_numpy(faces)


def ensure_marching_cubes() -> str:
    """Returns which implementation TripoSR will use."""
    try:
        import torchmcubes  # noqa: F401
        return "torchmcubes"
    except ImportError:
        mod = types.ModuleType("torchmcubes")
        mod.marching_cubes = marching_cubes
        sys.modules["torchmcubes"] = mod
        try:
            import mcubes  # noqa: F401
            return "shim:PyMCubes"
        except ImportError:
            return "shim:scikit-image"
