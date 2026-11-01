"""Hardware detection and GPU-memory helpers.

torch is imported lazily so the CV-only parts of the project (and the unit tests)
work on machines without PyTorch.
"""
from __future__ import annotations

import gc
import platform
import shutil
from typing import Any, Dict


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def resolve_device(preference: str = "auto") -> str:
    if preference in ("cpu", "cuda", "mps"):
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return 0.0


def free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or type(exc).__name__ == "OutOfMemoryError"


def hardware_report() -> Dict[str, Any]:
    """Collects the facts required by the hackathon README (GPU, VRAM, CUDA, RAM, disk)."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                kb = int(fh.readline().split()[1])
            info["ram_gb"] = round(kb / 1024 ** 2, 1)
        except Exception:
            info["ram_gb"] = None
    total, _, free = shutil.disk_usage(".")
    info["disk_free_gb"] = round(free / 1024 ** 3, 1)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1024 ** 3, 1)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:  # torch missing
        info["torch"] = f"not installed ({exc.__class__.__name__})"
        info["cuda_available"] = False
    return info


def peak_vram_gb_and_reset() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            torch.cuda.reset_peak_memory_stats()
            return round(peak, 2)
    except Exception:
        pass
    return 0.0


def memory_snapshot() -> str:
    """'RAM 5.1/12.7 GB used (process 3.2) | VRAM 4.3 GB' - logged per stage to diagnose OOM kills."""
    parts = []
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) / 1024 ** 2
        used = info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))
        rss = 0.0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024 ** 2
        parts.append(f"RAM {used:.1f}/{info['MemTotal']:.1f} GB used (this process {rss:.1f})")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            parts.append(f"VRAM {(total - free) / 1024 ** 3:.1f}/{total / 1024 ** 3:.1f} GB used")
    except Exception:
        pass
    return " | ".join(parts)
