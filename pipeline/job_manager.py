"""Per-request job state: one folder per job holding every intermediate representation."""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.file_handler import save_json
from utils.logging_utils import StageTimer, get_logger

log = get_logger("jobs")


@dataclass
class Job:
    id: str
    dir: Path
    created: float = field(default_factory=time.time)
    inputs: Dict[str, Any] = field(default_factory=dict)
    preprocess: Any = None                 # PreprocessResult
    files: Dict[str, str] = field(default_factory=dict)
    description: Any = None                # SketchDescription
    prompts: Any = None                    # Prompts
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected: Optional[int] = None
    prepared: Any = None                   # PreparedImage
    recon: Any = None                      # ReconResult
    mesh: Any = None                       # ProcessedMesh
    timer: StageTimer = field(default_factory=StageTimer)
    warnings: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def note(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {msg}")
        log.info("[%s] %s", self.id, msg)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def report(self) -> Dict[str, Any]:
        p = self.preprocess
        d = self.description
        rep: Dict[str, Any] = {
            "job_id": self.id,
            "inputs": self.inputs,
            "timings_s": self.timer.as_dict(),
            "warnings": self.warnings,
            "files": self.files,
            "log": self.log,
            "error": self.error,
        }
        if p is not None:
            rep["preprocessing"] = {"input_type": p.characteristics.input_type if p.characteristics else None,
                                    "operations": p.operations, "stats": p.stats}
        if d is not None:
            rep["sketch_understanding"] = d.to_dict()
        if self.prompts is not None:
            rep["prompts"] = {"positive": self.prompts.positive, "negative": self.prompts.negative}
        if self.candidates:
            rep["candidates"] = [{k: v for k, v in c.items() if k not in ("image", "mask")} for c in self.candidates]
            rep["selected_candidate"] = self.selected
        if self.prepared is not None:
            rep["image_preparation"] = {"method": self.prepared.method, **self.prepared.stats}
        if self.recon is not None:
            rep["reconstruction"] = {"backend": self.recon.backend, "textured": self.recon.textured,
                                     "notes": self.recon.notes, **self.recon.timings}
        if self.mesh is not None:
            rep["mesh"] = {"before_cleanup": self.mesh.report_before, "final": self.mesh.report_after,
                           "actions": self.mesh.actions, "export_check": self.mesh.export_check}
        return rep

    def save_report(self) -> Path:
        path = save_json(self.report(), self.dir / "report.json")
        self.files["report"] = str(path)
        return path


class JobManager:
    def __init__(self, root: Path, keep: int = 50):
        self.root = Path(root) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = Job(id=job_id, dir=self.root / job_id)
        job.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job_id] = job
        self._prune()
        return job

    def get(self, job_id: Optional[str]) -> Optional[Job]:
        if not job_id:
            return None
        return self._jobs.get(job_id)

    def _prune(self) -> None:
        dirs = sorted([d for d in self.root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
        for d in dirs[:-self.keep] if len(dirs) > self.keep else []:
            shutil.rmtree(d, ignore_errors=True)
            with self._lock:
                self._jobs.pop(d.name, None)
