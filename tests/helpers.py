"""Small real-Program fixtures and unambiguous integration imports."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import uuid
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_integration(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shinka_supervisor = load_integration(
    "release_shinka_supervisor", "integrations/shinka-takehome/run_repeat_flat_breadth.py"
)


def make_program(
    *,
    program_id: str | None = None,
    generation: int = 1,
    score: float = 0.5,
    correct: bool = True,
    parent_id: str | None = None,
    source_job_id: str | None = "job-1",
    island_idx: int | None = 0,
    code: str = "print('x')\n",
    side_effects_applied: bool = True,
    text_feedback: str = "ok",
    patch_type: str = "diff",
    code_diff: str | None = "--- a\n+++ b\n",
):
    """Build a real shinka Program (not a mock) so field drift is caught."""
    from shinka.database import Program

    metadata: dict[str, Any] = {"patch_type": patch_type, "api_costs": 0.25}
    if source_job_id is not None:
        metadata["source_job_id"] = source_job_id
    if side_effects_applied:
        metadata["postprocess_side_effects_applied"] = True
    return Program(
        id=program_id or str(uuid.uuid4()),
        code=code,
        generation=generation,
        correct=correct,
        combined_score=score,
        public_metrics={"cycles": 1000.0},
        private_metrics={},
        text_feedback=text_feedback,
        timestamp=1_000.0 + generation,
        parent_id=parent_id,
        island_idx=island_idx,
        code_diff=code_diff,
        metadata=metadata,
    )
