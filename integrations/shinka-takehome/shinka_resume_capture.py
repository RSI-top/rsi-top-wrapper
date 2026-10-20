"""Capture completion-level SQLite, controller, and RNG state without draining.

The native trace and resume points preserve the state needed to inspect a
completed search. In-flight provider calls are never serialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import shinka_causal_checkpoint as scc
import trajectory


CAPTURE_SCHEMA = "shinka_resume_point_v1"
BRANCH_STATE_SCHEMA = "shinka_resume_branch_state_v1"
CAPTURE_DIR = "resume_points"
BRANCH_STATE_NAME = "resume_branch_state.json"
CAPTURE_MANIFEST_NAME = "resume_capture_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_payload(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "digest"}
    return _sha256_bytes(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _completed_from_db(runner: Any) -> int:
    count = len(runner.db.get_all_programs())
    island_copies = max(0, int(getattr(runner.db_config, "num_islands", 1)) - 1)
    return max(0, count - island_copies)


def _task_generation(task: Any) -> int | None:
    try:
        name = str(task.get_name())
    except Exception:  # noqa: BLE001 - optional diagnostic only
        return None
    if name.startswith("proposal_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return None
    return None


def capture_open_attempts(runner: Any) -> list[dict[str, Any]]:
    """Union recorder scheduling with native active/running registries."""
    rec = trajectory.recorder()
    rows: dict[tuple[int, str | None], dict[str, Any]] = {}
    for row in rec.open_attempts():
        key = (int(row["generation"]), row.get("task_id"))
        rows[key] = dict(row)

    for task_id, task in dict(getattr(runner, "active_proposal_tasks", {})).items():
        generation = _task_generation(task)
        if generation is None or rec.is_terminal(generation, str(task_id)):
            continue
        rows.setdefault(
            (generation, str(task_id)),
            {
                "attempt_key": f"{os.environ.get('SHINKA_RUN_INSTANCE_ID')}:{generation}:{task_id}",
                "generation": generation,
                "task_id": str(task_id),
                "scheduled_at": None,
            },
        )

    native_jobs: dict[tuple[int, str | None], Any] = {}
    for job in list(getattr(runner, "running_jobs", []) or []):
        native_jobs[(int(job.generation), getattr(job, "proposal_task_id", None))] = job
    for job in dict(getattr(runner, "submitted_jobs", {}) or {}).values():
        native_jobs[(int(job.generation), getattr(job, "proposal_task_id", None))] = job
    for job in dict(getattr(runner, "failed_jobs_for_retry", {}) or {}).values():
        native_jobs[(int(job.generation), getattr(job, "proposal_task_id", None))] = job
    for key, job in native_jobs.items():
        if rec.is_terminal(key[0], key[1]):
            continue
        row = rows.setdefault(
            key,
            {
                "attempt_key": (
                    f"{os.environ.get('SHINKA_RUN_INSTANCE_ID')}:{key[0]}:"
                    f"{key[1] or 'generation'}"
                ),
                "generation": key[0],
                "task_id": key[1],
                "scheduled_at": None,
            },
        )
        row["source_job_id"] = str(getattr(job, "job_id", "")) or None

    return sorted(
        rows.values(), key=lambda row: (int(row["generation"]), str(row.get("task_id")))
    )


class ResumeCapture:
    def __init__(self, *, results: Path, checkpointer: Any, recorder: Any) -> None:
        self.results = Path(results)
        self.checkpointer = checkpointer
        self.recorder = recorder
        self.root = self.results / CAPTURE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

        config_path = self.results / "causal_config.json"
        if not config_path.exists():
            atomic_json(config_path, checkpointer.config)
        atomic_json(
            self.results / CAPTURE_MANIFEST_NAME,
            {
                "schema": "shinka_resume_capture_manifest_v1",
                "capture_schema": CAPTURE_SCHEMA,
                "config_digest": checkpointer.config["digest"],
                "evaluator_closure_digest": checkpointer.closure["digest"],
                "semantics": (
                    "no dispatch fence; one SQLite backup after each committed "
                    "completion's native post-add maintenance"
                ),
            },
        )

    def install(self, runner: Any) -> None:
        native_apply = runner._apply_persisted_program_side_effects

        async def apply_then_capture(persisted_event: Any) -> Any:
            result = await native_apply(persisted_event)
            completion_seq = getattr(persisted_event, "_resume_completion_seq", None)
            if completion_seq is not None:
                self.capture(runner, persisted_event, int(completion_seq))
            return result

        runner._apply_persisted_program_side_effects = apply_then_capture

    def capture(self, runner: Any, persisted_event: Any, completion_seq: int) -> None:
        final = self.root / f"seq_{completion_seq:08d}"
        if (final / "COMPLETE").is_file():
            return
        staging = self.root / f".seq_{completion_seq:08d}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            database = staging / "programs.sqlite"
            db_doc = scc.snapshot_database(self.results / "programs.sqlite", database)
            db_facts = scc.database_facts(database)
            controller = scc.capture_controller_state(runner)
            controller["completed_generations"] = _completed_from_db(runner)
            controller["digest"] = _digest_payload(controller)
            program = persisted_event.program
            state: dict[str, Any] = {
                "schema": CAPTURE_SCHEMA,
                "completion_seq": completion_seq,
                "program_id": str(program.id),
                "source_job_id": str(persisted_event.job.job_id),
                "source_run_instance_id": os.environ.get("SHINKA_RUN_INSTANCE_ID"),
                "trace_prefix": trajectory.trace_prefix_facts(),
                "open_attempts": capture_open_attempts(runner),
                "completed_program": program.to_dict(),
                "controller_state": controller,
                "database": {**db_doc, "facts_digest": db_facts["digest"]},
                "config_digest": self.checkpointer.config["digest"],
                "evaluator_closure_digest": self.checkpointer.closure["digest"],
            }
            state["digest"] = _digest_payload(state)
            atomic_json(staging / "state.json", state)
            (staging / "COMPLETE").write_text(state["digest"] + "\n", encoding="utf-8")
            if final.exists():
                shutil.rmtree(final)
            os.replace(staging, final)
            if self.recorder is not None:
                self.recorder(
                    "resume_point_written",
                    completion_seq=completion_seq,
                    program_id=str(program.id),
                    open_attempts=len(state["open_attempts"]),
                    state_digest=state["digest"],
                    path=str(final),
                )
        except Exception as exc:  # noqa: BLE001 - capture must not kill vanilla search
            shutil.rmtree(staging, ignore_errors=True)
            if self.recorder is not None:
                self.recorder(
                    "resume_point_capture_failed",
                    completion_seq=completion_seq,
                    error=f"{type(exc).__name__}: {exc}",
                )


def install_resume_capture(
    runner: Any, *, results: Path, checkpointer: Any, recorder: Any
) -> ResumeCapture:
    capture = ResumeCapture(
        results=Path(results), checkpointer=checkpointer, recorder=recorder
    )
    capture.install(runner)
    return capture


def load_branch_state(results: Path) -> dict[str, Any] | None:
    path = Path(results) / BRANCH_STATE_NAME
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != BRANCH_STATE_SCHEMA:
        raise scc.ContinuationRefused(
            f"unknown resume branch state schema {state.get('schema')!r}"
        )
    if _digest_payload(state) != state.get("digest"):
        raise scc.ContinuationRefused("resume branch state digest mismatch")
    database = Path(results) / "programs.sqlite"
    if not database.is_file() or scc.digest_file(database)["sha256"] != state.get(
        "database_sha256"
    ):
        raise scc.ContinuationRefused(
            "resume branch database no longer matches its state"
        )
    if scc.database_facts(database)["digest"] != state.get("database_facts_digest"):
        raise scc.ContinuationRefused(
            "resume branch sampler-visible database state drifted"
        )
    controller = state.get("controller_state") or {}
    if _digest_payload(controller) != controller.get("digest"):
        raise scc.ContinuationRefused("resume branch controller state digest mismatch")
    return state
