"""Lossless, append-only tracing for an unmodified ShinkaEvolve checkout.

The SQLite database remains the authority for programs that actually became
part of the search tree.  This module records the information SQLite does not:
physical LLM requests/responses, scheduling, failures, and a single durable
completion order shared by successful commits and terminal-only attempts.

All integration is installed with runtime monkeypatches.  Nothing under
``vendor/shinkaevolve`` or ``vendor/takehome`` is modified.
"""

from __future__ import annotations

import contextvars
import fcntl
import hashlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 2

#: FORWARD-ONLY exact audit-context capture (``shinka_db_state_v1``). One
#: record per committed canonical completion, emitted synchronously on the
#: controller thread AFTER the canonical completion is recorded and BEFORE the
#: meta-audit manager (if any) observes the persist — so the record is exactly
#: what ``capture_cutoff_snapshot()`` would have seen at that instant,
#: deferred-maintenance timing included. It carries the full bounded
#: ``_program_row`` census (no program code — code is proven from the
#: canonical child payloads, the recorded root program, and island-copy
#: origins), the authoritative global/island frontier identity and
#: fingerprint, the budget, the in-flight state, and a self-fingerprint.
#: Traces WITHOUT these records (any run before this capture existed) cannot
#: prove historical audit context and the offline replayer fails closed on
#: them. This capture is observation-only: a failure emits an explicit
#: ``db_state_capture_failed`` marker (itself a fail-closed gap for replay)
#: and never alters evolution.
DB_STATE_SCHEMA = "shinka_db_state_v1"
_WRITE_LOCK = threading.Lock()
_ATTEMPT_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "shinka_attempt_context", default={}
)
_RECORDER: "CanonicalCompletionRecorder | None" = None


def _trace_path() -> str | None:
    return os.environ.get("SHINKA_ATTEMPT_TRACE")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return repr(value)


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite_timestamp(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed == parsed and abs(parsed) != float("inf") else 0.0


def prompt_sha256(system_message: str | None, messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {"system": system_message or "", "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return sha256_text(canonical) or ""


def emit_event(event: str, **payload: Any) -> str | None:
    """Append one complete JSON object with process and thread locking."""
    path = _trace_path()
    if not path:
        return None
    event_id = str(uuid.uuid4())
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event": event,
        "timestamp": time.time(),
        "run_instance_id": os.environ.get("SHINKA_RUN_INSTANCE_ID"),
        "pid": os.getpid(),
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
    path_obj = Path(path).resolve()
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with path_obj.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return event_id


def trace_prefix_facts() -> dict[str, Any]:
    """Hash the fsync-durable trace prefix under the same writer lock."""
    path = _trace_path()
    if not path:
        raise RuntimeError("SHINKA_ATTEMPT_TRACE is not set")
    path_obj = Path(path).resolve()
    digest = hashlib.sha256()
    size = 0
    with _WRITE_LOCK:
        with path_obj.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def db_state_sha256(state: dict[str, Any]) -> str:
    """Deterministic self-fingerprint of one canonical state payload."""
    canonical = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _db_state_payload(
    runner: Any, *, completion_seq: int, source_job_id: str, program_id: str
) -> dict[str, Any]:
    """Capture native database, frontier, budget, and in-flight state."""
    import state_snapshot as sma

    db = runner.db
    best, islands = sma.frontier_from_db(db)
    fingerprint = sma.frontier_fingerprint(best[1] if best else None)
    rows = sma._sort_rows(
        [sma._program_row(program) for program in db.get_all_programs()]
    )
    return {
        "schema": DB_STATE_SCHEMA,
        "completion_seq": int(completion_seq),
        "source_job_id": str(source_job_id),
        "program_id": str(program_id),
        "program_rows": rows,
        "frontier": {
            "global_best_program_id": best[1] if best else None,
            "global_best_score": best[0] if best else None,
            "island_best": {
                key: {
                    "program_id": islands[key][1],
                    "combined_score": islands[key][0],
                }
                for key in sorted(islands)
            },
            "fingerprint": fingerprint,
        },
        "budget": sma.budget_snapshot(runner),
        "inflight_state": sma.inflight_snapshot(runner),
    }


def emit_db_state(
    runner: Any, *, completion_seq: int, source_job_id: str, program_id: str
) -> str | None:
    """Append one ``shinka_db_state_v1`` record; never raises.

    Observation only: a capture failure emits an explicit
    ``db_state_capture_failed`` marker (a fail-closed gap for any future
    replay) and evolution continues untouched.
    """
    if not _trace_path():
        return None
    try:
        state = _db_state_payload(
            runner,
            completion_seq=completion_seq,
            source_job_id=source_job_id,
            program_id=program_id,
        )
        return emit_event(
            "db_state",
            db_state_schema=DB_STATE_SCHEMA,
            completion_seq=int(completion_seq),
            source_job_id=str(source_job_id),
            program_id=str(program_id),
            state=state,
            state_sha256=db_state_sha256(state),
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break the run
        logging.getLogger("trajectory").exception(
            "db-state capture failed (evolution continues)"
        )
        try:
            emit_event(
                "db_state_capture_failed",
                db_state_schema=DB_STATE_SCHEMA,
                completion_seq=int(completion_seq),
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 — marker emission is best-effort
            pass
        return None


def _read_trace_rows(*, event: str | None = None) -> Iterator[dict[str, Any]]:
    path = _trace_path()
    if not path or not Path(path).is_file():
        return
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if event is not None and f'"{event}"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and (
                    event is None or row.get("event") == event
                ):
                    yield row
    except OSError:
        return


def current_attempt() -> dict[str, Any]:
    return dict(_ATTEMPT_CONTEXT.get())


def push_attempt(**updates: Any):
    merged = current_attempt()
    merged.update(updates)
    return _ATTEMPT_CONTEXT.set(merged)


def pop_attempt(token: contextvars.Token) -> None:
    _ATTEMPT_CONTEXT.reset(token)


def _program_ref(program: Any) -> dict[str, Any] | None:
    if program is None:
        return None
    code = getattr(program, "code", None)
    return {
        "id": getattr(program, "id", None),
        "generation": getattr(program, "generation", None),
        "parent_id": getattr(program, "parent_id", None),
        "code_sha256": sha256_text(code) if isinstance(code, str) else None,
        "combined_score": getattr(program, "combined_score", None),
        "public_metrics": getattr(program, "public_metrics", None),
        "correct": getattr(program, "correct", None),
    }


def _bound_arguments(
    function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict:
    try:
        return dict(inspect.signature(function).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _attempt_key(generation: int, task_id: str | None = None) -> str:
    run_id = os.environ.get("SHINKA_RUN_INSTANCE_ID") or "unidentified-run"
    return f"{run_id}:{generation}:{task_id or 'generation'}"


class CanonicalCompletionRecorder:
    """Durably sequence commits and terminal-only attempts exactly once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scheduled: dict[str, dict[str, Any]] = {}
        self._terminal_keys: set[str] = set()
        self._committed_source_jobs: set[str] = set()
        self._committed_programs: set[str] = set()
        self._next = 0
        for row in _read_trace_rows(event="canonical_completion"):
            if row.get("event") == "canonical_completion":
                seq = row.get("completion_seq")
                if isinstance(seq, int) and not isinstance(seq, bool):
                    self._next = max(self._next, seq + 1)
                key = row.get("attempt_key")
                if isinstance(key, str):
                    self._terminal_keys.add(key)
                source_job_id = row.get("source_job_id")
                if isinstance(source_job_id, str):
                    self._committed_source_jobs.add(source_job_id)
                child = row.get("child")
                if isinstance(child, dict) and isinstance(child.get("id"), str):
                    self._committed_programs.add(child["id"])

    def scheduled(self, *, generation: int, task_id: str, **payload: Any) -> str:
        key = _attempt_key(generation, task_id)
        with self._lock:
            self._scheduled[key] = {
                "generation": generation,
                "task_id": task_id,
                "scheduled_at": time.time(),
                **payload,
            }
        emit_event(
            "attempt_scheduled",
            attempt_key=key,
            generation=generation,
            task_id=task_id,
            **payload,
        )
        return key

    def _record(
        self,
        *,
        attempt_key: str,
        generation: int,
        status: str,
        terminal: str,
        **payload: Any,
    ) -> int | None:
        with self._lock:
            if attempt_key in self._terminal_keys:
                return None
            scheduled = self._scheduled.get(attempt_key, {})
            seq = self._next
            emit_event(
                "canonical_completion",
                completion_seq=seq,
                commit_index=seq,
                attempt_key=attempt_key,
                attempt_id=attempt_key,
                generation=generation,
                task_id=scheduled.get("task_id"),
                status=status,
                terminal=terminal,
                scheduled_at=scheduled.get("scheduled_at"),
                seconds_since_scheduled=(
                    time.time() - scheduled["scheduled_at"]
                    if scheduled.get("scheduled_at") is not None
                    else None
                ),
                **payload,
            )
            self._next += 1
            self._terminal_keys.add(attempt_key)
            return seq

    def committed(
        self,
        *,
        generation: int,
        task_id: str | None,
        source_job_id: str,
        program: Any,
        job: Any,
        evaluation_artifacts: dict[str, Any],
    ) -> int | None:
        program_id = str(getattr(program, "id", ""))
        with self._lock:
            if (
                source_job_id in self._committed_source_jobs
                or program_id in self._committed_programs
            ):
                return None
        key = _attempt_key(generation, task_id)
        program_dict = (
            program.to_dict() if hasattr(program, "to_dict") else _json_default(program)
        )
        seq = self._record(
            attempt_key=key,
            generation=generation,
            status="completed",
            terminal="program_committed",
            source_job_id=source_job_id,
            scheduled_parent_id=getattr(job, "parent_id", None),
            archive_inspiration_ids=getattr(job, "archive_insp_ids", None),
            top_k_inspiration_ids=getattr(job, "top_k_insp_ids", None),
            api_cost_usd=float(
                (getattr(job, "meta_patch_data", None) or {}).get("api_costs", 0.0)
                or 0.0
            ),
            embed_cost_usd=float(getattr(job, "embed_cost", 0.0) or 0.0),
            novelty_cost_usd=float(getattr(job, "novelty_cost", 0.0) or 0.0),
            child=program_dict,
            evaluation_artifacts=evaluation_artifacts,
        )
        if seq is not None:
            with self._lock:
                self._committed_source_jobs.add(source_job_id)
                self._committed_programs.add(program_id)
            emit_event(
                "program_committed",
                completion_seq=seq,
                attempt_key=key,
                generation=generation,
                source_job_id=source_job_id,
                program=program_dict,
                sqlite_path=os.environ.get("SHINKA_SQLITE_PATH"),
            )
        return seq

    def terminal(
        self,
        *,
        generation: int,
        task_id: str | None,
        terminal: str,
        status: str = "failed",
        **payload: Any,
    ) -> int | None:
        key = _attempt_key(generation, task_id)
        return self._record(
            attempt_key=key,
            generation=generation,
            status=status,
            terminal=terminal,
            **payload,
        )

    def recover_committed(self, program: Any) -> int | None:
        """Close the tiny crash window between SQLite commit and trace append."""
        metadata = getattr(program, "metadata", None) or {}
        source_job_id = str(metadata.get("source_job_id") or "")
        program_id = str(getattr(program, "id", ""))
        if not source_job_id or int(getattr(program, "generation", 0) or 0) == 0:
            return None
        with self._lock:
            if (
                source_job_id in self._committed_source_jobs
                or program_id in self._committed_programs
            ):
                return None
        generation = int(getattr(program, "generation", 0) or 0)
        key = f"recovered:{source_job_id}"
        seq = self._record(
            attempt_key=key,
            generation=generation,
            status="completed",
            terminal="program_committed_recovered",
            source_job_id=source_job_id,
            scheduled_parent_id=getattr(program, "parent_id", None),
            recovered=True,
            recovery_reason="SQLite row existed without canonical trace event",
            child=program.to_dict()
            if hasattr(program, "to_dict")
            else _json_default(program),
            api_cost_usd=float(metadata.get("api_costs", 0.0) or 0.0),
            embed_cost_usd=float(metadata.get("embed_cost", 0.0) or 0.0),
            novelty_cost_usd=float(metadata.get("novelty_cost", 0.0) or 0.0),
        )
        if seq is not None:
            with self._lock:
                self._committed_source_jobs.add(source_job_id)
                self._committed_programs.add(program_id)
            emit_event(
                "program_commit_recovered",
                completion_seq=seq,
                source_job_id=source_job_id,
                program_id=program_id,
                generation=generation,
            )
        return seq

    def is_terminal(self, generation: int, task_id: str | None) -> bool:
        with self._lock:
            return _attempt_key(generation, task_id) in self._terminal_keys

    def open_attempts(self) -> list[dict[str, Any]]:
        """Return the attempts scheduled before a snapshot but not terminal yet."""
        with self._lock:
            return [
                {"attempt_key": key, **dict(row)}
                for key, row in sorted(self._scheduled.items())
                if key not in self._terminal_keys
            ]

    def finalize_open(self, reason: str) -> None:
        with self._lock:
            open_rows = [
                (key, dict(row))
                for key, row in self._scheduled.items()
                if key not in self._terminal_keys
            ]
        for key, row in open_rows:
            self._record(
                attempt_key=key,
                generation=int(row["generation"]),
                status="abandoned",
                terminal="abandoned_at_shutdown",
                reason=reason,
            )


def recorder() -> CanonicalCompletionRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = CanonicalCompletionRecorder()
    return _RECORDER


def _read_evaluation_artifacts(job: Any) -> dict[str, Any]:
    root = Path(str(getattr(job, "results_dir", "")))
    out: dict[str, Any] = {
        "results_dir": str(root),
        "generated_code_path": str(getattr(job, "exec_fname", "")),
    }
    for name in ("metrics.json", "correct.json"):
        path = root / name
        if path.is_file():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                out[name] = {"read_error": str(exc)}
    for name in ("stdout.log", "stderr.log"):
        path = root / name
        if path.is_file():
            try:
                out[name] = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                out[name] = {"read_error": str(exc)}
    return out


def install_runner_trace() -> None:
    """Instrument Shinka's async controller at durable state boundaries."""
    from shinka.core import ShinkaEvolveRunner

    cls = ShinkaEvolveRunner
    if getattr(cls, "_shinka_complete_trace", False):
        return

    original_setup = cls._setup_async

    async def traced_setup(self):
        await original_setup(self)
        programs = sorted(
            self.db.get_all_programs(),
            key=lambda program: (
                finite_timestamp(getattr(program, "timestamp", 0.0)),
                int(getattr(program, "generation", 0) or 0),
            ),
        )
        root = next(
            (
                program
                for program in programs
                if int(getattr(program, "generation", 0) or 0) == 0
                and not getattr(program, "parent_id", None)
            ),
            None,
        )
        if root is not None:
            emit_event("root_program_committed", program=root.to_dict())
        for program in programs:
            recorder().recover_committed(program)

    cls._setup_async = traced_setup

    original_generate = cls._generate_proposal_async

    async def traced_generate(self, generation: int, task_id: str):
        rec = recorder()
        rec.scheduled(
            generation=generation,
            task_id=task_id,
            total_api_cost_at_schedule=getattr(self, "total_api_cost", None),
        )
        token = push_attempt(
            generation=generation,
            task_id=task_id,
            attempt_key=_attempt_key(generation, task_id),
        )
        try:
            result = await original_generate(self, generation, task_id)
            if result is None and not rec.is_terminal(generation, task_id):
                rec.terminal(
                    generation=generation,
                    task_id=task_id,
                    terminal="proposal_returned_none",
                    reason="proposal returned without an evaluation job",
                )
            return result
        except BaseException as exc:
            if not rec.is_terminal(generation, task_id):
                rec.terminal(
                    generation=generation,
                    task_id=task_id,
                    terminal="proposal_exception",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
            raise
        finally:
            pop_attempt(token)

    cls._generate_proposal_async = traced_generate

    for method_name in ("_run_patch_async", "_run_fix_patch_async"):
        original = getattr(cls, method_name)

        async def traced_patch(
            self, *args, __original=original, __method_name=method_name, **kwargs
        ):
            bound = _bound_arguments(__original, (self, *args), kwargs)
            generation = bound.get("generation", current_attempt().get("generation"))
            parent = bound.get("parent_program") or bound.get("incorrect_program")
            archive = (
                bound.get("archive_programs")
                or bound.get("ancestor_inspirations")
                or []
            )
            top_k = bound.get("top_k_programs") or []
            context = {
                "generation": generation,
                "patch_mode": __method_name,
                "parent": _program_ref(parent),
                "parent_id": getattr(parent, "id", None),
                "archive_inspirations": [_program_ref(p) for p in archive],
                "top_k_inspirations": [_program_ref(p) for p in top_k],
                "novelty_attempt": bound.get("novelty_attempt"),
                "resample_attempt": bound.get("resample_attempt"),
            }
            token = push_attempt(**context)
            emit_event("patch_generation_started", **current_attempt())
            try:
                result = await __original(self, *args, **kwargs)
                emit_event(
                    "patch_generation_completed", **current_attempt(), result=result
                )
                return result
            except BaseException as exc:
                emit_event(
                    "patch_generation_failed",
                    **current_attempt(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            finally:
                pop_attempt(token)

        setattr(cls, method_name, traced_patch)

    original_submit = cls._submit_evaluation_job_with_slot

    async def traced_submit(self, *args, **kwargs):
        emit_event(
            "evaluation_submit_started",
            **current_attempt(),
            exec_fname=kwargs.get("exec_fname"),
            results_dir=kwargs.get("results_dir"),
        )
        try:
            result = await original_submit(self, *args, **kwargs)
        except BaseException as exc:
            emit_event(
                "evaluation_submit_failed",
                **current_attempt(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        emit_event(
            "evaluation_submitted",
            **current_attempt(),
            scheduler_result=result,
            exec_fname=kwargs.get("exec_fname"),
            results_dir=kwargs.get("results_dir"),
        )
        return result

    cls._submit_evaluation_job_with_slot = traced_submit

    original_failed = cls._record_terminal_failed_proposal

    async def traced_failed(self, *args, **kwargs):
        bound = _bound_arguments(original_failed, (self, *args), kwargs)
        await original_failed(self, *args, **kwargs)
        generation = int(
            bound.get("generation", current_attempt().get("generation", -1))
        )
        task_id = current_attempt().get("task_id")
        failure_path = Path(str(bound.get("exec_fname", ""))).parent / "failure.json"
        failure_payload = None
        if failure_path.is_file():
            try:
                failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                failure_payload = {"read_error": str(exc), "path": str(failure_path)}
        recorder().terminal(
            generation=generation,
            task_id=task_id,
            terminal="failed_proposal",
            scheduled_parent_id=getattr(bound.get("parent_program"), "id", None),
            failure_stage=bound.get("failure_stage"),
            failure_reason=bound.get("failure_reason"),
            api_cost_usd=float(bound.get("api_costs", 0.0) or 0.0),
            embed_cost_usd=float(bound.get("embed_cost", 0.0) or 0.0),
            novelty_cost_usd=float(bound.get("novelty_cost", 0.0) or 0.0),
            failure_artifact=failure_payload,
        )

    cls._record_terminal_failed_proposal = traced_failed

    original_persist = cls._persist_completed_job

    async def traced_persist(self, job):
        result = await original_persist(self, job)
        if not getattr(result, "success", False):
            emit_event(
                "program_persist_retry",
                generation=getattr(job, "generation", None),
                source_job_id=str(getattr(job, "job_id", "")),
            )
            return result

        persisted_event = getattr(result, "persisted_event", None)
        program = getattr(persisted_event, "program", None) if persisted_event else None
        if program is None and not getattr(job, "discard_if_completed", False):
            getter = getattr(self.async_db, "get_program_by_source_job_id_async", None)
            if getter is not None:
                program = await getter(str(job.job_id))
        if program is not None:
            seq = recorder().committed(
                generation=int(job.generation),
                task_id=getattr(job, "proposal_task_id", None),
                source_job_id=str(job.job_id),
                program=program,
                job=job,
                evaluation_artifacts=_read_evaluation_artifacts(job),
            )
            if seq is not None:
                # The resume-capture wrapper runs after native post-add maintenance.
                # Carry the canonical boundary through without changing the vendor
                # method contract.
                setattr(persisted_event, "_resume_completion_seq", seq)
                # FORWARD capture: the exact audit-visible state, recorded
                # after the canonical completion and BEFORE any meta-audit
                # manager observes this persist (the meta wrapper is outer).
                emit_db_state(
                    self,
                    completion_seq=seq,
                    source_job_id=str(job.job_id),
                    program_id=str(getattr(program, "id", "")),
                )
        elif getattr(job, "discard_if_completed", False):
            recorder().terminal(
                generation=int(job.generation),
                task_id=getattr(job, "proposal_task_id", None),
                status="discarded",
                terminal="surplus_discarded",
                source_job_id=str(job.job_id),
            )
        return result

    cls._persist_completed_job = traced_persist

    original_start_proposals = cls._start_proposals

    async def traced_start_proposals(self, num_proposals):
        # Controller dispatch safe point (tracing-only marker): online, the
        # meta-audit adapter's reconcile() runs at ENTRY to _start_proposals —
        # its wrapper is installed OUTSIDE this one — BEFORE any native
        # generation assignment. Emitting immediately before the native call
        # records that exact boundary whether or not a meta-audit manager is
        # installed, so an offline replay resolves audit results ONLY here.
        # ``attempt_scheduled`` events remain dispatch/in-flight evidence, not
        # resolution boundaries (they are emitted later, inside proposal
        # generation).
        emit_event("controller_dispatch_safe_point", num_proposals=num_proposals)
        return await original_start_proposals(self, num_proposals)

    cls._start_proposals = traced_start_proposals
    cls._shinka_complete_trace = True
    emit_event("trajectory_instrumentation_installed", controller=cls.__qualname__)


def finalize_open_attempts(reason: str = "runner shutdown") -> None:
    if _RECORDER is not None:
        _RECORDER.finalize_open(reason)
