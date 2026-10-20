"""Opt-in exact-causal-checkpoint writer for ShinkaEvolve. Zero LLM/provider calls.

Phase A found this campaign unforkable at exact ``t0``: ``programs.sqlite`` is a
single final-state store whose ``archive`` table is eviction-based and untimestamped,
and the run recorded no digest of the scorer it invoked. This module is the forward
fix, built on the shared ``causal_checkpoint_manifest_v1`` contract.

**The barrier is the hard part, and it is not `_persist_completed_job` returning.**

Shinka inserts programs with ``defer_maintenance=True`` and replays archive / best /
island / migration maintenance later; four proposal and four evaluation jobs may be
in flight; completed jobs are batched; background side effects, prompt-percentile
recomputation, embedding work and DB retries all run behind their own queues and
logical slot pools. A snapshot taken when a job merely finished would capture a
database whose sampler-visible state is still being written.

So a checkpoint may claim exact t0 only at a **controller-owned quiescent barrier**:

1. dispatch is fenced — ``_start_proposals`` is wrapped and returns without
   assigning a generation, so no new proposal can begin;
2. every in-flight channel is drained to empty: ``running_jobs``,
   ``active_proposal_tasks``, ``proposal_queue``, ``failed_jobs_for_retry``,
   completed-job batch tasks, background side-effect queue and workers, the
   prompt-percentile recompute task, and all three ``LogicalSlotPool``s
   (sampling / evaluation / postprocess);
3. deferred database maintenance is replayed and durable;
4. no API, embedding or evaluation call is outstanding — implied by (2), and
   asserted rather than assumed;
5. the canonical terminal and cost prefix through the boundary are durable.

**Checkpoint-enabled source runs therefore briefly drain at each barrier.** That is
a real behavioural difference and it is stated rather than hidden: with the flag off
no wrapper is installed at all and timing is native.

Nothing here ever serialises an active API call and calls the result forkable.
"""

from __future__ import annotations
import asyncio
import hashlib
import importlib.util
import json
import os
import pickle
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHARED = Path(__file__).resolve().parents[2] / "shared" / "meta-agent-audit"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))
from meta_audit.causal_checkpoint import (
    COMPLETE_MARKER_NAME,
    CausalCheckpointError,
    CausalCheckpointManifest,
    digest_file,
    read_checkpoint,
    verify_checkpoint,
    write_checkpoint,
)

BASELINE = "shinka"
CLOSURE_PROJECT_FILES = ("evaluate.py", "eval_build.py", "eval_score.py")
CLOSURE_OPTIONAL_PROJECT_FILES = (
    "trajectory.py",
)
CLOSURE_TAKEHOME_ROOT_FILES = ("problem.py", "frozen_problem.py")
SEALED_TAKEHOME_RELATIVE = "vendor/takehome"
PROVENANCE_DIRNAME = "evaluator_provenance"
SEAL_SELF_MANIFEST = "closure.json"
ORIGIN_DIRNAME = "causal_origin"
CHECKPOINT_DIRNAME = "causal_checkpoints"
SCORING_ENV_VARS = (
    "SHINKA_TAKEHOME_DIR",
    "SHINKA_BUILD_TIMEOUT",
    "SHINKA_SCORE_TIMEOUT",
    "SHINKA_SCORE_TRIALS",
)
CHILD_EXECUTION_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}
DB_SNAPSHOT = "programs.snapshot.sqlite"
BANDIT_SNAPSHOT = "bandit_state.pkl"
MODE_NEW = "new_campaign"
MODE_RESUME = "resume_same_campaign"
MODE_FORK = "resume_restored_fork"
CONTROLLER_FIELDS = (
    "assigned_generations",
    "avg_proposal_cost",
    "best_program_id",
    "completed_generations",
    "completed_proposal_costs",
    "cost_limit_reached",
    "next_generation_to_submit",
    "total_api_cost",
    "total_proposals_generated",
    "_evaluation_seconds_ewma",
    "_proposal_timing_samples",
    "_sampling_seconds_ewma",
)
CONFIG_FIELDS = (
    "embedding_model",
    "llm_models",
    "max_api_costs",
    "max_db_workers",
    "max_evaluation_jobs",
    "max_novelty_attempts",
    "max_proposal_jobs",
    "num_generations",
    "patch_types",
    "seed",
)
UNSUPPORTED_WHEN_ENABLED = {
    "evolve_prompts": "prompt evolution keeps evolved prompt state and percentile recomputation state that this writer does not pin",
    "meta_rec_interval": "meta recommendations keep meta-agent state and a separate cost stream that this writer does not pin",
    "novelty_llm_models": "LLM novelty judging keeps judge state and a separate cost stream that this writer does not pin",
}


class ShinkaCheckpointError(CausalCheckpointError):
    """A Shinka checkpoint cannot be written or continued."""


class ContinuationRefused(ShinkaCheckpointError):
    """A resume or fork drifted from its checkpoint. Nothing may be dispatched."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _document_digest(payload: Any) -> str:
    return _sha256_bytes(
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    )


def interpreter_identity() -> dict[str, Any]:
    return {
        "digest": _sha256_text(f"{sys.executable}\n{sys.version}"),
        "executable": sys.executable,
        "version": sys.version,
    }


def dependency_identity() -> dict[str, Any]:
    """The installed dependency set that executes the evaluator subprocesses."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "digest": None,
            "error": f"{type(exc).__name__}: {exc}",
            "method": "pip_freeze",
        }
    if result.returncode != 0:
        return {
            "digest": None,
            "error": f"pip freeze exited {result.returncode}",
            "method": "pip_freeze",
        }
    frozen = "\n".join(
        sorted((line.strip() for line in result.stdout.splitlines() if line.strip()))
    )
    return {
        "digest": _sha256_text(frozen),
        "method": "pip_freeze",
        "n_packages": len(frozen.splitlines()),
    }


def assert_supported_features(evo_config: Any) -> dict[str, Any]:
    """Fail closed on an enabled feature whose state this writer cannot pin."""
    enabled = []
    report = {}
    for name, why in UNSUPPORTED_WHEN_ENABLED.items():
        value = getattr(evo_config, name, None)
        report[name] = value
        truthy = (
            bool(value)
            if not isinstance(value, (int, float))
            else value not in (0, None)
        )
        if truthy:
            enabled.append(f"{name} ({why})")
    if enabled:
        raise ShinkaCheckpointError(
            f"refusing to write causal checkpoints for a run with unsupported enabled features: {enabled}. Pinning them exactly is not implemented, and a checkpoint that silently dropped them would not be exact."
        )
    return report


def seal_closure(
    results: Path, *, here: Path, takehome: Path, recorder: Any = None
) -> dict:
    """Copy the transitive, runnable evaluator/session closure into a run-owned dir."""
    provenance = Path(results) / PROVENANCE_DIRNAME
    provenance.mkdir(parents=True, exist_ok=True)
    os.environ["SHINKA_TAKEHOME_DIR"] = str(sealed_takehome(provenance))
    os.environ.update(CHILD_EXECUTION_ENV)
    sealed: dict[str, dict[str, Any]] = {}

    def seal(source: Path, relative: str, *, required: bool = True) -> bool:
        if not source.is_file():
            if required:
                raise ShinkaCheckpointError(
                    f"cannot seal the evaluator closure: {source} is missing. The scorer must be recorded at campaign start, not reconstructed later."
                )
            return False
        target = provenance / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sealed[relative] = {**digest_file(target), "source": str(source)}
        return True

    for name in CLOSURE_PROJECT_FILES:
        seal(here / name, name)
    for name in CLOSURE_OPTIONAL_PROJECT_FILES:
        seal(here / name, name, required=False)
    for name in CLOSURE_TAKEHOME_ROOT_FILES:
        seal(takehome / name, f"{SEALED_TAKEHOME_RELATIVE}/{name}")
    interpreter = interpreter_identity()
    dependencies = dependency_identity()
    scoring_env = {name: os.environ.get(name) for name in SCORING_ENV_VARS}
    child_env = {name: os.environ.get(name) for name in CHILD_EXECUTION_ENV}
    for name, payload in (
        ("interpreter.json", interpreter),
        ("dependencies.json", dependencies),
        ("scoring_env.json", scoring_env),
        ("child_execution_env.json", child_env),
    ):
        (provenance / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        sealed[name] = digest_file(provenance / name)
    if not dependencies.get("digest"):
        raise ShinkaCheckpointError(
            f"the dependency/environment identity could not be recorded ({dependencies.get('error')}); refusing to write checkpoints that would overclaim"
        )
    closure = {
        "dependencies": dependencies["digest"],
        "entrypoint": sealed["evaluate.py"]["sha256"],
        "files": {name: entry["sha256"] for name, entry in sorted(sealed.items())},
        "frozen_reference": _sha256_text(
            json.dumps(
                {
                    k: v["sha256"]
                    for k, v in sealed.items()
                    if k.startswith(f"{SEALED_TAKEHOME_RELATIVE}/")
                },
                sort_keys=True,
            )
        ),
        "interpreter": interpreter["digest"],
        "invocation": "the SEALED evaluate.py is the LocalJobConfig.eval_program_path and SHINKA_TAKEHOME_DIR points inside the seal, so every future evaluation runs the sealed eval_build.py / eval_score.py against the sealed take-home",
        "phase_code": _sha256_text(
            json.dumps(
                [sealed[n]["sha256"] for n in CLOSURE_PROJECT_FILES[1:]], sort_keys=True
            )
        ),
        "child_execution_env": child_env,
        "provenance_dir": PROVENANCE_DIRNAME,
        "scoring_env": scoring_env,
        "sealed_at_campaign_start": True,
        "sealed_closure_is_executed": True,
        "session_context_dir": SEALED_TAKEHOME_RELATIVE,
        "session_context_files": sorted(
            (k for k in sealed if k.startswith(f"{SEALED_TAKEHOME_RELATIVE}/"))
        ),
    }
    closure["digest"] = _sha256_text(json.dumps(closure, sort_keys=True))
    (provenance / SEAL_SELF_MANIFEST).write_text(
        json.dumps({"closure": closure, "sealed": sealed}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    sealed[SEAL_SELF_MANIFEST] = digest_file(provenance / SEAL_SELF_MANIFEST)
    if recorder is not None:
        recorder(
            "evaluator_closure_sealed",
            closure_digest=closure["digest"],
            provenance_dir=PROVENANCE_DIRNAME,
            sealed_files=sorted(sealed),
        )
    return {"closure": closure, "provenance": provenance, "sealed": sealed}


def verify_provenance_exact(provenance: Path, expected: Mapping[str, Any]) -> list[str]:
    """The seal must hold EXACTLY its sealed files. Extras and symlinks are refused."""
    provenance = Path(provenance)
    problems: list[str] = []
    observed: dict[str, str] = {}
    for path in sorted(provenance.rglob("*")):
        relative = str(path.relative_to(provenance))
        if path.is_symlink():
            problems.append(f"unexpected symlink in the seal: {relative}")
        elif path.is_dir():
            continue
        elif path.is_file():
            observed[relative] = _sha256_bytes(path.read_bytes())
        else:
            problems.append(f"unexpected non-regular entry in the seal: {relative}")
    for relative in sorted(set(observed) - set(expected) - {SEAL_SELF_MANIFEST}):
        problems.append(f"unexpected file in the seal: {relative}")
    if SEAL_SELF_MANIFEST not in observed:
        problems.append(f"sealed file missing: {SEAL_SELF_MANIFEST}")
    for relative in sorted(set(expected) - set(observed)):
        problems.append(f"sealed file missing: {relative}")
    for relative in sorted((set(expected) & set(observed)) - {SEAL_SELF_MANIFEST}):
        want = expected[relative]
        want = want.get("sha256") if isinstance(want, Mapping) else want
        if observed[relative] != want:
            problems.append(f"sealed file drifted: {relative}")
    return problems


def sealed_eval_program(provenance: Path) -> Path:
    """The ``eval_program_path`` every future evaluation must use."""
    return Path(provenance) / "evaluate.py"


def sealed_takehome(provenance: Path) -> Path:
    """The ``SHINKA_TAKEHOME_DIR`` every future evaluation must use."""
    return Path(provenance) / SEALED_TAKEHOME_RELATIVE


def load_sealed_evaluator(provenance: Path):
    """Import the sealed ``evaluate.py`` to prove the seal is runnable, not just present."""
    provenance = Path(provenance)
    entrypoint = sealed_eval_program(provenance)
    if not entrypoint.is_file():
        raise ShinkaCheckpointError(f"no sealed evaluator at {entrypoint}")
    override = os.environ.get("SHINKA_TAKEHOME_DIR")
    expected = sealed_takehome(provenance).resolve()
    if override and Path(override).resolve() != expected:
        raise ShinkaCheckpointError(
            f"SHINKA_TAKEHOME_DIR={override!r} would redirect the sealed evaluator away from {expected}; refusing to score against an unsealed take-home"
        )
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(
            "shinka_sealed_evaluate", entrypoint
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def in_flight_report(runner: Any) -> dict[str, int]:
    """Every channel that must be empty before a checkpoint may claim exact t0."""

    def size(name: str) -> int:
        value = getattr(runner, name, None)
        if value is None:
            return 0
        if isinstance(value, (list, dict, set, tuple)):
            return len(value)
        if hasattr(value, "qsize"):
            return int(value.qsize())
        return int(bool(value))

    def slots(name: str) -> int:
        pool = getattr(runner, name, None)
        if pool is None:
            return 0
        for attr in ("in_use", "active", "acquired", "held"):
            held = getattr(pool, attr, None)
            if held is not None:
                return len(held) if hasattr(held, "__len__") else int(held)
        capacity = getattr(pool, "capacity", None)
        available = getattr(pool, "available", None)
        if capacity is not None and available is not None:
            return int(capacity) - int(available)
        return 0

    native_counter = getattr(runner, "_get_background_side_effect_work_count", None)
    if callable(native_counter):
        side_effect_backlog = int(native_counter() or 0)
    else:
        side_effect_backlog = max(
            int(getattr(runner, "_background_side_effects_pending", 0) or 0),
            int(getattr(runner, "_background_side_effects_busy_count", 0) or 0)
            or int(bool(getattr(runner, "_background_side_effects_busy", False))),
        )
    report = {
        "active_proposal_tasks": size("active_proposal_tasks"),
        "background_side_effect_backlog": side_effect_backlog,
        "completed_job_batch_tasks": len(
            [
                t
                for t in getattr(runner, "_completed_job_batch_tasks", set()) or set()
                if not t.done()
            ]
        ),
        "completed_jobs_pending": int(
            getattr(runner, "_completed_jobs_pending", 0) or 0
        ),
        "failed_jobs_for_retry": size("failed_jobs_for_retry"),
        "prompt_percentile_recompute_pending": int(
            bool(getattr(runner, "_prompt_percentile_recompute_pending", False))
        ),
        "proposal_queue": size("proposal_queue"),
        "running_jobs": size("running_jobs"),
        "side_effect_event_queue": size("side_effect_event_queue"),
        "slots_evaluation": slots("evaluation_slot_pool"),
        "slots_postprocess": slots("postprocess_slot_pool"),
        "slots_sampling": slots("sampling_slot_pool"),
    }
    task = getattr(runner, "_prompt_percentile_recompute_task", None)
    report["prompt_percentile_recompute_task"] = int(
        task is not None and (not task.done())
    )
    return report


def failed_background_workers(runner: Any) -> list[str]:
    """Side-effect workers that died. An idle worker is fine; a dead one is not.

    A crashed worker would leave the queue permanently undrained, so quiescence must
    never be declared while one is dead — fail closed rather than wait forever.
    """
    failed: list[str] = []
    for task in getattr(runner, "_background_side_effect_tasks", set()) or set():
        if not task.done():
            continue
        if task.cancelled():
            failed.append(f"{task.get_name()}: cancelled")
            continue
        exception = task.exception()
        if exception is not None:
            failed.append(f"{task.get_name()}: {type(exception).__name__}: {exception}")
    return failed


def is_quiescent(runner: Any) -> bool:
    return not any(in_flight_report(runner).values())


class QuiescentBarrier:
    """Fences dispatch, drains every in-flight channel, then releases.

    The fence works by wrapping ``_start_proposals``: while requested, it returns
    without assigning a generation, so no new proposal can begin and
    ``next_generation_to_submit`` does not move. Outside the barrier the wrapper is a
    pass-through, so native concurrency and search are untouched.
    """

    def __init__(
        self,
        runner: Any,
        *,
        drain_timeout: float = 900.0,
        poll_interval: float = 0.25,
        recorder: Any = None,
    ) -> None:
        self.runner = runner
        self.drain_timeout = drain_timeout
        self.poll_interval = poll_interval
        self.recorder = recorder
        self.fenced = False
        self._native_start_proposals = runner._start_proposals
        self.fence_events: list[dict[str, Any]] = []

        async def fenced_start_proposals(num_proposals: int):
            if self.fenced:
                return
            return await self._native_start_proposals(num_proposals)

        runner._start_proposals = fenced_start_proposals

    def uninstall(self) -> None:
        self.runner._start_proposals = self._native_start_proposals

    async def drain(self) -> dict[str, Any]:
        """Fence dispatch and wait for quiescence. Raises on timeout.

        EVERY failure path unfences before raising. The fence suppresses dispatch, so
        leaving it up after a failed drain would stall the whole run indefinitely --
        the caller only disables checkpointing, it cannot know the fence is still on.
        """
        self.fenced = True
        try:
            return await self._drain_fenced()
        except BaseException:
            self.fenced = False
            raise

    async def _drain_fenced(self) -> dict[str, Any]:
        started = time.monotonic()
        entry = in_flight_report(self.runner)
        report = dict(entry)
        while any(report.values()):
            dead = failed_background_workers(self.runner)
            if dead:
                raise ShinkaCheckpointError(
                    f"a background side-effect worker died, so the queue can never drain: {dead}"
                )
            if time.monotonic() - started > self.drain_timeout:
                raise ShinkaCheckpointError(
                    f"barrier drain timed out after {self.drain_timeout}s with work still in flight: { {k: v for k, v in report.items() if v} }"
                )
            await asyncio.sleep(self.poll_interval)
            report = in_flight_report(self.runner)
        await self._flush_deferred_maintenance()
        elapsed = time.monotonic() - started
        self.assert_workers_healthy()
        event = {
            "drain_seconds": round(elapsed, 3),
            "in_flight_at_entry": entry,
            "live_background_workers": len(
                [
                    t
                    for t in getattr(
                        self.runner, "_background_side_effect_tasks", set()
                    )
                    or set()
                    if not t.done()
                ]
            ),
        }
        self.fence_events.append(event)
        if self.recorder is not None:
            self.recorder("causal_barrier_quiescent", **event)
        return event

    def assert_workers_healthy(self) -> None:
        dead = failed_background_workers(self.runner)
        if dead:
            raise ShinkaCheckpointError(
                f"background side-effect worker(s) died: {dead}"
            )

    async def _flush_deferred_maintenance(self) -> None:
        """Replay every deferred post-insert maintenance path, then drain again."""
        database = getattr(self.runner, "db", None) or getattr(
            self.runner, "database", None
        )
        for name in ("replay_deferred_maintenance", "check_scheduled_operations"):
            target = getattr(database, name, None) if database is not None else None
            if target is None:
                continue
            result = target()
            if asyncio.iscoroutine(result):
                await result
        waiter = getattr(self.runner, "_wait_for_background_side_effects", None)
        if waiter is not None:
            self.assert_workers_healthy()
            try:
                await asyncio.wait_for(waiter(), timeout=self.drain_timeout)
            except asyncio.TimeoutError as exc:
                raise ShinkaCheckpointError(
                    f"waiting for background side effects exceeded {self.drain_timeout}s; failed workers: {failed_background_workers(self.runner) or 'none'}"
                ) from exc
        started = time.monotonic()
        while any(in_flight_report(self.runner).values()):
            dead = failed_background_workers(self.runner)
            if dead:
                raise ShinkaCheckpointError(
                    f"a background side-effect worker died while re-draining: {dead}"
                )
            if time.monotonic() - started > self.drain_timeout:
                raise ShinkaCheckpointError(
                    "deferred maintenance re-enqueued work that did not drain"
                )
            await asyncio.sleep(self.poll_interval)

    def release(self) -> None:
        self.fenced = False


def snapshot_database(source: Path, target: Path) -> dict[str, Any]:
    """Transactionally consistent SQLite snapshot via the online backup API.

    ``VACUUM INTO`` would also work, but the backup API copies a consistent view even
    with an open connection and does not rewrite page layout, so the snapshot stays
    comparable. The source is opened read-only through a URI so the historical file is
    never modified.
    """
    source, target = (Path(source), Path(target))
    if not source.is_file():
        raise ShinkaCheckpointError(f"no program database at {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        destination = sqlite3.connect(str(target))
        try:
            connection.backup(destination)
        finally:
            destination.close()
    finally:
        connection.close()
    return {**digest_file(target), "method": "sqlite_backup_api"}


ATTEMPT_TRACE_NAME = "attempts.jsonl"
ATTEMPT_PREFIX_NAME = "attempts_prefix.jsonl"
COMMITTED_TERMINALS = ("program_committed", "program_committed_recovered")


def read_canonical_completions(trace: Path) -> list[dict[str, Any]]:
    """Every fsync-durable ``canonical_completion`` row, with its byte extent.

    Rows are returned in file order; ``_end_offset`` is the byte offset just past the
    row's newline, which is what lets a boundary pin the exact prefix it used.
    """
    trace = Path(trace)
    if not trace.is_file():
        return []
    rows: list[dict[str, Any]] = []
    offset = 0
    with trace.open("rb") as handle:
        for raw in handle:
            offset += len(raw)
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except ValueError:
                break
            if isinstance(row, dict) and row.get("event") == "canonical_completion":
                rows.append({**row, "_end_offset": offset})
    return rows


def canonical_terminal(
    results: Path, *, database: Path | None = None
) -> dict[str, Any]:
    """The latest fsync-durable canonical terminal, cross-checked against the snapshot.

    ``trajectory.py`` assigns one monotonic ``completion_seq`` across committed
    programs **and** terminal-only attempts (failed proposals, exceptions, discards,
    abandonment). SQLite only ever sees the commits, so its last row is *not* the
    canonical terminal: a run whose generation 6 failed before persistence still ends
    at ``prog-5`` in the database while controller and budget state already include
    generation 6. Naming a checkpoint ``prog-5`` there would claim a t0 the state has
    moved past.

    A boundary may therefore be named by a ``Program.id`` and marked eligible only if
    ALL of these hold:

    * the latest canonical completion is ``status == "completed"`` with a committed
      terminal;
    * its ``child.id`` is present in the snapshot;
    * that program is ``correct`` with a non-null ``combined_score``;
    * no later trace terminal and no later database program exist.

    Anything else is reported non-eligible with a precise reason, and the caller skips
    honestly. Trace/DB disagreement fails closed rather than picking a side.
    """
    results = Path(results)
    trace = results / ATTEMPT_TRACE_NAME
    database = Path(database) if database is not None else results / "programs.sqlite"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        db_rows = connection.execute(
            "select id, generation, timestamp, combined_score, correct from programs order by rowid"
        ).fetchall()
    finally:
        connection.close()
    eligible_ids: list[str] = []
    db_by_id: dict[str, tuple] = {}
    for row in db_rows:
        db_by_id[str(row[0])] = row
        if row[4] and row[3] is not None:
            eligible_ids.append(str(row[0]))
    db_last_id = str(db_rows[-1][0]) if db_rows else None
    completions = read_canonical_completions(trace)
    base = {
        "db_last_program_id": db_last_id,
        "n_db_programs": len(db_rows),
        "n_eligible": len(eligible_ids),
        "n_trace_completions": len(completions),
        "trace": str(trace),
        "trace_present": trace.is_file(),
    }
    if not completions:
        if not trace.is_file():
            return {
                **base,
                "present": False,
                "eligible": False,
                "reason_not_eligible": f"no run-owned {ATTEMPT_TRACE_NAME}: the canonical completion order is defined by the trajectory, and SQLite rowid alone is not it",
            }
        return {
            **base,
            "present": False,
            "eligible": False,
            "reason_not_eligible": "the trajectory records no canonical_completion yet",
        }
    latest = max(
        completions,
        key=lambda r: (
            r.get("completion_seq") if isinstance(r.get("completion_seq"), int) else -1
        ),
    )
    trajectory_seq = latest.get("completion_seq")
    child = latest.get("child") if isinstance(latest.get("child"), dict) else {}
    child_id = child.get("id") if isinstance(child.get("id"), str) else None
    terminal = {
        **base,
        "attempt_key": latest.get("attempt_key"),
        "generation": latest.get("generation"),
        "identity": child_id,
        "prefix_bytes": latest["_end_offset"],
        "present": True,
        "status": latest.get("status"),
        "terminal": latest.get("terminal"),
        "trajectory_completion_seq": trajectory_seq,
    }
    committed = (
        latest.get("status") == "completed"
        and latest.get("terminal") in COMMITTED_TERMINALS
    )
    if not committed:
        return {
            **terminal,
            "eligible": False,
            "latest_eligible_identity": eligible_ids[-1] if eligible_ids else None,
            "reason_not_eligible": f"the latest canonical terminal is status={latest.get('status')!r} terminal={latest.get('terminal')!r}, not a committed program",
        }
    if child_id is None:
        return {
            **terminal,
            "eligible": False,
            "reason_not_eligible": "the committed terminal carries no child.id",
        }
    if child_id not in db_by_id:
        raise ShinkaCheckpointError(
            f"trace/DB disagreement: the latest canonical terminal committed {child_id!r} but the snapshot does not contain it"
        )
    if db_last_id != child_id:
        raise ShinkaCheckpointError(
            f"trace/DB disagreement: the latest canonical terminal is {child_id!r} but the database's last program is {db_last_id!r}; a later database state exists than the trajectory records"
        )
    _pid, _gen, _ts, combined_score, correct = db_by_id[child_id]
    if not correct or combined_score is None:
        return {
            **terminal,
            "eligible": False,
            "latest_eligible_identity": eligible_ids[-1] if eligible_ids else None,
            "reason_not_eligible": "the committed terminal is an incorrect program"
            if not correct
            else "the committed terminal has no combined_score",
        }
    if eligible_ids[-1] != child_id:
        raise ShinkaCheckpointError(
            f"trace/DB disagreement: the latest canonical terminal is {child_id!r} but the last eligible program is {eligible_ids[-1]!r}"
        )
    return {
        **terminal,
        "combined_score": combined_score,
        "correct": True,
        "eligible": True,
        "eligible_completion_seq": len(eligible_ids) - 1,
    }


def attempt_prefix(results: Path, prefix_bytes: int, target: Path) -> dict[str, Any]:
    """Copy the exact trace prefix that established the boundary."""
    trace = Path(results) / ATTEMPT_TRACE_NAME
    data = trace.read_bytes()[: int(prefix_bytes)]
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_bytes(data)
    completions = read_canonical_completions(Path(target))
    return {
        "bytes": len(data),
        "n_canonical_completions": len(completions),
        "sha256": _sha256_bytes(data),
        "last_completion_seq": completions[-1].get("completion_seq")
        if completions
        else None,
    }


def database_facts(path: Path) -> dict[str, Any]:
    """Integrity, schema, table counts, ordered archive membership, metadata values."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        integrity = connection.execute("pragma integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ShinkaCheckpointError(
                f"{path}: sqlite integrity_check said {integrity!r}"
            )
        tables = sorted(
            (
                row[0]
                for row in connection.execute(
                    "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
                )
            )
        )
        schema = sorted(
            (
                f"{row[0]}::{row[1]}"
                for row in connection.execute(
                    "select name, sql from sqlite_master where type='table' and name not like 'sqlite_%'"
                )
            )
        )
        counts = {
            table: connection.execute(f"select count(*) from '{table}'").fetchone()[0]
            for table in tables
        }
        archive: list[str] = []
        if "archive" in tables:
            archive = [
                str(row[0])
                for row in connection.execute(
                    "select program_id from archive order by program_id"
                )
            ]
        metadata: dict[str, Any] = {}
        if "metadata_store" in tables:
            metadata = {
                str(k): v
                for k, v in connection.execute(
                    "select key, value from metadata_store order by key"
                )
            }
        islands: dict[str, int] = {}
        generations: dict[str, int] = {}
        if "programs" in tables:
            columns = {
                row[1] for row in connection.execute("pragma table_info(programs)")
            }
            if "island_idx" in columns:
                islands = {
                    str(row[0]): row[1]
                    for row in connection.execute(
                        "select island_idx, count(*) from programs group by island_idx order by island_idx"
                    )
                }
            if "generation" in columns:
                generations = {
                    str(row[0]): row[1]
                    for row in connection.execute(
                        "select generation, count(*) from programs group by generation order by generation"
                    )
                }
    finally:
        connection.close()
    facts = {
        "archive_ordered": archive,
        "integrity_check": integrity,
        "island_membership_counts": islands,
        "metadata_store": metadata,
        "programs_per_generation": generations,
        "schema": schema,
        "table_counts": counts,
        "tables": tables,
    }
    facts["digest"] = _sha256_text(json.dumps(facts, sort_keys=True))
    return facts


def capture_controller_state(runner: Any) -> dict[str, Any]:
    """Everything SQLite cannot restore, plus both global RNG states."""
    missing = [name for name in CONTROLLER_FIELDS if not hasattr(runner, name)]
    if missing:
        raise ShinkaCheckpointError(
            f"the runner does not expose required controller fields: {missing}; refusing to write a checkpoint that would silently omit them"
        )
    state: dict[str, Any] = {}
    for name in CONTROLLER_FIELDS:
        value = getattr(runner, name)
        if isinstance(value, set):
            value = sorted(value)
        state[name] = value
    python_random = random.getstate()
    state["python_random_state"] = json.loads(
        json.dumps([python_random[0], list(python_random[1]), python_random[2]])
    )
    try:
        import numpy as np

        numpy_state = np.random.get_state()
        state["numpy_random_state"] = {
            "available": True,
            "kind": numpy_state[0],
            "keys_digest": _sha256_bytes(numpy_state[1].tobytes()),
            "keys_hex": numpy_state[1].tobytes().hex(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
    except ImportError:
        state["numpy_random_state"] = {
            "available": False,
            "reason": "numpy is not importable in this interpreter, so it holds no global RNG state that could influence sampling",
        }
    state["digest"] = _sha256_text(json.dumps(state, sort_keys=True))
    return state


def restore_controller_state(runner: Any, state: Mapping[str, Any]) -> None:
    """Reinstate controller and RNG state on a restored runner."""
    for name in CONTROLLER_FIELDS:
        if name not in state:
            raise ShinkaCheckpointError(f"controller state is missing {name}")
        value = state[name]
        if name == "assigned_generations":
            value = set(value)
        setattr(runner, name, value)
    payload = state.get("python_random_state")
    if payload is not None:
        random.setstate((payload[0], tuple((int(x) for x in payload[1])), payload[2]))
    numpy_state = state.get("numpy_random_state")
    if numpy_state and numpy_state.get("available"):
        import numpy as np

        keys = np.frombuffer(bytes.fromhex(numpy_state["keys_hex"]), dtype=np.uint32)
        np.random.set_state(
            (
                numpy_state["kind"],
                keys,
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )


def bandit_state_facts(results: Path) -> dict[str, Any]:
    """Digest ``bandit_state.pkl`` if present. Absence is explicit, never implied."""
    path = Path(results) / "bandit_state.pkl"
    if not path.is_file():
        return {
            "present": False,
            "reason": "no bandit_state.pkl in the results directory",
        }
    data = path.read_bytes()
    facts = {"present": True, "sha256": _sha256_bytes(data), "bytes": len(data)}
    try:
        loaded = pickle.loads(data)
        facts["unpickles"] = True
        facts["type"] = type(loaded).__name__
    except Exception as exc:
        facts["unpickles"] = False
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def capture_budget(
    runner: Any, *, attempt_costs: Mapping[str, Any] | None = None
) -> dict:
    """All-in budget at the boundary.

    ``total_api_cost`` is the runner's own accumulator and already folds in LLM,
    embedding, novelty and native meta spend. Failed proposals can contribute cost
    that never reaches a persisted program, so ``completed_proposal_costs`` and any
    attempt-record total are pinned alongside it rather than recomputed from the
    database. The in-flight reservation set is pinned as empty, which the barrier has
    already proven.
    """
    evo_config = getattr(runner, "evo_config", None)
    in_flight = in_flight_report(runner)
    if any(in_flight.values()):
        raise ShinkaCheckpointError(
            f"refusing to pin a budget with work in flight: { {k: v for k, v in in_flight.items() if v} }"
        )
    budget = {
        "all_in_spent_usd": float(getattr(runner, "total_api_cost", 0.0)),
        "avg_proposal_cost": float(getattr(runner, "avg_proposal_cost", 0.0)),
        "completed_proposal_costs": list(
            getattr(runner, "completed_proposal_costs", [])
        ),
        "cost_components": ["llm", "openrouter_embedding", "novelty", "native_meta"],
        "cost_limit_reached": bool(getattr(runner, "cost_limit_reached", False)),
        "in_flight_reservations": [],
        "max_api_costs": getattr(evo_config, "max_api_costs", None),
        "n_completed_proposals": len(getattr(runner, "completed_proposal_costs", [])),
    }
    if attempt_costs is not None:
        budget["attempt_record_costs"] = dict(attempt_costs)
    budget["digest"] = _sha256_text(json.dumps(budget, sort_keys=True))
    return budget


def pipeline_target(runner: Any) -> dict[str, Any]:
    """The next pipeline target, from the pinned timing EWMAs.

    Read through the runner's own helper when it exposes one, so the restored value is
    whatever Shinka would compute rather than a reimplementation.
    """
    for name in (
        "_get_pipeline_target",
        "_compute_pipeline_target",
        "_pipeline_target",
    ):
        helper = getattr(runner, name, None)
        if callable(helper):
            try:
                return {"source": name, "target": helper()}
            except Exception as exc:
                return {"source": name, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "evaluation_seconds_ewma": getattr(runner, "_evaluation_seconds_ewma", None),
        "proposal_timing_samples": getattr(runner, "_proposal_timing_samples", None),
        "sampling_seconds_ewma": getattr(runner, "_sampling_seconds_ewma", None),
        "source": "pinned_ewmas",
    }


def capture_config(
    *,
    evo_config: Any,
    db_config: Any,
    raw_config: Mapping[str, Any],
    task_sys_msg: str,
    cli: Mapping[str, Any],
    closure: Mapping[str, Any],
    here: Path,
) -> dict[str, Any]:
    """Everything outside the database that shapes the next search decision."""

    def fields(obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        source = getattr(obj, "__dict__", None) or {}
        return {
            k: v
            for k, v in sorted(source.items())
            if not k.startswith("_")
            and isinstance(v, (str, int, float, bool, list, type(None)))
        }

    runtime_sources = {}
    for name in (
        "run_evo.py",
        "trajectory.py",
        "state_snapshot.py",
        "shinka_resume_capture.py",
    ):
        path = Path(here) / name
        if path.is_file():
            runtime_sources[name] = digest_file(path)["sha256"]
    vendor = Path(here) / "vendor" / "shinkaevolve" / "shinka"
    if vendor.is_dir():
        runtime_sources["vendor_shinka_tree"] = _sha256_text(
            json.dumps(
                {
                    str(p.relative_to(vendor)): _sha256_bytes(p.read_bytes())
                    for p in sorted(vendor.rglob("*.py"))
                    if "__pycache__" not in p.parts
                },
                sort_keys=True,
            )
        )
    config = {
        "cli": {k: v for k, v in sorted(dict(cli).items())},
        "db_config": fields(db_config),
        "effective": {name: getattr(evo_config, name, None) for name in CONFIG_FIELDS},
        "evo_config": fields(evo_config),
        "raw_config_digest": _sha256_text(
            json.dumps(normalise_raw_config(raw_config), sort_keys=True)
        ),
        "runtime_sources": runtime_sources,
        "scoring_env": {name: os.environ.get(name) for name in SCORING_ENV_VARS},
        "session_context_digest": _sha256_text(
            json.dumps(
                {
                    k: v
                    for k, v in closure.get("files", {}).items()
                    if k.startswith(f"{SEALED_TAKEHOME_RELATIVE}/")
                },
                sort_keys=True,
            )
        ),
        "task_sys_msg_sha256": _sha256_text(task_sys_msg),
        "unsupported_feature_state": {
            name: getattr(evo_config, name, None) for name in UNSUPPORTED_WHEN_ENABLED
        },
    }
    config["digest"] = _sha256_text(json.dumps(config, sort_keys=True))
    return config


PATH_VALUED_KEYS = {
    "cli": ("config", "output", "takehome"),
    "db_config": ("db_path",),
    "effective": ("results_dir",),
    "evo_config": ("db_path", "init_program_path", "results_dir"),
    "scoring_env": ("SHINKA_TAKEHOME_DIR",),
}
RAW_CONFIG_PATH_KEYS = ("results_dir", "db_path", "init_program_path", "output")


def normalise_raw_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop path-valued keys so a fork's digest matches t0's for the same settings."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                k: strip(v)
                for k, v in sorted(value.items())
                if k not in RAW_CONFIG_PATH_KEYS
            }
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(dict(raw_config))


def compare_config(
    pinned: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    allowed_runtime_source_drift: frozenset[str] = frozenset(),
) -> list[str]:
    """Section-by-section differences between two effective configurations."""
    problems: list[str] = []
    for section in (
        "cli",
        "db_config",
        "effective",
        "evo_config",
        "runtime_sources",
        "scoring_env",
        "unsupported_feature_state",
    ):
        want, got = (pinned.get(section) or {}, observed.get(section) or {})
        skip = PATH_VALUED_KEYS.get(section, ())
        for key in sorted(set(want) | set(got)):
            if key in skip:
                continue
            if section == "runtime_sources" and key in allowed_runtime_source_drift:
                continue
            if want.get(key) != got.get(key):
                problems.append(
                    f"{section}.{key}: t0={want.get(key)!r} now={got.get(key)!r}"
                )
    for scalar in (
        "raw_config_digest",
        "session_context_digest",
        "task_sys_msg_sha256",
    ):
        if pinned.get(scalar) != observed.get(scalar):
            problems.append(f"{scalar} changed")
    return problems


class ShinkaCausalCheckpointer:
    """Writes one exact causal checkpoint per quiescent barrier. Opt-in.

    Also owns the sealed evaluator: with checkpointing on,
    ``LocalJobConfig.eval_program_path`` and ``SHINKA_TAKEHOME_DIR`` must point at the
    run-owned copies, which :meth:`bind_job_config` does.
    """

    def __init__(
        self,
        *,
        results: Path,
        here: Path,
        takehome: Path,
        evo_config: Any,
        db_config: Any,
        raw_config: Mapping[str, Any],
        task_sys_msg: str,
        cli: Mapping[str, Any],
        cadence_generations: int = 5,
        drain_timeout: float = 900.0,
        allow_non_eligible_boundary: bool = False,
        allowed_runtime_source_drift: frozenset[str] = frozenset(),
        recorder: Any = None,
    ) -> None:
        self.results = Path(results)
        self.here = Path(here)
        self.takehome = Path(takehome)
        self.recorder = recorder
        self.cadence = max(1, int(cadence_generations))
        self.drain_timeout = float(drain_timeout)
        self.allow_non_eligible_boundary = bool(allow_non_eligible_boundary)
        self.root = self.results / CHECKPOINT_DIRNAME
        self.provenance = self.results / PROVENANCE_DIRNAME
        self.disabled_reason: str | None = None
        self.barrier: QuiescentBarrier | None = None
        self.written: list[dict[str, Any]] = []
        self.cli = dict(cli)
        self._setup_complete = asyncio.Event()
        self.state_applied: dict[str, Any] | None = None
        self.reference_boundary: str | None = None
        self.resume_branch_state: dict[str, Any] | None = None
        self.feature_state = assert_supported_features(evo_config)
        self.mode = detect_mode(self.results)
        self.continuation = None
        if self.mode == MODE_NEW:
            sealed = seal_closure(
                self.results, here=self.here, takehome=self.takehome, recorder=recorder
            )
            self.closure = sealed["closure"]
            self.sealed_files = sealed["sealed"]
            load_sealed_evaluator(self.provenance)
            self.config = capture_config(
                evo_config=evo_config,
                db_config=db_config,
                raw_config=raw_config,
                task_sys_msg=task_sys_msg,
                cli=self.cli,
                closure=self.closure,
                here=self.here,
            )
        else:
            observed_config = capture_config(
                evo_config=evo_config,
                db_config=db_config,
                raw_config=raw_config,
                task_sys_msg=task_sys_msg,
                cli=self.cli,
                closure={},
                here=self.here,
            )
            self.continuation = validate_continuation(
                self.results,
                observed_config=observed_config,
                mode=self.mode,
                allowed_runtime_source_drift=allowed_runtime_source_drift,
            )
            self.closure = self.continuation["closure"]
            self.sealed_files = self.continuation["sealed_files"]
            self.config = self.continuation["config"]
            if self.continuation.get("manifest") is None:
                from shinka_resume_capture import load_branch_state

                self.resume_branch_state = load_branch_state(self.results)
                if (
                    self.resume_branch_state is not None
                    and self.resume_branch_state.get("config_digest")
                    != self.config.get("digest")
                ):
                    raise ContinuationRefused(
                        "resume branch state was produced under a different config"
                    )
            if recorder is not None:
                recorder(
                    "causal_continuation_validated",
                    mode=self.mode,
                    evaluator_closure_digest=self.closure["digest"],
                    config_digest=self.config["digest"],
                    checks=self.continuation["report"]["checks"],
                )
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def disable(self, reason: str) -> None:
        self.disabled_reason = reason
        if self.recorder is not None:
            self.recorder("causal_checkpoint_disabled", reason=reason)

    @property
    def eval_program_path(self) -> Path:
        return sealed_eval_program(self.provenance)

    @property
    def session_takehome(self) -> Path:
        return sealed_takehome(self.provenance)

    def bind_job_config(self, job_config: Any) -> dict[str, Any]:
        """Point every future evaluation at the sealed evaluator, take-home AND env.

        Called before the runner is constructed, so no evaluation can ever run the
        mutable project copy while checkpointing is on.

        The pinned ``child_execution_env`` is validated and re-applied here, not only
        at seal time: a fresh fork process goes through ``validate_continuation`` and
        this method, and would otherwise run evaluator subprocesses **without**
        ``PYTHONDONTWRITEBYTECODE`` — which would drop ``__pycache__`` into the seal
        and break exact-seal validation on the next resume.
        """
        previous = {
            "eval_program_path": getattr(job_config, "eval_program_path", None),
            "SHINKA_TAKEHOME_DIR": os.environ.get("SHINKA_TAKEHOME_DIR"),
            **{name: os.environ.get(name) for name in CHILD_EXECUTION_ENV},
        }
        pinned_child_env = dict(self.closure.get("child_execution_env") or {})
        missing = [name for name in CHILD_EXECUTION_ENV if name not in pinned_child_env]
        if missing:
            raise ShinkaCheckpointError(
                f"the sealed closure does not pin the child execution environment {missing}; refusing to run evaluator subprocesses whose bytecode behaviour is unknown"
            )
        for name, value in sorted(pinned_child_env.items()):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        job_config.eval_program_path = str(self.eval_program_path)
        os.environ["SHINKA_TAKEHOME_DIR"] = str(self.session_takehome)
        bound = {
            "eval_program_path": job_config.eval_program_path,
            "SHINKA_TAKEHOME_DIR": os.environ["SHINKA_TAKEHOME_DIR"],
            "child_execution_env": {
                name: os.environ.get(name) for name in CHILD_EXECUTION_ENV
            },
            "previous": previous,
        }
        if self.recorder is not None:
            self.recorder("causal_evaluator_bound", **bound)
        return bound

    def install(self, runner: Any, *, periodic: bool = True) -> None:
        """Restore pinned state, and optionally run periodic drain barriers.

        Restored controller/RNG state is applied by wrapping ``_setup_async``, NOT
        before ``run()``: native setup loads the restored database and bandit state and
        then overwrites ``total_api_cost``, ``completed_generations`` and
        ``next_generation_to_submit``, so applying earlier would be silently undone.
        The wrapper applies the pinned state immediately after native setup returns,
        which in ``run_async`` is before ``_verify_database_ready()``, before the
        proposal coordinator and job monitor exist, and before any proposal.

        The barrier loop is gated on that same event, so it can never observe a
        pre-restored completed count while setup is still running.
        """
        self.barrier = (
            QuiescentBarrier(
                runner, drain_timeout=self.drain_timeout, recorder=self.recorder
            )
            if periodic
            else None
        )
        self.runner = runner
        self.state_applied: dict[str, Any] | None = None
        native_run_async = runner.run_async
        native_setup = getattr(runner, "_setup_async", None)
        if native_setup is not None:

            async def setup_then_restore(*args, **kwargs):
                result = await native_setup(*args, **kwargs)
                self.state_applied = self.apply_restored_controller_state(runner)
                self._setup_complete.set()
                return result

            runner._setup_async = setup_then_restore
        else:
            self._setup_complete.set()
        if periodic:

            async def run_async_with_barrier():
                task = asyncio.create_task(self._barrier_loop(), name="causal_barrier")
                try:
                    return await native_run_async()
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            runner.run_async = run_async_with_barrier

    async def _barrier_loop(self) -> None:
        """Request a barrier once progress passes a monotonic threshold.

        Waits for setup first: reading ``completed_generations`` while native setup is
        still running would observe a pre-restored value.

        The threshold is monotonic rather than ``completed % cadence == 0``. Async
        batches can jump over an exact multiple, so a modulus test would silently turn
        cadence 5 into "every 20 completions" — or never fire at all. It is initialised
        from the verified reference so a resume at an existing boundary does not
        immediately try to rewrite the same checkpoint.

        A skipped non-eligible terminal advances the threshold to the current count, so
        the loop waits for NEW durable progress instead of busy-looping on every tick.
        """
        await self._setup_complete.wait()
        threshold = self._initial_cadence_threshold()
        while self.enabled:
            await asyncio.sleep(1.0)
            completed = int(getattr(self.runner, "completed_generations", 0) or 0)
            if completed < 1 or completed < threshold:
                continue
            try:
                payload = await self.checkpoint_at_barrier()
            except ShinkaCheckpointError as exc:
                self.disable(f"barrier checkpoint failed: {type(exc).__name__}: {exc}")
                return
            if payload is None:
                threshold = completed + 1
                continue
            threshold = completed + self.cadence

    def _initial_cadence_threshold(self) -> int:
        """Where the cadence starts, from the verified reference if there is one."""
        reference = None
        if self.continuation is not None:
            reference = self.continuation.get("manifest")
        if reference is None:
            reference = latest_own_checkpoint(self.results)
        if reference is not None:
            already = int(
                reference["boundary"].get("completed_generations")
                or reference["boundary"].get("completion_seq")
                or 0
            )
            self.reference_boundary = reference["boundary"].get("completion_identity")
            return already + self.cadence
        return self.cadence

    async def checkpoint_at_barrier(self) -> dict[str, Any] | None:
        """Fence, drain to quiescence, snapshot, release. Never snapshots mid-flight."""
        if not self.enabled or self.barrier is None:
            return None
        event = await self.barrier.drain()
        try:
            if not is_quiescent(self.runner):
                raise ShinkaCheckpointError("the barrier released before quiescence")
            return self.write(self.runner, drain_event=event)
        finally:
            self.barrier.release()

    def write(
        self, runner: Any, *, drain_event: Mapping[str, Any] | None = None
    ) -> dict:
        """Write one checkpoint. Only valid at a proven-quiescent barrier."""
        in_flight = in_flight_report(runner)
        if any(in_flight.values()):
            raise ShinkaCheckpointError(
                f"refusing to write a checkpoint with work in flight: { {k: v for k, v in in_flight.items() if v} }"
            )
        controller = capture_controller_state(runner)
        budget = capture_budget(runner)
        database_path = self.results / "programs.sqlite"
        terminal = canonical_terminal(self.results, database=database_path)
        if not terminal.get("present"):
            if self.recorder is not None:
                self.recorder(
                    "causal_checkpoint_skipped_no_canonical_terminal",
                    reason=terminal.get("reason_not_eligible"),
                )
            return None
        boundary_id = terminal["identity"]
        if not terminal["eligible"] and (not self.allow_non_eligible_boundary):
            if self.recorder is not None:
                self.recorder(
                    "causal_checkpoint_skipped_non_eligible_terminal",
                    identity=boundary_id,
                    reason_not_eligible=terminal["reason_not_eligible"],
                    latest_eligible_identity=terminal.get("latest_eligible_identity"),
                )
            return None
        directory = self.root / boundary_id
        if directory.exists():
            if self.recorder is not None:
                self.recorder(
                    "causal_checkpoint_skipped_existing_boundary", identity=boundary_id
                )
            return None
        staging = self.root / f".staging-{boundary_id}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            snapshot = snapshot_database(database_path, staging / DB_SNAPSHOT)
            prefix = attempt_prefix(
                self.results, terminal["prefix_bytes"], staging / ATTEMPT_PREFIX_NAME
            )
            if prefix["last_completion_seq"] != terminal["trajectory_completion_seq"]:
                raise ShinkaCheckpointError(
                    f"the pinned trace prefix does not end at the boundary terminal ({prefix['last_completion_seq']} != {terminal['trajectory_completion_seq']})"
                )
            facts = database_facts(staging / DB_SNAPSHOT)
            live_facts = database_facts(database_path)
            if facts["digest"] != live_facts["digest"]:
                raise ShinkaCheckpointError(
                    "the snapshot's sampler-visible facts differ from the live database; the barrier did not reach a consistent state"
                )
            payload_files: dict[str, str] = {
                ATTEMPT_PREFIX_NAME: str(staging / ATTEMPT_PREFIX_NAME),
                DB_SNAPSHOT: str(staging / DB_SNAPSHOT),
            }
            for relative in sorted(self.sealed_files):
                payload_files[f"{PROVENANCE_DIRNAME}/{relative}"] = str(
                    self.provenance / relative
                )
            bandit = bandit_state_facts(self.results)
            if bandit["present"]:
                shutil.copyfile(
                    self.results / "bandit_state.pkl", staging / BANDIT_SNAPSHOT
                )
                payload_files[BANDIT_SNAPSHOT] = str(staging / BANDIT_SNAPSHOT)
            files = {rel: digest_file(src) for rel, src in payload_files.items()}
            documents = {
                name: {
                    "path": path,
                    "payload": value,
                    "sha256": _document_digest(value),
                }
                for name, (path, value) in {
                    "controller_state": ("causal_controller_state.json", controller),
                    "budget": ("causal_budget.json", budget),
                    "config": ("causal_config.json", self.config),
                    "database_facts": ("causal_database_facts.json", facts),
                    "canonical_terminal": (
                        "causal_canonical_terminal.json",
                        {"prefix": prefix, "terminal": terminal},
                    ),
                    "bandit_state": ("causal_bandit_state.json", bandit),
                    "barrier": (
                        "causal_barrier.json",
                        {
                            "cadence_generations": self.cadence,
                            "drain": dict(drain_event or {}),
                            "in_flight_at_write": in_flight,
                            "quiescent": True,
                            "semantics": "dispatch fenced via _start_proposals; running_jobs, active_proposal_tasks, proposal_queue, failed_jobs_for_retry, completed-job batches, background side effects, prompt percentile recompute and all three logical slot pools drained to empty; deferred DB maintenance replayed and durable",
                        },
                    ),
                }.items()
            }
            components = {
                "archive": _sha256_text(
                    json.dumps(facts["archive_ordered"], sort_keys=True)
                ),
                "beam_search_parent": _sha256_text(
                    json.dumps(
                        facts["metadata_store"].get("beam_search_parent_id"),
                        sort_keys=True,
                    )
                ),
                "budget_state": budget["digest"],
                "candidate_state": _sha256_text(
                    json.dumps(
                        facts["metadata_store"].get("best_program_id"), sort_keys=True
                    )
                ),
                "concurrency_state": _sha256_text(
                    json.dumps(in_flight, sort_keys=True)
                ),
                "evaluator_closure": self.closure["digest"],
                "generation_counter": _sha256_text(
                    json.dumps(
                        {
                            "completed_generations": controller[
                                "completed_generations"
                            ],
                            "next_generation_to_submit": controller[
                                "next_generation_to_submit"
                            ],
                            "programs_per_generation": facts["programs_per_generation"],
                        },
                        sort_keys=True,
                    )
                ),
                "island_state": _sha256_text(
                    json.dumps(facts["island_membership_counts"], sort_keys=True)
                ),
                "prompt_state": self.config["task_sys_msg_sha256"],
                "search_population": snapshot["sha256"],
                "settings": self.config["digest"],
            }
            manifest = CausalCheckpointManifest(
                baseline=BASELINE,
                boundary={
                    "completion_identity": boundary_id,
                    "completion_seq": int(
                        terminal.get(
                            "eligible_completion_seq", terminal["n_eligible"] - 1
                        )
                    ),
                    "completed_generations": int(controller["completed_generations"]),
                    "eligible_terminal": terminal["eligible"],
                    "next_generation_to_submit": int(
                        controller["next_generation_to_submit"]
                    ),
                    "quiescent_barrier": True,
                    "terminal": terminal,
                    "trajectory_completion_seq": terminal["trajectory_completion_seq"],
                },
                state_components=components,
                files=files,
                documents=documents,
                evaluator_closure=self.closure,
                prompt_provenance={
                    "task_sys_msg_sha256": self.config["task_sys_msg_sha256"],
                    "raw_config_digest": self.config["raw_config_digest"],
                },
                budget_state=budget,
                concurrency_state={
                    "pending_sessions": [],
                    "in_flight": in_flight,
                    "quiescent_barrier": True,
                },
                settings={
                    "config_digest": self.config["digest"],
                    **{
                        k: v
                        for k, v in self.config["effective"].items()
                        if isinstance(v, (str, int, float, bool, type(None)))
                    },
                },
                trees={
                    "attempt_prefix": prefix["sha256"],
                    "controller_state": controller["digest"],
                    "database_facts": facts["digest"],
                    "database_snapshot": snapshot["sha256"],
                },
            )
            payload = write_checkpoint(directory, manifest, payload_files=payload_files)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self.written.append(
            {"boundary": boundary_id, "manifest_digest": payload["manifest_digest"]}
        )
        if self.recorder is not None:
            self.recorder(
                "causal_checkpoint_written",
                boundary=boundary_id,
                manifest_digest=payload["manifest_digest"],
                n_files=len(files),
                completed_generations=controller["completed_generations"],
            )
        return payload


def detect_mode(results: Path) -> str:
    results = Path(results)
    if (results / ORIGIN_DIRNAME / "manifest.json").is_file():
        return MODE_FORK
    if (results / PROVENANCE_DIRNAME / SEAL_SELF_MANIFEST).is_file():
        return MODE_RESUME
    return MODE_NEW


def latest_own_checkpoint(results: Path) -> dict[str, Any] | None:
    """The most recent SHARED-VERIFIER-verified checkpoint this campaign wrote.

    A directory without the atomic ``COMPLETE`` marker is an interrupted write and is
    skipped by design — that is what the marker is for. A directory that IS marked
    complete but no longer verifies is a different thing entirely: the campaign's own
    anchor has been corrupted or tampered with. Skipping that one would silently
    downgrade the resume to "no checkpoint yet", so no pinned controller/RNG/bandit
    state would be reapplied and the run would continue from native setup's
    recomputed state. That is a fail-open, so it refuses instead.
    """
    root = Path(results) / CHECKPOINT_DIRNAME
    if not root.is_dir():
        return None
    best, best_dir = (None, None)
    broken: list[str] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        complete = (directory / COMPLETE_MARKER_NAME).is_file()
        try:
            verify_checkpoint(directory)
            payload = read_checkpoint(directory)
        except CausalCheckpointError as exc:
            if complete:
                broken.append(f"{directory.name}: {exc}")
            continue
        if best is None or int(payload["boundary"]["completion_seq"]) > int(
            best["boundary"]["completion_seq"]
        ):
            best, best_dir = (payload, directory)
    if broken:
        raise ContinuationRefused(
            "this campaign has completed checkpoint(s) that no longer verify, so its own causal anchor cannot be trusted: "
            + "; ".join(broken)
        )
    if best is None:
        return None
    return {**best, "_checkpoint_dir": str(best_dir)}


def validate_continuation(
    results: Path,
    *,
    observed_config: Mapping[str, Any],
    mode: str,
    allowed_runtime_source_drift: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate a resume/fork before setup, dispatch or any model call. Never reseals."""
    results = Path(results)
    provenance = results / PROVENANCE_DIRNAME
    findings: list[str] = []
    report: dict[str, Any] = {"mode": mode, "results": str(results), "checks": {}}
    if not (provenance / SEAL_SELF_MANIFEST).is_file():
        raise ContinuationRefused(
            f"{mode}: no sealed closure at {provenance / SEAL_SELF_MANIFEST}; a resume must not reseal, so there is nothing to continue from"
        )
    sealed = json.loads((provenance / SEAL_SELF_MANIFEST).read_text(encoding="utf-8"))
    closure, sealed_files = (sealed["closure"], sealed["sealed"])
    manifest, reference_source = (None, None)
    if mode == MODE_FORK:
        origin = results / ORIGIN_DIRNAME
        manifest = json.loads((origin / "manifest.json").read_text(encoding="utf-8"))
        marker = (origin / "COMPLETE").read_text(encoding="utf-8").strip()
        if marker != manifest["manifest_digest"]:
            findings.append(
                "the origin COMPLETE marker disagrees with the origin manifest"
            )
        reference_source = "origin_manifest"
    else:
        manifest = latest_own_checkpoint(results)
        if manifest is not None:
            reference_source = "latest_own_checkpoint"
            report["reference_checkpoint"] = manifest.get("_checkpoint_dir")
    report["reference_source"] = reference_source or "seal_only_no_checkpoint_yet"
    if manifest is not None:
        report["boundary"] = manifest["boundary"]
        report["manifest_digest"] = manifest["manifest_digest"]
    if manifest is not None:
        if closure["digest"] != manifest["evaluator_closure"]["digest"]:
            findings.append(
                f"sealed closure {closure['digest'][:12]}… != reference {manifest['evaluator_closure']['digest'][:12]}… ({reference_source})"
            )
        entry = manifest["files"].get(f"{PROVENANCE_DIRNAME}/{SEAL_SELF_MANIFEST}")
        if entry is None:
            findings.append(f"the reference manifest does not pin {SEAL_SELF_MANIFEST}")
        elif (
            _sha256_bytes((provenance / SEAL_SELF_MANIFEST).read_bytes())
            != entry["sha256"]
        ):
            findings.append(f"{SEAL_SELF_MANIFEST} drifted from the {reference_source}")
        for relative, expected in sorted(manifest["files"].items()):
            if not relative.startswith(f"{PROVENANCE_DIRNAME}/"):
                continue
            inside = relative[len(PROVENANCE_DIRNAME) + 1 :]
            path = provenance / inside
            if not path.is_file():
                findings.append(f"sealed file missing vs the reference: {inside}")
            elif _sha256_bytes(path.read_bytes()) != expected["sha256"]:
                findings.append(f"sealed file drifted vs the reference: {inside}")
    report["checks"]["seal_anchored_to_reference"] = manifest is not None and (
        not any(("reference" in f for f in findings))
    )
    context_files = [
        n for n in sealed_files if n.startswith(f"{SEALED_TAKEHOME_RELATIVE}/")
    ]
    if not context_files:
        findings.append("the seal contains no session-context take-home")
    for name in context_files:
        if not (provenance / name).is_file():
            findings.append(f"sealed session-context file missing: {name}")
    report["checks"]["session_context_sealed"] = not any(
        ("session-context" in f for f in findings)
    )
    problems = verify_provenance_exact(provenance, sealed_files)
    findings.extend(problems)
    report["checks"]["seal_exact"] = not problems
    try:
        os.environ["SHINKA_TAKEHOME_DIR"] = str(sealed_takehome(provenance))
        load_sealed_evaluator(provenance)
        report["checks"]["sealed_evaluator_usable"] = True
    except CausalCheckpointError as exc:
        findings.append(f"the sealed evaluator is not usable: {exc}")
        report["checks"]["sealed_evaluator_usable"] = False
    recorded_interpreter = json.loads((provenance / "interpreter.json").read_text())
    observed_interpreter = interpreter_identity()
    if observed_interpreter["digest"] != recorded_interpreter["digest"]:
        findings.append(
            f"interpreter drift: {observed_interpreter['executable']} != {recorded_interpreter['executable']}"
        )
    report["checks"]["interpreter_matches"] = (
        observed_interpreter["digest"] == recorded_interpreter["digest"]
    )
    recorded_dependencies = json.loads((provenance / "dependencies.json").read_text())
    observed_dependencies = dependency_identity()
    if not observed_dependencies.get("digest"):
        findings.append(
            f"the dependency set could not be read ({observed_dependencies.get('error')})"
        )
    elif observed_dependencies["digest"] != recorded_dependencies["digest"]:
        findings.append("dependency drift")
    report["checks"]["dependencies_match"] = (
        observed_dependencies.get("digest") == recorded_dependencies["digest"]
    )
    pinned_config = None
    config_path = results / "causal_config.json"
    if manifest is not None:
        pinned_config = manifest["documents"]["config"]["payload"]
    elif config_path.is_file():
        pinned_config = json.loads(config_path.read_text(encoding="utf-8"))
    if pinned_config is None:
        if mode == MODE_FORK:
            findings.append("no pinned configuration to compare against")
        report["checks"]["config_matches_t0"] = None
    else:
        expected_context = pinned_config.get("session_context_digest")
        merged = dict(observed_config)
        merged["session_context_digest"] = _sha256_text(
            json.dumps(
                {
                    k: v
                    for k, v in closure.get("files", {}).items()
                    if k.startswith(f"{SEALED_TAKEHOME_RELATIVE}/")
                },
                sort_keys=True,
            )
        )
        problems = compare_config(
            pinned_config,
            merged,
            allowed_runtime_source_drift=allowed_runtime_source_drift,
        )
        findings.extend((f"config: {problem}" for problem in problems))
        report["checks"]["config_matches_t0"] = not problems
        report["allowed_runtime_source_drift"] = sorted(allowed_runtime_source_drift)
        report["expected_session_context_digest"] = expected_context
    if manifest is not None:
        database = results / "programs.sqlite"
        if not database.is_file():
            findings.append("the restored results directory has no programs.sqlite")
            report["checks"]["database_matches"] = False
        else:
            expected = manifest["documents"]["database_facts"]["payload"]
            observed = database_facts(database)
            if observed["digest"] != expected["digest"]:
                for key in (
                    "tables",
                    "table_counts",
                    "archive_ordered",
                    "metadata_store",
                    "island_membership_counts",
                    "programs_per_generation",
                    "schema",
                ):
                    if observed.get(key) != expected.get(key):
                        findings.append(f"database drift in {key}")
            report["checks"]["database_matches"] = (
                observed["digest"] == expected["digest"]
            )
        state_path = results / "causal_controller_state.json"
        if state_path.is_file():
            observed_state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_state = manifest["documents"]["controller_state"]["payload"]
            recomputed = _sha256_text(
                json.dumps(
                    {k: v for k, v in observed_state.items() if k != "digest"},
                    sort_keys=True,
                )
            )
            if recomputed != observed_state.get("digest"):
                findings.append(
                    "controller state is internally inconsistent: its content does not hash to its own recorded digest"
                )
            if recomputed != expected_state.get("digest"):
                differing = sorted(
                    (
                        k
                        for k in set(observed_state) | set(expected_state)
                        if k != "digest"
                        and observed_state.get(k) != expected_state.get(k)
                    )
                )
                findings.append(
                    f"controller state drifted from the reference: {differing}"
                )
            report["checks"]["controller_state_matches"] = (
                recomputed == expected_state.get("digest")
            )
        elif mode == MODE_FORK:
            findings.append("the restored fork has no causal_controller_state.json")
            report["checks"]["controller_state_matches"] = False
        document_problems: list[str] = []
        for _name, entry in sorted(
            manifest["documents"].items() if mode == MODE_FORK else []
        ):
            path = results / entry["path"]
            if not path.is_file():
                document_problems.append(f"restored document missing: {entry['path']}")
            elif _sha256_bytes(path.read_bytes()) != entry["sha256"]:
                document_problems.append(f"restored document drifted: {entry['path']}")
        findings.extend(document_problems)
        report["checks"]["restored_documents_match"] = (
            not document_problems if mode == MODE_FORK else None
        )
        in_flight = manifest["concurrency_state"].get("in_flight") or {}
        if any(in_flight.values()):
            findings.append(f"the reference records in-flight work: {in_flight}")
        report["checks"]["in_flight_empty"] = not any(in_flight.values())
    if findings:
        raise ContinuationRefused(f"{mode}: " + "; ".join(findings))
    report["validated"] = True
    return {
        "closure": closure,
        "config": pinned_config or observed_config,
        "manifest": manifest,
        "report": report,
        "sealed_files": sealed_files,
    }


def _apply_restored_controller_state(self, runner: Any) -> dict[str, Any] | None:
    """Reinstate pinned controller/RNG/bandit state after native setup.

    Applied for a **resume as well as a fork** whenever a verified reference checkpoint
    exists. A same-campaign resume is not safe to leave alone: native
    ``_setup_async`` recomputes ``total_api_cost`` from persisted programs, which drops
    the cost of failed attempts that never produced a program, and it has no way to
    recover the RNG state or the proposal-timing EWMAs at all. A new campaign, and a
    resume with no verified reference, have nothing to restore.
    """
    if self.mode == MODE_NEW or self.continuation is None:
        return None
    manifest = self.continuation.get("manifest")
    branch = None
    if manifest is None:
        from shinka_resume_capture import load_branch_state

        branch = getattr(self, "resume_branch_state", None) or load_branch_state(
            self.results
        )
        if branch is None:
            return None
        state = branch["controller_state"]
    else:
        state = manifest["documents"]["controller_state"]["payload"]
    restore_controller_state(runner, state)
    applied = {
        "applied_in_mode": self.mode,
        "completed_generations": state["completed_generations"],
        "controller_state_digest": state["digest"],
        "next_generation_to_submit": state["next_generation_to_submit"],
        "python_random_restored": state.get("python_random_state") is not None,
        "numpy_random_restored": state.get("numpy_random_state") is not None,
        "total_api_cost": state["total_api_cost"],
    }
    applied["source"] = (
        "resume_branch_state" if branch is not None else "causal_checkpoint"
    )
    applied["bandit"] = (
        {"present": False, "action": "unsupported_features_disabled"}
        if branch is not None
        else self.reconcile_bandit_state(manifest)
    )
    if self.recorder is not None:
        self.recorder("causal_controller_state_restored", **applied)
    return applied


def _reconcile_bandit_state(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Never silently load a bandit that is not the one the checkpoint pinned.

    A same-campaign resume finds whatever ``bandit_state.pkl`` the live directory has.
    If it disagrees with the reference, restore the reference copy from the verified
    checkpoint; if neither is available, refuse rather than continue with an unknown
    arm-selection state.
    """
    expected = manifest["documents"]["bandit_state"]["payload"]
    live = self.results / "bandit_state.pkl"
    observed = bandit_state_facts(self.results)
    if not expected.get("present"):
        if observed.get("present"):
            raise ContinuationRefused(
                "the reference checkpoint pinned NO bandit state but this run has a bandit_state.pkl; refusing to continue with an unpinned bandit"
            )
        return {"present": False, "action": "none_pinned_none_present"}
    if observed.get("present") and observed.get("sha256") == expected.get("sha256"):
        return {
            "present": True,
            "action": "verified_matching",
            "sha256": expected["sha256"],
        }
    reference_dir = manifest.get("_checkpoint_dir")
    source = Path(reference_dir) / BANDIT_SNAPSHOT if reference_dir else None
    if source is not None and source.is_file():
        if _sha256_bytes(source.read_bytes()) != expected["sha256"]:
            raise ContinuationRefused(
                "the reference checkpoint's bandit snapshot does not match its own manifest digest"
            )
        shutil.copyfile(source, live)
        return {
            "present": True,
            "action": "restored_from_reference",
            "sha256": expected["sha256"],
        }
    raise ContinuationRefused(
        f"the reference pinned bandit state {expected['sha256'][:12]}… but the live copy is {observed.get('sha256') or 'absent'!r} and no reference snapshot is available to restore from"
    )


ShinkaCausalCheckpointer.apply_restored_controller_state = (
    _apply_restored_controller_state
)
ShinkaCausalCheckpointer.reconcile_bandit_state = _reconcile_bandit_state
del _apply_restored_controller_state, _reconcile_bandit_state
