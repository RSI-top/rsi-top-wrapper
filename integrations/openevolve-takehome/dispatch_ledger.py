"""Recover the dispatch and completion ledger for exact checkpoint branches."""

from __future__ import annotations

import json

import os

from dataclasses import dataclass

from typing import Any

_EVT_SCHEDULED = "attempt_scheduled"

_EVT_COMPLETION = "canonical_completion"

_EVT_COMMITTED = "program_committed"

_EVT_COMMIT_FAILED = "program_commit_failed"

_EVT_META_ELIGIBLE = "meta_eligible_completion"


def attempt_id_for(instance_id: str | None, iteration: int) -> str:
    """Canonical composite launch identity (matches canonical_completion)."""
    return f"{instance_id}:{iteration}"


def _instance_of(attempt_id: str | None) -> str | None:
    if not attempt_id:
        return None
    return attempt_id.rsplit(":", 1)[0]


def fitness_of(metrics: Any, feature_dimensions: list[str]) -> float | None:
    """Global-best fitness exactly as the upstream database computes it."""
    if not isinstance(metrics, dict) or not metrics:
        return None
    from openevolve.utils.metrics_utils import get_fitness_score

    try:
        return float(get_fitness_score(metrics, feature_dimensions))
    except Exception:
        return None


@dataclass
class DispatchRecord:
    attempt_id: str
    iteration: int
    instance_id: str | None
    island: int | None
    parent_id: str | None
    inspiration_ids: tuple[str, ...]
    submitted_at: float | None
    submitted_after_seq: int
    status: str = "inflight"  # inflight | consumed | orphaned
    completion_seq: int | None = None


@dataclass
class CompletionRecord:
    completion_seq: int
    attempt_id: str | None
    instance_id: str | None
    iteration: int
    island: int | None
    parent_id: str | None
    status: str
    terminal: str
    evaluation_valid: bool
    at: float | None = None
    error: str | None = None
    failure_stage: str | None = None
    child: dict[str, Any] | None = None  # id/metrics/generation/code_sha256/changes
    # Launch metadata for THIS exact attempt (copied from its dispatch record).
    inspiration_ids: tuple[str, ...] | None = None
    submitted_at: float | None = None
    submitted_after_seq: int | None = None
    committed: bool = False
    frontier_improved: bool = False
    eligible: bool = False
    #: True when database.add raised for this consumed valid child: terminal,
    #: settled, never committed and never eligible.
    commit_failed: bool = False
    #: True when this completion belongs to a prior process instance and lies
    #: beyond the checkpoint boundary the current process resumed from — its
    #: effects are NOT part of the current live database.
    rolled_back_on_resume: bool = False


class DispatchLedger:
    """Causal ledger of submissions and controller-consumed completions.

    Keyed by the collision-free composite ``attempt_id``
    (``run_instance_id:iteration``): iteration numbers repeat across
    checkpoint rollback/resume, and a completion is only ever paired with the
    dispatch of its own process instance. Live updates come from the traced
    controller hooks; on construction the ledger is recovered from
    attempts.jsonl (torn/malformed lines skipped). Completion sequences come
    exclusively from the wrapper's CanonicalCompletionRecorder and are never
    reassigned or reused here. Dispatches recorded by earlier process
    instances that never reached a terminal are marked ``orphaned`` (their
    worker died with the old pool) and are excluded from cutoff-time
    in-flight metadata.
    """

    def __init__(self, current_instance_id: str) -> None:
        self._instance_id = current_instance_id
        self.dispatches: dict[str, DispatchRecord] = {}
        self.completions: dict[int, CompletionRecord] = {}
        self.last_completion_seq: int = -1

    # -- live updates (controller thread) -----------------------------------

    def record_dispatch(
        self,
        *,
        attempt_id: str,
        iteration: int,
        island: int | None,
        parent_id: str | None,
        inspiration_ids: tuple[str, ...],
        at: float | None,
    ) -> None:
        self.dispatches[attempt_id] = DispatchRecord(
            attempt_id=attempt_id,
            iteration=iteration,
            instance_id=_instance_of(attempt_id),
            island=island,
            parent_id=parent_id,
            inspiration_ids=tuple(inspiration_ids),
            submitted_at=at,
            submitted_after_seq=self.last_completion_seq,
        )

    def record_completion(
        self,
        *,
        attempt_id: str | None,
        completion_seq: int,
        iteration: int,
        island: int | None,
        parent_id: str | None,
        status: str,
        terminal: str,
        evaluation_valid: bool,
        at: float | None = None,
        error: str | None = None,
        failure_stage: str | None = None,
        child: dict[str, Any] | None = None,
    ) -> CompletionRecord:
        if completion_seq in self.completions:  # never reuse a sequence
            return self.completions[completion_seq]
        dispatch = self.dispatches.get(attempt_id) if attempt_id else None
        record = CompletionRecord(
            completion_seq=completion_seq,
            attempt_id=attempt_id,
            instance_id=_instance_of(attempt_id),
            iteration=iteration,
            island=island,
            parent_id=parent_id,
            status=status,
            terminal=terminal,
            evaluation_valid=bool(evaluation_valid),
            at=at,
            error=error,
            failure_stage=failure_stage,
            child=child,
            inspiration_ids=(dispatch.inspiration_ids if dispatch else None),
            submitted_at=(dispatch.submitted_at if dispatch else None),
            submitted_after_seq=(dispatch.submitted_after_seq if dispatch else None),
        )
        self.completions[completion_seq] = record
        self.last_completion_seq = max(self.last_completion_seq, completion_seq)
        if dispatch is not None and dispatch.completion_seq is None:
            dispatch.status = "consumed"
            dispatch.completion_seq = completion_seq
        return record


    def mark_commit_failed(self, completion_seq: int) -> None:
        """database.add raised for this consumed child: settled, never committed."""
        record = self.completions.get(completion_seq)
        if record is None:
            return
        record.commit_failed = True
        record.committed = False
        record.eligible = False


    # -- recovery ------------------------------------------------------------

    @staticmethod
    def _child_summary(raw_child: Any) -> dict[str, Any] | None:
        if not isinstance(raw_child, dict):
            return None
        metadata = raw_child.get("metadata")
        changes = metadata.get("changes") if isinstance(metadata, dict) else None
        return {
            "id": raw_child.get("id"),
            "metrics": raw_child.get("metrics"),
            "generation": raw_child.get("generation"),
            "code_sha256": raw_child.get("code_sha256"),
            "changes": changes,
        }

    @classmethod
    def recover(
        cls, attempts_path: str | os.PathLike | None, current_instance_id: str
    ) -> "DispatchLedger":
        ledger = cls(current_instance_id)
        if not attempts_path or not os.path.isfile(attempts_path):
            return ledger
        with open(attempts_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue  # torn/foreign line: skip, never guess
                if not isinstance(record, dict):
                    continue
                event = record.get("event")
                if event == _EVT_SCHEDULED:
                    iteration = record.get("iteration")
                    if not isinstance(iteration, int):
                        continue
                    # Backward compatible: old traces lack attempt_id on
                    # attempt_scheduled — derive it from the record's own
                    # run_instance_id + iteration (never from another
                    # instance's reused iteration number).
                    attempt_id = record.get("attempt_id") or attempt_id_for(
                        record.get("run_instance_id"), iteration
                    )
                    ledger.record_dispatch(
                        attempt_id=attempt_id,
                        iteration=iteration,
                        island=record.get("target_island"),
                        parent_id=record.get("parent_id"),
                        inspiration_ids=tuple(record.get("inspiration_ids") or ()),
                        at=record.get("timestamp"),
                    )
                elif event == _EVT_COMPLETION:
                    seq = record.get("completion_seq")
                    iteration = record.get("iteration")
                    if not isinstance(seq, int) or not isinstance(iteration, int):
                        continue
                    attempt_id = record.get("attempt_id") or attempt_id_for(
                        record.get("run_instance_id"), iteration
                    )
                    ledger.record_completion(
                        attempt_id=attempt_id,
                        completion_seq=seq,
                        iteration=iteration,
                        island=record.get("scheduled_target_island"),
                        parent_id=record.get("scheduled_parent_id"),
                        status=str(record.get("status")),
                        terminal=str(record.get("terminal")),
                        evaluation_valid=bool(record.get("evaluation_valid")),
                        at=record.get("timestamp"),
                        error=record.get("error"),
                        failure_stage=record.get("failure_stage"),
                        child=cls._child_summary(record.get("child")),
                    )
                elif event == _EVT_COMMITTED:
                    seq = record.get("completion_seq")
                    if isinstance(seq, int) and seq in ledger.completions:
                        ledger.completions[seq].committed = True
                elif event == _EVT_COMMIT_FAILED:
                    seq = record.get("completion_seq")
                    if isinstance(seq, int) and seq in ledger.completions:
                        ledger.mark_commit_failed(seq)
                elif event == _EVT_META_ELIGIBLE:
                    seq = record.get("completion_seq")
                    if isinstance(seq, int) and seq in ledger.completions:
                        ledger.completions[seq].frontier_improved = bool(
                            record.get("frontier_improved")
                        )
                        ledger.completions[seq].eligible = bool(record.get("eligible"))
        # Dispatches from dead prior processes can never complete: not inflight.
        for dispatch in ledger.dispatches.values():
            if (
                dispatch.completion_seq is None
                and dispatch.instance_id != current_instance_id
            ):
                dispatch.status = "orphaned"
        return ledger

