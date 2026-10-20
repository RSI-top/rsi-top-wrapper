"""Reconstruct causal attempt history and code geometry from native traces."""

from __future__ import annotations

import hashlib

import json

import re

from pathlib import Path

from typing import Any, Iterable

import history_utils as base

from code_geometry import (
    STATE_WINDOW as DASHBOARD_STATE_WINDOW,
    implementation_spread,
    signed_fingerprint,
)

ARM_DIVERSITY = "diversity"

ATTEMPT_WINDOW = 24

AXIS_ADAPTIVE_UNFOLD_MIN = 2

ADAPTIVE_UNFOLD_MIN = 2 * AXIS_ADAPTIVE_UNFOLD_MIN

MIN_CODE_UNFOLDS = 3

MEANINGFUL_GAIN = 0.002

ATTEMPTS_EVIDENCE_ID = "global:recent_attempts"

RECENT_ATTEMPT_ROOT = "recent_attempts"

SEARCH_HISTORY_ROOT = "search_history"

SEARCH_HISTORY_INDEX = f"{SEARCH_HISTORY_ROOT}/index.jsonl"

SEARCH_HISTORY_DASHBOARD = f"{SEARCH_HISTORY_ROOT}/dashboard.json"

DASHBOARD_TRANSITION_WARMUP = 6

if ATTEMPT_WINDOW <= 0:
    raise ValueError("ATTEMPT_WINDOW must be positive")

if ADAPTIVE_UNFOLD_MIN < 0:
    raise ValueError("ADAPTIVE_UNFOLD_MIN cannot be negative")

if ADAPTIVE_UNFOLD_MIN >= ATTEMPT_WINDOW:
    raise ValueError(
        "ADAPTIVE_UNFOLD_MIN must leave room for at least one required unfold"
    )

if MIN_CODE_UNFOLDS < 0:
    raise ValueError("MIN_CODE_UNFOLDS cannot be negative")


def _cycles(metrics: Any) -> float | None:
    return base.finite(metrics.get("cycles")) if isinstance(metrics, dict) else None


def _code_sha256(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _terminal_class(
    *, valid: bool, child: dict[str, Any], event: dict[str, Any]
) -> str:
    if valid:
        return "valid"
    terminal = str(event.get("terminal") or event.get("status") or "").lower()
    if terminal == "worker_error" or not child:
        return "worker_error"
    return "invalid"


def _choose_required_unfolds(
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose shared anchor records without using State Diversity."""
    if not attempts:
        return []

    chosen: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    def add(row: dict[str, Any] | None, role: str) -> None:
        if row is None:
            return
        seq = int(row["seq"])
        if seq not in chosen:
            chosen[seq] = {
                "evidence_id": row["evidence_id"],
                "completion_seq": seq,
                "roles": [],
            }
            order.append(seq)
        if role not in chosen[seq]["roles"]:
            chosen[seq]["roles"].append(role)

    add(attempts[-1], "latest_terminal")

    valid = [
        row
        for row in attempts
        if row.get("valid") is True
        and _cycles({"cycles": row.get("cycles")}) is not None
    ]
    strongest_valid = (
        min(valid, key=lambda row: (float(row["cycles"]), -int(row["seq"])))
        if valid
        else None
    )
    add(strongest_valid, "best_valid_cycles")

    failure_options = (
        ("no_op_representative", lambda row: row.get("no_op") is True),
        (
            "exact_duplicate_representative",
            lambda row: row.get("exact_duplicate") is True,
        ),
        (
            "invalid_representative",
            lambda row: row.get("terminal_class") == "invalid",
        ),
        (
            "worker_error_representative",
            lambda row: row.get("terminal_class") == "worker_error",
        ),
    )
    for role, predicate in failure_options:
        candidates = [
            row for row in attempts if predicate(row) and int(row["seq"]) not in chosen
        ]
        if candidates:
            add(max(candidates, key=lambda row: int(row["seq"])), role)
            break

    return [chosen[seq] for seq in order]


def _build_history_from_events(
    *,
    events: list[dict[str, Any]],
    root_code: str,
    root_cycles: float,
    started_at: float,
    cutoff_seq: int,
    cutoff_at: float,
) -> dict[str, Any]:
    """Fold a causal prefix into search state and the recent attempt window."""
    schedules: dict[str, dict[str, Any]] = {}
    completion_events: dict[int, dict[str, Any]] = {}
    attempt_by_seq: dict[int, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []

    root_hash = _code_sha256(root_code)
    seen_code_hashes: dict[str, str] = {root_hash: "root"}
    program_code_hashes: dict[str, str] = {"root": root_hash}

    best_cycles = float(root_cycles)
    best_program_id = "root"
    progress_anchor_cycles = float(root_cycles)
    progress_anchor_seq = 0
    progress_anchor_at = float(started_at)
    last_meaningful: dict[str, Any] = {
        "completion_seq": 0,
        "timestamp": float(started_at),
        "program_id": "root",
        "cycles": float(root_cycles),
        "previous_anchor_cycles": None,
        "relative_gain": None,
    }
    meaningful_frontier_improvement_count = 0

    for event in events:
        name = event.get("event")
        if name == "attempt_scheduled":
            attempt_id = str(event.get("attempt_id"))
            schedules[attempt_id] = event
            parent_id = event.get("parent_id")
            parent_hash = event.get("parent_code_sha256")
            if isinstance(parent_id, str) and isinstance(parent_hash, str):
                program_code_hashes.setdefault(parent_id, parent_hash)
            continue

        seq = event.get("completion_seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq > cutoff_seq:
            continue

        if name == "canonical_completion":
            child = event.get("child") if isinstance(event.get("child"), dict) else {}
            schedule = schedules.get(str(event.get("attempt_id"))) or {}
            code = child.get("code") if isinstance(child.get("code"), str) else None
            code_hash = _code_sha256(code) if code is not None else None
            cycles = _cycles(child.get("metrics"))
            parent_cycles = _cycles(schedule.get("parent_metrics"))
            valid = bool(
                event.get("evaluation_valid") is True
                and event.get("failure_stage") in (None, "")
                and cycles is not None
            )
            terminal_class = _terminal_class(valid=valid, child=child, event=event)
            parent_id = schedule.get("parent_id") or event.get("parent_id")
            parent_id = str(parent_id) if parent_id is not None else None
            parent_hash = schedule.get("parent_code_sha256")
            if not isinstance(parent_hash, str) and parent_id is not None:
                parent_hash = program_code_hashes.get(parent_id)
            if not isinstance(parent_hash, str):
                parent_code = schedule.get("parent_code")
                parent_hash = (
                    _code_sha256(parent_code) if isinstance(parent_code, str) else None
                )

            duplicate_of = seen_code_hashes.get(code_hash) if code_hash else None
            exact_duplicate = duplicate_of is not None if code_hash else None
            no_op = (
                code_hash == parent_hash
                if code_hash is not None and parent_hash is not None
                else None
            )
            evidence_id = f"attempt:seq:{seq:08d}"
            if code_hash is not None and code_hash not in seen_code_hashes:
                seen_code_hashes[code_hash] = evidence_id

            metadata = (
                event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            )
            changes = str(
                metadata.get("changes")
                or child.get("changes_description")
                or "No recorded change summary."
            )
            observed_at = base.finite(event.get("timestamp"))
            attempt = {
                "attempt_id": str(event.get("attempt_id") or f"seq-{seq}"),
                "evidence_id": evidence_id,
                "seq": seq,
                "timestamp": observed_at,
                "elapsed_minutes": max(
                    0.0,
                    (
                        (observed_at if observed_at is not None else cutoff_at)
                        - started_at
                    )
                    / 60.0,
                ),
                "terminal": event.get("terminal") or event.get("status"),
                "terminal_class": terminal_class,
                "valid": valid,
                "failure_stage": event.get("failure_stage"),
                "code_bearing": code is not None,
                "code": code,
                "code_sha256": code_hash,
                "exact_duplicate": exact_duplicate,
                "duplicate_of": duplicate_of,
                "no_op": no_op,
                "cycles": cycles,
                "parent_cycles": parent_cycles,
                "frontier_before_cycles": float(best_cycles),
                "gap_to_frontier_cycles": (
                    float(cycles - best_cycles)
                    if valid and cycles is not None
                    else None
                ),
                "frontier_improvement": bool(
                    valid and cycles is not None and cycles < best_cycles
                ),
                "meaningful_frontier_improvement": False,
                "program_id": child.get("id"),
                "parent_id": parent_id,
                "target_island": (
                    schedule.get("target_island")
                    if schedule.get("target_island") is not None
                    else event.get("scheduled_target_island")
                ),
                "changes": changes,
                "committed": False,
                "source_events": [row for row in (schedule, event) if row],
            }
            attempt_by_seq[seq] = attempt
            completion_events[seq] = {
                "event": event,
                "schedule": schedule,
                "summary": {
                    "seq": seq,
                    "timestamp": observed_at,
                    "valid": valid,
                    "cycles": cycles,
                    "parent_cycles": parent_cycles,
                    "program_id": child.get("id"),
                    "parent_id": parent_id,
                },
            }
            continue

        if name != "program_committed":
            continue

        completion = completion_events.get(seq)
        attempt = attempt_by_seq.get(seq)
        if completion is None:
            continue
        if attempt is not None:
            attempt["source_events"].append(event)
            attempt["committed"] = True

        canonical = completion["event"]
        summary = completion["summary"]
        child = (
            canonical.get("child") if isinstance(canonical.get("child"), dict) else {}
        )
        code = child.get("code") if isinstance(child.get("code"), str) else None
        cycles = _cycles(event.get("metrics")) or summary.get("cycles")
        if not summary["valid"] or code is None or cycles is None:
            continue

        program_id = str(
            event.get("program_id") or summary.get("program_id") or f"seq-{seq}"
        )
        code_hash = _code_sha256(code)
        program_code_hashes[program_id] = code_hash
        if attempt is not None:
            attempt["program_id"] = program_id

        frontier_before = best_cycles
        frontier_improvement = cycles < best_cycles
        meaningful = False
        if frontier_improvement:
            best_cycles = float(cycles)
            best_program_id = program_id
            gain = (progress_anchor_cycles - best_cycles) / abs(progress_anchor_cycles)
            meaningful = gain > MEANINGFUL_GAIN
            if meaningful:
                previous = progress_anchor_cycles
                progress_anchor_cycles = best_cycles
                progress_anchor_seq = seq
                observed_at = base.finite(summary.get("timestamp"))
                progress_anchor_at = (
                    observed_at if observed_at is not None else float(cutoff_at)
                )
                last_meaningful = {
                    "completion_seq": seq,
                    "timestamp": progress_anchor_at,
                    "program_id": program_id,
                    "cycles": best_cycles,
                    "previous_anchor_cycles": previous,
                    "relative_gain": gain,
                }
                meaningful_frontier_improvement_count += 1

        if attempt is not None:
            attempt["frontier_before_cycles"] = float(frontier_before)
            attempt["gap_to_frontier_cycles"] = (
                float(cycles - frontier_before)
                if attempt.get("terminal_class") == "valid"
                else None
            )
            attempt["frontier_improvement"] = frontier_improvement
            attempt["meaningful_frontier_improvement"] = meaningful

        metadata = (
            event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        )
        changes = str(
            metadata.get("changes")
            or child.get("changes_description")
            or "No recorded change summary."
        )
        observed_at = base.finite(summary.get("timestamp"))
        entries.append(
            {
                **summary,
                "program_id": program_id,
                "elapsed_minutes": max(
                    0.0,
                    (
                        (observed_at if observed_at is not None else cutoff_at)
                        - started_at
                    )
                    / 60.0,
                ),
                "cycles": float(cycles),
                "frontier_before_cycles": float(frontier_before),
                "gap_to_frontier_cycles": float(cycles - frontier_before),
                "frontier_improvement": frontier_improvement,
                "meaningful_frontier_improvement": meaningful,
                "changes": changes,
                "code": code,
                "source_events": tuple(
                    row for row in (completion["schedule"], canonical, event) if row
                ),
                "evidence_id": f"attempt:seq:{seq:08d}",
            }
        )

    entries.sort(key=lambda row: int(row["seq"]))
    all_attempts = sorted(attempt_by_seq.values(), key=lambda row: int(row["seq"]))
    for attempt in all_attempts:
        attempt["source_events"] = tuple(attempt["source_events"])

    recent_attempts = all_attempts[-ATTEMPT_WINDOW:]
    window_size = len(recent_attempts)
    valid_count = sum(row["terminal_class"] == "valid" for row in recent_attempts)
    invalid_count = sum(row["terminal_class"] == "invalid" for row in recent_attempts)
    worker_error_count = sum(
        row["terminal_class"] == "worker_error" for row in recent_attempts
    )

    def gap_ratio(row: dict[str, Any]) -> float | None:
        gap = base.finite(row.get("gap_to_frontier_cycles"))
        frontier = base.finite(row.get("frontier_before_cycles"))
        if gap is None or frontier is None or frontier == 0:
            return None
        return gap / abs(frontier)

    def parent_improved(row: dict[str, Any]) -> bool:
        cycles = base.finite(row.get("cycles"))
        parent_cycles = base.finite(row.get("parent_cycles"))
        return bool(
            row.get("terminal_class") == "valid"
            and cycles is not None
            and parent_cycles is not None
            and cycles < parent_cycles
        )

    def yield_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        valid_rows = [row for row in rows if row["terminal_class"] == "valid"]
        gaps = [
            float(gap)
            for row in valid_rows
            if (gap := base.finite(row.get("gap_to_frontier_cycles"))) is not None
        ]
        ratios = [ratio for row in valid_rows if (ratio := gap_ratio(row)) is not None]
        return {
            "attempt_count": len(rows),
            "valid_count": len(valid_rows),
            "parent_improving_valid_count": sum(parent_improved(row) for row in rows),
            "frontier_tie_count": sum(abs(gap) <= 1e-9 for gap in gaps),
            "near_frontier_2pct_count": sum(ratio <= 0.02 for ratio in ratios),
            "near_frontier_10pct_count": sum(ratio <= 0.10 for ratio in ratios),
            "best_gap_to_frontier_cycles": min(gaps) if gaps else None,
        }

    island_rows: dict[str, list[dict[str, Any]]] = {}
    for row in recent_attempts:
        island = row.get("target_island")
        island_rows.setdefault("unknown" if island is None else str(island), []).append(
            row
        )
    population_yield = yield_summary(recent_attempts)
    population_yield.update(
        {
            "distinct_parent_count": len(
                {
                    str(row["parent_id"])
                    for row in recent_attempts
                    if row.get("parent_id") is not None
                }
            ),
            "island_yield": {
                island: yield_summary(rows)
                for island, rows in sorted(island_rows.items())
            },
        }
    )
    recent_attempt_summary = {
        "window_size": window_size,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "worker_error_count": worker_error_count,
        "valid_rate": valid_count / window_size if window_size else None,
        "code_bearing_count": sum(bool(row["code_bearing"]) for row in recent_attempts),
        "exact_duplicate_count": sum(
            row["exact_duplicate"] is True for row in recent_attempts
        ),
        "no_op_count": sum(row["no_op"] is True for row in recent_attempts),
        "population_yield": population_yield,
    }
    required_unfolds = _choose_required_unfolds(recent_attempts)
    required_roles = {
        int(row["completion_seq"]): list(row["roles"]) for row in required_unfolds
    }
    for attempt in recent_attempts:
        attempt["required_unfold_roles"] = required_roles.get(int(attempt["seq"]), [])

    plateau_entries = [row for row in entries if row["seq"] > progress_anchor_seq]
    current_frontier_entry = next(
        (row for row in reversed(entries) if row["program_id"] == best_program_id),
        None,
    )
    return {
        "entries": entries,
        "entry_by_seq": {int(entry["seq"]): entry for entry in entries},
        "attempt_by_seq": {int(row["seq"]): row for row in all_attempts},
        "recent_attempts": recent_attempts,
        "recent_attempt_summary": recent_attempt_summary,
        "required_unfolds": required_unfolds,
        "root_code": root_code,
        "root_cycles": float(root_cycles),
        "started_at": float(started_at),
        "current_frontier_cycles": best_cycles,
        "current_frontier_program_id": best_program_id,
        "current_frontier_evidence_id": (
            current_frontier_entry["evidence_id"]
            if current_frontier_entry is not None
            else None
        ),
        "run_elapsed_seconds": max(0.0, cutoff_at - started_at),
        "valid_committed_trajectory_count": len(entries),
        "meaningful_frontier_improvement_count": meaningful_frontier_improvement_count,
        "last_meaningful": last_meaningful,
        "plateau_anchor_seq": progress_anchor_seq,
        "plateau_wallclock_age_seconds": max(0.0, cutoff_at - progress_anchor_at),
        "plateau_valid_committed_trajectory_count": len(plateau_entries),
    }


def _serialize_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    seq = int(attempt["seq"])
    valid = attempt.get("terminal_class") == "valid"
    record = {
        "evidence_id": attempt["evidence_id"],
        "completion_seq": seq,
        "trajectory_location": (
            f"{RECENT_ATTEMPT_ROOT}/seq_{seq:08d}/trajectory.jsonl"
        ),
        "elapsed_minutes": attempt.get("elapsed_minutes"),
        "terminal_class": attempt.get("terminal_class"),
        "failure_stage": attempt.get("failure_stage"),
        "program_id": attempt.get("program_id"),
        "parent_id": attempt.get("parent_id"),
        "cycles": attempt.get("cycles") if valid else None,
        "parent_cycles": attempt.get("parent_cycles"),
        "frontier_before_cycles": attempt.get("frontier_before_cycles"),
        "gap_to_frontier_cycles": attempt.get("gap_to_frontier_cycles"),
        "meaningful_frontier_improvement": attempt.get(
            "meaningful_frontier_improvement"
        ),
        "target_island": attempt.get("target_island"),
        "exact_duplicate": attempt.get("exact_duplicate"),
        "duplicate_of": attempt.get("duplicate_of"),
        "no_op": attempt.get("no_op"),
        "required_unfold_roles": list(attempt.get("required_unfold_roles") or []),
    }
    changes = str(attempt.get("changes") or "").strip()
    if changes and changes != "No recorded change summary.":
        record["changes"] = changes
    if attempt.get("code_bearing") is True:
        record["code_location"] = f"{RECENT_ATTEMPT_ROOT}/seq_{seq:08d}/code.py"
    return record


def _performance_timeline(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress per-attempt performance while preserving completion order."""
    timeline: list[dict[str, Any]] = []
    for record in records:
        candidate_cycles = base.finite(record.get("cycles"))
        parent_cycles = base.finite(record.get("parent_cycles"))
        timeline.append(
            {
                "evidence_id": record["evidence_id"],
                "completion_seq": int(record["completion_seq"]),
                "elapsed_minutes": record.get("elapsed_minutes"),
                "terminal_class": record.get("terminal_class"),
                "candidate_cycles": candidate_cycles,
                "parent_cycles": parent_cycles,
                "parent_gain_cycles": (
                    parent_cycles - candidate_cycles
                    if parent_cycles is not None and candidate_cycles is not None
                    else None
                ),
                "gap_to_frontier_cycles": record.get("gap_to_frontier_cycles"),
                "meaningful_frontier_improvement": record.get(
                    "meaningful_frontier_improvement"
                ),
            }
        )
    return timeline


def _render_global_history(
    *,
    candidate: dict[str, Any],
    history: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    attempts = list(history["recent_attempts"])
    records = [_serialize_attempt(row) for row in attempts]
    last_meaningful = history["last_meaningful"]
    last_timestamp = base.finite(last_meaningful.get("timestamp"))
    search_state = {
        "initial_cycles": history["root_cycles"],
        "current_frontier": {
            "program_id": history["current_frontier_program_id"],
            "evidence_id": history["current_frontier_evidence_id"],
            "cycles": history["current_frontier_cycles"],
        },
        "run": {
            "elapsed_minutes": history["run_elapsed_seconds"] / 60.0,
            "valid_committed_trajectory_count": history[
                "valid_committed_trajectory_count"
            ],
            "meaningful_frontier_improvement_count": history[
                "meaningful_frontier_improvement_count"
            ],
        },
        "last_meaningful_frontier_improvement": {
            "completion_seq": int(last_meaningful["completion_seq"]),
            "elapsed_minutes": (
                max(0.0, last_timestamp - history["started_at"]) / 60.0
                if last_timestamp is not None
                else None
            ),
            "program_id": last_meaningful["program_id"],
            "previous_frontier_cycles": last_meaningful["previous_anchor_cycles"],
            "new_frontier_cycles": last_meaningful["cycles"],
            "relative_gain": last_meaningful["relative_gain"],
        },
        "current_plateau": {
            "starts_after_completion_seq": history["plateau_anchor_seq"],
            "elapsed_minutes": history["plateau_wallclock_age_seconds"] / 60.0,
            "valid_committed_trajectory_count": history[
                "plateau_valid_committed_trajectory_count"
            ],
        },
    }
    recent_summary = dict(history["recent_attempt_summary"])
    population_yield = recent_summary.pop("population_yield")
    performance_timeline = _performance_timeline(records)
    attempt_window = {
        "summary": recent_summary,
        "required_unfolds": history["required_unfolds"],
        "record_count": len(records),
        "records": records,
    }
    sections = [
        "[global:search_state]\n" + json.dumps(search_state, indent=2, sort_keys=True),
        f"[{ATTEMPTS_EVIDENCE_ID}]\n"
        + json.dumps(attempt_window, indent=2, sort_keys=True),
    ]
    rendered = "\n\n".join(sections)
    metadata = {
        "attempt_evidence_ids": [row["evidence_id"] for row in records],
        "attempt_completion_seqs": [int(row["completion_seq"]) for row in records],
        "required_unfold_ids": [
            row["evidence_id"] for row in history["required_unfolds"]
        ],
        "required_unfolds": history["required_unfolds"],
        "population_yield": population_yield,
        "performance_timeline": performance_timeline,
        "records": records,
        "global_history_token_proxy": len(re.findall(r"\w+|[^\w\s]", rendered)),
        "candidate_cutoff_seq": int(candidate["completion_seq"]),
    }
    return rendered, metadata


def _state_dynamics(
    candidate: dict[str, Any], recent_completion_seqs: Iterable[int]
) -> tuple[str, dict[str, Any]]:
    state = candidate["state_context"]
    root = state["root_relative_state_spread"]
    frontier = state["frontier_relative_state_spread"]
    root_values = list(root.get("values_oldest_to_newest") or [])
    frontier_values = list(frontier.get("values_oldest_to_newest") or [])
    seqs = [int(seq) for seq in state.get("window_completion_seqs") or []]
    if len(root_values) != len(frontier_values) or len(root_values) != len(seqs):
        raise base.ExperimentError(
            "State Dynamics S/F values and completion seqs are misaligned"
        )

    recent_set = {int(seq) for seq in recent_completion_seqs}
    observations: list[dict[str, Any]] = []
    previous_s: float | None = None
    previous_f: float | None = None
    for seq, s_value, f_value in zip(seqs, root_values, frontier_values):
        s_float = float(s_value)
        f_float = float(f_value)
        observations.append(
            {
                "completion_seq": seq,
                "S": s_float,
                "F": f_float,
                "delta_S_from_previous": (
                    s_float - previous_s if previous_s is not None else None
                ),
                "delta_F_from_previous": (
                    f_float - previous_f if previous_f is not None else None
                ),
                "unfoldable_in_attempt_window": seq in recent_set,
            }
        )
        previous_s = s_float
        previous_f = f_float

    document: dict[str, Any] = {
        "context_id": "state:dynamics",
        "purpose": "routing_only",
        "scope": "causal observations at or before the frozen cutoff",
        "definition": {
            "S": "root-relative implementation-state spread",
            "F": (
                "implementation-state spread relative to the latest meaningful "
                "frontier, frozen throughout the current reference epoch"
            ),
        },
        "F_reference": state.get("frontier_reference"),
        "availability": {
            "observation_count": len(observations),
            "observations_in_attempt_window": sum(
                bool(row["unfoldable_in_attempt_window"]) for row in observations
            ),
        },
        "observations_oldest_to_newest": observations,
    }
    return (
        "[state:dynamics]\n" + json.dumps(document, indent=2),
        document,
    )


def _attempt_files(
    attempts: Iterable[dict[str, Any]], *, root_name: str
) -> dict[str, str]:
    """Build immutable trajectory/code read targets for selected attempts."""

    def null_unverified_cycles(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: None if key == "cycles" else null_unverified_cycles(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [null_unverified_cycles(item) for item in value]
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                return value
            if not isinstance(decoded, (dict, list)):
                return value
            return json.dumps(
                null_unverified_cycles(decoded),
                sort_keys=True,
                separators=(",", ":"),
            )
        return value

    files: dict[str, str] = {}
    for attempt in attempts:
        seq = int(attempt["seq"])
        root = f"{root_name}/seq_{seq:08d}"
        events = attempt.get("source_events")
        if not isinstance(events, tuple) or not events:
            raise base.ExperimentError(f"attempt seq {seq} has no source events")

        projected_events = []
        for event in events:
            projected = dict(event)
            child = projected.get("child")
            if isinstance(child, dict):
                projected_child = dict(child)
                if isinstance(projected_child.get("code"), str):
                    projected_child.pop("code")
                    projected_child["code_location"] = f"{root}/code.py"
                if attempt.get("terminal_class") != "valid":
                    metrics = projected_child.get("metrics")
                    if isinstance(metrics, dict):
                        projected_child["metrics"] = {
                            **metrics,
                            "cycles": None,
                        }
                projected["child"] = projected_child
            if attempt.get("terminal_class") != "valid":
                metrics = projected.get("metrics")
                if isinstance(metrics, dict):
                    projected["metrics"] = {**metrics, "cycles": None}
                artifacts = projected.get("artifacts")
                if isinstance(artifacts, dict):
                    projected["artifacts"] = null_unverified_cycles(artifacts)
            projected_events.append(projected)
        files[f"{root}/trajectory.jsonl"] = "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"), default=str) + "\n"
            for event in projected_events
        )
        code = attempt.get("code")
        if isinstance(code, str):
            files[f"{root}/code.py"] = code
    return files


def _intervention_history_files(history: dict[str, Any]) -> dict[str, str]:
    """Expose the causal attempt history without expanding the audit prompt."""
    attempts = sorted(
        history["attempt_by_seq"].values(), key=lambda row: int(row["seq"])
    )
    files = _attempt_files(attempts, root_name=SEARCH_HISTORY_ROOT)
    index_rows = []
    for attempt in attempts:
        seq = int(attempt["seq"])
        cycles = base.finite(attempt.get("cycles"))
        parent_cycles = base.finite(attempt.get("parent_cycles"))
        changes = re.sub(r"\s+", " ", str(attempt.get("changes") or "")).strip()
        record = {
            "evidence_id": attempt["evidence_id"],
            "completion_seq": seq,
            "elapsed_minutes": attempt.get("elapsed_minutes"),
            "terminal_class": attempt.get("terminal_class"),
            "committed": attempt.get("committed") is True,
            "code_bearing": attempt.get("code_bearing") is True,
            "cycles": cycles,
            "parent_cycles": parent_cycles,
            "parent_gain_cycles": (
                parent_cycles - cycles
                if parent_cycles is not None and cycles is not None
                else None
            ),
            "gap_to_frontier_cycles": attempt.get("gap_to_frontier_cycles"),
            "frontier_improvement": attempt.get("frontier_improvement"),
            "meaningful_frontier_improvement": attempt.get(
                "meaningful_frontier_improvement"
            ),
            "program_id": attempt.get("program_id"),
            "parent_id": attempt.get("parent_id"),
            "target_island": attempt.get("target_island"),
            "exact_duplicate": attempt.get("exact_duplicate"),
            "duplicate_of": attempt.get("duplicate_of"),
            "no_op": attempt.get("no_op"),
            "changes_excerpt": changes[:800],
            "trajectory_location": (
                f"{SEARCH_HISTORY_ROOT}/seq_{seq:08d}/trajectory.jsonl"
            ),
        }
        if attempt.get("code_bearing") is True:
            record["code_location"] = f"{SEARCH_HISTORY_ROOT}/seq_{seq:08d}/code.py"
        index_rows.append(record)
    files[SEARCH_HISTORY_INDEX] = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        for row in index_rows
    )
    return files


def _full_history_dashboard(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute the complete causal dashboard through the frozen cutoff."""
    attempts = sorted(
        history["attempt_by_seq"].values(), key=lambda row: int(row["seq"])
    )
    root_code = str(history["root_code"])
    root_hash = _code_sha256(root_code)
    program_codes: dict[str, str] = {"root": root_code}
    for attempt in attempts:
        code = attempt.get("code")
        program_id = attempt.get("program_id")
        if (
            attempt.get("committed") is True
            and isinstance(program_id, str)
            and program_id
            and isinstance(code, str)
        ):
            program_codes[program_id] = code
        parent_id = attempt.get("parent_id")
        for event in attempt.get("source_events") or ():
            if (
                event.get("event") == "attempt_scheduled"
                and event.get("parent_code_sha256") == root_hash
                and isinstance(parent_id, str)
                and parent_id
            ):
                program_codes[parent_id] = root_code

    attempt_by_program = {
        str(attempt["program_id"]): attempt
        for attempt in attempts
        if isinstance(attempt.get("program_id"), str) and attempt.get("program_id")
    }
    evolution_pairs: dict[str, tuple[str, str]] = {}
    evolution_path_value = history.get("evolution_trace_path")
    evolution_path = Path(str(evolution_path_value)) if evolution_path_value else None
    if evolution_path is not None and evolution_path.is_file():
        with evolution_path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                child_id = str(event.get("child_id") or "")
                attempt = attempt_by_program.get(child_id)
                if attempt is None:
                    continue
                parent_code = event.get("parent_code")
                child_code = event.get("child_code")
                if not isinstance(parent_code, str) or not isinstance(child_code, str):
                    continue
                expected_parent_id = str(attempt.get("parent_id") or "")
                journal_parent_id = str(event.get("parent_id") or "")
                if journal_parent_id and journal_parent_id != expected_parent_id:
                    raise base.ExperimentError(
                        f"evolution parent mismatch for {child_id}: "
                        f"{journal_parent_id} != {expected_parent_id}"
                    )
                known_parent_code = program_codes.get(expected_parent_id)
                if known_parent_code is not None and _code_sha256(
                    parent_code
                ) != _code_sha256(known_parent_code):
                    raise base.ExperimentError(
                        f"evolution parent code mismatch for {child_id}"
                    )
                expected_hash = attempt.get("code_sha256")
                if expected_hash and _code_sha256(child_code) != expected_hash:
                    raise base.ExperimentError(
                        f"evolution child code mismatch for {child_id}"
                    )
                pair = (parent_code, child_code)
                previous_pair = evolution_pairs.get(child_id)
                if previous_pair is not None and previous_pair != pair:
                    raise base.ExperimentError(
                        f"conflicting evolution rows for {child_id}"
                    )
                evolution_pairs[child_id] = pair

    state_fingerprints = []
    state_codes: list[str] = []
    transition_fingerprints = []
    frontier_cycles = float(history["root_cycles"])
    progress_anchor_cycles = float(history["root_cycles"])
    meaningful_frontier_code = root_code
    meaningful_frontier_program_id = "root"
    meaningful_frontier_seq: int | None = None
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        seq = int(attempt["seq"])
        code = attempt.get("code")
        cycles = base.finite(attempt.get("cycles"))
        parent_cycles = base.finite(attempt.get("parent_cycles"))
        parent_id = attempt.get("parent_id")
        eligible = bool(
            attempt.get("terminal_class") == "valid"
            and attempt.get("committed") is True
            and cycles is not None
            and isinstance(code, str)
            and isinstance(parent_id, str)
            and parent_id
        )
        frontier_move = bool(
            eligible and cycles is not None and cycles < frontier_cycles
        )
        if frontier_move:
            frontier_cycles = float(cycles)
        meaningful = attempt.get("meaningful_frontier_improvement") is True
        if meaningful and not frontier_move:
            raise base.ExperimentError(
                f"meaningful progress at completion {seq} is not a frontier move"
            )
        relative_gain = None
        if frontier_move:
            gain = (progress_anchor_cycles - frontier_cycles) / abs(
                progress_anchor_cycles
            )
            if meaningful:
                relative_gain = gain
                progress_anchor_cycles = frontier_cycles
                meaningful_frontier_code = str(code)
                meaningful_frontier_program_id = str(attempt.get("program_id") or "")
                meaningful_frontier_seq = seq

        state_spread = None
        transition_spread = None
        frontier_spread = None
        if eligible:
            child_code = str(code)
            state_codes.append(child_code)
            state_fingerprints.append(signed_fingerprint(root_code, child_code))
            if len(state_fingerprints) >= DASHBOARD_STATE_WINDOW:
                state_spread = float(
                    implementation_spread(state_fingerprints[-DASHBOARD_STATE_WINDOW:])
                )
                frontier_spread = float(
                    implementation_spread(
                        [
                            signed_fingerprint(meaningful_frontier_code, item)
                            for item in state_codes[-DASHBOARD_STATE_WINDOW:]
                        ]
                    )
                )

            program_id = str(attempt.get("program_id") or "")
            if evolution_path is not None:
                pair = evolution_pairs.get(program_id)
                parent_code = pair[0] if pair is not None else None
            else:
                parent_code = program_codes.get(str(parent_id))
            if parent_code is not None:
                transition_fingerprints.append(
                    signed_fingerprint(parent_code, child_code)
                )
                if len(transition_fingerprints) >= DASHBOARD_TRANSITION_WARMUP:
                    transition_spread = float(
                        implementation_spread(
                            transition_fingerprints[-DASHBOARD_STATE_WINDOW:]
                        )
                    )

        source_events = attempt.get("source_events") or ()
        schedule = next(
            (
                event
                for event in source_events
                if event.get("event") == "attempt_scheduled"
            ),
            {},
        )
        row = {
            "evidence_id": attempt["evidence_id"],
            "seq": seq,
            "elapsed_hours": float(attempt.get("elapsed_minutes") or 0.0) / 60.0,
            "iteration": schedule.get("iteration"),
            "terminal_class": attempt.get("terminal_class"),
            "valid": attempt.get("terminal_class") == "valid",
            "committed": attempt.get("committed") is True,
            "code_bearing": attempt.get("code_bearing") is True,
            "eligible": eligible,
            "program_id": attempt.get("program_id"),
            "parent_id": parent_id,
            "target_island": attempt.get("target_island"),
            "cycles": cycles,
            "parent_cycles": parent_cycles,
            "parent_gain_cycles": (
                parent_cycles - cycles
                if parent_cycles is not None and cycles is not None
                else None
            ),
            "frontier_before_cycles": attempt.get("frontier_before_cycles"),
            "gap_to_frontier_cycles": attempt.get("gap_to_frontier_cycles"),
            "verified_frontier_cycles": frontier_cycles,
            "frontier_move": frontier_move,
            "meaningful_frontier_improvement": attempt.get(
                "meaningful_frontier_improvement"
            ),
            "relative_gain": relative_gain,
            "F_reference_program_id": meaningful_frontier_program_id,
            "F_reference_seq": meaningful_frontier_seq,
            "exact_duplicate": attempt.get("exact_duplicate"),
            "duplicate_of": attempt.get("duplicate_of"),
            "no_op": attempt.get("no_op"),
            "changes_excerpt": re.sub(
                r"\s+", " ", str(attempt.get("changes") or "")
            ).strip()[:800],
            "S": state_spread,
            "T": transition_spread,
            "F": frontier_spread,
            "trajectory_location": (
                f"{SEARCH_HISTORY_ROOT}/seq_{seq:08d}/trajectory.jsonl"
            ),
        }
        if attempt.get("code_bearing") is True:
            row["code_location"] = f"{SEARCH_HISTORY_ROOT}/seq_{seq:08d}/code.py"
        rows.append(row)
    return rows


def _serialize_dashboard(rows: list[dict[str, Any]]) -> str:
    """Keep a valid JSON list with one compact, pageable observation per line."""
    return (
        "[\n"
        + ",\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
            for row in rows
        )
        + "\n]\n"
    )


def _validate_unfold_requirements(history: dict[str, Any]) -> None:
    attempts = list(history["recent_attempts"])
    required_ids = {row["evidence_id"] for row in history["required_unfolds"]}
    available_adaptive = len(attempts) - len(required_ids)
    if ADAPTIVE_UNFOLD_MIN > available_adaptive:
        raise base.ExperimentError(
            "unfold configuration is impossible for this candidate: "
            f"{ADAPTIVE_UNFOLD_MIN} adaptive attempts are required but only "
            f"{available_adaptive} remain after the required unfolds"
        )

    code_ids = {
        row["evidence_id"] for row in attempts if row.get("code_bearing") is True
    }
    required_code_reads = len(required_ids & code_ids)
    minimum_code_reads = max(
        required_code_reads,
        min(MIN_CODE_UNFOLDS, len(code_ids)),
    )
    minimum_tool_calls = len(attempts) + minimum_code_reads
    tool_limit = base.TOOL_CALL_LIMIT
    if minimum_tool_calls > tool_limit:
        raise base.ExperimentError(
            "unfold configuration exceeds the audit tool budget before any "
            f"optional reads: {minimum_tool_calls} required calls > {tool_limit}"
        )
