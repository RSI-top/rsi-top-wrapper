"""Build the causal M25 history and simulator profiles for planning."""

from __future__ import annotations

from release_paths import release_resource

import hashlib

import json

import math

import os

import re

import subprocess

import sys

import tempfile

import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Any, Mapping

import search_history

import history_utils as audit_base

from metric_scaled_dashboard import (
    build_metric_observations,
    compact_observation,
    serialize_llm_dashboard_jsonl,
)

from window_health_metrics import compute_window_health_metrics

ROUTING_DASHBOARD = f"{search_history.SEARCH_HISTORY_ROOT}/dashboard_routing.json"

ROUTING_SUMMARY = f"{search_history.SEARCH_HISTORY_ROOT}/dashboard_routing_summary.json"

WINDOW_HEALTH_METRICS = f"{search_history.SEARCH_HISTORY_ROOT}/window_health_metrics.json"

METRIC_DEFINITIONS = f"{search_history.SEARCH_HISTORY_ROOT}/metric_definitions.json"

LLM_DASHBOARD = f"{search_history.SEARCH_HISTORY_ROOT}/dashboard_llm.jsonl"

METRIC_DEFINITIONS_SOURCE = release_resource("metrics.json")

ENGINE_PROFILES = f"{search_history.SEARCH_HISTORY_ROOT}/engine_profiles.json"

ROUTING_DASHBOARD_SCHEMA = "oe_intervention_dashboard_routing_v1"

ROUTING_SUMMARY_SCHEMA = "oe_intervention_dashboard_routing_summary_v1"

ENGINE_PROFILES_SCHEMA = "oe_intervention_engine_profiles_v2"

ENGINE_SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
}


class PlannerContextError(RuntimeError):
    """Causal history cannot be materialized exactly."""


def _read_trace_events(attempts_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with attempts_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # torn tail line of the live trace: never guess
            if isinstance(event, dict):
                events.append(event)
    return events


def _db_state_program_ids(event: Mapping[str, Any]) -> frozenset[str] | None:
    state = event.get("state")
    if (
        event.get("event") != "db_state"
        or event.get("db_state_schema") != "oe_db_state_v1"
        or not isinstance(state, dict)
        or state.get("schema") != "oe_db_state_v1"
        or not isinstance(state.get("program_ids"), list)
    ):
        return None
    values = state["program_ids"]
    if any(not isinstance(value, str) or not value for value in values):
        return None
    expected = event.get("state_sha256")
    canonical = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not isinstance(expected, str) or expected != actual:
        return None
    return frozenset(values)


def live_program_ids_at_cutoff(
    events: list[dict[str, Any]], cutoff_seq: int
) -> frozenset[str] | None:
    """Return exact native database membership at one completion cutoff.

    New traces record a full ``oe_db_state_v1`` snapshot immediately after
    every successful ``database.add``.  A committing cutoff must have its own
    correlated snapshot; a non-committing cutoff inherits the last snapshot
    that precedes it.  Missing or malformed capture data stays unknown rather
    than being approximated from a neighbouring checkpoint.
    """

    target_index = next(
        (
            index
            for index, event in enumerate(events)
            if event.get("event") == "canonical_completion"
            and event.get("completion_seq") == int(cutoff_seq)
        ),
        None,
    )
    if target_index is None:
        return None

    target = events[target_index]
    committed = False
    for event in events[target_index + 1 :]:
        if event.get("event") == "canonical_completion":
            break
        if event.get("event") == "program_committed" and event.get(
            "completion_seq"
        ) == int(cutoff_seq):
            committed = True
        if (
            event.get("completion_seq") == int(cutoff_seq)
            and event.get("capture") == "program_committed"
        ):
            return _db_state_program_ids(event)
    if committed:
        return None

    for event in reversed(events[:target_index]):
        live = _db_state_program_ids(event)
        if live is not None:
            return live
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _rank_within_frontier_epoch(
    rows: list[dict[str, Any]],
    *,
    value_field: str,
    rank_field: str,
    descending: bool,
) -> None:
    """Attach deterministic ordinal ranks without making a quality score."""
    epochs: dict[tuple[Any, Any], list[tuple[float, int, dict[str, Any]]]] = {}
    for row in rows:
        value = _finite_number(row.get(value_field))
        if value is None or row.get("eligible") is not True:
            row[rank_field] = None
            continue
        epoch = (row.get("F_reference_seq"), row.get("F_reference_program_id"))
        epochs.setdefault(epoch, []).append((value, int(row["seq"]), row))
    for entries in epochs.values():
        entries.sort(key=lambda item: ((-item[0] if descending else item[0]), item[1]))
        for rank, (_, _, row) in enumerate(entries, 1):
            row[rank_field] = rank


def build_dashboard_routing(
    dashboard_rows: list[dict[str, Any]],
    *,
    history: dict[str, Any],
    live_program_ids: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Decorate the causal dashboard with queryable, non-decisional signals.

    The base v4 dashboard remains byte-identical for the locked stage-1 audit.
    This v2-only view adds local deltas/ranks, rolling-window membership,
    branch survival/descendant outcomes, and simple source-diff sizes.  None
    of these fields selects an unfold or asserts semantic compatibility.
    """
    rows = [dict(row) for row in dashboard_rows]
    live = set(live_program_ids) if live_program_ids is not None else None
    attempts = history.get("attempt_by_seq") or {}
    program_codes: dict[str, str] = {"root": str(history["root_code"])}
    for attempt in attempts.values():
        program_id = attempt.get("program_id")
        code = attempt.get("code")
        if isinstance(program_id, str) and program_id and isinstance(code, str):
            program_codes[program_id] = code

    eligible_window: list[str] = []
    previous: dict[tuple[Any, Any, str], float] = {}
    for row in rows:
        row["routing_schema"] = ROUTING_DASHBOARD_SCHEMA
        program_id = row.get("program_id")
        row["live_at_cutoff"] = (
            bool(isinstance(program_id, str) and program_id in live)
            if live is not None
            else None
        )
        if row.get("eligible") is True:
            eligible_window.append(str(row["evidence_id"]))
        window = eligible_window[-search_history.DASHBOARD_STATE_WINDOW :]
        row["rolling_window_evidence_ids"] = list(window)
        row["rolling_window_start_seq"] = (
            int(window[0].rsplit(":", 1)[1]) if window else None
        )
        epoch = (row.get("F_reference_seq"), row.get("F_reference_program_id"))
        for axis in ("S", "F", "T"):
            value = _finite_number(row.get(axis))
            key = (*epoch, axis)
            prior = previous.get(key)
            row[f"delta_{axis}"] = (
                value - prior if value is not None and prior is not None else None
            )
            row[f"abs_delta_{axis}"] = (
                abs(row[f"delta_{axis}"]) if row[f"delta_{axis}"] is not None else None
            )
            if value is not None:
                previous[key] = value

        attempt = attempts.get(int(row["seq"])) or {}
        source_events = attempt.get("source_events") or ()
        schedule = next(
            (
                event
                for event in source_events
                if event.get("event") == "attempt_scheduled"
            ),
            {},
        )
        inspiration_ids = [
            str(item)
            for item in (schedule.get("inspiration_ids") or [])
            if isinstance(item, str) and item
        ]
        row["inspiration_ids"] = inspiration_ids
        row["inspiration_count"] = len(inspiration_ids)
        row["frontier_reference_was_inspiration"] = (
            row.get("F_reference_program_id") in inspiration_ids
        )
        row["parent_code_sha256"] = schedule.get("parent_code_sha256")
        row["child_code_sha256"] = attempt.get("code_sha256")

        # The canonical-completion row stores the worker's change summary
        # under child.metadata, while the base v4 fold looked only at the
        # event's top-level metadata. Recover that already-causal text for the
        # Stage-2 routing view without changing Stage 1 or synthesizing a
        # semantic label.
        changes = str(attempt.get("changes") or "").strip()
        if not changes or changes == "No recorded change summary.":
            for event in reversed(source_events):
                metadata = event.get("metadata")
                child = event.get("child")
                child_metadata = (
                    child.get("metadata") if isinstance(child, dict) else None
                )
                candidate = (
                    metadata.get("changes") if isinstance(metadata, dict) else None
                ) or (
                    child_metadata.get("changes")
                    if isinstance(child_metadata, dict)
                    else None
                )
                if isinstance(candidate, str) and candidate.strip():
                    changes = candidate.strip()
                    break
        row["changes_excerpt"] = re.sub(r"\s+", " ", changes).strip()[:800]
        recorded_change_count = len(re.findall(r"(?m)\bChange\s+\d+\s*:", changes))
        row["recorded_change_count"] = recorded_change_count or None
        row["multi_edit_transition"] = recorded_change_count > 1

        child_code = attempt.get("code")
        parent_code = program_codes.get(str(row.get("parent_id") or ""))
        frontier_code = program_codes.get(
            str(row.get("F_reference_program_id") or "root")
        )
        row["transition_changed_lines"] = (
            sum(search_history.signed_fingerprint(parent_code, child_code).values())
            if isinstance(parent_code, str) and isinstance(child_code, str)
            else None
        )
        row["frontier_changed_lines"] = (
            sum(search_history.signed_fingerprint(frontier_code, child_code).values())
            if isinstance(frontier_code, str) and isinstance(child_code, str)
            else None
        )

    # Historical lineage outcomes through this exact cutoff.  They help route
    # attention to branches that seeded later yield, but remain observational.
    row_by_program = {
        str(row["program_id"]): row
        for row in rows
        if isinstance(row.get("program_id"), str) and row.get("program_id")
    }
    children: dict[str, list[str]] = {}
    for program_id, row in row_by_program.items():
        parent_id = row.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            children.setdefault(parent_id, []).append(program_id)

    def descendants(program_id: str) -> set[str]:
        found: set[str] = set()
        stack = list(children.get(program_id, ()))
        while stack:
            child = stack.pop()
            if child in found:
                continue
            found.add(child)
            stack.extend(children.get(child, ()))
        return found

    for program_id, row in row_by_program.items():
        descendant_ids = descendants(program_id)
        descendant_rows = [
            row_by_program[item] for item in descendant_ids if item in row_by_program
        ]
        valid_descendants = [
            item for item in descendant_rows if item.get("eligible") is True
        ]
        descendant_cycles = [
            value
            for item in valid_descendants
            for value in [_finite_number(item.get("cycles"))]
            if value is not None
        ]
        row["valid_descendant_count"] = len(valid_descendants)
        row["best_descendant_cycles"] = (
            min(descendant_cycles) if descendant_cycles else None
        )
        row["descendant_frontier_move_count"] = sum(
            item.get("frontier_move") is True for item in descendant_rows
        )
        row["has_live_descendant"] = (
            any(item in live for item in descendant_ids) if live is not None else None
        )

    for value_field, rank_field, descending in (
        ("parent_gain_cycles", "epoch_parent_gain_rank", True),
        ("gap_to_frontier_cycles", "epoch_frontier_gap_rank", False),
        ("S", "epoch_S_high_rank", True),
        ("S", "epoch_S_low_rank", False),
        ("F", "epoch_F_high_rank", True),
        ("F", "epoch_F_low_rank", False),
        ("T", "epoch_T_high_rank", True),
        ("T", "epoch_T_low_rank", False),
        ("abs_delta_S", "epoch_abs_delta_S_rank", True),
        ("abs_delta_F", "epoch_abs_delta_F_rank", True),
        ("abs_delta_T", "epoch_abs_delta_T_rank", True),
    ):
        _rank_within_frontier_epoch(
            rows,
            value_field=value_field,
            rank_field=rank_field,
            descending=descending,
        )
    return rows


def build_dashboard_routing_summary(
    routing_rows: list[dict[str, Any]], *, limit: int = 12
) -> dict[str, Any]:
    """Compact, query-ready views of independent routing signals.

    Categories are deliberately separate: no combined score, inferred recipe,
    or automatic branch choice is produced.  The judge must inspect overlap,
    then unfold trajectory and code before drawing a semantic conclusion.
    """
    eligible = [row for row in routing_rows if row.get("eligible") is True]
    latest = eligible[-1] if eligible else None
    current_epoch = (
        [
            row
            for row in eligible
            if (
                row.get("F_reference_seq"),
                row.get("F_reference_program_id"),
            )
            == (
                latest.get("F_reference_seq"),
                latest.get("F_reference_program_id"),
            )
        ]
        if latest is not None
        else []
    )

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row.get(key)
            for key in (
                "evidence_id",
                "seq",
                "program_id",
                "parent_id",
                "cycles",
                "parent_cycles",
                "parent_gain_cycles",
                "gap_to_frontier_cycles",
                "frontier_move",
                "live_at_cutoff",
                "valid_descendant_count",
                "best_descendant_cycles",
                "descendant_frontier_move_count",
                "has_live_descendant",
                "S",
                "F",
                "T",
                "delta_S",
                "delta_F",
                "delta_T",
                "transition_changed_lines",
                "frontier_changed_lines",
                "inspiration_ids",
                "inspiration_count",
                "frontier_reference_was_inspiration",
                "parent_code_sha256",
                "child_code_sha256",
                "recorded_change_count",
                "multi_edit_transition",
                "terminal_class",
                "failure_stage",
                "exact_duplicate",
                "duplicate_of",
                "no_op",
                "changes_excerpt",
                "trajectory_location",
                "code_location",
            )
        }

    def top(
        rows: list[dict[str, Any]],
        field: str,
        *,
        descending: bool,
        live_only: bool = False,
    ) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in rows
            if _finite_number(row.get(field)) is not None
            and (not live_only or row.get("live_at_cutoff") is True)
        ]
        candidates.sort(
            key=lambda row: (
                -float(row[field]) if descending else float(row[field]),
                int(row["seq"]),
            )
        )
        return candidates[:limit]

    frontier_advances = [
        row for row in routing_rows if row.get("frontier_move") is True
    ][-limit:]
    failures_or_regressions = [
        row
        for row in routing_rows
        if (
            row.get("terminal_class") in {"invalid", "worker_error"}
            or (
                _finite_number(row.get("parent_gain_cycles")) is not None
                and float(row["parent_gain_cycles"]) <= 0
            )
        )
    ][-limit:]
    shortlist_rows = {
        "recent_verified_frontier_advances": frontier_advances,
        "current_epoch_live_near_frontier": top(
            current_epoch,
            "gap_to_frontier_cycles",
            descending=False,
            live_only=True,
        ),
        "current_epoch_live_parent_gains": top(
            current_epoch,
            "parent_gain_cycles",
            descending=True,
            live_only=True,
        ),
        "current_epoch_high_S_windows": top(current_epoch, "S", descending=True),
        "current_epoch_low_S_windows": top(current_epoch, "S", descending=False),
        "current_epoch_high_F_windows": top(current_epoch, "F", descending=True),
        "current_epoch_low_F_windows": top(current_epoch, "F", descending=False),
        "current_epoch_high_T_windows": top(current_epoch, "T", descending=True),
        "current_epoch_low_T_windows": top(current_epoch, "T", descending=False),
        "current_epoch_large_delta_S_windows": top(
            current_epoch, "abs_delta_S", descending=True
        ),
        "current_epoch_large_delta_F_windows": top(
            current_epoch, "abs_delta_F", descending=True
        ),
        "current_epoch_large_delta_T_windows": top(
            current_epoch, "abs_delta_T", descending=True
        ),
        "recent_failures_or_regressions": failures_or_regressions,
    }
    unique_rows: dict[str, dict[str, Any]] = {}
    for category in shortlist_rows.values():
        for row in category:
            unique_rows.setdefault(str(row["evidence_id"]), row)
    profiles: dict[str, list[str]] = {}
    for name, category in shortlist_rows.items():
        for row in category:
            profiles.setdefault(str(row["evidence_id"]), []).append(name)
    return {
        "schema": ROUTING_SUMMARY_SCHEMA,
        "semantics": (
            "Independent deterministic shortlists only; overlap routes "
            "attention but is not a quality score or compatibility claim."
        ),
        "current_frontier_epoch": {
            "F_reference_seq": latest.get("F_reference_seq") if latest else None,
            "F_reference_program_id": (
                latest.get("F_reference_program_id") if latest else None
            ),
            "eligible_row_count": len(current_epoch),
        },
        "shortlists": {
            name: [str(row["evidence_id"]) for row in category]
            for name, category in shortlist_rows.items()
        },
        "profiles": profiles,
        "records": {
            evidence_id: compact(row) for evidence_id, row in unique_rows.items()
        },
        "rolling_windows": {
            evidence_id: list(row.get("rolling_window_evidence_ids") or [])
            for evidence_id, row in unique_rows.items()
        },
    }


def build_stage_context(
    *,
    attempts_path: str | os.PathLike,
    cutoff_seq: int,
    cutoff_at: float,
    root_code: str,
    rolled_back_seqs: frozenset[int] | set[int] = frozenset(),
    live_program_ids: frozenset[str] | set[str] | None = None,
    preloaded_events: list[dict[str, Any]] | None = None,
    engine_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    engine_failures: Mapping[str, str] | None = None,
    include_metric_dashboard: bool = True,
) -> dict[str, Any]:
    """Freeze the v4 DIVERSITY-arm audit context at one causal cutoff.

    The causal prefix is the live attempts trace with (a) every event beyond
    the cutoff ignored by the shared fold itself and (b) completions the
    manager classified as rolled back on resume excluded — the same branch
    the live database represents. Everything downstream is the UNCHANGED v4
    materialization: the shared history fold, the canonical dashboard fold
    ``_full_history_dashboard`` (root-relative S, parent-relative T,
    frozen-meaningful-frontier-relative F with its per-row reference
    program/seq; only valid committed code-bearing states enter the
    geometry, failures/invalid stay raw trajectory evidence), the
    PAIRED-FROZEN ``paired_v4_state_context`` W12 window over those rows,
    and ``_state_dynamics`` for the rendered State Dynamics block. The
    search-history index is replaced by ``search_history/dashboard.json``
    exactly as the v4 diversity arm does. ``cutoff_at`` must be the
    canonical timestamp of the exact cutoff completion. Any missing
    precondition raises (fail closed): this builder never approximates.
    """
    path = Path(attempts_path)
    if not path.is_file():
        raise PlannerContextError(f"attempts trace unavailable: {path}")
    excluded = {int(seq) for seq in rolled_back_seqs}
    events: list[dict[str, Any]] = []
    started: dict[str, Any] | None = None
    source_events = (
        preloaded_events if preloaded_events is not None else _read_trace_events(path)
    )
    if live_program_ids is None:
        live_program_ids = live_program_ids_at_cutoff(source_events, cutoff_seq)
    for event in source_events:
        name = event.get("event")
        if name == "run_started":
            at = event.get("timestamp")
            if isinstance(at, (int, float)) and (
                started is None or float(at) < float(started["timestamp"])
            ):
                started = event
            continue
        seq = event.get("completion_seq")
        if isinstance(seq, int) and not isinstance(seq, bool) and seq in excluded:
            continue  # not part of the current live branch
        events.append(event)
    if started is None:
        raise PlannerContextError(
            "attempts trace has no run_started event — the v4 context cannot "
            "be materialized exactly"
        )
    root_cycles = next(
        (
            value
            for event in events
            if event.get("event") == "attempt_scheduled"
            for value in [search_history._cycles(event.get("parent_metrics"))]
            if value is not None
        ),
        None,
    )
    if root_cycles is None:
        raise PlannerContextError(
            "attempts trace has no finite root cycles — the v4 context "
            "cannot be materialized exactly"
        )
    history = search_history._build_history_from_events(
        events=events,
        root_code=root_code,
        root_cycles=float(root_cycles),
        started_at=float(started["timestamp"]),
        cutoff_seq=int(cutoff_seq),
        cutoff_at=float(cutoff_at),
    )
    # The canonical dashboard fold verifies parent/child code identity
    # against the native evolution trace when it exists beside the attempts
    # trace (post-cutoff children are outside the fold and ignored).
    evolution_trace = path.parent / "evolution_trace.jsonl"
    history["evolution_trace_path"] = (
        str(evolution_trace) if evolution_trace.is_file() else None
    )
    search_history._validate_unfold_requirements(history)
    global_text, metadata = search_history._render_global_history(
        candidate={"completion_seq": int(cutoff_seq)}, history=history
    )
    files = {
        **search_history._intervention_history_files(history),
        **search_history._attempt_files(
            history["recent_attempts"], root_name=search_history.RECENT_ATTEMPT_ROOT
        ),
    }
    # Diversity arm, exactly as v4: the complete causal S/T/F dashboard
    # (with each row's frozen F reference program/seq) replaces the
    # search-history index, and the State Dynamics block is the offline
    # ``state_context`` W12/epoch selection over the SAME dashboard rows.
    dashboard_rows = search_history._full_history_dashboard(history)
    renderer_rows = [
        {
            "seq": int(row["seq"]),
            "node": row.get("program_id"),
            "s": row.get("S"),
            "fs": row.get("F"),
            "fReferenceProgram": row.get("F_reference_program_id"),
            "fReferenceSeq": row.get("F_reference_seq"),
        }
        for row in dashboard_rows
    ]
    state_ctx = paired_v4_state_context(renderer_rows, int(cutoff_seq))
    dynamics_text, dynamics_doc = search_history._state_dynamics(
        {"state_context": state_ctx}, metadata["attempt_completion_seqs"]
    )
    files.pop(search_history.SEARCH_HISTORY_INDEX, None)
    files[search_history.SEARCH_HISTORY_DASHBOARD] = search_history._serialize_dashboard(dashboard_rows)
    routing_rows = build_dashboard_routing(
        dashboard_rows,
        history=history,
        live_program_ids=live_program_ids,
    )
    window_health_metrics = compute_window_health_metrics(routing_rows)
    routing_summary = build_dashboard_routing_summary(routing_rows)
    routing_summary["window_health_metrics"] = window_health_metrics
    files[ROUTING_DASHBOARD] = search_history._serialize_dashboard(routing_rows)
    files[WINDOW_HEALTH_METRICS] = (
        json.dumps(
            window_health_metrics,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    files[ROUTING_SUMMARY] = (
        json.dumps(
            routing_summary,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if include_metric_dashboard:
        metric_header, metric_observations = build_metric_observations(
            routing_rows,
            history=history,
            engine_profiles=engine_profiles,
            engine_failures=engine_failures,
            live_program_ids=live_program_ids,
            definitions_path=METRIC_DEFINITIONS_SOURCE,
        )
        files[METRIC_DEFINITIONS] = METRIC_DEFINITIONS_SOURCE.read_text(
            encoding="utf-8"
        )
        files[LLM_DASHBOARD] = serialize_llm_dashboard_jsonl(
            metric_header,
            [
                compact_observation(metric_header, observation)
                for observation in metric_observations
            ],
        )
    return {
        "cutoff_seq": int(cutoff_seq),
        "arm": search_history.ARM_DIVERSITY,
        "global_history_context": global_text,
        "performance_timeline": metadata["performance_timeline"],
        "population_yield": metadata["population_yield"],
        "attempt_evidence_ids": list(metadata["attempt_evidence_ids"]),
        "required_unfold_ids": list(metadata["required_unfold_ids"]),
        "available_code_ids": sorted(
            row["evidence_id"]
            for row in metadata["records"]
            if row.get("code_location")
        ),
        "state_dynamics_context": dynamics_text,
        "state_dynamics": dynamics_doc,
        # The FULL causal dashboard attempt namespace (attempt:seq:* for
        # every terminal attempt at or before the cutoff): the ONLY legal
        # ids for stage-2 lane evidence_attempt_ids, plus the code-bearing
        # subset used to enforce stage-2 code unfolds.
        "dashboard_attempt_evidence_ids": [
            row["evidence_id"] for row in dashboard_rows
        ],
        "dashboard_live_membership_known": live_program_ids is not None,
        "dashboard_live_attempt_ids": sorted(
            row["evidence_id"]
            for row in routing_rows
            if row.get("live_at_cutoff") is True
        ),
        "dashboard_code_attempt_ids": sorted(
            row["evidence_id"] for row in dashboard_rows if row.get("code_location")
        ),
        "dashboard_child_code_sha256_by_evidence": {
            row["evidence_id"]: row["child_code_sha256"]
            for row in routing_rows
            if row.get("code_location")
            and isinstance(row.get("child_code_sha256"), str)
            and row["child_code_sha256"]
        },
        "files": files,
    }


STATE_DYNAMICS_SEMANTICS = "paired_v4_latest_full_w12_unfiltered"


def paired_v4_state_context(
    rows: list[dict[str, Any]], cutoff_seq: int
) -> dict[str, Any]:
    """The paired-frozen state_context over renderer-shaped dashboard rows.

    Carries exactly the fields the State Dynamics renderer consumes (window
    seqs + S/F value series; deliberately no frontier_reference). The old
    function's trend/percentile summaries never reached the stage-1 prompt
    and are not reproduced.
    """
    available = [
        row
        for row in rows
        if int(row["seq"]) <= int(cutoff_seq)
        and audit_base.finite(row.get("s")) is not None
        and audit_base.finite(row.get("fs")) is not None
    ]
    recent = available[-audit_base.STATE_WINDOW :]
    return {
        "semantics": STATE_DYNAMICS_SEMANTICS,
        "causal_selection": (
            "latest full-W12 observations at or before candidate cutoff"
        ),
        "window_completion_seqs": [int(row["seq"]) for row in recent],
        "root_relative_state_spread": {
            "values_oldest_to_newest": [float(row["s"]) for row in recent]
        },
        "frontier_relative_state_spread": {
            "values_oldest_to_newest": [float(row["fs"]) for row in recent]
        },
    }


def _instruction_profile(instrs: Any, *, scratch_words: Any = None) -> dict[str, Any]:
    """Mechanical emitted-stream resource counts; never a bottleneck label."""
    if not isinstance(instrs, list) or not all(
        isinstance(bundle, dict) for bundle in instrs
    ):
        raise ValueError("built instruction stream is not a list of bundles")
    op_counts = {
        engine: sum(
            len(bundle.get(engine) or ())
            for bundle in instrs
            if isinstance(bundle.get(engine) or (), list)
        )
        for engine in ENGINE_SLOT_LIMITS
    }
    occupied = {
        engine: sum(bool(bundle.get(engine)) for bundle in instrs)
        for engine in ENGINE_SLOT_LIMITS
    }
    lower_bounds = {
        engine: math.ceil(op_counts[engine] / limit)
        for engine, limit in ENGINE_SLOT_LIMITS.items()
    }
    maximum = max(lower_bounds.values(), default=0)
    profile = {
        "emitted_bundle_count": len(instrs),
        "engine_op_counts": op_counts,
        "engine_occupied_bundle_counts": occupied,
        "engine_slot_lower_bounds": lower_bounds,
        "max_engine_slot_lower_bound": maximum,
        "max_engine_slot_lower_bound_engines": sorted(
            engine for engine, value in lower_bounds.items() if value == maximum
        ),
    }
    if type(scratch_words) is int and scratch_words >= 0:
        profile["scratch_words"] = scratch_words
    return profile


def _profile_delta(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Mechanical child-minus-parent profile delta; no bottleneck inference."""
    result: dict[str, Any] = {}
    for field in ("emitted_bundle_count", "scratch_words"):
        before = parent.get(field)
        after = child.get(field)
        if type(before) is int and type(after) is int:
            result[f"delta_{field}"] = after - before
    for field in (
        "engine_op_counts",
        "engine_occupied_bundle_counts",
        "engine_slot_lower_bounds",
    ):
        before = parent.get(field)
        after = child.get(field)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        delta = {
            engine: after[engine] - before[engine]
            for engine in ENGINE_SLOT_LIMITS
            if type(before.get(engine)) is int and type(after.get(engine)) is int
        }
        if delta:
            result[f"delta_{field}"] = delta
    return result


def _profile_one_code(code_path: Path, output_path: Path, cwd: Path) -> dict[str, Any]:
    build = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().with_name("eval_build.py")),
            str(code_path),
            str(Path(__file__).resolve().parent / "vendor" / "takehome"),
            str(output_path),
        ],
        cwd=cwd,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    if build.returncode != 0 or not output_path.is_file():
        raise ValueError(f"instruction build failed (exit {build.returncode})")
    built = json.loads(output_path.read_text())
    return _instruction_profile(
        built["instrs"], scratch_words=built.get("scratch_words")
    )


_ENGINE_PROFILE_CACHE: dict[str, dict[str, Any]] = {}

_ENGINE_PROFILE_FAILURE_CACHE: dict[str, str] = {}

_ENGINE_PROFILE_CACHE_LOCK = threading.Lock()


def profile_source_code(source: str) -> dict[str, Any]:
    """Build one source profile once per process, keyed by exact source hash."""
    code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with _ENGINE_PROFILE_CACHE_LOCK:
        cached = _ENGINE_PROFILE_CACHE.get(code_hash)
        cached_failure = _ENGINE_PROFILE_FAILURE_CACHE.get(code_hash)
    if cached is not None:
        return dict(cached)
    if cached_failure is not None:
        raise ValueError(cached_failure)
    with tempfile.TemporaryDirectory(prefix="oe-engine-profile-") as temporary:
        root = Path(temporary)
        code_path = root / "program.py"
        output_path = root / "built.json"
        code_path.write_text(source, encoding="utf-8")
        try:
            profile = _profile_one_code(code_path, output_path, root)
        except ValueError as exc:
            with _ENGINE_PROFILE_CACHE_LOCK:
                _ENGINE_PROFILE_FAILURE_CACHE.setdefault(code_hash, str(exc))
            raise
    with _ENGINE_PROFILE_CACHE_LOCK:
        cached = _ENGINE_PROFILE_CACHE.setdefault(code_hash, profile)
    return dict(cached)


def _materialize_engine_profiles(view: Path, root: Path) -> None:
    """Profile live/shortlisted code and measured shortlist transitions."""
    catalog = json.loads(
        (view / "workspace/meta/branch_catalog.json").read_text(encoding="utf-8")
    )
    summary = json.loads((view / ROUTING_SUMMARY).read_text(encoding="utf-8"))
    routing_rows = json.loads((view / ROUTING_DASHBOARD).read_text(encoding="utf-8"))
    routing_by_program = {
        str(row["program_id"]): row
        for row in routing_rows
        if isinstance(row.get("program_id"), str)
        and row.get("program_id")
        and isinstance(row.get("code_location"), str)
    }
    routing_by_evidence = {
        str(row["evidence_id"]): row
        for row in routing_rows
        if isinstance(row.get("evidence_id"), str)
    }
    sources: dict[str, set[str]] = {}
    evidence_profiles = summary.get("profiles") or {}
    selected_records = summary.get("records") or {}
    for evidence_id, record in selected_records.items():
        program_id = record.get("program_id")
        if isinstance(program_id, str) and program_id:
            sources.setdefault(program_id, set()).update(
                str(item) for item in evidence_profiles.get(evidence_id, ())
            )
        parent_id = record.get("parent_id")
        if isinstance(parent_id, str) and parent_id:
            sources.setdefault(parent_id, set()).add(f"parent_of:{evidence_id}")
    catalog_by_id = {str(row["id"]): row for row in catalog.get("programs", ())}
    for program_id, row in catalog_by_id.items():
        sources.setdefault(program_id, set()).add("live_catalog")
        if row.get("is_global_best"):
            sources.setdefault(program_id, set()).add("global_best")
        if row.get("is_island_best"):
            sources.setdefault(program_id, set()).add("island_best")

    code_paths: dict[str, Path] = {
        program_id: view / str(row["code_location"])
        for program_id, row in routing_by_program.items()
    }
    expected_hashes: dict[str, str] = {
        program_id: str(row.get("child_code_sha256") or "")
        for program_id, row in routing_by_program.items()
    }
    cycles_by_program: dict[str, Any] = {
        program_id: row.get("cycles") for program_id, row in routing_by_program.items()
    }
    for program_id, row in catalog_by_id.items():
        code_paths[program_id] = view / str(row["code_path"])
        expected_hashes[program_id] = str(row["code_sha256"])
        cycles_by_program[program_id] = (row.get("metrics") or {}).get("cycles")

    by_hash: dict[str, list[str]] = {}
    for program_id in sorted(sources):
        code_path = code_paths.get(program_id)
        if code_path is None or not code_path.is_file():
            continue
        actual_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
        expected_hash = expected_hashes.get(program_id)
        if expected_hash and actual_hash != expected_hash:
            continue
        by_hash.setdefault(actual_hash, []).append(program_id)

    build_root = root / "engine-profile-build"
    build_root.mkdir()
    profile_by_hash: dict[str, dict[str, Any]] = {}
    failure_by_hash: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(by_hash)))) as pool:
        futures = {
            pool.submit(
                profile_source_code,
                code_paths[ids[0]].read_text(encoding="utf-8"),
            ): code_hash
            for code_hash, ids in by_hash.items()
        }
        for future in as_completed(futures):
            code_hash = futures[future]
            try:
                profile_by_hash[code_hash] = future.result()
            except Exception as exc:  # one unbuildable branch must not hide peers
                failure_by_hash[code_hash] = f"{type(exc).__name__}: {exc}"

    profiles: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for code_hash, program_ids in by_hash.items():
        for program_id in program_ids:
            common = {
                "program_id": program_id,
                "code_sha256": code_hash,
                "cycles": cycles_by_program.get(program_id),
                "selection_sources": sorted(sources[program_id]),
            }
            if code_hash in profile_by_hash:
                profiles[program_id] = {**common, **profile_by_hash[code_hash]}
            else:
                failures[program_id] = {
                    **common,
                    "error": failure_by_hash.get(code_hash, "profile unavailable"),
                }
    measured_transitions = []
    for evidence_id in selected_records:
        row = routing_by_evidence.get(str(evidence_id))
        if row is None or row.get("eligible") is not True:
            continue
        child_id = row.get("program_id")
        parent_id = row.get("parent_id")
        if child_id not in profiles or parent_id not in profiles:
            continue
        child_cycles = _finite_number(row.get("cycles"))
        parent_cycles = _finite_number(row.get("parent_cycles"))
        if child_cycles is None or parent_cycles is None:
            continue
        measured_transitions.append(
            {
                "attempt_id": str(evidence_id),
                "parent_program_id": parent_id,
                "child_program_id": child_id,
                "delta_cycles": child_cycles - parent_cycles,
                **_profile_delta(profiles[parent_id], profiles[child_id]),
            }
        )
    document = {
        "schema": ENGINE_PROFILES_SCHEMA,
        "semantics": (
            "Mechanical emitted-instruction counts and exact builder scratch "
            "high-water for buildable live branches and shortlisted historical "
            "programs. Measured transition deltas are child minus parent, so "
            "negative delta_cycles is an improvement. Slot lower bounds ignore "
            "dependencies and packing; all fields route code inspection and "
            "never prove a bottleneck or recipe."
        ),
        "workload": {"forest_height": 10, "batch_size": 256, "rounds": 16},
        "engine_slot_limits": ENGINE_SLOT_LIMITS,
        "profiles": profiles,
        "measured_transitions": measured_transitions,
        "failures": failures,
    }
    target = view / ENGINE_PROFILES
    target.write_text(
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
