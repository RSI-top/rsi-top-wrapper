"""Materialize planner inputs, run Pi sessions, and validate portfolios."""

from __future__ import annotations

import hashlib

import json

import math

import os

import re

import shutil

import signal

import subprocess

import time

from pathlib import Path

from typing import Any

from metric_scaled_dashboard import (
    load_definitions,
    profile_programs,
    validate_dashboard_jsonl,
    write_llm_dashboard_jsonl,
)

from pi_cli import (
    build_pi_argv,
    build_pi_subprocess_env,
    canonical_view_access_paths,
    parse_pi_stream,
    remove_python_bytecode,
)

DIRECTION_KEYS = {
    "target_branch_id",
    "mechanism_hypothesis",
    "task",
    "constraints",
    "donor_branch_ids",
    "evidence_attempt_ids",
}

PROPOSAL_REASONING_KEYS = {
    "type",
    "frontier_case",
    "evidence_strength",
    "novel_axis",
    "falsifier",
}

ENGINE_PROFILES_PATH = "search_history/engine_profiles.json"

ENGINE_PROFILES_SCHEMA = "oe_stage2_engine_profiles_v1"

PRIVATE_ENGINE_PROFILES_ENV = "OE_PRIVATE_ENGINE_PROFILES_PATH"

PRIVATE_ENGINE_PROFILES_PATH = "controller_private/engine_profiles.json"

PRIVATE_HISTORY_ENV = "OE_PRIVATE_HISTORY_ROOT"

PRIVATE_HISTORY_PATH = "controller_private/search_history"

RAW_HISTORY_INDEX = "search_history/attempt_paths.jsonl"

INTERNAL_DASHBOARD_PATHS = (
    "search_history/dashboard.json",
    "search_history/dashboard.jsonl",
    "search_history/dashboard_routing.json",
    "search_history/dashboard_routing_summary.json",
    "search_history/window_health_metrics.json",
)

INTERNAL_HISTORY_DIRECTORIES = ("recent_attempts", "selected_history")

PROFILE_WORKERS = 8

UNFOLD_SELECTION_MODES = ("registered",)

UNFOLD_SELECTION_EXTENSION = Path(__file__).with_name(
    "pi_unfold_selection_extension.ts"
)

PUBLIC_FILES_EXTENSION = Path(__file__).with_name("pi_public_files_extension.ts")

REGISTERED_PI_TOOLS = (
    "read_public_file,list_public_directory,compare_attempts,"
    "register_unfold_selection,read_registered_attempt"
)

REGISTERED_UNFOLD_PROMPT = """## Registered selective unfolding

Use `read_public_file` for the metric definitions, complete dashboard, task
files, branch catalog, and live program code. Then register each
historical attempt before first reading or comparing its trajectory or code.

Use `register_unfold_selection` in batches. For each attempt provide:

- `attempt_id`;
- metric IDs that actually influenced the selection, or an empty list;
- one concise selection reason.

Additional adaptive unfolds require another registration. Registration records
the routing decision. Then use `read_registered_attempt` to read that attempt's
trajectory and, only when needed, its code. Raw history is not exposed through
ordinary filesystem tools. `compare_attempts` remains available for focused
pairwise checks. A historical attempt first encountered only as the other side
of a comparison is recorded automatically as a non-metric-attributed dependency;
do not register it afterward merely to repair the bookkeeping order.
"""


def prompt_for_unfold_selection_mode(
    prompt: str, mode: str, *, require_dashboard: bool = True
) -> str:
    if mode != "registered" or not require_dashboard:
        raise ValueError(
            "the release requires registered unfolding from the M25 dashboard"
        )
    separator = "\n" if prompt.endswith("\n") else "\n\n"
    return prompt + separator + REGISTERED_UNFOLD_PROMPT


def _attempt_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:attempt:seq:)?(\d{1,8})", value)
    if match is None:
        return None
    return f"attempt:seq:{int(match.group(1)):08d}"


def normalize_portfolio_evidence_ids(
    document: Any, evidence_ids: set[str]
) -> list[dict[str, Any]]:
    """Canonicalize unambiguous attempt-ID padding without changing evidence."""
    corrections: list[dict[str, Any]] = []
    if not isinstance(document, dict) or not isinstance(
        document.get("directions"), list
    ):
        return corrections
    for direction_index, direction in enumerate(document["directions"]):
        if not isinstance(direction, dict):
            continue
        cited = direction.get("evidence_attempt_ids")
        if not isinstance(cited, list):
            continue
        for evidence_index, raw in enumerate(cited):
            if not isinstance(raw, str) or not re.fullmatch(
                r"attempt:seq:\d{1,8}", raw
            ):
                continue
            canonical = _attempt_id(raw)
            if canonical is None or canonical == raw or canonical not in evidence_ids:
                continue
            cited[evidence_index] = canonical
            corrections.append(
                {
                    "direction_index": direction_index,
                    "evidence_index": evidence_index,
                    "raw": raw,
                    "canonical": canonical,
                }
            )
    return corrections


def normalize_portfolio_direction_fields(document: Any) -> list[dict[str, Any]]:
    """Move an unambiguously nested constraints list back to its direction."""
    corrections: list[dict[str, Any]] = []
    if not isinstance(document, dict) or not isinstance(
        document.get("directions"), list
    ):
        return corrections
    for direction_index, direction in enumerate(document["directions"]):
        if not isinstance(direction, dict) or "constraints" in direction:
            continue
        frontier_case = direction.get("frontier_case")
        if not isinstance(frontier_case, dict) or not isinstance(
            frontier_case.get("constraints"), list
        ):
            continue
        direction["constraints"] = frontier_case.pop("constraints")
        corrections.append(
            {
                "direction_index": direction_index,
                "field": "constraints",
                "from": "frontier_case.constraints",
                "to": "constraints",
            }
        )
    return corrections


def merge_pi_evidence_metadata(
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge read/tool ledgers from consecutive turns of one Pi session."""
    list_fields = (
        "tool_names",
        "successful_read_paths",
        "direct_read_paths",
        "tool_internal_read_paths",
    )
    event_fields = (
        "read_events",
        "compare_events",
        "unfold_selection_events",
    )
    merged: dict[str, Any] = {field: [] for field in (*list_fields, *event_fields)}
    event_offset = 0
    for call in calls:
        for field in list_fields:
            merged[field].extend(call.get(field) or [])
        local_max = -1
        for field in event_fields:
            for raw_event in call.get(field) or []:
                event = dict(raw_event)
                for key in ("start_event_index", "end_event_index"):
                    if key in event:
                        local = int(event[key])
                        local_max = max(local_max, local)
                        event[key] = local + event_offset
                merged[field].append(event)
        event_offset += max(local_max + 1, 1)
    merged["session_call_count"] = len(calls)
    return merged


def _dashboard_metric_ids(view: Path) -> set[str]:
    path = view / "search_history/dashboard_llm.jsonl"
    with path.open(encoding="utf-8") as handle:
        header = next(
            (json.loads(line) for line in handle if line.strip()),
            None,
        )
    if not isinstance(header, dict) or not isinstance(header.get("metric_order"), list):
        raise ValueError("dashboard header has no metric_order")
    result = {str(metric_id) for metric_id in header["metric_order"]}
    profiles = header.get("mechanistic_profile_order") or []
    if not isinstance(profiles, list):
        raise ValueError("dashboard header has an invalid mechanistic_profile_order")
    result.update(str(profile_id) for profile_id in profiles)
    return result


def _validate_registered_unfolds(
    *,
    view: Path,
    read_events: list[dict[str, Any]],
    compare_events: list[dict[str, Any]],
    unfold_selection_events: list[dict[str, Any]],
    require_dashboard: bool = True,
) -> dict[str, Any]:
    metric_ids = _dashboard_metric_ids(view) if require_dashboard else set()
    prerequisite_ends: dict[str, list[int]] = {}
    historical_read_events: list[tuple[str, int]] = []
    for event in read_events:
        path = canonical_view_access_paths([str(event.get("path") or "")], view).pop()
        end_index = int(event.get("end_event_index", -1))
        if path in {
            "search_history/metric_definitions.json",
            "search_history/dashboard_llm.jsonl",
        }:
            prerequisite_ends.setdefault(path, []).append(end_index)
        match = re.fullmatch(
            r"search_history/seq_(\d{8})/(?:trajectory\.jsonl|code\.py)", path
        )
        if match is not None:
            historical_read_events.append(
                (
                    f"attempt:seq:{match.group(1)}",
                    int(event.get("start_event_index", -1)),
                )
            )

    prerequisite_paths = (
        {
            "search_history/metric_definitions.json",
            "search_history/dashboard_llm.jsonl",
        }
        if require_dashboard
        else {RAW_HISTORY_INDEX}
    )
    if not require_dashboard:
        prerequisite_ends[RAW_HISTORY_INDEX] = [
            int(event.get("end_event_index", -1))
            for event in read_events
            if canonical_view_access_paths([str(event.get("path") or "")], view).pop()
            == RAW_HISTORY_INDEX
        ]
        prerequisite_ends = {
            path: ends for path, ends in prerequisite_ends.items() if ends
        }
    if set(prerequisite_ends) != prerequisite_paths:
        raise ValueError("registered unfolding prerequisites were not read")

    selections: list[dict[str, Any]] = []
    registered_at: dict[str, int] = {}
    metrics_by_attempt: dict[str, set[str]] = {}
    ignored_unknown_metric_ids: dict[str, set[str]] = {}
    ignored_unavailable_attempt_ids: set[str] = set()
    for event in unfold_selection_events:
        batch = event.get("selections")
        if not isinstance(batch, list) or not batch:
            raise ValueError("registered unfold selection batch is empty")
        seen_in_batch: set[str] = set()
        for selection in batch:
            if not isinstance(selection, dict) or set(selection) != {
                "attempt_id",
                "metric_ids",
                "selection_reason",
            }:
                raise ValueError("registered unfold selection has an invalid shape")
            attempt_id = selection["attempt_id"]
            if not isinstance(attempt_id, str) or not re.fullmatch(
                r"attempt:seq:\d{8}", attempt_id
            ):
                raise ValueError(f"invalid registered attempt ID: {attempt_id!r}")
            if attempt_id in seen_in_batch:
                raise ValueError(f"duplicate registered attempt in batch: {attempt_id}")
            seen_in_batch.add(attempt_id)
            selected_metrics = selection["metric_ids"]
            if (
                not isinstance(selected_metrics, list)
                or any(
                    not isinstance(metric_id, str) or not metric_id
                    for metric_id in selected_metrics
                )
                or len(selected_metrics) != len(set(selected_metrics))
            ):
                raise ValueError(f"invalid metric IDs for {attempt_id}")
            if not require_dashboard and selected_metrics:
                raise ValueError(
                    "no-metric registered unfolding requires empty metric_ids"
                )
            unknown = sorted(set(selected_metrics) - metric_ids)
            if unknown:
                ignored_unknown_metric_ids.setdefault(attempt_id, set()).update(unknown)
                selected_metrics = [
                    metric_id
                    for metric_id in selected_metrics
                    if metric_id in metric_ids
                ]
            reason = selection["selection_reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"empty selection reason for {attempt_id}")
            seq = int(attempt_id.rsplit(":", 1)[1])
            if not _history_trajectory_exists(view, seq):
                ignored_unavailable_attempt_ids.add(attempt_id)
                continue
            registration_index = int(event.get("start_event_index", -1))
            registered_at[attempt_id] = min(
                registered_at.get(attempt_id, registration_index), registration_index
            )
            metrics_by_attempt.setdefault(attempt_id, set()).update(selected_metrics)
            selections.append(
                {
                    "attempt_id": attempt_id,
                    "metric_ids": list(selected_metrics),
                    "selection_reason": reason.strip(),
                    "registration_event_index": registration_index,
                    "registration_source": "explicit",
                    "registered_before_use": True,
                }
            )

    consumed: set[str] = set()
    posthoc_compare_dependencies: set[str] = set()
    for attempt_id, start_index in historical_read_events:
        if registered_at.get(attempt_id, start_index) >= start_index:
            raise ValueError(
                f"historical evidence read before successful registration: {attempt_id}"
            )
        consumed.add(attempt_id)
    for event in compare_events:
        start_index = int(event.get("start_event_index", -1))
        for raw in (event.get("left"), event.get("right")):
            attempt_id = _attempt_id(raw)
            if attempt_id is None:
                continue
            if attempt_id not in registered_at:
                seq = int(attempt_id.rsplit(":", 1)[1])
                if not _history_trajectory_exists(view, seq):
                    raise ValueError(
                        f"compared attempt is absent from the frozen view: {attempt_id}"
                    )
                end_index = int(event.get("end_event_index", -1))
                registered_at[attempt_id] = end_index
                metrics_by_attempt.setdefault(attempt_id, set())
                selections.append(
                    {
                        "attempt_id": attempt_id,
                        "metric_ids": [],
                        "selection_reason": "Implicit dependency of compare_attempts.",
                        "registration_event_index": end_index,
                        "registration_source": "posthoc_compare_dependency",
                        "registered_before_use": False,
                    }
                )
                posthoc_compare_dependencies.add(attempt_id)
            elif (
                attempt_id not in posthoc_compare_dependencies
                and registered_at[attempt_id] >= start_index
            ):
                # A focused comparison may discover a parent, child, or donor
                # that was not part of the prospective metric-guided routing
                # decision.  A later explicit registration cannot make that
                # selection prospective, but it should not invalidate an
                # otherwise usable portfolio.
                posthoc_compare_dependencies.add(attempt_id)
                metrics_by_attempt[attempt_id].clear()
                for selection in selections:
                    if selection["attempt_id"] != attempt_id:
                        continue
                    selection["metric_ids"] = []
                    selection["registration_source"] = "explicit_after_compare"
                    selection["registered_before_use"] = False
            consumed.add(attempt_id)

    registered = set(registered_at)
    metric_attributed = {
        attempt_id for attempt_id, values in metrics_by_attempt.items() if values
    }
    return {
        "unfold_selection_mode": "registered",
        "unfold_selections": selections,
        "registered_unfold_count": len(registered),
        "metric_attributed_unfold_count": len(metric_attributed),
        "nonmetric_attributed_unfold_count": len(registered - metric_attributed),
        "posthoc_compare_dependency_count": len(posthoc_compare_dependencies),
        "ignored_unknown_metric_ids": {
            attempt_id: sorted(values)
            for attempt_id, values in sorted(ignored_unknown_metric_ids.items())
        },
        "ignored_unavailable_attempt_ids": sorted(ignored_unavailable_attempt_ids),
        "unused_registered_attempt_ids": sorted(registered - consumed),
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _private_engine_profiles_path(view: Path) -> Path:
    return view.parent / PRIVATE_ENGINE_PROFILES_PATH


def _private_history_path(view: Path) -> Path:
    return view.parent / PRIVATE_HISTORY_PATH


def _history_trajectory_exists(view: Path, seq: int) -> bool:
    relative = Path(f"seq_{seq:08d}/trajectory.jsonl")
    return (view / "search_history" / relative).is_file() or (
        _private_history_path(view) / relative
    ).is_file()


def sanitize_planner_view(
    view: Path, *, isolate_raw_history: bool = False
) -> Path | None:
    """Hide internal intermediate dashboards and keep profiles private to compare_attempts."""
    visible_profiles = view / ENGINE_PROFILES_PATH
    private_profiles = _private_engine_profiles_path(view)
    if visible_profiles.is_file():
        private_profiles.parent.mkdir(parents=True, exist_ok=True)
        visible_profiles.replace(private_profiles)
    if isolate_raw_history:
        private_history = _private_history_path(view)
        private_history.mkdir(parents=True, exist_ok=True)
        for source in sorted((view / "search_history").glob("seq_*")):
            if source.is_dir():
                source.replace(private_history / source.name)
    for relative in INTERNAL_DASHBOARD_PATHS:
        path = view / relative
        if path.is_file():
            path.unlink()
    for relative in INTERNAL_HISTORY_DIRECTORIES:
        path = view / relative
        if path.is_dir():
            shutil.rmtree(path)
    remove_python_bytecode(view)
    return private_profiles if private_profiles.is_file() else None


def _materialize_engine_profiles(
    *,
    view: Path,
    observations: list[dict[str, Any]],
    catalog_rows: list[dict[str, Any]],
    definitions_path: Path,
) -> dict[str, Any] | None:
    """Build one neutral profile cache for historical and live programs."""
    definitions, _ = load_definitions(definitions_path)
    if not definitions.get("mechanistic_profiles"):
        return None

    program_codes: dict[str, str] = {}
    cycles_by_program: dict[str, float] = {}

    def add_program(program_id: Any, relative_path: Any, cycles: Any) -> None:
        if not isinstance(program_id, str) or not program_id:
            return
        if not isinstance(relative_path, str) or not relative_path:
            return
        path = view / relative_path
        if not path.is_file():
            return
        code = path.read_text(encoding="utf-8")
        previous = program_codes.get(program_id)
        if previous is not None and previous != code:
            raise RuntimeError(f"conflicting code for program {program_id}")
        program_codes[program_id] = code
        if (
            isinstance(cycles, (int, float))
            and not isinstance(cycles, bool)
            and math.isfinite(float(cycles))
        ):
            cycles_by_program[program_id] = float(cycles)

    for observation in observations:
        evidence = observation.get("evidence") or {}
        add_program(
            observation.get("program_id"),
            evidence.get("code"),
            observation.get("cycles"),
        )
    for row in catalog_rows:
        add_program(
            row.get("id"),
            row.get("code_path"),
            (row.get("metrics") or {}).get("cycles"),
        )

    profiles, failures = profile_programs(
        program_codes,
        definitions=definitions,
        workers=PROFILE_WORKERS,
    )
    for program_id, profile in profiles.items():
        if program_id in cycles_by_program:
            profile["cycles"] = cycles_by_program[program_id]
    document = {
        "schema": ENGINE_PROFILES_SCHEMA,
        "semantics": (
            "Deterministic emitted-instruction profiles. Absolute values and "
            "right-minus-left deltas are mechanistic evidence, not causal "
            "claims or optimization directions."
        ),
        "profiles": profiles,
        "failures": failures,
    }
    target = view / ENGINE_PROFILES_PATH
    target.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "engine_profile_count": len(profiles),
        "engine_profile_failure_count": len(failures),
    }


def materialize_view(
    *,
    context_root: Path,
    dashboard_root: Path | None,
    checkpoint: Path,
    output: Path,
    prompt_path: Path,
    num_directions: int,
    isolate_raw_history: bool = False,
) -> tuple[Path, str, dict[str, Any], set[str]]:
    view = output / "bundle"
    if view.exists():
        raise RuntimeError(f"output bundle already exists: {view}")

    history_source = context_root / "search_history"
    history_target = view / "search_history"
    history_target.mkdir(parents=True)
    history_directories = [
        source for source in sorted(history_source.glob("seq_*")) if source.is_dir()
    ]
    for source in history_directories:
        if source.is_dir():
            shutil.copytree(source, history_target / source.name)
    if dashboard_root is None:
        try:
            history_sequences = [
                int(source.name.removeprefix("seq_")) for source in history_directories
            ]
        except ValueError as exc:
            raise RuntimeError("raw history contains an invalid seq directory") from exc
        if not history_sequences:
            raise RuntimeError("raw history contains no attempt directories")
        cutoff_seq = max(history_sequences)
        evidence_ids = {f"attempt:seq:{seq:08d}" for seq in history_sequences}
        for seq in history_sequences:
            trajectory = history_target / f"seq_{seq:08d}" / "trajectory.jsonl"
            if not trajectory.is_file():
                raise RuntimeError(f"missing raw-history trajectory: {trajectory}")
        header = {"schema": "oe_raw_history_view_v1", "metric_order": []}
        observations = [{"cutoff_seq": seq} for seq in history_sequences]
        (view / RAW_HISTORY_INDEX).write_text(
            "".join(
                json.dumps(
                    {
                        "attempt_id": f"attempt:seq:{seq:08d}",
                        "trajectory_path": (
                            f"search_history/seq_{seq:08d}/trajectory.jsonl"
                        ),
                        "code_path": (
                            f"search_history/seq_{seq:08d}/code.py"
                            if (history_target / f"seq_{seq:08d}" / "code.py").is_file()
                            else None
                        ),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for seq in history_sequences
            ),
            encoding="utf-8",
        )
    else:
        dashboard_path = dashboard_root / "dashboard.jsonl"
        definitions_path = dashboard_root / "metric_definitions.json"
        header, observations = validate_dashboard_jsonl(
            dashboard_path,
            definitions_path=definitions_path,
        )
        cutoff_seq = int(observations[-1]["cutoff_seq"])
        shutil.copy2(dashboard_path, history_target / "dashboard.jsonl")
        shutil.copy2(definitions_path, history_target / "metric_definitions.json")
        write_llm_dashboard_jsonl(
            history_target / "dashboard_llm.jsonl", header, observations
        )
        evidence_ids = set()
        for observation in observations:
            seq = int(observation["cutoff_seq"])
            evidence_ids.add(f"attempt:seq:{seq:08d}")
            evidence = observation.get("evidence") or {}
            for field in ("trajectory", "code"):
                relpath = evidence.get(field)
                if field == "code" and relpath is None:
                    continue
                if not isinstance(relpath, str) or not (view / relpath).is_file():
                    raise RuntimeError(f"missing dashboard evidence: {relpath!r}")

        allowed_directories = {
            f"seq_{int(item.rsplit(':', 1)[1]):08d}" for item in evidence_ids
        }
        for directory in history_target.glob("seq_*"):
            if directory.name not in allowed_directories:
                shutil.rmtree(directory)

    metadata = _json(checkpoint / "metadata.json")
    islands = metadata.get("islands") or []
    island_of = {
        str(program_id): island
        for island, members in enumerate(islands)
        for program_id in members
    }
    best_id = str(metadata.get("best_program_id") or "")
    island_bests = {
        str(program_id)
        for program_id in (metadata.get("island_best_programs") or [])
        if program_id
    }

    programs_target = view / "workspace" / "programs"
    programs_target.mkdir(parents=True)
    catalog_rows: list[dict[str, Any]] = []
    for source in sorted((checkpoint / "programs").glob("*.json")):
        program = _json(source)
        program_id = str(program["id"])
        code = program.get("code")
        if not isinstance(code, str) or not code:
            continue
        code_path = programs_target / f"{program_id}.py"
        code_path.write_text(code, encoding="utf-8")
        metrics = program.get("metrics") or {}
        metadata_island = (program.get("metadata") or {}).get("island")
        catalog_rows.append(
            {
                "id": program_id,
                "parent_id": program.get("parent_id"),
                "island": island_of.get(program_id, metadata_island),
                "generation": program.get("generation"),
                "metrics": metrics,
                "fitness": metrics.get("combined_score"),
                "is_global_best": program_id == best_id,
                "is_island_best": program_id in island_bests,
                "code_path": f"workspace/programs/{program_id}.py",
                "code_sha256": _sha256(code),
            }
        )
    if not catalog_rows or best_id not in {row["id"] for row in catalog_rows}:
        raise RuntimeError("checkpoint did not produce a valid live branch catalog")

    catalog = {
        "schema": "oe_branch_catalog_v1",
        "cutoff_seq": cutoff_seq,
        "programs": catalog_rows,
    }
    catalog_path = view / "workspace" / "meta" / "branch_catalog.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile_summary = None
    if dashboard_root is not None:
        profile_summary = _materialize_engine_profiles(
            view=view,
            observations=observations,
            catalog_rows=catalog_rows,
            definitions_path=definitions_path,
        )
    sanitize_planner_view(view, isolate_raw_history=isolate_raw_history)

    integration_dir = Path(__file__).resolve().parent
    reference = view / "workspace" / "reference"
    reference.mkdir(parents=True)
    shutil.copy2(integration_dir / "vendor" / "takehome" / "problem.py", reference)
    shutil.copy2(integration_dir / "evaluator.py", reference)

    prompt = prompt_path.read_text(encoding="utf-8").replace(
        "{{num_directions}}", str(num_directions)
    )
    if "{{num_directions}}" in prompt:
        raise RuntimeError("direction-count placeholder was not fully resolved")
    (output / "prompt.md").write_text(prompt, encoding="utf-8")
    materialization = {
        "schema": header["schema"],
        "cutoff_seq": cutoff_seq,
        "observation_count": len(observations),
        "metric_count": len(header["metric_order"]),
        "history_mode": "raw" if dashboard_root is None else "dashboard",
        "live_branch_count": len(catalog_rows),
        "best_program_id": best_id,
        "num_directions": num_directions,
        "raw_history_visibility": (
            "controller_private" if isolate_raw_history else "planner"
        ),
    }
    if profile_summary is not None:
        materialization.update(profile_summary)
    (output / "materialization.json").write_text(
        json.dumps(materialization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return view, prompt, catalog, evidence_ids


def validate_portfolio(
    document: Any,
    *,
    catalog: dict[str, Any],
    evidence_ids: set[str],
    num_directions: int,
) -> dict[str, Any]:
    if not isinstance(document, dict) or frozenset(document) not in {
        frozenset({"exhausted_patterns", "directions"}),
        frozenset({"diagnosis", "exhausted_patterns", "directions"}),
    }:
        raise ValueError("top-level portfolio fields are invalid")
    exhausted = document["exhausted_patterns"]
    if (
        not isinstance(exhausted, list)
        or len(exhausted) > 3
        or not all(isinstance(item, str) and item.strip() for item in exhausted)
    ):
        raise ValueError("exhausted_patterns is invalid")
    directions = document["directions"]
    if not isinstance(directions, list) or len(directions) != num_directions:
        raise ValueError(f"directions must contain exactly {num_directions} items")

    live = {str(row["id"]): row for row in catalog["programs"]}
    tasks: set[str] = set()
    normalized_directions: list[dict[str, Any]] = []
    for index, direction in enumerate(directions):
        if not isinstance(direction, dict) or frozenset(direction) not in {
            frozenset(DIRECTION_KEYS),
            frozenset(DIRECTION_KEYS | PROPOSAL_REASONING_KEYS),
        }:
            raise ValueError(f"direction {index} fields are invalid")
        target = direction["target_branch_id"]
        if target not in live:
            raise ValueError(f"direction {index} target is not live")
        for field in ("mechanism_hypothesis", "task"):
            if not isinstance(direction[field], str) or not direction[field].strip():
                raise ValueError(f"direction {index} has empty {field}")
        if direction["task"] in tasks:
            raise ValueError("portfolio contains duplicate tasks")
        tasks.add(direction["task"])
        if not isinstance(direction["constraints"], list):
            raise ValueError(f"direction {index} constraints are invalid")
        donors = direction["donor_branch_ids"]
        if (
            not isinstance(donors, list)
            or len(donors) > 2
            or any(donor not in live for donor in donors)
            or len(set(donors)) != len(donors)
            or target in donors
        ):
            raise ValueError(f"direction {index} donors are invalid")
        evidence = direction["evidence_attempt_ids"]
        if not isinstance(evidence, list) or any(
            item not in evidence_ids for item in evidence
        ):
            raise ValueError(f"direction {index} evidence is invalid")
        if donors and not evidence:
            raise ValueError(f"direction {index} donor evidence is invalid")
        normalized_directions.append({key: direction[key] for key in DIRECTION_KEYS})
    return {
        "exhausted_patterns": list(exhausted),
        "directions": normalized_directions,
    }


def validate_reads(
    document: dict[str, Any],
    read_paths: list[str],
    *,
    view: Path,
    require_dashboard: bool = True,
    max_history_unfolds: int | None = None,
    tool_names: list[str] | None = None,
    direct_read_paths: list[str] | None = None,
    tool_internal_read_paths: list[str] | None = None,
    unfold_selection_mode: str = "registered",
    read_events: list[dict[str, Any]] | None = None,
    compare_events: list[dict[str, Any]] | None = None,
    unfold_selection_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if unfold_selection_mode not in UNFOLD_SELECTION_MODES:
        raise ValueError(f"unknown unfold selection mode: {unfold_selection_mode!r}")
    private_profiles = _private_engine_profiles_path(view)
    private_resolved = private_profiles.resolve()
    ordinary_reads = []
    for raw in read_paths:
        try:
            if Path(raw).resolve() == private_resolved:
                continue
        except OSError:
            pass
        ordinary_reads.append(raw)
    reads = canonical_view_access_paths(ordinary_reads, view)
    direct_reads = canonical_view_access_paths(direct_read_paths or [], view)
    historical_comparison_required = False
    allowed_direct = {
        "workspace/reference/problem.py",
        "workspace/reference/evaluator.py",
        "workspace/meta/branch_catalog.json",
        "search_history/metric_definitions.json",
        "search_history/dashboard_llm.jsonl",
        RAW_HISTORY_INDEX,
    }
    forbidden_direct = sorted(
        path
        for path in direct_reads
        if path not in allowed_direct
        and not re.fullmatch(r"workspace/programs/[^/]+\.py", path)
    )
    private_profile_reads = [
        path
        for path in direct_read_paths or []
        if path.replace("\\", "/").endswith("/engine_profiles.json")
        or path.replace("\\", "/") == "engine_profiles.json"
    ]
    if private_profile_reads:
        forbidden_direct.extend(private_profile_reads)
    forbidden_alias_reads = []
    for path in direct_read_paths or []:
        normalized = "/" + path.replace("\\", "/")
        if any(
            f"/{directory}/" in normalized for directory in INTERNAL_HISTORY_DIRECTORIES
        ):
            forbidden_alias_reads.append(path)
    if forbidden_alias_reads:
        forbidden_direct.extend(forbidden_alias_reads)
    if forbidden_direct:
        raise ValueError(
            f"forbidden direct evidence reads: {sorted(set(forbidden_direct))}"
        )
    required = {
        "workspace/reference/problem.py",
        "workspace/reference/evaluator.py",
        "workspace/meta/branch_catalog.json",
    }
    if require_dashboard:
        required.update(
            {
                "search_history/metric_definitions.json",
                "search_history/dashboard_llm.jsonl",
            }
        )
        historical_comparison_required = any(
            direction["evidence_attempt_ids"] for direction in document["directions"]
        )
        if private_profiles.is_file() and historical_comparison_required:
            if "compare_attempts" not in (tool_names or []):
                raise ValueError(
                    "mechanistic profile was not consumed through compare_attempts"
                )
            internal = {Path(path).resolve() for path in tool_internal_read_paths or []}
            if private_profiles.resolve() not in internal:
                raise ValueError(
                    "compare_attempts did not consume the private mechanistic profile"
                )
    else:
        required.add(RAW_HISTORY_INDEX)
    for direction in document["directions"]:
        required.add(f"workspace/programs/{direction['target_branch_id']}.py")
        for donor in direction["donor_branch_ids"]:
            required.add(f"workspace/programs/{donor}.py")
        for evidence_id in direction["evidence_attempt_ids"]:
            seq = int(evidence_id.rsplit(":", 1)[1])
            prefix = f"search_history/seq_{seq:08d}"
            required.add(f"{prefix}/trajectory.jsonl")
    missing = sorted(required - reads)
    if missing:
        raise ValueError(f"required evidence was not read: {missing}")
    history_unfolds = {
        part
        for path in reads
        for part in path.split("/")
        if part.startswith("seq_") and part.removeprefix("seq_").isdigit()
    }
    if max_history_unfolds is not None and len(history_unfolds) > max_history_unfolds:
        raise ValueError(
            f"history unfold limit exceeded: {len(history_unfolds)} > "
            f"{max_history_unfolds}"
        )
    unfold_contract = _validate_registered_unfolds(
        view=view,
        read_events=read_events or [],
        compare_events=compare_events or [],
        unfold_selection_events=unfold_selection_events or [],
        require_dashboard=require_dashboard,
    )
    unfold_contract["unfold_selection_mode"] = unfold_selection_mode
    return {
        "read_count": len(reads),
        "history_unfold_count": len(history_unfolds),
        "required_reads": sorted(required),
        "required_tools": (
            ["compare_attempts"]
            if require_dashboard
            and historical_comparison_required
            and _private_engine_profiles_path(view).is_file()
            else []
        ),
        **unfold_contract,
    }


def run_pi(
    *,
    view: Path,
    prompt: str,
    output: Path,
    executable: Path,
    model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: float,
    pi_tools: str | None = None,
    num_directions: int = 4,
    unfold_selection_mode: str = "registered",
    session_id: str | None = None,
    session_home: Path | None = None,
    session_tmp: Path | None = None,
    user_message: str | None = None,
    require_dashboard: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_prompt = prompt_for_unfold_selection_mode(
        prompt, unfold_selection_mode, require_dashboard=require_dashboard
    )
    if pi_tools is not None:
        raise ValueError("registered unfolding requires the registered tool set")
    if not os.environ.get(api_key_env):
        raise RuntimeError(f"{api_key_env} is not set")
    home = session_home or output / "pi-home"
    temporary = session_tmp or output / "pi-tmp"
    home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    private_profiles = _private_engine_profiles_path(view)
    private_env: dict[str, str] = {}
    if private_profiles.is_file():
        private_env[PRIVATE_ENGINE_PROFILES_ENV] = str(private_profiles.resolve())
    private_history = _private_history_path(view)
    if not private_history.is_dir():
        raise RuntimeError("registered unfolding requires private raw history")
    private_env[PRIVATE_HISTORY_ENV] = str(private_history.resolve())
    env = build_pi_subprocess_env(
        executable.as_posix(),
        home=home,
        tmpdir=temporary,
        extra=private_env or None,
        provider=provider,
        api_key_env=api_key_env,
    )
    config_dir = home / ".pi" / "agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    if provider != "deepseek":
        if provider == "google":
            (config_dir / "settings.json").write_text(
                json.dumps(
                    {
                        "retry": {
                            "enabled": False,
                            "provider": {
                                "timeoutMs": 600_000,
                                "maxRetries": 0,
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provider_config = {
                "baseUrl": base_url,
                "api": "google-generative-ai",
                "apiKey": f"${api_key_env}",
                "models": [
                    {
                        "id": model,
                        "name": model,
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": 1_000_000,
                        "maxTokens": 65_536,
                        # Match the worker's list-price estimate, not an invoice.
                        "cost": {
                            "input": 2.0,
                            "output": 12.0,
                            "cacheRead": 2.0,
                            "cacheWrite": 2.0,
                        },
                    }
                ],
            }
        elif provider == "glm":
            provider_config = {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": f"${api_key_env}",
                "models": [
                    {
                        "id": model,
                        "name": f"{model} via BigModel",
                        "reasoning": True,
                        "contextWindow": 1_000_000,
                        "maxTokens": 128_000,
                        "cost": {
                            "input": 0.8 / 7.0,
                            "output": 2.8 / 7.0,
                            "cacheRead": 0.23 / 7.0,
                            "cacheWrite": 0.8 / 7.0,
                        },
                        "thinkingLevelMap": {
                            "low": "low",
                            "high": "high",
                            "max": "max",
                        },
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": True,
                            "maxTokensField": "max_tokens",
                        },
                    }
                ],
            }
        else:
            provider_config = {
                "baseUrl": base_url,
                "api": "openai-responses",
                "apiKey": f"${api_key_env}",
                "authHeader": True,
                "models": [
                    {
                        "id": model,
                        "name": f"{model} via MAST",
                        "reasoning": True,
                        "contextWindow": 400000,
                        "maxTokens": 16384,
                        "cost": {
                            "input": 2.5,
                            "output": 15.0,
                            "cacheRead": 0.25,
                            "cacheWrite": 3.125,
                        },
                        "thinkingLevelMap": {"high": "high"},
                    }
                ],
            }
        (config_dir / "models.json").write_text(
            json.dumps(
                {"providers": {provider: provider_config}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    # Pi loads --system-prompt files directly into system context. Avoid the
    # OS per-argument size limit without truncating or folding failed history.
    prompt_path = output / "system_prompt.txt"
    prompt_path.write_text(effective_prompt, encoding="utf-8")
    argv = build_pi_argv(
        model,
        str(prompt_path.resolve()),
        executable.as_posix(),
        extra_extensions=(UNFOLD_SELECTION_EXTENSION, PUBLIC_FILES_EXTENSION),
        extra_tools=("register_unfold_selection", "read_registered_attempt"),
        tools=REGISTERED_PI_TOOLS,
        session_id=session_id,
        user_message=user_message
        or (
            f"Generate the {num_directions}-direction intervention portfolio now. Return "
            "only the required JSON object, with no prose or Markdown fence."
        ),
        provider=provider,
        reasoning=os.environ.get("AUDIT_PI_REASONING", "high"),
    )
    started = time.monotonic()
    timed_out = False
    stdout, stderr = "", ""
    transport_retries = 0
    while True:
        process = subprocess.Popen(
            argv,
            cwd=view,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            chunk, errors = process.communicate(
                timeout=max(0.1, timeout_seconds - (time.monotonic() - started))
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            chunk, errors = process.communicate()
        stdout += chunk
        stderr += errors
        (output / "pi.jsonl").write_text(stdout, encoding="utf-8")
        last_message = {}
        if provider == "google":
            for line in chunk.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                message = event.get("message") or {}
                if (
                    event.get("type") == "message_end"
                    and message.get("role") == "assistant"
                ):
                    last_message = message
        transient = re.search(
            r"\b(?:429|502|503|504)\b", str(last_message.get("errorMessage", ""))
        )
        if (
            not transient
            or timed_out
            or time.monotonic() - started + 5 >= timeout_seconds
        ):
            break
        transport_retries += 1
        # Resume the same Pi session without a JSON/portfolio repair instruction.
        time.sleep(5)
    duration = time.monotonic() - started
    (output / "pi.jsonl").write_text(stdout, encoding="utf-8")
    (output / "pi.stderr.txt").write_text(stderr, encoding="utf-8")
    parsed = parse_pi_stream(stdout)
    metadata = {
        "provider": provider,
        "model": model,
        "reasoning": os.environ.get("AUDIT_PI_REASONING"),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "tool_calls": parsed["tool_calls"],
        "input_tokens": parsed["input_tokens"],
        "output_tokens": parsed["output_tokens"],
        "cache_read_tokens": parsed["cache_read_tokens"],
        "cache_write_tokens": parsed["cache_write_tokens"],
        "catalog_cost_estimate_usd": parsed["cli_cost_usd"],
        "session_id": parsed["session_id"],
        "session_persistence": session_id is not None,
        "transport_retries": transport_retries,
        "tool_names": parsed["tool_names"],
        "successful_read_paths": parsed["successful_read_paths"],
        "direct_read_paths": parsed["direct_read_paths"],
        "tool_internal_read_paths": parsed["tool_internal_read_paths"],
        "read_events": parsed["read_events"],
        "compare_events": parsed["compare_events"],
        "unfold_selection_events": parsed["unfold_selection_events"],
        "unfold_selection_mode": unfold_selection_mode,
    }
    if timed_out or process.returncode != 0 or not parsed["final_text"]:
        exc = RuntimeError(
            f"Pi failed: timed_out={timed_out}, returncode={process.returncode}, "
            f"stderr={stderr[-1000:]!r}"
        )
        exc.pi_metadata = metadata
        raise exc
    try:
        document = json.loads(parsed["final_text"])
    except json.JSONDecodeError as exc:
        # A malformed final answer does not invalidate successful tool reads.
        exc.pi_metadata = metadata
        raise
    return document, metadata
