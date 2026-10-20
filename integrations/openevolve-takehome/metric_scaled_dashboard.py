#!/usr/bin/env python3
"""Build the append-only OpenEvolve metric-scaled dashboard.

This module is deliberately independent of the auditor prompt and read path.
It folds a causal terminal-attempt prefix into atomic U/B/R/G metrics
defined in ``metric_definitions.json``.  Each metric has exactly one value plus
percentile, robust-z, availability, and evidence provenance.
"""

from __future__ import annotations

from release_paths import release_resource

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SHARED_ROOT = HERE.parents[1] / "shared" / "meta-agent-audit"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from code_geometry import (
    implementation_spread,
    signed_fingerprint,
    weighted_multiset_jaccard,
)


DEFAULT_DEFINITIONS = release_resource("metrics.json")
SCHEMA = "oe_metric_dashboard_v4"
LLM_SCHEMA = "oe_metric_dashboard_llm_v2"
HEADER_RECORD = "dashboard_header"
OBSERVATION_RECORD = "observation"
DEFAULT_WINDOW_SIZE = 12
DEFAULT_HISTORY_GATE_SECONDS = 3600.0
DEFAULT_MINIMUM_REFERENCE_N = 12
DEFAULT_BRANCH_MATURITY_STEPS = 12
ENGINE_NAMES = ("alu", "valu", "load", "store", "flow")
ENGINE_METRIC_ORDER = ("valu", "load", "store", "flow", "alu")
ENGINE_CAPACITY_PROFILE = "engine_capacity"
SCOPE_PREFIX = {
    "unary": "U",
    "binary": "B",
    "regional": "R",
    "global": "G",
}
STATUS_CODES = {
    "available": "a",
    "partial": "p",
    "warming_up": "w",
    "unavailable": "u",
}
STATUS_CODE_DEFINITIONS = {
    "a": {"status": "available", "eligible": True},
    "p": {"status": "partial", "eligible": True},
    "w": {"status": "warming_up", "eligible": False},
    "u": {"status": "unavailable", "eligible": False},
}
METRIC_WINDOW_SOURCE = {
    "R1": "terminal_W12",
    "R2": "eligible_state_W12",
    "R3": "eligible_state_W12",
    "R4": "terminal_W12",
    "R5": "terminal_W12",
    "R6": "eligible_state_W12",
    "R7": "terminal_W12",
    "R8": "terminal_W12",
    "R9": "terminal_W12",
    "R10": "terminal_W12",
    "R11": "terminal_W12",
    "R12": "terminal_W12",
    "R13": "terminal_W12",
    "R14": "terminal_W12",
    "R15": "terminal_W12",
    "R16": "terminal_W12",
    "R17": "terminal_W12",
    "R18": "terminal_W12_and_live_database_at_audit_cutoff",
    "G2": "terminal_W12",
}
METRIC_REFERENCE_ROLE = {
    "B1": "parent",
    "B2": "frontier",
    "B3": "parent",
    "B4": "frontier",
    "B5": "nearest_history",
    "B6": "parent",
    "B7": "parent",
    "B8": "parent",
    "B9": "parent",
    "B10": "parent",
    "B11": "frontier",
    "B12": "frontier",
    "B13": "parent",
    "B15": "parent",
    "B16": "parent",
    "B17": "frontier",
    "R2": "frontier",
    "R3": "root",
    "R17": "frontier",
    "R18": "live_database_at_audit_cutoff",
    "G3": "frontier",
    "G10": "descendants_at_audit_cutoff",
    "G11": "descendants_at_audit_cutoff",
    "G13": "direct_native_reuse",
    "G14": "best_direct_native_reuse",
}


class MetricDashboardError(ValueError):
    """The metric definitions or causal source rows are inconsistent."""


def _finite(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _hhi(
    values: Iterable[str],
) -> tuple[int, float | None, float | None, str | None, float | None]:
    counts = Counter(str(item) for item in values if item is not None)
    total = sum(counts.values())
    if not total:
        return 0, None, None, None, None
    shares = {key: count / total for key, count in counts.items()}
    hhi = sum(share * share for share in shares.values())
    dominant = min(
        (key for key, share in shares.items() if share == max(shares.values())),
        default=None,
    )
    return len(counts), hhi, 1.0 / hhi, dominant, max(shares.values())


def _percentile(value: float, reference: Sequence[float]) -> float:
    less = sum(item < value for item in reference)
    equal = sum(item == value for item in reference)
    return (less + 0.5 * equal) / len(reference)


def _robust_z(value: float, reference: Sequence[float]) -> float | None:
    if len(reference) < 6:
        return None
    median = statistics.median(reference)
    mad = statistics.median(abs(item - median) for item in reference)
    if mad <= 0.0:
        return None
    return (value - median) / (1.4826 * mad)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_definitions(
    path: str | Path = DEFAULT_DEFINITIONS,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise MetricDashboardError(f"invalid metric definitions: {exc}") from exc
    metrics = document.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise MetricDashboardError("metric definitions must contain metrics")
    if document.get("scope_order") != list(SCOPE_PREFIX):
        raise MetricDashboardError("metric scope order is invalid")
    for scope, prefix in SCOPE_PREFIX.items():
        actual = [
            metric_id
            for metric_id, definition in metrics.items()
            if definition.get("scope") == scope
        ]
        try:
            indices = [
                int(metric_id[len(prefix) :])
                for metric_id in actual
                if metric_id.startswith(prefix)
            ]
        except ValueError as exc:
            raise MetricDashboardError(
                f"{scope} metric definitions have invalid ids"
            ) from exc
        if (
            len(indices) != len(actual)
            or any(index <= 0 for index in indices)
            or indices != sorted(set(indices))
        ):
            raise MetricDashboardError(
                f"{scope} metric definitions must be uniquely ordered"
            )
    required_fields = {
        "name",
        "scope",
        "definition",
        "formula",
        "high_means",
        "low_means",
        "scaling_reference",
    }
    for metric_id, definition in metrics.items():
        if definition.get("scope") not in SCOPE_PREFIX:
            raise MetricDashboardError(f"{metric_id} has an invalid scope")
        missing = required_fields - set(definition)
        if missing:
            raise MetricDashboardError(
                f"{metric_id} is missing definition fields: {sorted(missing)}"
            )
        if "components" in definition or "rankable_components" in definition:
            raise MetricDashboardError(f"{metric_id} is not an atomic metric")
    profiles = document.get("mechanistic_profiles") or {}
    if not isinstance(profiles, dict):
        raise MetricDashboardError("mechanistic profile definitions must be an object")
    for profile_id, definition in profiles.items():
        if not isinstance(definition, dict):
            raise MetricDashboardError(f"{profile_id} profile definition is invalid")
        columns = definition.get("columns")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(item, str) and item for item in columns)
            or len(columns) != len(set(columns))
        ):
            raise MetricDashboardError(f"{profile_id} profile columns are invalid")
        if not isinstance(definition.get("role"), str) or not definition["role"]:
            raise MetricDashboardError(f"{profile_id} profile role is invalid")
    return document, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def dashboard_header(
    definitions: Mapping[str, Any], definitions_sha256: str
) -> dict[str, Any]:
    scaling = definitions.get("scaling_contract") or {}
    metric_order = list(definitions["metrics"])
    result = {
        "record_type": HEADER_RECORD,
        "schema": SCHEMA,
        "definition_version": definitions["definition_version"],
        "definitions_sha256": definitions_sha256,
        "metric_order": metric_order,
        "observation_metric_columns": [
            "values",
            "percentiles",
            "robust_z",
            "status_codes",
            "reference_n",
        ],
        "status_codes": STATUS_CODE_DEFINITIONS,
        "metric_window_source": {
            metric_id: source
            for metric_id, source in METRIC_WINDOW_SOURCE.items()
            if metric_id in metric_order
        },
        "metric_reference_role": {
            metric_id: role
            for metric_id, role in METRIC_REFERENCE_ROLE.items()
            if metric_id in metric_order
        },
        "evidence_ref_encoding": (
            "An integer N means attempt:seq:N zero-padded to eight digits; "
            "other strings are literal evidence references."
        ),
        "window_size": DEFAULT_WINDOW_SIZE,
        "history_gate_seconds": scaling.get(
            "history_gate_seconds", DEFAULT_HISTORY_GATE_SECONDS
        ),
        "minimum_reference_n": scaling.get(
            "minimum_reference_n", DEFAULT_MINIMUM_REFERENCE_N
        ),
    }
    profiles = definitions.get("mechanistic_profiles") or {}
    if profiles:
        result["mechanistic_profile_order"] = list(profiles)
        result["mechanistic_profile_columns"] = {
            profile_id: list(definition["columns"])
            for profile_id, definition in profiles.items()
        }
    return result


def _availability(
    *,
    status: str,
    eligible: bool,
    n_expected: int | None = None,
    n_used: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"eligible": bool(eligible), "status": status}
    if n_expected is not None:
        result["n_expected"] = int(n_expected)
    if n_used is not None:
        result["n_used"] = int(n_used)
        result["missing_rate"] = (
            max(0.0, (n_expected - n_used) / n_expected) if n_expected else None
        )
    if reason:
        result["reason"] = str(reason)
    return result


def _metric(
    metric_id: str,
    value: Any,
    *,
    availability: Mapping[str, Any],
    window_ids: Sequence[str] = (),
    reference_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value": value,
        "percentile": None,
        "robust_z": None,
        "availability": dict(availability),
        "window_evidence_ids": list(window_ids),
        "reference_ids": list(reference_ids),
    }


def _attempts(history: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    if history is None:
        return {}
    source = history.get("attempt_by_seq")
    if not isinstance(source, Mapping):
        return {}
    return {
        int(key): value for key, value in source.items() if isinstance(value, Mapping)
    }


def _program_codes(history: Mapping[str, Any] | None) -> dict[str, str]:
    if history is None:
        return {}
    root = history.get("root_code")
    result = {"root": str(root)} if isinstance(root, str) else {}
    root_hash = _sha256_text(root) if isinstance(root, str) else None
    for attempt in _attempts(history).values():
        program_id = attempt.get("program_id")
        code = attempt.get("code")
        if isinstance(program_id, str) and program_id and isinstance(code, str):
            result[program_id] = code
        for event in attempt.get("source_events") or ():
            if (
                not isinstance(event, Mapping)
                or event.get("event") != "attempt_scheduled"
            ):
                continue
            parent_id = event.get("parent_id")
            parent_code = event.get("parent_code")
            if isinstance(parent_id, str) and isinstance(parent_code, str):
                result.setdefault(parent_id, parent_code)
            elif (
                isinstance(parent_id, str)
                and root_hash is not None
                and event.get("parent_code_sha256") == root_hash
            ):
                result.setdefault(parent_id, str(root))
    return result


def _profile_from_instrs(
    instrs: Any,
    *,
    scratch_words: Any,
    slot_limits: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(instrs, list) or not all(
        isinstance(item, dict) for item in instrs
    ):
        raise MetricDashboardError("built instruction stream is not a bundle list")
    op_counts = {
        engine: sum(
            len(bundle.get(engine) or ())
            for bundle in instrs
            if isinstance(bundle.get(engine) or (), list)
        )
        for engine in ENGINE_NAMES
    }
    occupied = {
        engine: sum(bool(bundle.get(engine)) for bundle in instrs)
        for engine in ENGINE_NAMES
    }
    lower_bounds = {
        engine: math.ceil(op_counts[engine] / int(slot_limits[engine]))
        for engine in ENGINE_NAMES
    }
    maximum = max(lower_bounds.values(), default=0)
    dependency_lower_bound = _dependency_critical_path_lower_bound(instrs)
    result: dict[str, Any] = {
        "emitted_bundle_count": len(instrs),
        "engine_op_counts": op_counts,
        "engine_occupied_bundle_counts": occupied,
        "engine_slot_lower_bounds": lower_bounds,
        "max_engine_slot_lower_bound": maximum,
        "max_engine_slot_lower_bound_engines": sorted(
            key for key, value in lower_bounds.items() if value == maximum
        ),
        "dependency_critical_path_lower_bound": dependency_lower_bound,
    }
    if type(scratch_words) is int and scratch_words >= 0:
        result["scratch_words"] = scratch_words
    return result


def _scratch_range(address: Any, width: int = 1) -> set[int]:
    if type(address) is not int or address < 0:
        raise MetricDashboardError(f"invalid scratch address: {address!r}")
    return set(range(address, address + width))


def _slot_scratch_accesses(engine: str, slot: Any) -> tuple[set[int], set[int]]:
    """Return scratch reads/writes for one frozen-takehome ISA slot."""
    if not isinstance(slot, (list, tuple)) or not slot:
        raise MetricDashboardError(f"invalid {engine} slot: {slot!r}")
    op = slot[0]
    if engine == "debug":
        return set(), set()
    if engine == "alu" and len(slot) == 4:
        _, dest, left, right = slot
        return _scratch_range(left) | _scratch_range(right), _scratch_range(dest)
    if engine == "valu":
        if op == "vbroadcast" and len(slot) == 3:
            _, dest, source = slot
            return _scratch_range(source), _scratch_range(dest, 8)
        if op == "multiply_add" and len(slot) == 5:
            _, dest, left, right, addend = slot
            reads = (
                _scratch_range(left, 8)
                | _scratch_range(right, 8)
                | _scratch_range(addend, 8)
            )
            return reads, _scratch_range(dest, 8)
        if len(slot) == 4:
            _, dest, left, right = slot
            return (
                _scratch_range(left, 8) | _scratch_range(right, 8),
                _scratch_range(dest, 8),
            )
    if engine == "load":
        if op == "const" and len(slot) == 3:
            return set(), _scratch_range(slot[1])
        if op == "load" and len(slot) == 3:
            return _scratch_range(slot[2]), _scratch_range(slot[1])
        if op == "load_offset" and len(slot) == 4:
            _, dest, address, offset = slot
            if type(offset) is not int:
                raise MetricDashboardError(f"invalid load_offset: {slot!r}")
            return _scratch_range(address + offset), _scratch_range(dest + offset)
        if op == "vload" and len(slot) == 3:
            return _scratch_range(slot[2]), _scratch_range(slot[1], 8)
    if engine == "store":
        if op == "store" and len(slot) == 3:
            return _scratch_range(slot[1]) | _scratch_range(slot[2]), set()
        if op == "vstore" and len(slot) == 3:
            return _scratch_range(slot[1]) | _scratch_range(slot[2], 8), set()
    if engine == "flow":
        if op == "select" and len(slot) == 5:
            _, dest, condition, left, right = slot
            return (
                _scratch_range(condition)
                | _scratch_range(left)
                | _scratch_range(right),
                _scratch_range(dest),
            )
        if op == "add_imm" and len(slot) == 4:
            return _scratch_range(slot[2]), _scratch_range(slot[1])
        if op == "vselect" and len(slot) == 5:
            _, dest, condition, left, right = slot
            return (
                _scratch_range(condition, 8)
                | _scratch_range(left, 8)
                | _scratch_range(right, 8),
                _scratch_range(dest, 8),
            )
        if op in {"halt", "pause", "jump"}:
            return set(), set()
        if op == "trace_write" and len(slot) == 2:
            return _scratch_range(slot[1]), set()
        if op in {"cond_jump", "cond_jump_rel"} and len(slot) == 3:
            return _scratch_range(slot[1]), set()
        if op == "jump_indirect" and len(slot) == 2:
            return _scratch_range(slot[1]), set()
        if op == "coreid" and len(slot) == 2:
            return set(), _scratch_range(slot[1])
    raise MetricDashboardError(f"unsupported {engine} slot: {slot!r}")


def _dependency_critical_path_lower_bound(instrs: Sequence[Mapping[str, Any]]) -> int:
    """Longest fixed-scratch RAW/WAR/WAW chain; dynamic memory aliases are ignored."""
    last_writer: dict[int, int] = {}
    readers_since_write: dict[int, set[int]] = defaultdict(set)
    depth_by_bundle: dict[int, int] = {}
    maximum = 0
    for bundle_index, bundle in enumerate(instrs):
        reads: set[int] = set()
        writes: set[int] = set()
        has_runtime_slot = False
        for engine, slots in bundle.items():
            if not isinstance(slots, list):
                raise MetricDashboardError(f"invalid {engine} slots: {slots!r}")
            if engine != "debug" and slots:
                has_runtime_slot = True
            for slot in slots:
                slot_reads, slot_writes = _slot_scratch_accesses(engine, slot)
                reads.update(slot_reads)
                writes.update(slot_writes)
        if not has_runtime_slot:
            continue

        predecessors = {
            last_writer[address] for address in reads | writes if address in last_writer
        }
        for address in writes:
            predecessors.update(readers_since_write.get(address, ()))
        depth = 1 + max((depth_by_bundle[index] for index in predecessors), default=0)
        depth_by_bundle[bundle_index] = depth
        maximum = max(maximum, depth)

        for address in reads - writes:
            readers_since_write[address].add(bundle_index)
        for address in writes:
            last_writer[address] = bundle_index
            readers_since_write[address].clear()
    return maximum


def _profile_code(
    code: str,
    *,
    eval_build: Path,
    takehome_dir: Path,
    slot_limits: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="oe-metric-profile-") as directory:
        root = Path(directory)
        candidate = root / "candidate.py"
        output = root / "instrs.json"
        candidate.write_text(code, encoding="utf-8")
        build = subprocess.run(
            [
                sys.executable,
                str(eval_build),
                str(candidate),
                str(takehome_dir),
                str(output),
            ],
            cwd=root,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        if build.returncode != 0 or not output.is_file():
            raise MetricDashboardError(
                f"instruction build failed (exit {build.returncode})"
            )
        built = json.loads(output.read_text(encoding="utf-8"))
        return _profile_from_instrs(
            built.get("instrs"),
            scratch_words=built.get("scratch_words"),
            slot_limits=slot_limits,
        )


def profile_programs(
    program_codes: Mapping[str, str],
    *,
    definitions: Mapping[str, Any],
    workers: int = 1,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build each unique program once; return values and explicit failures."""
    if workers <= 0:
        raise MetricDashboardError("profile workers must be positive")
    profile_definition = (definitions.get("mechanistic_profiles") or {}).get(
        ENGINE_CAPACITY_PROFILE, {}
    )
    parameters = profile_definition.get("parameters") or (
        definitions.get("metrics", {}).get("U3", {}).get("parameters")
    )
    if not isinstance(parameters, Mapping):
        raise MetricDashboardError("engine profile parameters are missing")
    slot_limits = parameters["engine_slot_limits"]
    by_hash: dict[str, list[str]] = defaultdict(list)
    code_by_hash: dict[str, str] = {}
    for program_id, code in program_codes.items():
        if not isinstance(code, str):
            continue
        digest = _sha256_text(code)
        by_hash[digest].append(str(program_id))
        code_by_hash[digest] = code

    profiles_by_hash: dict[str, dict[str, Any]] = {}
    failures_by_hash: dict[str, str] = {}
    eval_build = HERE / "eval_build.py"
    takehome_dir = HERE / "vendor" / "takehome"
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(by_hash)))) as pool:
        futures = {
            pool.submit(
                _profile_code,
                code_by_hash[digest],
                eval_build=eval_build,
                takehome_dir=takehome_dir,
                slot_limits=slot_limits,
            ): digest
            for digest in by_hash
        }
        for future in as_completed(futures):
            digest = futures[future]
            try:
                profiles_by_hash[digest] = future.result()
            except Exception as exc:  # one invalid program cannot hide peers
                failures_by_hash[digest] = f"{type(exc).__name__}: {exc}"

    profiles: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for digest, program_ids in by_hash.items():
        for program_id in program_ids:
            if digest in profiles_by_hash:
                profiles[program_id] = {
                    "code_sha256": digest,
                    **profiles_by_hash[digest],
                }
            else:
                failures[program_id] = failures_by_hash.get(
                    digest, "profile unavailable"
                )
    return profiles, failures


def _source_event_number(
    attempt: Mapping[str, Any], names: Sequence[str]
) -> float | None:
    for event in reversed(tuple(attempt.get("source_events") or ())):
        if not isinstance(event, Mapping):
            continue
        containers = [event]
        for key in ("metadata", "usage", "cost", "timing"):
            value = event.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
        for container in containers:
            for name in names:
                value = _finite(container.get(name))
                if value is not None:
                    return value
    return None


def _recorded_change_count(
    row: Mapping[str, Any], attempt: Mapping[str, Any]
) -> int | None:
    recorded = row.get("recorded_change_count")
    if type(recorded) is int and recorded >= 0:
        return recorded
    changes = attempt.get("changes") or row.get("changes_excerpt")
    if not isinstance(changes, str) or not changes.strip():
        return None
    markers = re.findall(r"(?m)^\s*Change\s+\d+\s*:", changes)
    return len(markers) if markers else 1


def _failure_signature(
    row: Mapping[str, Any], attempt: Mapping[str, Any]
) -> str | None:
    if row.get("terminal_class") == "valid":
        return None
    for value in (
        attempt.get("failure_stage"),
        row.get("failure_stage"),
        attempt.get("terminal"),
        row.get("terminal_class"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown_failure"


def _attempt_timestamp(row: Mapping[str, Any], attempt: Mapping[str, Any]) -> float:
    timestamp = _finite(attempt.get("timestamp"))
    if timestamp is not None:
        return timestamp
    elapsed = _finite(row.get("elapsed_hours"))
    return (elapsed or 0.0) * 3600.0


def _code_similarity(
    left_program_id: str | None,
    right_program_id: str | None,
    fingerprints: Mapping[str, Mapping[Any, int]],
) -> float | None:
    if left_program_id not in fingerprints or right_program_id not in fingerprints:
        return None
    return weighted_multiset_jaccard(
        fingerprints[str(left_program_id)], fingerprints[str(right_program_id)]
    )


def build_metric_observations(
    dashboard_rows: Sequence[Mapping[str, Any]],
    *,
    history: Mapping[str, Any] | None,
    engine_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    engine_failures: Mapping[str, str] | None = None,
    live_program_ids: Iterable[str] | None = None,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    branch_maturity_steps: int = DEFAULT_BRANCH_MATURITY_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one header and one causal metric observation per terminal row."""
    if window_size <= 0 or branch_maturity_steps <= 0:
        raise MetricDashboardError("window and maturity sizes must be positive")
    definitions, definitions_sha = load_definitions(definitions_path)
    header = dashboard_header(definitions, definitions_sha)
    history_gate = float(header["history_gate_seconds"])
    minimum_reference_n = int(header["minimum_reference_n"])
    rows = [dict(row) for row in dashboard_rows]
    rows.sort(key=lambda item: int(item["seq"]))
    if not rows or any(type(row.get("seq")) is not int for row in rows):
        raise MetricDashboardError("dashboard rows require integer seq values")
    if len({int(row["seq"]) for row in rows}) != len(rows):
        raise MetricDashboardError("dashboard completion seq values must be unique")

    attempts = _attempts(history)
    codes = _program_codes(history)
    root_code = codes.get("root")
    fingerprints = (
        {
            program_id: signed_fingerprint(root_code, code)
            for program_id, code in codes.items()
        }
        if root_code is not None
        else {}
    )
    profiles = dict(engine_profiles or {})
    profile_failures = dict(engine_failures or {})
    live_program_id_set = (
        {str(item) for item in live_program_ids if isinstance(item, str)}
        if live_program_ids is not None
        else {
            str(row["program_id"])
            for row in rows
            if row.get("live_at_cutoff") is True
            and isinstance(row.get("program_id"), str)
        }
    )
    live_membership_known = live_program_ids is not None or any(
        isinstance(row.get("live_at_cutoff"), bool) for row in rows
    )
    metric_ids = set(definitions["metrics"])

    # Cutoff-scoped direct-reuse facts.  These are computed once over the
    # frozen causal prefix, just like G10/G11 descendant outcomes.
    gap_steps_by_seq: dict[int, float] = {}
    recent_frontier_steps: list[float] = []
    direct_reuse_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row in rows:
        source_seq = int(source_row["seq"])
        source_attempt = dict(attempts.get(source_seq) or {})
        if source_row.get("meaningful_frontier_improvement") is True:
            before = _finite(source_row.get("frontier_before_cycles"))
            after = _finite(source_row.get("verified_frontier_cycles"))
            if before is not None and after is not None and before > after:
                recent_frontier_steps.append(before - after)
        recent_step = (
            statistics.median(recent_frontier_steps[-5:])
            if recent_frontier_steps
            else None
        )
        source_cycles = _finite(source_row.get("cycles"))
        source_frontier = _finite(source_row.get("frontier_before_cycles"))
        if (
            source_row.get("eligible") is True
            and source_cycles is not None
            and source_frontier is not None
            and recent_step not in (None, 0.0)
        ):
            gap_steps_by_seq[source_seq] = (
                source_cycles - source_frontier
            ) / recent_step

        reused_ids = {
            str(value)
            for value in (
                source_row.get("parent_id"),
                *(source_row.get("inspiration_ids") or ()),
            )
            if isinstance(value, str) and value
        }
        for reused_id in reused_ids:
            direct_reuse_rows[reused_id].append(source_row)

    program_evidence: dict[str, str] = {"root": "root"}
    program_birth_seq: dict[str, int] = {"root": 0}
    ancestors_by_program: dict[str, set[str]] = {"root": set()}
    descendant_count: Counter[str] = Counter()
    previous_implementation_ids: list[str] = ["root"] if root_code is not None else []
    last_meaningful_at = _finite(history.get("started_at")) if history else 0.0
    progress_anchor_cycles: float | None = None
    dominant_parent_previous: str | None = None
    dominant_parent_dwell = 0
    observations: list[dict[str, Any]] = []
    terminal_prefix: list[dict[str, Any]] = []
    eligible_prefix: list[dict[str, Any]] = []
    failure_signatures_seen: set[str] = set()

    for row in rows:
        seq = int(row["seq"])
        attempt = dict(attempts.get(seq) or {})
        timestamp = _attempt_timestamp(row, attempt)
        elapsed_seconds = max(0.0, (_finite(row.get("elapsed_hours")) or 0.0) * 3600.0)
        evidence_id = str(row.get("evidence_id") or f"attempt:seq:{seq:08d}")
        program_id = row.get("program_id")
        program_id = (
            str(program_id) if isinstance(program_id, str) and program_id else None
        )
        row_live = row.get("live_at_cutoff")
        live_at_cutoff = (
            bool(program_id is not None and program_id in live_program_id_set)
            if live_program_ids is not None
            else row_live
            if isinstance(row_live, bool)
            else None
        )
        parent_id = row.get("parent_id")
        parent_id = str(parent_id) if isinstance(parent_id, str) and parent_id else None
        cycles = _finite(row.get("cycles"))
        parent_cycles = _finite(row.get("parent_cycles"))
        frontier_before = _finite(row.get("frontier_before_cycles"))
        if progress_anchor_cycles is None and frontier_before is not None:
            progress_anchor_cycles = frontier_before
        eligible = row.get("eligible") is True
        failure_signature = _failure_signature(row, attempt)
        row["_failure_signature"] = failure_signature
        row["_failure_repeated_before"] = (
            failure_signature in failure_signatures_seen
            if failure_signature is not None
            else False
        )
        terminal_prefix.append(row)
        if eligible:
            eligible_prefix.append(row)

        if program_id is not None:
            program_evidence[program_id] = evidence_id
            program_birth_seq.setdefault(program_id, seq)
            if parent_id is not None:
                ancestors = {parent_id} | ancestors_by_program.get(parent_id, set())
                ancestors_by_program[program_id] = ancestors
                for ancestor in ancestors:
                    descendant_count[ancestor] += 1

        if row.get("meaningful_frontier_improvement") is True:
            last_meaningful_at = timestamp

        # U1
        u1_value = cycles if eligible else None
        u1_available = eligible and cycles is not None

        # U3
        profile = profiles.get(program_id or "")
        if profile is not None:
            u3_value = _finite(profile.get("max_engine_slot_lower_bound"))
            u3_availability = _availability(
                status="available" if u3_value is not None else "unavailable",
                eligible=u3_value is not None,
                reason=None if u3_value is not None else "capacity_lower_bound_missing",
            )
        else:
            reason = profile_failures.get(program_id or "", "profile_not_materialized")
            u3_value = None
            u3_availability = _availability(
                status="unavailable", eligible=False, reason=reason
            )
        engine_lower_bounds = {
            engine: _finite(
                (profile or {}).get("engine_slot_lower_bounds", {}).get(engine)
            )
            for engine in ENGINE_METRIC_ORDER
        }
        emitted_bundle_count = _finite((profile or {}).get("emitted_bundle_count"))
        scratch_words = _finite((profile or {}).get("scratch_words"))
        recorded_change_count = _recorded_change_count(row, attempt)
        multi_edit_transition = row.get("multi_edit_transition")
        if not isinstance(multi_edit_transition, bool):
            multi_edit_transition = (
                recorded_change_count > 1 if recorded_change_count is not None else None
            )
        dependency_lower_bound = _finite(
            (profile or {}).get("dependency_critical_path_lower_bound")
        )
        combined_schedule_lower_bound = (
            max(u3_value, dependency_lower_bound)
            if u3_value is not None and dependency_lower_bound is not None
            else None
        )
        residual_to_modeled_lower_bounds = (
            cycles - combined_schedule_lower_bound
            if eligible
            and cycles is not None
            and combined_schedule_lower_bound is not None
            and cycles >= combined_schedule_lower_bound
            else None
        )
        engine_availability = {
            engine: _availability(
                status="available" if value is not None else "unavailable",
                eligible=value is not None,
                reason=(
                    None
                    if value is not None
                    else profile_failures.get(
                        program_id or "", "engine_profile_not_materialized"
                    )
                ),
            )
            for engine, value in engine_lower_bounds.items()
        }

        # U4
        u4_value = _source_event_number(
            attempt, ("model_cost_usd", "estimated_cost_usd", "cost_usd")
        )

        # U5
        required = {
            "terminal_class": row.get("terminal_class"),
            "evidence_id": evidence_id,
            "timestamp": timestamp,
        }
        if row.get("terminal_class") == "valid":
            required.update(
                {
                    "parent_id": parent_id,
                    "program_id": program_id,
                    "cycles_if_valid": cycles,
                }
            )
        present = sum(value is not None for value in required.values())
        u5_value = present / len(required)

        # B1/B2
        parent_gain = (
            parent_cycles - cycles
            if parent_cycles is not None and cycles is not None
            else None
        )
        frontier_gap = (
            cycles - frontier_before
            if cycles is not None and frontier_before is not None
            else None
        )
        authoritative_frontier_gap = frontier_gap if eligible else None
        authoritative_parent_gain = parent_gain if eligible else None
        b2_value = (
            frontier_gap / abs(frontier_before)
            if frontier_gap is not None and frontier_before not in (None, 0.0)
            else None
        )
        frontier_gain = (
            -authoritative_frontier_gap
            if authoritative_frontier_gap is not None
            else None
        )
        parent_gain_fraction = (
            authoritative_parent_gain / abs(parent_cycles)
            if authoritative_parent_gain is not None
            and parent_cycles not in (None, 0.0)
            else None
        )
        meaningful_relative_gain = (
            (progress_anchor_cycles - cycles) / abs(progress_anchor_cycles)
            if eligible
            and cycles is not None
            and progress_anchor_cycles not in (None, 0.0)
            else None
        )

        # B3
        parent_similarity = _code_similarity(parent_id, program_id, fingerprints)
        b3_value = 1.0 - parent_similarity if parent_similarity is not None else None

        parent_profile = profiles.get(parent_id or "")
        parent_emitted_bundle_count = _finite(
            (parent_profile or {}).get("emitted_bundle_count")
        )
        parent_scratch_words = _finite((parent_profile or {}).get("scratch_words"))
        delta_emitted_bundle_count = (
            emitted_bundle_count - parent_emitted_bundle_count
            if emitted_bundle_count is not None
            and parent_emitted_bundle_count is not None
            else None
        )
        delta_scratch_words = (
            scratch_words - parent_scratch_words
            if scratch_words is not None and parent_scratch_words is not None
            else None
        )
        engine_parent_lower_bounds = {
            engine: _finite(
                (parent_profile or {}).get("engine_slot_lower_bounds", {}).get(engine)
            )
            for engine in ENGINE_METRIC_ORDER
        }
        engine_lower_bound_deltas = {
            engine: (
                engine_lower_bounds[engine] - engine_parent_lower_bounds[engine]
                if engine_lower_bounds[engine] is not None
                and engine_parent_lower_bounds[engine] is not None
                else None
            )
            for engine in ENGINE_METRIC_ORDER
        }

        # B4
        frontier_program = row.get("F_reference_program_id")
        frontier_program = (
            str(frontier_program)
            if isinstance(frontier_program, str) and frontier_program
            else "root"
        )
        frontier_similarity = _code_similarity(
            frontier_program, program_id, fingerprints
        )
        b4_value = (
            1.0 - frontier_similarity if frontier_similarity is not None else None
        )

        # B5: strictly earlier implementations only.
        nearest_similarity = None
        nearest_program_id = None
        if program_id in fingerprints:
            candidates = [
                prior for prior in previous_implementation_ids if prior in fingerprints
            ]
            if candidates:
                nearest_program_id = min(
                    candidates,
                    key=lambda prior: (
                        -weighted_multiset_jaccard(
                            fingerprints[program_id], fingerprints[prior]
                        ),
                        prior,
                    ),
                )
                nearest_similarity = weighted_multiset_jaccard(
                    fingerprints[program_id], fingerprints[nearest_program_id]
                )
        if program_id in fingerprints:
            previous_implementation_ids.append(program_id)

        terminal_window = terminal_prefix[-window_size:]
        terminal_window_ids = [str(item.get("evidence_id")) for item in terminal_window]
        valid_window = [
            item for item in terminal_window if item.get("eligible") is True
        ]
        n = len(terminal_window)

        # R1
        closures = []
        for item in valid_window:
            item_cycles = _finite(item.get("cycles"))
            item_parent = _finite(item.get("parent_cycles"))
            item_frontier = _finite(item.get("frontier_before_cycles"))
            if item_cycles is None:
                continue
            if item_frontier not in (None, 0.0):
                parent_frontier_gap = (
                    item_parent - item_frontier if item_parent is not None else None
                )
                if parent_frontier_gap is not None and parent_frontier_gap > 0.0:
                    closure = (item_parent - item_cycles) / parent_frontier_gap
                    closures.append(min(1.0, max(0.0, closure)))
        r1_value = _ratio(sum(closures), n)

        # R2/R3: a complete eligible-state window only.
        state_window = eligible_prefix[-window_size:]
        state_window_ids = [str(item.get("evidence_id")) for item in state_window]
        state_programs = [
            str(item.get("program_id"))
            for item in state_window
            if str(item.get("program_id")) in fingerprints
        ]
        geometry_complete = (
            eligible
            and program_id in fingerprints
            and len(state_programs) == window_size
        )
        s_value = None
        f_value = None
        if geometry_complete:
            s_value = implementation_spread(
                [fingerprints[item] for item in state_programs]
            )
            if frontier_program in codes:
                f_value = implementation_spread(
                    [
                        signed_fingerprint(codes[frontier_program], codes[item])
                        for item in state_programs
                    ]
                )
        transition_fingerprints = []
        transition_complete = eligible and len(state_window) == window_size
        if transition_complete:
            for item in state_window:
                item_parent = item.get("parent_id")
                item_program = item.get("program_id")
                if item_parent not in codes or item_program not in codes:
                    transition_complete = False
                    transition_fingerprints = []
                    break
                transition_fingerprints.append(
                    signed_fingerprint(
                        codes[str(item_parent)], codes[str(item_program)]
                    )
                )
        transition_value = (
            implementation_spread(transition_fingerprints)
            if transition_complete
            else None
        )
        # R4
        parents = [
            str(item.get("parent_id"))
            for item in terminal_window
            if item.get("parent_id")
        ]
        _, parent_hhi, _, dominant_parent, _ = _hhi(parents)

        # R5
        first_frontier = (
            _finite(terminal_window[0].get("frontier_before_cycles"))
            if terminal_window
            else None
        )
        last_frontier = (
            _finite(terminal_window[-1].get("verified_frontier_cycles"))
            if terminal_window
            else None
        )
        r5_value = (
            first_frontier - last_frontier
            if first_frontier is not None and last_frontier is not None
            else None
        )
        r7_value = _ratio(
            sum(
                item.get("exact_duplicate") is True or item.get("no_op") is True
                for item in terminal_window
            ),
            n,
        )
        r8_value = _ratio(len(valid_window), n)
        islands = [
            str(item.get("target_island"))
            for item in terminal_window
            if item.get("target_island") is not None
        ]
        _, r9_value, _, _, _ = _hhi(islands)
        r10_value = _ratio(
            sum(
                item.get("meaningful_frontier_improvement") is True
                for item in valid_window
            ),
            len(valid_window),
        )
        comparable_valid = [
            item
            for item in valid_window
            if _finite(item.get("parent_gain_cycles")) is not None
        ]
        r11_value = _ratio(
            sum(
                _finite(item.get("parent_gain_cycles")) > 0.0
                for item in comparable_valid
            ),
            len(comparable_valid),
        )
        recent_frontier_steps = [
            (_finite(item.get("frontier_before_cycles")) or 0.0)
            - (_finite(item.get("verified_frontier_cycles")) or 0.0)
            for item in terminal_prefix
            if item.get("meaningful_frontier_improvement") is True
            and _finite(item.get("frontier_before_cycles")) is not None
            and _finite(item.get("verified_frontier_cycles")) is not None
            and _finite(item.get("frontier_before_cycles"))
            > _finite(item.get("verified_frontier_cycles"))
        ][-5:]
        recent_frontier_step = (
            statistics.median(recent_frontier_steps) if recent_frontier_steps else None
        )
        gap_in_recent_step_units = (
            authoritative_frontier_gap / recent_frontier_step
            if authoritative_frontier_gap is not None
            and recent_frontier_step not in (None, 0.0)
            else None
        )
        valid_gaps = [
            _finite(item.get("gap_to_frontier_cycles"))
            for item in valid_window
            if _finite(item.get("gap_to_frontier_cycles")) is not None
        ]
        r12_value = (
            _ratio(
                sum(gap <= recent_frontier_step for gap in valid_gaps),
                len(valid_gaps),
            )
            if recent_frontier_step is not None
            else None
        )
        r13_value = _ratio(
            sum(
                (_finite(item.get("parent_gain_cycles")) or 0.0) <= 0.0
                for item in comparable_valid
            ),
            len(comparable_valid),
        )
        r14_value = _ratio(
            sum(item.get("exact_duplicate") is True for item in terminal_window),
            n,
        )
        window_failure_signatures = [
            str(item["_failure_signature"])
            for item in terminal_window
            if item.get("_failure_signature") is not None
        ]
        failure_counts = Counter(window_failure_signatures)
        r15_value = (
            max(failure_counts.values()) / len(window_failure_signatures)
            if window_failure_signatures
            else None
        )
        r16_value = (
            sum(
                item.get("_failure_signature") is not None
                and item.get("_failure_repeated_before") is True
                for item in terminal_window
            )
            / len(window_failure_signatures)
            if window_failure_signatures
            else None
        )

        elite_programs: list[str] = []
        if recent_frontier_step is not None and frontier_program in codes:
            for item in valid_window:
                item_program = item.get("program_id")
                item_cycles = _finite(item.get("cycles"))
                item_frontier = _finite(item.get("frontier_before_cycles"))
                if (
                    isinstance(item_program, str)
                    and item_program in codes
                    and item_cycles is not None
                    and item_frontier is not None
                    and item_cycles - item_frontier <= recent_frontier_step
                ):
                    elite_programs.append(item_program)
        r17_value = (
            implementation_spread(
                [
                    signed_fingerprint(codes[frontier_program], codes[item])
                    for item in elite_programs
                ]
            )
            if n == window_size and len(elite_programs) >= 2
            else None
        )

        # Exact native membership is available only at the frozen audit cutoff.
        # Do not manufacture historical live sets for earlier dashboard rows.
        is_audit_cutoff = seq == int(rows[-1]["seq"])
        underexplored_ids: list[str] = []
        r18_value = None
        if (
            is_audit_cutoff
            and live_membership_known
            and n == window_size
            and recent_frontier_step is not None
        ):
            current_frontier = _finite(row.get("verified_frontier_cycles"))
            window_start_seq = int(terminal_window[0]["seq"])
            recently_expanded = {
                str(value)
                for item in terminal_window
                for value in (
                    item.get("parent_id"),
                    *(item.get("inspiration_ids") or ()),
                )
                if isinstance(value, str) and value
            }
            latest_by_program = {
                str(item["program_id"]): item
                for item in terminal_prefix
                if isinstance(item.get("program_id"), str)
                and item.get("eligible") is True
            }
            if current_frontier is not None:
                underexplored_ids = sorted(
                    program_id
                    for program_id in live_program_id_set
                    if program_id != frontier_program
                    and program_id in latest_by_program
                    and program_birth_seq.get(program_id, seq) < window_start_seq
                    and program_id not in recently_expanded
                    and _finite(latest_by_program[program_id].get("cycles")) is not None
                    and _finite(latest_by_program[program_id].get("cycles"))
                    - current_frontier
                    <= recent_frontier_step
                )
                r18_value = float(len(underexplored_ids))

        # G1
        g1_value = (
            timestamp - last_meaningful_at if last_meaningful_at is not None else None
        )

        # G2
        if dominant_parent == dominant_parent_previous:
            dominant_parent_dwell += 1
        else:
            dominant_parent_previous = dominant_parent
            dominant_parent_dwell = 1 if dominant_parent is not None else 0
        g2_value = float(dominant_parent_dwell)

        # G3
        mature = [
            item
            for item, born in program_birth_seq.items()
            if item != "root" and seq - born >= branch_maturity_steps
        ]
        surviving = [item for item in mature if descendant_count[item] > 0]
        g3_value = _ratio(len(surviving), len(mature))

        # G4/G5: causal history through the current terminal attempt.
        redundant_count = sum(
            item.get("exact_duplicate") is True or item.get("no_op") is True
            for item in terminal_prefix
        )
        eligible_history = [
            item for item in terminal_prefix if item.get("eligible") is True
        ]
        meaningful_history_count = sum(
            item.get("meaningful_frontier_improvement") is True
            for item in eligible_history
        )
        g4_value = _ratio(redundant_count, len(terminal_prefix))
        g5_value = _ratio(meaningful_history_count, len(eligible_history))
        g6_value = elapsed_seconds
        g7_value = float(len(terminal_prefix))
        g8_value = _finite(row.get("verified_frontier_cycles"))
        frontier_reference_seq = row.get("F_reference_seq")
        g9_value = (
            float(seq - frontier_reference_seq)
            if type(frontier_reference_seq) is int
            else None
        )
        g10_value = _finite(row.get("best_descendant_cycles"))
        g11_value = _finite(row.get("descendant_frontier_move_count"))
        direct_reuses = list(direct_reuse_rows.get(program_id or "", ()))
        direct_reuse_ids = [str(item.get("evidence_id")) for item in direct_reuses]
        g13_value = (
            float(len(direct_reuses))
            if program_id is not None and row.get("committed") is True
            else None
        )
        reuse_gaps = [
            (gap_steps_by_seq[int(item["seq"])], item)
            for item in direct_reuses
            if int(item["seq"]) in gap_steps_by_seq
        ]
        best_reuse = min(reuse_gaps, key=lambda item: item[0]) if reuse_gaps else None
        g14_value = best_reuse[0] if best_reuse is not None else None
        best_reuse_id = (
            str(best_reuse[1].get("evidence_id")) if best_reuse is not None else None
        )

        common_window_availability = _availability(
            status="available" if n == window_size else "warming_up",
            eligible=n == window_size,
            n_expected=window_size,
            n_used=n,
        )
        geometry_availability = _availability(
            status="available" if geometry_complete else "warming_up",
            eligible=geometry_complete,
            n_expected=window_size,
            n_used=len(state_programs),
        )
        transition_availability = _availability(
            status="available" if transition_complete else "warming_up",
            eligible=transition_complete,
            n_expected=window_size,
            n_used=len(transition_fingerprints),
        )
        island_availability = _availability(
            status=(
                "available"
                if n == window_size and r9_value is not None
                else "warming_up"
            ),
            eligible=n == window_size and r9_value is not None,
            n_expected=window_size,
            n_used=len(islands),
        )
        conversion_availability = _availability(
            status=(
                "available"
                if n == window_size and r10_value is not None
                else "warming_up"
            ),
            eligible=n == window_size and r10_value is not None,
            n_expected=window_size,
            n_used=len(valid_window),
        )
        observation = {
            "record_type": OBSERVATION_RECORD,
            "schema": SCHEMA,
            "definition_version": definitions["definition_version"],
            "definitions_sha256": definitions_sha,
            "cutoff_seq": seq,
            "cutoff_at": timestamp,
            "elapsed_seconds": elapsed_seconds,
            "evidence_id": evidence_id,
            "program_id": program_id,
            "live_at_cutoff": live_at_cutoff,
            "parent_id": parent_id,
            "direct_reuse_refs": [
                _compact_evidence_ref(item) for item in direct_reuse_ids
            ],
            "best_direct_reuse_ref": (
                _compact_evidence_ref(best_reuse_id)
                if best_reuse_id is not None
                else None
            ),
            "frontier_epoch": {
                "F_reference_seq": row.get("F_reference_seq"),
                "F_reference_program_id": frontier_program,
            },
            "evidence": {
                "trajectory": row.get("trajectory_location"),
                "code": row.get("code_location"),
            },
            "metrics": {
                "unary": {
                    "U1": _metric(
                        "U1",
                        u1_value,
                        availability=_availability(
                            status="available" if u1_available else "unavailable",
                            eligible=u1_available,
                            reason=None
                            if u1_available
                            else "no_authoritative_valid_cycles",
                        ),
                    ),
                    "U2": _metric(
                        "U2",
                        1.0 if eligible else 0.0,
                        availability=_availability(status="available", eligible=True),
                    ),
                    **(
                        {"U3": _metric("U3", u3_value, availability=u3_availability)}
                        if "U3" in metric_ids
                        else {}
                    ),
                    "U4": _metric(
                        "U4",
                        u4_value,
                        availability=_availability(
                            status=(
                                "available" if u4_value is not None else "unavailable"
                            ),
                            eligible=u4_value is not None,
                            reason=(
                                None
                                if u4_value is not None
                                else "model_cost_not_recorded"
                            ),
                        ),
                    ),
                    "U5": _metric(
                        "U5",
                        u5_value,
                        availability=_availability(
                            status="available"
                            if present == len(required)
                            else "partial",
                            eligible=present > 0,
                            n_expected=len(required),
                            n_used=present,
                        ),
                    ),
                    **{
                        f"U{index}": _metric(
                            f"U{index}",
                            engine_lower_bounds[engine],
                            availability=engine_availability[engine],
                        )
                        for index, engine in enumerate(ENGINE_METRIC_ORDER, start=6)
                        if f"U{index}" in metric_ids
                    },
                    **(
                        {
                            "U11": _metric(
                                "U11",
                                emitted_bundle_count,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if emitted_bundle_count is not None
                                        else "unavailable"
                                    ),
                                    eligible=emitted_bundle_count is not None,
                                    reason=(
                                        None
                                        if emitted_bundle_count is not None
                                        else profile_failures.get(
                                            program_id or "",
                                            "engine_profile_not_materialized",
                                        )
                                    ),
                                ),
                            )
                        }
                        if "U11" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U12": _metric(
                                "U12",
                                scratch_words,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if scratch_words is not None
                                        else "unavailable"
                                    ),
                                    eligible=scratch_words is not None,
                                    reason=(
                                        None
                                        if scratch_words is not None
                                        else profile_failures.get(
                                            program_id or "",
                                            "scratch_words_not_materialized",
                                        )
                                    ),
                                ),
                            )
                        }
                        if "U12" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U13": _metric(
                                "U13",
                                (
                                    1.0
                                    if program_id in live_program_id_set
                                    else 0.0
                                    if live_program_id_set and program_id is not None
                                    else None
                                ),
                                availability=_availability(
                                    status=(
                                        "available"
                                        if live_program_id_set
                                        and program_id is not None
                                        else "unavailable"
                                    ),
                                    eligible=bool(
                                        live_program_id_set and program_id is not None
                                    ),
                                    reason=(
                                        None
                                        if live_program_id_set
                                        and program_id is not None
                                        else "historical_live_set_unavailable"
                                    ),
                                ),
                            )
                        }
                        if "U13" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U14": _metric(
                                "U14",
                                1.0 if row.get("committed") is True else 0.0,
                                availability=_availability(
                                    status="available", eligible=True
                                ),
                            )
                        }
                        if "U14" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U15": _metric(
                                "U15",
                                dependency_lower_bound,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if dependency_lower_bound is not None
                                        else "unavailable"
                                    ),
                                    eligible=dependency_lower_bound is not None,
                                    reason=(
                                        None
                                        if dependency_lower_bound is not None
                                        else profile_failures.get(
                                            program_id or "",
                                            "dependency_profile_not_materialized",
                                        )
                                    ),
                                ),
                            )
                        }
                        if "U15" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U16": _metric(
                                "U16",
                                residual_to_modeled_lower_bounds,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if residual_to_modeled_lower_bounds is not None
                                        else "unavailable"
                                    ),
                                    eligible=residual_to_modeled_lower_bounds
                                    is not None,
                                    reason=(
                                        None
                                        if residual_to_modeled_lower_bounds is not None
                                        else "valid_cycles_and_schedule_bounds_required"
                                    ),
                                ),
                            )
                        }
                        if "U16" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U17": _metric(
                                "U17",
                                float(recorded_change_count)
                                if recorded_change_count is not None
                                else None,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if recorded_change_count is not None
                                        else "unavailable"
                                    ),
                                    eligible=recorded_change_count is not None,
                                    reason=(
                                        None
                                        if recorded_change_count is not None
                                        else "recorded_changes_unavailable"
                                    ),
                                ),
                            )
                        }
                        if "U17" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "U18": _metric(
                                "U18",
                                (
                                    1.0
                                    if multi_edit_transition is True
                                    else 0.0
                                    if multi_edit_transition is False
                                    else None
                                ),
                                availability=_availability(
                                    status=(
                                        "available"
                                        if multi_edit_transition is not None
                                        else "unavailable"
                                    ),
                                    eligible=multi_edit_transition is not None,
                                    reason=(
                                        None
                                        if multi_edit_transition is not None
                                        else "recorded_changes_unavailable"
                                    ),
                                ),
                            )
                        }
                        if "U18" in metric_ids
                        else {}
                    ),
                },
                "binary": {
                    "B1": _metric(
                        "B1",
                        parent_gain,
                        availability=_availability(
                            status="available"
                            if parent_gain is not None
                            else "unavailable",
                            eligible=parent_gain is not None,
                        ),
                        reference_ids=[program_evidence.get(parent_id or "", parent_id)]
                        if parent_id
                        else [],
                    ),
                    "B2": _metric(
                        "B2",
                        b2_value,
                        availability=_availability(
                            status=(
                                "available" if b2_value is not None else "unavailable"
                            ),
                            eligible=b2_value is not None,
                        ),
                        reference_ids=[
                            program_evidence.get(frontier_program, frontier_program)
                        ],
                    ),
                    "B3": _metric(
                        "B3",
                        b3_value,
                        availability=_availability(
                            status="available"
                            if parent_similarity is not None
                            else "unavailable",
                            eligible=parent_similarity is not None,
                        ),
                        reference_ids=[program_evidence.get(parent_id or "", parent_id)]
                        if parent_id
                        else [],
                    ),
                    "B4": _metric(
                        "B4",
                        b4_value,
                        availability=_availability(
                            status="available"
                            if frontier_similarity is not None
                            else "unavailable",
                            eligible=frontier_similarity is not None,
                        ),
                        reference_ids=[
                            program_evidence.get(frontier_program, frontier_program)
                        ],
                    ),
                    "B5": _metric(
                        "B5",
                        nearest_similarity,
                        availability=_availability(
                            status="available"
                            if nearest_similarity is not None
                            else "unavailable",
                            eligible=nearest_similarity is not None,
                        ),
                        reference_ids=[
                            program_evidence.get(
                                nearest_program_id or "", nearest_program_id
                            )
                        ]
                        if nearest_program_id
                        else [],
                    ),
                    **{
                        f"B{index}": _metric(
                            f"B{index}",
                            engine_lower_bound_deltas[engine],
                            availability=_availability(
                                status=(
                                    "available"
                                    if engine_lower_bound_deltas[engine] is not None
                                    else "unavailable"
                                ),
                                eligible=engine_lower_bound_deltas[engine] is not None,
                                reason=(
                                    None
                                    if engine_lower_bound_deltas[engine] is not None
                                    else "parent_or_child_engine_profile_unavailable"
                                ),
                            ),
                            reference_ids=(
                                [program_evidence.get(parent_id or "", parent_id)]
                                if parent_id
                                else []
                            ),
                        )
                        for index, engine in enumerate(ENGINE_METRIC_ORDER, start=6)
                        if f"B{index}" in metric_ids
                    },
                    **(
                        {
                            "B11": _metric(
                                "B11",
                                authoritative_frontier_gap,
                                availability=_availability(
                                    status="available"
                                    if authoritative_frontier_gap is not None
                                    else "unavailable",
                                    eligible=authoritative_frontier_gap is not None,
                                ),
                                reference_ids=[
                                    program_evidence.get(
                                        frontier_program, frontier_program
                                    )
                                ],
                            )
                        }
                        if "B11" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B12": _metric(
                                "B12",
                                frontier_gain,
                                availability=_availability(
                                    status="available"
                                    if frontier_gain is not None
                                    else "unavailable",
                                    eligible=frontier_gain is not None,
                                ),
                                reference_ids=[
                                    program_evidence.get(
                                        frontier_program, frontier_program
                                    )
                                ],
                            )
                        }
                        if "B12" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B13": _metric(
                                "B13",
                                parent_gain_fraction,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if parent_gain_fraction is not None
                                        else "unavailable"
                                    ),
                                    eligible=parent_gain_fraction is not None,
                                ),
                                reference_ids=(
                                    [program_evidence.get(parent_id or "", parent_id)]
                                    if parent_id
                                    else []
                                ),
                            )
                        }
                        if "B13" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B14": _metric(
                                "B14",
                                meaningful_relative_gain,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if meaningful_relative_gain is not None
                                        else "unavailable"
                                    ),
                                    eligible=meaningful_relative_gain is not None,
                                    reason=(
                                        None
                                        if meaningful_relative_gain is not None
                                        else "progress_anchor_relative_gain_unavailable"
                                    ),
                                ),
                            )
                        }
                        if "B14" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B15": _metric(
                                "B15",
                                delta_emitted_bundle_count,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if delta_emitted_bundle_count is not None
                                        else "unavailable"
                                    ),
                                    eligible=delta_emitted_bundle_count is not None,
                                    reason=(
                                        None
                                        if delta_emitted_bundle_count is not None
                                        else "parent_or_child_engine_profile_unavailable"
                                    ),
                                ),
                                reference_ids=(
                                    [program_evidence.get(parent_id or "", parent_id)]
                                    if parent_id
                                    else []
                                ),
                            )
                        }
                        if "B15" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B16": _metric(
                                "B16",
                                delta_scratch_words,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if delta_scratch_words is not None
                                        else "unavailable"
                                    ),
                                    eligible=delta_scratch_words is not None,
                                    reason=(
                                        None
                                        if delta_scratch_words is not None
                                        else "parent_or_child_scratch_profile_unavailable"
                                    ),
                                ),
                                reference_ids=(
                                    [program_evidence.get(parent_id or "", parent_id)]
                                    if parent_id
                                    else []
                                ),
                            )
                        }
                        if "B16" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "B17": _metric(
                                "B17",
                                gap_in_recent_step_units,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if gap_in_recent_step_units is not None
                                        else "unavailable"
                                    ),
                                    eligible=gap_in_recent_step_units is not None,
                                    reason=(
                                        None
                                        if gap_in_recent_step_units is not None
                                        else "valid_gap_and_recent_frontier_step_required"
                                    ),
                                ),
                                reference_ids=[
                                    program_evidence.get(
                                        frontier_program, frontier_program
                                    )
                                ],
                            )
                        }
                        if "B17" in metric_ids
                        else {}
                    ),
                },
                "regional": {
                    "R1": _metric(
                        "R1",
                        r1_value,
                        availability=common_window_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R2": _metric(
                        "R2",
                        f_value,
                        availability=geometry_availability,
                        window_ids=state_window_ids,
                        reference_ids=[
                            program_evidence.get(frontier_program, frontier_program)
                        ],
                    ),
                    "R3": _metric(
                        "R3",
                        s_value,
                        availability=geometry_availability,
                        window_ids=state_window_ids,
                        reference_ids=["root"],
                    ),
                    "R4": _metric(
                        "R4",
                        parent_hhi,
                        availability=common_window_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R5": _metric(
                        "R5",
                        r5_value,
                        availability=common_window_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R6": _metric(
                        "R6",
                        transition_value,
                        availability=transition_availability,
                        window_ids=state_window_ids,
                    ),
                    "R7": _metric(
                        "R7",
                        r7_value,
                        availability=common_window_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R8": _metric(
                        "R8",
                        r8_value,
                        availability=common_window_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R9": _metric(
                        "R9",
                        r9_value,
                        availability=island_availability,
                        window_ids=terminal_window_ids,
                    ),
                    "R10": _metric(
                        "R10",
                        r10_value,
                        availability=conversion_availability,
                        window_ids=terminal_window_ids,
                    ),
                    **(
                        {
                            "R11": _metric(
                                "R11",
                                r11_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if n == window_size and r11_value is not None
                                        else "warming_up"
                                    ),
                                    eligible=n == window_size and r11_value is not None,
                                    n_expected=window_size,
                                    n_used=len(comparable_valid),
                                ),
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R11" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R12": _metric(
                                "R12",
                                r12_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if n == window_size and r12_value is not None
                                        else "warming_up"
                                    ),
                                    eligible=n == window_size and r12_value is not None,
                                    n_expected=window_size,
                                    n_used=len(valid_gaps),
                                    reason=(
                                        None
                                        if recent_frontier_step is not None
                                        else "recent_frontier_step_unavailable"
                                    ),
                                ),
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R12" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R13": _metric(
                                "R13",
                                r13_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if n == window_size and r13_value is not None
                                        else "warming_up"
                                    ),
                                    eligible=n == window_size and r13_value is not None,
                                    n_expected=window_size,
                                    n_used=len(comparable_valid),
                                ),
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R13" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R14": _metric(
                                "R14",
                                r14_value,
                                availability=common_window_availability,
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R14" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R15": _metric(
                                "R15",
                                r15_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if n == window_size and r15_value is not None
                                        else "warming_up"
                                        if n < window_size
                                        else "unavailable"
                                    ),
                                    eligible=n == window_size and r15_value is not None,
                                    n_expected=window_size,
                                    n_used=len(window_failure_signatures),
                                    reason=(
                                        None
                                        if r15_value is not None
                                        else "no_failed_attempts_in_window"
                                    ),
                                ),
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R15" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R16": _metric(
                                "R16",
                                r16_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if n == window_size and r16_value is not None
                                        else "warming_up"
                                        if n < window_size
                                        else "unavailable"
                                    ),
                                    eligible=n == window_size and r16_value is not None,
                                    n_expected=window_size,
                                    n_used=len(window_failure_signatures),
                                    reason=(
                                        None
                                        if r16_value is not None
                                        else "no_failed_attempts_in_window"
                                    ),
                                ),
                                window_ids=terminal_window_ids,
                            )
                        }
                        if "R16" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R17": _metric(
                                "R17",
                                r17_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if r17_value is not None
                                        else "warming_up"
                                        if n < window_size
                                        else "unavailable"
                                    ),
                                    eligible=r17_value is not None,
                                    n_expected=window_size,
                                    n_used=len(elite_programs),
                                    reason=(
                                        None
                                        if r17_value is not None
                                        else "at_least_two_near_frontier_valid_codes_required"
                                    ),
                                ),
                                window_ids=terminal_window_ids,
                                reference_ids=[
                                    program_evidence.get(
                                        frontier_program, frontier_program
                                    )
                                ],
                            )
                        }
                        if "R17" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "R18": _metric(
                                "R18",
                                r18_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if r18_value is not None
                                        else "warming_up"
                                        if n < window_size
                                        else "unavailable"
                                    ),
                                    eligible=r18_value is not None,
                                    n_expected=window_size,
                                    n_used=(
                                        len(underexplored_ids)
                                        if r18_value is not None
                                        else 0
                                    ),
                                    reason=(
                                        None
                                        if r18_value is not None
                                        else "exact_live_membership_available_only_at_audit_cutoff"
                                        if not is_audit_cutoff
                                        or not live_membership_known
                                        else "recent_frontier_step_unavailable"
                                    ),
                                ),
                                window_ids=terminal_window_ids,
                                reference_ids=[
                                    program_evidence.get(item, item)
                                    for item in underexplored_ids
                                ],
                            )
                        }
                        if "R18" in metric_ids
                        else {}
                    ),
                },
                "global": {
                    "G1": _metric(
                        "G1",
                        g1_value,
                        availability=_availability(status="available", eligible=True),
                    ),
                    "G2": _metric(
                        "G2",
                        g2_value,
                        availability=_availability(status="available", eligible=True),
                        window_ids=terminal_window_ids,
                    ),
                    "G3": _metric(
                        "G3",
                        g3_value,
                        availability=_availability(
                            status="available" if mature else "warming_up",
                            eligible=bool(mature),
                            n_expected=branch_maturity_steps,
                            n_used=min(seq, branch_maturity_steps),
                        ),
                        reference_ids=[
                            program_evidence.get(frontier_program, frontier_program)
                        ],
                    ),
                    "G4": _metric(
                        "G4",
                        g4_value,
                        availability=_availability(status="available", eligible=True),
                    ),
                    "G5": _metric(
                        "G5",
                        g5_value,
                        availability=_availability(
                            status="available"
                            if g5_value is not None
                            else "warming_up",
                            eligible=g5_value is not None,
                        ),
                    ),
                    **(
                        {
                            "G6": _metric(
                                "G6",
                                g6_value,
                                availability=_availability(
                                    status="available", eligible=True
                                ),
                            )
                        }
                        if "G6" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G7": _metric(
                                "G7",
                                g7_value,
                                availability=_availability(
                                    status="available", eligible=True
                                ),
                            )
                        }
                        if "G7" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G8": _metric(
                                "G8",
                                g8_value,
                                availability=_availability(
                                    status="available"
                                    if g8_value is not None
                                    else "unavailable",
                                    eligible=g8_value is not None,
                                ),
                            )
                        }
                        if "G8" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G9": _metric(
                                "G9",
                                g9_value,
                                availability=_availability(
                                    status="available"
                                    if g9_value is not None
                                    else "unavailable",
                                    eligible=g9_value is not None,
                                ),
                            )
                        }
                        if "G9" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G10": _metric(
                                "G10",
                                g10_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if g10_value is not None
                                        else "unavailable"
                                    ),
                                    eligible=g10_value is not None,
                                    reason=(
                                        None
                                        if g10_value is not None
                                        else "no_valid_descendant_by_audit_cutoff"
                                    ),
                                ),
                            )
                        }
                        if "G10" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G11": _metric(
                                "G11",
                                g11_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if g11_value is not None
                                        else "unavailable"
                                    ),
                                    eligible=g11_value is not None,
                                    reason=(
                                        None
                                        if g11_value is not None
                                        else "descendant_outcome_unavailable"
                                    ),
                                ),
                            )
                        }
                        if "G11" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G13": _metric(
                                "G13",
                                g13_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if g13_value is not None
                                        else "unavailable"
                                    ),
                                    eligible=g13_value is not None,
                                    reason=(
                                        None
                                        if g13_value is not None
                                        else "committed_program_required"
                                    ),
                                ),
                                reference_ids=direct_reuse_ids,
                            )
                        }
                        if "G13" in metric_ids
                        else {}
                    ),
                    **(
                        {
                            "G14": _metric(
                                "G14",
                                g14_value,
                                availability=_availability(
                                    status=(
                                        "available"
                                        if g14_value is not None
                                        else "unavailable"
                                    ),
                                    eligible=g14_value is not None,
                                    reason=(
                                        None
                                        if g14_value is not None
                                        else "no_valid_direct_reuse_with_recent_step_scale"
                                    ),
                                ),
                                reference_ids=(
                                    [best_reuse_id] if best_reuse_id else []
                                ),
                            )
                        }
                        if "G14" in metric_ids
                        else {}
                    ),
                },
            },
        }
        for scope_metrics in observation["metrics"].values():
            for unused_metric_id in set(scope_metrics) - metric_ids:
                del scope_metrics[unused_metric_id]
        if ENGINE_CAPACITY_PROFILE in (definitions.get("mechanistic_profiles") or {}):
            dominant_engines = [
                str(engine)
                for engine in (profile or {}).get(
                    "max_engine_slot_lower_bound_engines", ()
                )
                if engine in ENGINE_METRIC_ORDER
            ]
            profile_values = [
                u3_value,
                dominant_engines,
                [engine_lower_bounds[engine] for engine in ENGINE_METRIC_ORDER],
                [engine_lower_bound_deltas[engine] for engine in ENGINE_METRIC_ORDER],
                dependency_lower_bound,
                residual_to_modeled_lower_bounds,
            ]
            current_complete = (
                u3_value is not None
                and bool(dominant_engines)
                and all(
                    engine_lower_bounds[engine] is not None
                    for engine in ENGINE_METRIC_ORDER
                )
                and dependency_lower_bound is not None
            )
            transition_complete = all(
                engine_lower_bound_deltas[engine] is not None
                for engine in ENGINE_METRIC_ORDER
            )
            profile_status = (
                "available"
                if current_complete
                and transition_complete
                and residual_to_modeled_lower_bounds is not None
                else "partial"
                if profile is not None
                else "unavailable"
            )
            observation["mechanistic_profiles"] = {
                ENGINE_CAPACITY_PROFILE: {
                    "status": profile_status,
                    "values": profile_values,
                }
            }
        observations.append(observation)
        if row.get("meaningful_frontier_improvement") is True:
            updated_anchor = _finite(row.get("verified_frontier_cycles"))
            if updated_anchor is not None:
                progress_anchor_cycles = updated_anchor
        if failure_signature is not None:
            failure_signatures_seen.add(failure_signature)

    _attach_causal_scaling(
        observations,
        definitions=definitions,
        history_gate_seconds=history_gate,
        minimum_reference_n=minimum_reference_n,
    )
    return header, observations


def _attach_causal_scaling(
    observations: Sequence[dict[str, Any]],
    *,
    definitions: Mapping[str, Any],
    history_gate_seconds: float,
    minimum_reference_n: int,
) -> None:
    global_reference: dict[str, list[float]] = defaultdict(list)
    epoch_reference: dict[tuple[Any, Any], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metric_definitions = definitions["metrics"]
    for observation in observations:
        epoch_doc = observation["frontier_epoch"]
        epoch = (
            epoch_doc.get("F_reference_seq"),
            epoch_doc.get("F_reference_program_id"),
        )
        pending: list[tuple[str, float]] = []
        for scope in ("unary", "binary", "regional", "global"):
            for metric_id, metric in observation["metrics"][scope].items():
                if metric["availability"].get("eligible") is not True:
                    continue
                value = _finite(metric.get("value"))
                if value is None:
                    continue
                reference_kind = metric_definitions[metric_id].get(
                    "scaling_reference", "global"
                )
                if reference_kind == "audit_cutoff":
                    metric["availability"]["history_gate_open"] = (
                        observation["elapsed_seconds"] >= history_gate_seconds
                    )
                    metric["availability"]["reference_n"] = 0
                    continue
                reference = (
                    epoch_reference[epoch][metric_id]
                    if reference_kind == "frontier_epoch"
                    else global_reference[metric_id]
                )
                gate_open = observation["elapsed_seconds"] >= history_gate_seconds
                enough_history = len(reference) >= minimum_reference_n
                if gate_open and enough_history:
                    metric["percentile"] = _percentile(value, reference)
                    metric["robust_z"] = _robust_z(value, reference)
                metric["availability"]["history_gate_open"] = gate_open
                metric["availability"]["reference_n"] = len(reference)
                pending.append((metric_id, value))

        # Add current values only after all metrics used strictly earlier history.
        for metric_id, value in pending:
            global_reference[metric_id].append(value)
            epoch_reference[epoch][metric_id].append(value)


def _metrics_by_id(observation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for scope in observation.get("metrics", {}).values():
        if not isinstance(scope, Mapping):
            continue
        for metric_id, metric in scope.items():
            if isinstance(metric, Mapping):
                result[str(metric_id)] = metric
    return result


def _compact_evidence_ref(value: Any) -> int | str:
    text = str(value)
    prefix = "attempt:seq:"
    suffix = text[len(prefix) :] if text.startswith(prefix) else ""
    return int(suffix) if suffix.isdigit() else text


def compact_observation(
    header: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the information-equivalent, low-repetition dashboard wire row."""
    metrics = _metrics_by_id(observation)
    metric_order = [str(item) for item in header["metric_order"]]
    if set(metrics) != set(metric_order):
        raise MetricDashboardError("observation metrics do not match dashboard header")

    values: list[Any] = []
    percentiles: list[Any] = []
    robust_z: list[Any] = []
    status_codes: list[str] = []
    reference_counts: list[Any] = []
    availability_details: dict[str, dict[str, Any]] = {}
    window_membership: dict[str, list[str]] = {}
    references: dict[str, list[int | str]] = {}
    row_gate_open = float(observation.get("elapsed_seconds") or 0.0) >= float(
        header["history_gate_seconds"]
    )

    for metric_id in metric_order:
        metric = metrics[metric_id]
        availability = dict(metric.get("availability") or {})
        status = str(availability.get("status") or "unavailable")
        if status not in STATUS_CODES:
            raise MetricDashboardError(f"unsupported availability status: {status}")
        status_code = STATUS_CODES[status]
        expected_eligible = STATUS_CODE_DEFINITIONS[status_code]["eligible"]
        if bool(availability.get("eligible")) != expected_eligible:
            raise MetricDashboardError(
                f"{metric_id} availability status/eligibility disagree"
            )
        recorded_gate = availability.get("history_gate_open")
        if recorded_gate is not None and bool(recorded_gate) != row_gate_open:
            raise MetricDashboardError(f"{metric_id} history gate disagrees with row")
        values.append(metric.get("value"))
        percentiles.append(metric.get("percentile"))
        robust_z.append(metric.get("robust_z"))
        status_codes.append(status_code)
        reference_counts.append(availability.get("reference_n"))
        source = METRIC_WINDOW_SOURCE.get(metric_id)
        detail = {
            key: availability[key]
            for key in ("n_expected", "n_used", "reason")
            if key in availability
        }
        if detail.get("n_expected") == header.get("window_size") and source:
            window_size = len(metric.get("window_evidence_ids") or ())
            if detail.get("n_used") == window_size:
                detail.pop("n_expected", None)
                detail.pop("n_used", None)
        if detail:
            availability_details[metric_id] = detail

        window_ids = [str(item) for item in metric.get("window_evidence_ids") or ()]
        if source:
            previous = window_membership.setdefault(source, window_ids)
            if previous != window_ids:
                raise MetricDashboardError(
                    f"{metric_id} window evidence disagrees with {source}"
                )
        elif window_ids:
            raise MetricDashboardError(f"{metric_id} has an undeclared evidence window")

        role = METRIC_REFERENCE_ROLE.get(metric_id)
        reference_ids = [str(item) for item in metric.get("reference_ids") or ()]
        if role:
            compact_reference_ids = [
                _compact_evidence_ref(item) for item in reference_ids
            ]
            previous = references.setdefault(role, compact_reference_ids)
            if previous != compact_reference_ids:
                raise MetricDashboardError(
                    f"{metric_id} references disagree with {role}"
                )
        elif reference_ids:
            raise MetricDashboardError(f"{metric_id} has an undeclared reference role")

    nearest_history = references.get("nearest_history") or []
    if len(nearest_history) > 1:
        raise MetricDashboardError("nearest-history reference is not singular")
    direct_reuse_refs = references.get("direct_native_reuse") or []
    best_direct_reuse = references.get("best_direct_native_reuse") or []
    if len(best_direct_reuse) > 1:
        raise MetricDashboardError("best direct-reuse reference is not singular")

    result = {
        "cutoff_seq": observation.get("cutoff_seq"),
        "elapsed_seconds": observation.get("elapsed_seconds"),
        "program_id": observation.get("program_id"),
        "live_at_cutoff": observation.get("live_at_cutoff"),
        "parent_id": observation.get("parent_id"),
        "frontier_epoch": observation.get("frontier_epoch"),
        "evidence": observation.get("evidence"),
        "nearest_history_ref": nearest_history[0] if nearest_history else None,
        "direct_reuse_refs": direct_reuse_refs,
        "best_direct_reuse_ref": (best_direct_reuse[0] if best_direct_reuse else None),
        "metric_columns": [
            values,
            percentiles,
            robust_z,
            "".join(status_codes),
            reference_counts,
        ],
    }
    if availability_details:
        result["availability_details"] = availability_details
    profile_order = [
        str(item) for item in header.get("mechanistic_profile_order") or ()
    ]
    if profile_order:
        profiles = observation.get("mechanistic_profiles") or {}
        if not isinstance(profiles, Mapping) or set(profiles) != set(profile_order):
            raise MetricDashboardError(
                "observation mechanistic profiles do not match dashboard header"
            )
        profile_values: list[list[Any]] = []
        profile_statuses: list[str] = []
        columns_by_profile = header["mechanistic_profile_columns"]
        for profile_id in profile_order:
            profile = profiles[profile_id]
            if not isinstance(profile, Mapping):
                raise MetricDashboardError(f"{profile_id} profile is invalid")
            status = str(profile.get("status") or "unavailable")
            if status not in STATUS_CODES:
                raise MetricDashboardError(
                    f"{profile_id} profile has unsupported status: {status}"
                )
            values_for_profile = profile.get("values")
            if not isinstance(values_for_profile, list) or len(
                values_for_profile
            ) != len(columns_by_profile[profile_id]):
                raise MetricDashboardError(f"{profile_id} profile values are invalid")
            profile_values.append(values_for_profile)
            profile_statuses.append(STATUS_CODES[status])
        result["mechanistic_profile_values"] = profile_values
        result["mechanistic_profile_statuses"] = "".join(profile_statuses)
    return result


def serialize_dashboard_jsonl(
    header: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> str:
    records = [header, *(compact_observation(header, item) for item in observations)]
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def write_dashboard_jsonl(
    output: str | Path,
    header: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialize_dashboard_jsonl(header, observations), encoding="utf-8")


def _llm_number(value: Any) -> Any:
    """Quantize agent-view numbers while the canonical dashboard stays exact."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MetricDashboardError("LLM dashboard contains a non-finite number")
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [_llm_number(item) for item in value]
    return value


def serialize_llm_dashboard_jsonl(
    header: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> str:
    """Encode validated dashboard rows for a lower-token agent read path."""
    metric_order = [str(item) for item in header["metric_order"]]
    metric_index = {metric_id: index for index, metric_id in enumerate(metric_order)}
    program_to_seq = {
        str(row["program_id"]): int(row["cutoff_seq"])
        for row in observations
        if row.get("program_id") is not None
    }
    reasons = sorted(
        {
            str(detail["reason"])
            for row in observations
            for detail in (row.get("availability_details") or {}).values()
            if "reason" in detail
        }
    )
    reason_index = {reason: index for index, reason in enumerate(reasons)}
    row_columns = [
        "cutoff_seq",
        "elapsed_seconds",
        "program_id",
        "live_at_cutoff",
        "parent_ref",
        "frontier_ref",
        "nearest_history_ref",
        "direct_reuse_refs",
        "best_direct_reuse_ref",
        "code_available",
        "metric_columns",
        "mechanistic_profile_values",
        "mechanistic_profile_statuses",
        "availability_details",
    ]
    llm_header = {
        "record_type": "dashboard_llm_header",
        "schema": LLM_SCHEMA,
        "source_schema": header["schema"],
        "definition_version": header["definition_version"],
        "definitions_sha256": header["definitions_sha256"],
        "metric_order": metric_order,
        "metric_columns": header["observation_metric_columns"],
        "status_codes": header["status_codes"],
        "metric_window_source": header.get("metric_window_source", {}),
        "metric_reference_role": header.get("metric_reference_role", {}),
        "window_size": header["window_size"],
        "history_gate_seconds": header["history_gate_seconds"],
        "minimum_reference_n": header["minimum_reference_n"],
        "row_columns": row_columns,
        "live_at_cutoff_semantics": (
            "true=natively available at the frozen cutoff; "
            "false=historical only; null=unknown"
        ),
        "availability_detail_columns": [
            "metric_index",
            "n_used",
            "n_expected",
            "reason_code",
        ],
        "availability_reason_codes": reasons,
        "reference_encoding": (
            "An integer N refers to the row with cutoff_seq=N; other values "
            "are literal program or evidence references."
        ),
        "trajectory_path": "search_history/seq_{cutoff_seq:08d}/trajectory.jsonl",
        "code_path": "search_history/seq_{cutoff_seq:08d}/code.py",
        "numeric_precision": "six significant digits",
    }
    if header.get("mechanistic_profile_order"):
        llm_header["mechanistic_profile_order"] = header["mechanistic_profile_order"]
        llm_header["mechanistic_profile_columns"] = header[
            "mechanistic_profile_columns"
        ]

    rows: list[list[Any]] = []
    for observation in observations:
        seq = int(observation["cutoff_seq"])
        evidence = observation.get("evidence") or {}
        trajectory_path = f"search_history/seq_{seq:08d}/trajectory.jsonl"
        code_path = f"search_history/seq_{seq:08d}/code.py"
        if evidence.get("trajectory") != trajectory_path:
            raise MetricDashboardError(
                f"dashboard row {seq} has a nonstandard trajectory path"
            )
        if evidence.get("code") not in (None, code_path):
            raise MetricDashboardError(
                f"dashboard row {seq} has a nonstandard code path"
            )

        parent_id = observation.get("parent_id")
        parent_ref = program_to_seq.get(str(parent_id), parent_id)
        frontier = observation.get("frontier_epoch") or {}
        frontier_ref = frontier.get("F_reference_seq")
        if type(frontier_ref) is not int:
            frontier_program = frontier.get("F_reference_program_id")
            frontier_ref = program_to_seq.get(str(frontier_program), frontier_program)

        details = []
        for metric_id, detail in sorted(
            (observation.get("availability_details") or {}).items(),
            key=lambda item: metric_index[item[0]],
        ):
            unknown = set(detail) - {"n_expected", "n_used", "reason"}
            if unknown:
                raise MetricDashboardError(
                    f"dashboard row {seq} has unsupported availability details"
                )
            details.append(
                [
                    metric_index[metric_id],
                    detail.get("n_used"),
                    detail.get("n_expected"),
                    reason_index.get(str(detail["reason"]))
                    if "reason" in detail
                    else None,
                ]
            )
        rows.append(
            [
                seq,
                _llm_number(observation.get("elapsed_seconds")),
                observation.get("program_id"),
                observation.get("live_at_cutoff"),
                parent_ref,
                frontier_ref,
                observation.get("nearest_history_ref"),
                observation.get("direct_reuse_refs") or [],
                observation.get("best_direct_reuse_ref"),
                evidence.get("code") is not None,
                _llm_number(observation.get("metric_columns")),
                _llm_number(observation.get("mechanistic_profile_values")),
                observation.get("mechanistic_profile_statuses"),
                details,
            ]
        )

    return "".join(
        json.dumps(
            record,
            sort_keys=isinstance(record, dict),
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in [llm_header, *rows]
    )


def write_llm_dashboard_jsonl(
    output: str | Path,
    header: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        serialize_llm_dashboard_jsonl(header, observations), encoding="utf-8"
    )


def write_metric_definitions(
    output_directory: str | Path,
    *,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
) -> Path:
    """Materialize the one immutable definition source beside dashboard.jsonl."""
    source = Path(definitions_path)
    definitions, digest = load_definitions(source)
    raw = source.read_text(encoding="utf-8")
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise MetricDashboardError("metric definitions changed while materializing")
    target = Path(output_directory) / "metric_definitions.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    del definitions
    target.write_text(raw, encoding="utf-8")
    return target


def validate_dashboard_jsonl(
    path: str | Path,
    *,
    definitions_path: str | Path = DEFAULT_DEFINITIONS,
    expected_cutoff_seq: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the two-file contract before exposing it to an agent."""
    definitions, definitions_sha = load_definitions(definitions_path)
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != HEADER_RECORD:
        raise MetricDashboardError("dashboard has no valid header")
    header = records[0]
    if header.get("schema") != SCHEMA:
        raise MetricDashboardError("dashboard schema mismatch")
    if header.get("definition_version") != definitions.get("definition_version"):
        raise MetricDashboardError("dashboard definition version mismatch")
    if header.get("definitions_sha256") != definitions_sha:
        raise MetricDashboardError("dashboard definition digest mismatch")
    metric_order = list(definitions["metrics"])
    if header.get("metric_order") != metric_order:
        raise MetricDashboardError("dashboard metric order mismatch")
    profile_definitions = definitions.get("mechanistic_profiles") or {}
    profile_order = list(profile_definitions)
    if profile_order:
        expected_columns = {
            profile_id: list(definition["columns"])
            for profile_id, definition in profile_definitions.items()
        }
        if header.get("mechanistic_profile_order") != profile_order:
            raise MetricDashboardError("dashboard mechanistic profile order mismatch")
        if header.get("mechanistic_profile_columns") != expected_columns:
            raise MetricDashboardError("dashboard mechanistic profile columns mismatch")
    elif header.get("mechanistic_profile_order") or header.get(
        "mechanistic_profile_columns"
    ):
        raise MetricDashboardError("dashboard has undeclared mechanistic profiles")

    observations = records[1:]
    seqs: list[int] = []
    for record in observations:
        seq = record.get("cutoff_seq")
        if type(seq) is not int:
            raise MetricDashboardError("dashboard observation cutoff is invalid")
        seqs.append(seq)
        columns = record.get("metric_columns")
        if not isinstance(columns, list) or len(columns) != 5:
            raise MetricDashboardError("dashboard metric columns are invalid")
        values, percentiles, robust_values, statuses, reference_counts = columns
        if any(
            not isinstance(column, list) or len(column) != len(metric_order)
            for column in (values, percentiles, robust_values, reference_counts)
        ):
            raise MetricDashboardError("dashboard metric column length mismatch")
        if (
            not isinstance(statuses, str)
            or len(statuses) != len(metric_order)
            or any(code not in STATUS_CODE_DEFINITIONS for code in statuses)
        ):
            raise MetricDashboardError("dashboard metric status column is invalid")
        if profile_order:
            profile_values = record.get("mechanistic_profile_values")
            profile_statuses = record.get("mechanistic_profile_statuses")
            if (
                not isinstance(profile_values, list)
                or len(profile_values) != len(profile_order)
                or any(
                    not isinstance(values, list)
                    or len(values) != len(profile_definitions[profile_id]["columns"])
                    for profile_id, values in zip(profile_order, profile_values)
                )
            ):
                raise MetricDashboardError(
                    "dashboard mechanistic profile values are invalid"
                )
            if (
                not isinstance(profile_statuses, str)
                or len(profile_statuses) != len(profile_order)
                or any(code not in STATUS_CODE_DEFINITIONS for code in profile_statuses)
            ):
                raise MetricDashboardError(
                    "dashboard mechanistic profile statuses are invalid"
                )
        elif "mechanistic_profile_values" in record or (
            "mechanistic_profile_statuses" in record
        ):
            raise MetricDashboardError(
                "dashboard row has undeclared mechanistic profiles"
            )
    if seqs != sorted(seqs) or len(seqs) != len(set(seqs)):
        raise MetricDashboardError("dashboard observation prefix is invalid")
    if expected_cutoff_seq is not None and (
        not seqs or seqs[-1] != expected_cutoff_seq
    ):
        raise MetricDashboardError("dashboard cutoff does not match controller cutoff")
    return dict(header), [dict(item) for item in observations]


def append_dashboard_observations(
    output: str | Path,
    header: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> int:
    """Append only unseen seqs; reject definition drift or prefix disorder."""
    target = Path(output)
    if not target.exists():
        write_dashboard_jsonl(target, header, observations)
        return len(observations)
    records = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != HEADER_RECORD:
        raise MetricDashboardError("existing dashboard has no valid header")
    for field in ("schema", "definition_version", "definitions_sha256"):
        if records[0].get(field) != header.get(field):
            raise MetricDashboardError(f"dashboard header drift: {field}")
    existing = [int(record["cutoff_seq"]) for record in records[1:]]
    if existing != sorted(existing) or len(existing) != len(set(existing)):
        raise MetricDashboardError("existing dashboard seq prefix is invalid")
    last = existing[-1] if existing else -1
    existing_by_seq = {int(record["cutoff_seq"]): record for record in records[1:]}
    for item in observations:
        seq = int(item["cutoff_seq"])
        compact = compact_observation(header, item)
        if seq in existing_by_seq and compact != existing_by_seq[seq]:
            raise MetricDashboardError(
                f"dashboard append would rewrite existing observation {seq}"
            )
    new = [item for item in observations if int(item["cutoff_seq"]) > last]
    if not new:
        return 0
    with target.open("a", encoding="utf-8") as handle:
        for item in new:
            handle.write(
                json.dumps(
                    compact_observation(header, item),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    return len(new)


def _history_from_dashboard(
    rows: Sequence[Mapping[str, Any]], *, root_code: str, context_root: Path
) -> dict[str, Any]:
    attempts: dict[int, dict[str, Any]] = {}
    for row in rows:
        seq = int(row["seq"])
        code = None
        source_events: list[dict[str, Any]] = []
        completion: Mapping[str, Any] = {}
        location = row.get("code_location")
        if isinstance(location, str):
            path = context_root / location
            if path.is_file():
                code = path.read_text(encoding="utf-8")
        trajectory_location = row.get("trajectory_location")
        if isinstance(trajectory_location, str):
            trajectory_path = context_root / trajectory_location
            if trajectory_path.is_file():
                for line in trajectory_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if isinstance(event, dict):
                        source_events.append(event)
                        if event.get("event") == "canonical_completion":
                            completion = event
        attempts[seq] = {
            "seq": seq,
            "evidence_id": row.get("evidence_id"),
            "timestamp": (_finite(row.get("elapsed_hours")) or 0.0) * 3600.0,
            "terminal_class": row.get("terminal_class"),
            "committed": row.get("committed"),
            "code_bearing": row.get("code_bearing"),
            "program_id": row.get("program_id"),
            "parent_id": row.get("parent_id"),
            "code": code,
            "code_sha256": _sha256_text(code) if code is not None else None,
            "cycles": row.get("cycles"),
            "parent_cycles": row.get("parent_cycles"),
            "failure_stage": completion.get("failure_stage")
            or row.get("failure_stage"),
            "terminal": completion.get("terminal"),
            "changes": row.get("changes_excerpt"),
            "source_events": tuple(source_events),
        }
    return {"root_code": root_code, "started_at": 0.0, "attempt_by_seq": attempts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--root-code", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--profile-cache", type=Path)
    parser.add_argument("--skip-engine-profiles", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = json.loads(args.dashboard.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise MetricDashboardError("dashboard input must be a JSON list")
    root_code = args.root_code.read_text(encoding="utf-8")
    history = _history_from_dashboard(
        rows, root_code=root_code, context_root=args.context_root
    )
    definitions, _ = load_definitions(args.definitions)
    profiles: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    if not args.skip_engine_profiles:
        if args.profile_cache is not None and args.profile_cache.is_file():
            cached = json.loads(args.profile_cache.read_text(encoding="utf-8"))
            profiles = dict(cached.get("profiles") or {})
            failures = dict(cached.get("failures") or {})
        else:
            profiles, failures = profile_programs(
                _program_codes(history),
                definitions=definitions,
                workers=args.profile_workers,
            )
            if args.profile_cache is not None:
                args.profile_cache.parent.mkdir(parents=True, exist_ok=True)
                args.profile_cache.write_text(
                    json.dumps(
                        {"profiles": profiles, "failures": failures},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
    header, observations = build_metric_observations(
        rows,
        history=history,
        engine_profiles=profiles,
        engine_failures=failures,
        definitions_path=args.definitions,
    )
    write_dashboard_jsonl(args.output, header, observations)
    write_metric_definitions(args.output.parent, definitions_path=args.definitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
