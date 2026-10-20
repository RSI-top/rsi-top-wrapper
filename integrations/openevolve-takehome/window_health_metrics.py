#!/usr/bin/env python3
"""Causal quantitative compression for one OpenEvolve audit window."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "oe_window_health_metrics_v1"
WINDOW_SIZE = 12


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _ratio(count: int, total: int) -> float | None:
    return count / total if total else None


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    def at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "q25": at(0.25),
        "median": at(0.5),
        "q75": at(0.75),
        "max": ordered[-1],
    }


def _slope(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    center = (len(values) - 1) / 2.0
    denominator = sum((index - center) ** 2 for index in range(len(values)))
    if denominator == 0:
        return None
    mean = sum(values) / len(values)
    return (
        sum((index - center) * (value - mean) for index, value in enumerate(values))
        / denominator
    )


def _recent_frontier_step_scale(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    steps = []
    for row in rows:
        if row.get("meaningful_frontier_improvement") is not True:
            continue
        before = _number(row.get("frontier_before_cycles"))
        after = _number(row.get("cycles"))
        if before is not None and after is not None and before > after:
            steps.append(before - after)
    recent = steps[-5:]
    scale = _quantiles(recent)["median"] if recent else None
    return {
        "definition": "median cycle gain of the last up to five verified meaningful frontier moves",
        "step_count": len(recent),
        "cycles": scale,
    }


def compute_window_health_metrics(
    routing_rows: Sequence[Mapping[str, Any]],
    *,
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    """Compress the terminal W12 without looking beyond the supplied cutoff."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if len(routing_rows) < window_size:
        return {
            "schema": SCHEMA,
            "available": False,
            "required_window_size": window_size,
            "observed_terminal_rows": len(routing_rows),
        }
    rows = [dict(row) for row in routing_rows]
    window = rows[-window_size:]
    if any(not isinstance(row.get("seq"), int) for row in window):
        raise ValueError("every window row must have an integer seq")

    valid = [row for row in window if row.get("terminal_class") == "valid"]
    invalid_count = sum(row.get("terminal_class") == "invalid" for row in window)
    worker_error_count = sum(
        row.get("terminal_class") == "worker_error" for row in window
    )
    duplicate_count = sum(row.get("exact_duplicate") is True for row in window)
    no_op_count = sum(row.get("no_op") is True for row in window)
    regressions = [
        row
        for row in valid
        if _number(row.get("cycles")) is not None
        and _number(row.get("parent_cycles")) is not None
        and float(row["cycles"]) >= float(row["parent_cycles"])
    ]

    scale = _recent_frontier_step_scale(rows)
    step = _number(scale["cycles"])
    gaps = [
        value
        for row in valid
        if (value := _number(row.get("gap_to_frontier_cycles"))) is not None
    ]
    relative_gaps = [
        gap / abs(float(row["frontier_before_cycles"]))
        for row in valid
        if (gap := _number(row.get("gap_to_frontier_cycles"))) is not None
        and _number(row.get("frontier_before_cycles")) not in (None, 0.0)
    ]
    scaled_gaps = [gap / step for gap in gaps] if step and step > 0 else []

    gains = [
        float(row["parent_cycles"]) - float(row["cycles"])
        for row in valid
        if _number(row.get("cycles")) is not None
        and _number(row.get("parent_cycles")) is not None
    ]
    relative_gains = [
        (float(row["parent_cycles"]) - float(row["cycles"]))
        / abs(float(row["parent_cycles"]))
        for row in valid
        if _number(row.get("cycles")) is not None
        and _number(row.get("parent_cycles")) not in (None, 0.0)
    ]
    scaled_gains = [gain / step for gain in gains] if step and step > 0 else []

    islands = Counter(
        "unknown" if row.get("target_island") is None else str(row["target_island"])
        for row in window
    )
    shares = [count / len(window) for count in islands.values()]
    effective_islands = 1.0 / sum(share * share for share in shares)
    axis_series: dict[str, dict[str, Any]] = {}
    for axis in ("S", "F"):
        observations = [
            (int(row["seq"]), value)
            for row in window
            if (value := _number(row.get(axis))) is not None
        ]
        values = [value for _, value in observations]
        axis_series[axis] = {
            "observation_count": len(values),
            "latest": values[-1] if values else None,
            "distribution": _quantiles(values),
            "slope_per_observation": _slope(values),
            "change_first_to_last": (
                values[-1] - values[0] if len(values) >= 2 else None
            ),
            "observation_seqs": [seq for seq, _ in observations],
        }

    return {
        "schema": SCHEMA,
        "available": True,
        "cutoff_seq": int(window[-1]["seq"]),
        "window_size": window_size,
        "window_evidence_ids": [str(row.get("evidence_id")) for row in window],
        "scale": scale,
        "terminal_outcomes": {
            "valid_rate": _ratio(len(valid), len(window)),
            "invalid_rate": _ratio(invalid_count, len(window)),
            "worker_error_rate": _ratio(worker_error_count, len(window)),
            "exact_duplicate_rate": _ratio(duplicate_count, len(window)),
            "no_op_rate": _ratio(no_op_count, len(window)),
            "regression_rate_among_valid": _ratio(len(regressions), len(valid)),
            "counts": {
                "valid": len(valid),
                "invalid": invalid_count,
                "worker_error": worker_error_count,
                "exact_duplicate": duplicate_count,
                "no_op": no_op_count,
                "valid_regression_or_tie": len(regressions),
            },
        },
        "frontier_competitiveness": {
            "gap_cycles": _quantiles(gaps),
            "gap_fraction_of_frontier": _quantiles(relative_gaps),
            "gap_in_recent_step_units": _quantiles(scaled_gaps),
            "within_one_recent_step_rate_among_valid": (
                _ratio(sum(value <= 1.0 for value in scaled_gaps), len(scaled_gaps))
                if scaled_gaps
                else None
            ),
        },
        "parent_progress": {
            "gain_cycles": _quantiles(gains),
            "gain_fraction_of_parent": _quantiles(relative_gains),
            "gain_in_recent_step_units": _quantiles(scaled_gains),
            "positive_gain_rate_among_comparable_valid": (
                _ratio(sum(value > 0 for value in gains), len(gains)) if gains else None
            ),
        },
        "population_state": {
            "island_attempt_counts": dict(sorted(islands.items())),
            "island_attempt_shares": {
                key: count / len(window) for key, count in sorted(islands.items())
            },
            "dominant_island_share": max(shares),
            "effective_island_count": effective_islands,
            "state_dynamics": axis_series,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dashboard", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()
    rows = json.loads(args.dashboard.read_text(encoding="utf-8"))
    result = compute_window_health_metrics(rows, window_size=args.window_size)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
