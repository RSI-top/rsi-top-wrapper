from __future__ import annotations

import pytest

from window_health_metrics import SCHEMA, compute_window_health_metrics


def _rows(count: int = 16):
    rows = []
    frontier = 1000.0
    for seq in range(1, count + 1):
        meaningful = seq in {3, 9}
        cycles = frontier - 10.0 if meaningful else frontier + float(seq % 4)
        row = {
            "seq": seq,
            "evidence_id": f"attempt:seq:{seq:08d}",
            "terminal_class": "invalid" if seq == count - 1 else "valid",
            "cycles": cycles,
            "parent_cycles": frontier + 5.0,
            "frontier_before_cycles": frontier,
            "gap_to_frontier_cycles": cycles - frontier,
            "meaningful_frontier_improvement": meaningful,
            "exact_duplicate": seq == count,
            "no_op": False,
            "target_island": seq % 4,
            "S": 0.01 * seq,
            "F": 0.02 * seq,
        }
        rows.append(row)
        if meaningful:
            frontier = cycles
    return rows


def test_metrics_are_four_causal_w12_compressions():
    result = compute_window_health_metrics(_rows())

    assert result["schema"] == SCHEMA
    assert result["available"] is True
    assert result["window_evidence_ids"][0].endswith("00000005")
    assert result["window_evidence_ids"][-1].endswith("00000016")
    assert set(result) >= {
        "terminal_outcomes",
        "frontier_competitiveness",
        "parent_progress",
        "population_state",
    }
    assert result["terminal_outcomes"]["invalid_rate"] == pytest.approx(1 / 12)
    assert result["population_state"]["effective_island_count"] == 4.0
    assert result["scale"]["cycles"] == 10.0


def test_incomplete_w12_is_telemetry_only():
    result = compute_window_health_metrics(_rows(8))
    assert result == {
        "schema": SCHEMA,
        "available": False,
        "required_window_size": 12,
        "observed_terminal_rows": 8,
    }
