"""Focused zero-API tests for the definition/value dashboard boundary."""

from __future__ import annotations

import copy
import json

import pytest

import metric_scaled_dashboard as dashboard


def _profile(value: int) -> dict:
    return {
        "code_sha256": f"hash-{value}",
        "emitted_bundle_count": value,
        "engine_op_counts": {name: value for name in dashboard.ENGINE_NAMES},
        "engine_occupied_bundle_counts": {
            name: value // 2 for name in dashboard.ENGINE_NAMES
        },
        "engine_slot_lower_bounds": {
            name: value // 3 for name in dashboard.ENGINE_NAMES
        },
        "max_engine_slot_lower_bound": value // 3,
        "max_engine_slot_lower_bound_engines": ["valu"],
        "dependency_critical_path_lower_bound": value // 2,
        "scratch_words": value * 2,
    }


def _fixture(count: int = 14):
    root_code = "value = 0\n"
    rows = []
    attempts = {}
    profiles = {"root": _profile(100)}
    for seq in range(1, count + 1):
        program_id = f"p{seq}"
        parent_id = "root" if seq == 1 else f"p{seq - 1}"
        cycles = 2000.0 - seq
        code = f"value = {seq}\n"
        evidence_id = f"attempt:seq:{seq:08d}"
        rows.append(
            {
                "evidence_id": evidence_id,
                "seq": seq,
                "elapsed_hours": 1.0 + seq / 60.0,
                "terminal_class": "valid",
                "valid": True,
                "committed": True,
                "code_bearing": True,
                "eligible": True,
                "program_id": program_id,
                "parent_id": parent_id,
                "target_island": seq % 4,
                "cycles": cycles,
                "parent_cycles": 3000.0 if seq == 1 else 2001.0 - seq,
                "frontier_before_cycles": 1000.0,
                "gap_to_frontier_cycles": cycles - 1000.0,
                "verified_frontier_cycles": 1000.0,
                "frontier_move": False,
                "meaningful_frontier_improvement": False,
                "relative_gain": None,
                "F_reference_program_id": "root",
                "F_reference_seq": None,
                "exact_duplicate": False,
                "duplicate_of": None,
                "no_op": False,
                "trajectory_location": f"search_history/seq_{seq:08d}/trajectory.jsonl",
                "code_location": f"search_history/seq_{seq:08d}/code.py",
            }
        )
        attempts[seq] = {
            "seq": seq,
            "evidence_id": evidence_id,
            "timestamp": 3600.0 + seq * 60.0,
            "terminal_class": "valid",
            "committed": True,
            "code_bearing": True,
            "program_id": program_id,
            "parent_id": parent_id,
            "code": code,
            "code_sha256": dashboard._sha256_text(code),
            "cycles": cycles,
            "parent_cycles": rows[-1]["parent_cycles"],
            "changes": f"Change 1: set value to {seq}",
            "source_events": (),
        }
        profiles[program_id] = _profile(100 + seq)
    history = {
        "root_code": root_code,
        "started_at": 0.0,
        "attempt_by_seq": attempts,
    }
    return rows, history, profiles


def test_definitions_are_atomic_and_ordered_by_scope():
    definitions, digest = dashboard.load_definitions()
    assert len(digest) == 64
    expected = {
        "unary": ["U1", "U2", "U17"],
        "binary": ["B1", "B2", "B3", "B4", "B5", "B14", "B17"],
        "regional": [
            "R2", "R3", "R4", "R5", "R7", "R8", "R10", "R12", "R15",
            "R16", "R17",
        ],
        "global": ["G1", "G9", "G10", "G11"],
    }
    for scope in definitions["scope_order"]:
        actual = [
            key for key, value in definitions["metrics"].items()
            if value["scope"] == scope
        ]
        assert actual == expected[scope]
    assert len(definitions["metrics"]) == 25
    assert "R16" in definitions["metrics"]
    assert {"R1", "R9", "R18", "G2", "G13", "G14"}.isdisjoint(
        definitions["metrics"]
    )
    assert "U3" not in definitions["metrics"]
    assert list(definitions["mechanistic_profiles"]) == ["engine_capacity"]
    assert all(
        "components" not in definition
        and "rankable_components" not in definition
        for definition in definitions["metrics"].values()
    )


def test_metrics_are_atomic_and_scale_against_strict_history():
    rows, history, profiles = _fixture()
    header, observations = dashboard.build_metric_observations(
        rows, history=history, engine_profiles=profiles
    )

    assert header["definition_version"] == "oe_metric_dashboard_m25_v1"
    assert len(header["metric_order"]) == 25
    twelfth = observations[11]
    assert twelfth["metrics"]["regional"]["R2"]["availability"]["eligible"]
    assert not observations[10]["metrics"]["regional"]["R2"]["availability"]["eligible"]

    thirteenth = observations[12]
    assert thirteenth["metrics"]["unary"]["U17"]["value"] == 1.0
    assert thirteenth["metrics"]["binary"]["B3"]["value"] > 0.0
    assert thirteenth["metrics"]["regional"]["R7"]["value"] == 0.0
    assert thirteenth["metrics"]["regional"]["R8"]["value"] == 1.0
    assert thirteenth["metrics"]["regional"]["R16"]["value"] is None
    assert thirteenth["mechanistic_profiles"]["engine_capacity"]["values"][0] == 37.0

    for scope in thirteenth["metrics"].values():
        for metric in scope.values():
            assert "value" in metric
            assert "percentile" in metric
            assert "robust_z" in metric
            assert "availability" in metric
            assert "values" not in metric
            assert "scaled" not in metric

    serialized = dashboard.serialize_dashboard_jsonl(header, observations)
    assert "engine_profiles.json" not in serialized
    assert "profile_ref" not in serialized
    assert len(serialized.splitlines()) == 15
    compact = json.loads(serialized.splitlines()[-1])
    assert len(compact["metric_columns"][0]) == 25
    assert compact["nearest_history_ref"] == dashboard._compact_evidence_ref(
        observations[-1]["metrics"]["binary"]["B5"]["reference_ids"][0]
    )
    assert "evidence_sets" not in compact
    assert "references" not in compact
    assert "record_type" not in compact
    assert "evidence_id" not in compact
    assert "cutoff_at" not in compact
    assert "history_gate_open" not in compact
    assert len(serialized.splitlines()[-1]) < 2050


def test_append_is_idempotent_and_rejects_prefix_rewrite(tmp_path):
    rows, history, profiles = _fixture()
    header, observations = dashboard.build_metric_observations(
        rows, history=history, engine_profiles=profiles
    )
    output = tmp_path / "dashboard.jsonl"
    dashboard.write_dashboard_jsonl(output, header, observations[:12])
    assert dashboard.append_dashboard_observations(
        output, header, observations
    ) == 2
    assert dashboard.append_dashboard_observations(
        output, header, observations
    ) == 0

    changed = copy.deepcopy(observations)
    changed[0]["metrics"]["unary"]["U1"]["value"] = 1.0
    with pytest.raises(dashboard.MetricDashboardError, match="rewrite"):
        dashboard.append_dashboard_observations(output, header, changed)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["record_type"] == "dashboard_header"
    assert [item["cutoff_seq"] for item in records[1:]] == list(range(1, 15))

    definitions_path = dashboard.write_metric_definitions(tmp_path)
    assert dashboard._sha256_text(definitions_path.read_text()) == header[
        "definitions_sha256"
    ]
    validated_header, validated_rows = dashboard.validate_dashboard_jsonl(
        output,
        definitions_path=definitions_path,
        expected_cutoff_seq=14,
    )
    assert validated_header["metric_order"] == header["metric_order"]
    assert len(validated_rows) == 14


def test_llm_dashboard_is_smaller_and_semantically_aligned():
    rows, history, profiles = _fixture()
    rows[-2]["live_at_cutoff"] = False
    rows[-1]["live_at_cutoff"] = True
    header, observations = dashboard.build_metric_observations(
        rows, history=history, engine_profiles=profiles
    )
    canonical = dashboard.serialize_dashboard_jsonl(header, observations)
    canonical_rows = [json.loads(line) for line in canonical.splitlines()][1:]
    encoded = dashboard.serialize_llm_dashboard_jsonl(header, canonical_rows)
    records = [json.loads(line) for line in encoded.splitlines()]
    llm_header, llm_rows = records[0], records[1:]

    assert len(encoded) < len(canonical)
    assert llm_header["definitions_sha256"] == header["definitions_sha256"]
    assert llm_header["schema"] == "oe_metric_dashboard_llm_v2"
    assert llm_header["metric_order"] == header["metric_order"]
    decoded = [dict(zip(llm_header["row_columns"], row)) for row in llm_rows]
    assert decoded[0]["live_at_cutoff"] is None
    assert decoded[-2]["live_at_cutoff"] is False
    assert decoded[-1]["live_at_cutoff"] is True
    last = dict(zip(llm_header["row_columns"], llm_rows[-1]))
    assert last["cutoff_seq"] == canonical_rows[-1]["cutoff_seq"]
    assert last["program_id"] == canonical_rows[-1]["program_id"]
    assert last["parent_ref"] == canonical_rows[-2]["cutoff_seq"]
    assert last["metric_columns"][0] == pytest.approx(
        canonical_rows[-1]["metric_columns"][0], rel=2e-6, abs=1e-9
    )
    assert last["metric_columns"][3:] == canonical_rows[-1]["metric_columns"][3:]
    decoded_details = {}
    for metric_position, n_used, n_expected, reason_position in last[
        "availability_details"
    ]:
        detail = {}
        if n_used is not None:
            detail["n_used"] = n_used
        if n_expected is not None:
            detail["n_expected"] = n_expected
        if reason_position is not None:
            detail["reason"] = llm_header["availability_reason_codes"][
                reason_position
            ]
        decoded_details[llm_header["metric_order"][metric_position]] = detail
    assert decoded_details == canonical_rows[-1].get("availability_details", {})
    assert last["code_available"]
