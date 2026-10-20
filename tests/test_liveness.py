from __future__ import annotations

import hashlib
import json

from planner_context import (
    live_program_ids_at_cutoff,
)


def _state(program_ids: list[str], *, completion_seq: int | None = None) -> dict:
    state = {"schema": "oe_db_state_v1", "program_ids": program_ids}
    event = {
        "event": "db_state",
        "db_state_schema": "oe_db_state_v1",
        "capture": "program_committed",
        "completion_seq": completion_seq,
        "state": state,
        "state_sha256": hashlib.sha256(
            json.dumps(
                state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest(),
    }
    return event


def test_exact_liveness_uses_correlated_post_commit_state(tmp_path):
    events = [
        _state(["old"]),
        {"event": "canonical_completion", "completion_seq": 7},
        {"event": "program_committed", "completion_seq": 7},
        _state(["old", "new"], completion_seq=7),
        {"event": "canonical_completion", "completion_seq": 8},
    ]
    trace = tmp_path / "attempts.jsonl"
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    assert live_program_ids_at_cutoff(events, 7) == {"old", "new"}


def test_exact_liveness_inherits_only_without_a_commit():
    prior = _state(["still-live"])
    no_commit = [
        prior,
        {"event": "canonical_completion", "completion_seq": 9},
        {"event": "canonical_completion", "completion_seq": 10},
    ]
    missing_capture = [
        prior,
        {"event": "canonical_completion", "completion_seq": 9},
        {"event": "program_committed", "completion_seq": 9},
        {"event": "canonical_completion", "completion_seq": 10},
    ]

    assert live_program_ids_at_cutoff(no_commit, 9) == {"still-live"}
    assert live_program_ids_at_cutoff(missing_capture, 9) is None
