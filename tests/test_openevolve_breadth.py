from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from breadth_controller import OverlapPortfolioManager


def _make_planner_fixture(tmp_path):
    planner = tmp_path / "planner"
    catalog_dir = planner / "bundle/workspace/meta"
    catalog_dir.mkdir(parents=True)
    code = "def build_kernel():\n    pass\n"
    directions = [
        {
            "target_branch_id": "live",
            "mechanism_hypothesis": f"hypothesis {index}",
            "task": f"task {index}",
            "constraints": ["preserve correctness"],
            "donor_branch_ids": [],
            "evidence_attempt_ids": [],
        }
        for index in range(3)
    ]
    (planner / "portfolio.json").write_text(
        json.dumps(
            {
                "exhausted_patterns": [],
                "directions": directions,
            }
        ),
        encoding="utf-8",
    )
    (catalog_dir / "branch_catalog.json").write_text(
        json.dumps(
            {
                "programs": [
                    {
                        "id": "live",
                        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

def test_overlap_assigns_guided_beside_pretrigger_then_returns_to_native(tmp_path):
    _make_planner_fixture(tmp_path)
    control = tmp_path / "control"
    control.mkdir()
    (control / "handoff_000.json").write_text(
        json.dumps({"cohort_index": 0, "planner_dir": str(tmp_path / "planner")}),
        encoding="utf-8",
    )
    pending = tmp_path / "pending.json"
    pending.write_text(
        json.dumps(
            {
                "schema": "oe_pending_inflight_replay_v1",
                "records": [
                    {
                        "source_attempt_id": "old:7",
                        "iteration": 7,
                        "remaining_seconds": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Database:
        best_program_id = "live"
        programs = {
            "live": SimpleNamespace(
                code="def build_kernel():\n    pass\n",
                id="live",
                metrics={"cycles": 100, "combined_score": 0.01},
            )
        }

        def save(self, path, iteration):
            target = __import__("pathlib").Path(path)
            (target / "metadata.json").write_text(
                json.dumps({"last_iteration": iteration}), encoding="utf-8"
            )

    manager = OverlapPortfolioManager(
        control_dir=control,
        output_dir=tmp_path / "overlap",
        repeat_flat_seconds=0.001,
        repeat_flat_min_terminal_attempts=1,
        pending_inflight_path=pending,
    )
    manager.configure_native_concurrency(2)
    manager.bind_database(Database())
    replay = manager.initialize_replayed_inflight("new")
    manager.on_attempt_scheduled(
        attempt_id=replay[0]["attempt_id"],
        iteration=7,
        replayed_inflight=True,
    )
    manager.reconcile()
    for index in range(3):
        attempt_id = f"guided-{index}"
        assert manager.active_plan_for_dispatch(
            attempt_id=attempt_id, iteration=10 + index
        ) is not None
        manager.on_attempt_scheduled(attempt_id=attempt_id, iteration=10 + index)
    assert manager.ordinary_dispatch_fenced()
    for index in range(3):
        manager.on_completion_consumed(
            attempt_id=f"guided-{index}",
            completion_seq=index,
            iteration=10 + index,
            child=None,
        )
    assert not manager.ordinary_dispatch_fenced()
    assert manager.active_plan_for_dispatch(
        attempt_id="ordinary", iteration=13
    ) is None
    manager.on_attempt_scheduled(attempt_id="ordinary", iteration=13)
    assert manager.ordinary_dispatch_fenced()
    manager.on_completion_consumed(
        attempt_id=replay[0]["attempt_id"],
        completion_seq=3,
        iteration=7,
        child=None,
    )
    manager._flat_anchor_at -= 1
    manager._scheduled_at["ordinary"] -= 1
    manager.on_completion_consumed(
        attempt_id="ordinary", completion_seq=4, iteration=13, child=None
    )
    manager.controller_heartbeat()
    request = json.loads((control / "request_001.json").read_text())
    assert request["cutoff_seq"] == 4


def test_overlap_restores_a_selected_branch_evicted_while_planner_runs(tmp_path):
    _make_planner_fixture(tmp_path)
    control = tmp_path / "control"
    control.mkdir()

    program = SimpleNamespace(
        code="def build_kernel():\n    pass\n",
        id="live",
        metrics={"cycles": 100, "combined_score": 0.01},
    )
    database = SimpleNamespace(
        best_program_id="live",
        programs={"live": program},
        islands=[{"live"}],
    )
    manager = OverlapPortfolioManager(
        control_dir=control,
        output_dir=tmp_path / "overlap",
        repeat_flat_seconds=1,
    )
    manager.configure_native_concurrency(2)
    manager.bind_database(database)

    database.programs.pop("live")
    database.islands[0].remove("live")
    (control / "handoff_000.json").write_text(
        json.dumps({"cohort_index": 0, "planner_dir": str(tmp_path / "planner")}),
        encoding="utf-8",
    )

    manager.reconcile()

    assert database.programs["live"] is program
    assert "live" in database.islands[0]
    journal = [
        json.loads(line)
        for line in (tmp_path / "overlap/portfolio_cohort.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["kind"] for row in journal].count(
        "overlap_frozen_branch_restored"
    ) == 1


def test_overlap_repeat_requires_twelve_terminal_attempts(tmp_path):
    control = tmp_path / "control"
    control.mkdir()

    class Database:
        best_program_id = "live"
        programs = {
            "live": SimpleNamespace(
                code="def build_kernel():\n    pass\n",
                id="live",
                metrics={"cycles": 100, "combined_score": 0.01},
            )
        }

        def save(self, path, iteration):
            target = __import__("pathlib").Path(path)
            (target / "metadata.json").write_text(
                json.dumps({"last_iteration": iteration}), encoding="utf-8"
            )

    manager = OverlapPortfolioManager(
        control_dir=control,
        output_dir=tmp_path / "overlap",
        repeat_flat_seconds=0.001,
        repeat_flat_min_terminal_attempts=12,
        start_ordinary=True,
    )
    manager.configure_native_concurrency(4)
    manager.bind_database(Database())
    manager._flat_anchor_at -= 1.0
    for index in range(12):
        attempt_id = f"ordinary-{index}"
        manager.on_attempt_scheduled(attempt_id=attempt_id, iteration=index)
        manager._scheduled_at[attempt_id] = manager._flat_anchor_at

    for index in range(11):
        manager.on_completion_consumed(
            attempt_id=f"ordinary-{index}",
            completion_seq=index,
            iteration=index,
            child=None,
        )
    manager.controller_heartbeat()
    assert not (control / "request_000.json").exists()

    manager.on_completion_consumed(
        attempt_id="ordinary-11",
        completion_seq=11,
        iteration=11,
        child=None,
    )
    manager.controller_heartbeat()
    request = json.loads((control / "request_000.json").read_text())
    assert request["cutoff_seq"] == 11
    journal = [
        json.loads(line)
        for line in (tmp_path / "overlap/portfolio_cohort.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ready = next(row for row in journal if row["kind"] == "repeat_flat_ready")
    assert ready["terminal_attempts_since_anchor"] == 12
    assert ready["minimum_terminal_attempts"] == 12
