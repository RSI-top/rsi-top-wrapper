"""Minimal zero-API checks for the opt-in 1108 breadth adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.helpers import make_program, shinka_supervisor
_failed_directions = shinka_supervisor._failed_directions
from shinka.core.sampler import PromptSampler
from shinka_breadth import ShinkaBreadthGuidance, ShinkaOverlapBreadthManager


def _payload(target: str, donors: list[str]) -> dict:
    return {
        "plan_version": "branch_recipe_portfolio_guidance_v2",
        "audit_id": "audit-1",
        "lane_id": "intervention_direction",
        "lane_task": "change the schedule in one bounded way",
        "target_branch_id": target,
        "donor_branch_ids": donors,
        "diagnosed_bottleneck": "the current schedule leaves a measured gap",
        "compatibility": ["preserve correctness"],
    }


def test_dashboard_resume_clock_excludes_offline_gap_only(tmp_path):
    manager = ShinkaOverlapBreadthManager(results_dir=tmp_path, control_dir=tmp_path / "control")
    root = make_program(program_id="root", generation=0, parent_id=None)
    root.public_metrics = {"cycles": 2000}
    child = root.to_dict()
    events = [
        {"event": "run_started", "timestamp": 100},
        {"event": "root_program_committed", "program": child},
        {"event": "canonical_completion", "timestamp": 200, "scheduled_at": 110,
         "completion_seq": 1, "generation": 1, "child": child},
        {"event": "run_started", "timestamp": 1000},
    ]
    (tmp_path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    manager.runner = SimpleNamespace(db=SimpleNamespace(get_all_programs=lambda: [root]))
    manager._initialize_trace()
    assert manager._dashboard_time(200) == 200
    assert manager._dashboard_time(1030) == 230
    live = {**events[2], "timestamp": 1030, "scheduled_at": 1005, "completion_seq": 2}
    normalized = manager._normalized_events(live)
    assert [row["timestamp"] for row in normalized] == [205, 230, 230]
    assert live["timestamp"] == 1030
    manager._snapshot_database = lambda _: None
    manager._request_portfolio(2, 1030)
    request = json.loads((tmp_path / "control/request_000.json").read_text())
    assert request["cutoff_at"] == 230
    assert request["source_cutoff_at"] == 1030


def test_guided_unit_uses_target_native_context_and_user_task_only_once():
    target = make_program(program_id="target")
    donor = make_program(program_id="donor")
    native_archive = make_program(program_id="target-archive")
    native_top_k = make_program(program_id="target-top-k")
    guidance = ShinkaBreadthGuidance(
        payload=_payload("target", ["donor"]),
        frozen_programs={"target": target, "donor": donor},
        archive_slots=1,
        top_k_slots=1,
    )

    sampled_for = []

    async def sample_context(parent, archive_slots, top_k_slots):
        sampled_for.append((parent.id, archive_slots, top_k_slots))
        return [native_archive], [native_top_k]

    parent, archive, top_k, needs_fix = asyncio.run(
        guidance.sample_inputs(
            sample_context=sample_context,
        )
    )
    assert sampled_for == [("target", 1, 1)]
    assert parent is target
    assert archive == [donor]
    assert top_k == [native_archive]
    assert needs_fix is False
    system, user, patch_type = guidance.build_prompt_override(
        prompt_sampler=PromptSampler(task_sys_msg="NATIVE PROBLEM"),
        parent_program=parent,
        archive_programs=archive,
        top_k_programs=top_k,
        needs_fix=False,
        patch_type="diff",
    )
    assembled = system + "\n" + user
    assert patch_type == "diff"
    assert assembled.count("Execute one selected intervention direction") == 1
    assert "Execute one selected intervention direction" not in system
    assert "Execute one selected intervention direction" in user
    assert "Suggest a new idea to improve" not in assembled
    assert "SEARCH/REPLACE" in assembled
    assert "NATIVE PROBLEM" in system

    for native_mode in ("full", "cross"):
        mode_system, mode_user, selected = guidance.build_prompt_override(
            prompt_sampler=PromptSampler(task_sys_msg="NATIVE PROBLEM"),
            parent_program=parent,
            archive_programs=archive,
            top_k_programs=top_k,
            needs_fix=False,
            patch_type=native_mode,
        )
        mode_prompt = mode_system + "\n" + mode_user
        assert selected == native_mode
        assert mode_prompt.count("Execute one selected intervention direction") == 1
        assert "Execute one selected intervention direction" not in mode_system
        assert "Execute one selected intervention direction" in mode_user
        assert "Perform a cross-over" not in mode_prompt
        assert "Rewrite the program to improve" not in mode_prompt

    fix_system, fix_user, selected = guidance.build_prompt_override(
        prompt_sampler=PromptSampler(task_sys_msg="NATIVE PROBLEM"),
        parent_program=parent,
        archive_programs=archive,
        top_k_programs=top_k,
        needs_fix=True,
        patch_type="fix",
    )
    fix_prompt = fix_system + "\n" + fix_user
    assert selected == "fix"
    assert fix_prompt.count("Execute one selected intervention direction") == 1
    assert "Execute one selected intervention direction" not in fix_system
    assert "Execute one selected intervention direction" in fix_user
    assert "Analyze the error output above and fix the program" not in fix_prompt


@pytest.mark.parametrize("mode", ["diff", "full", "cross", "fix"])
def test_guided_task_braces_remain_literal(mode):
    target = make_program(program_id="target")
    payload = _payload("target", [])
    payload["lane_task"] = (
        'Test {0,1,2}, {multiply_add, +, ^, <<, >>}, and {0,2,4,6,8,15}; '
        'keep {language}, {code_content}, and {"limits": [1, 2]} literal.'
    )
    guidance = ShinkaBreadthGuidance(
        payload=payload, frozen_programs={"target": target},
        archive_slots=1, top_k_slots=1,
    )
    system, user, selected = guidance.build_prompt_override(
        prompt_sampler=PromptSampler(task_sys_msg="NATIVE PROBLEM"),
        parent_program=target,
        archive_programs=[make_program(program_id="inspiration")],
        top_k_programs=[], needs_fix=mode == "fix", patch_type=mode,
    )
    assert selected == mode
    assert (system + user).count(payload["lane_task"]) == 1
    assert "{intervention_task}" not in system + user


@pytest.mark.parametrize("one_cohort", [False, True])
def test_variable_burst_returns_immediately_to_native_width(tmp_path, one_cohort):
    manager = ShinkaOverlapBreadthManager(
        results_dir=tmp_path,
        control_dir=tmp_path / "control",
        start_ordinary=True,
        stop_after_guided_cohort=one_cohort,
    )
    manager.runner = SimpleNamespace(
        running_jobs={"old": object()},
        active_proposal_tasks={},
        db_config=SimpleNamespace(
            num_archive_inspirations=1, num_top_k_inspirations=1
        ),
    )
    manager.phase = "assigning_guided"
    manager._directions = [
        _payload("target", []) for _ in range(3)
    ]
    manager._frozen_programs = {"target": make_program(program_id="target")}
    assert manager.pipeline_target(4) == 4  # one old + three selected
    assert manager.prepare_start(3) == 3
    manager.reserve_generations(10, 3)
    assert manager.phase == "burst_draining_to_native_width"
    assert sorted(manager._guidance_by_generation) == [10, 11, 12]
    if one_cohort:
        assert manager.pipeline_target(4) == 0
        assert manager.prepare_start(1) == 0
        assert manager.phase == "burst_draining_to_native_width"
        return
    assert manager.pipeline_target(4) == 4
    assert manager.prepare_start(1) == 1
    assert manager.phase == "ordinary"


def test_repeat_trigger_requires_both_20_minutes_and_12_terminals(tmp_path):
    manager = ShinkaOverlapBreadthManager(
        results_dir=tmp_path,
        control_dir=tmp_path / "control",
        start_ordinary=True,
    )
    manager.phase = "ordinary"
    manager._flat_anchor_at = 0.0
    for index in range(12):
        attempt = f"a-{index}"
        manager._ordinary_attempts.add(attempt)
        manager._scheduled_at[attempt] = index * 100.0
        manager._terminal_at[attempt] = (index + 1) * 100.0
    requests = []
    manager._request_portfolio = lambda seq, at: requests.append((seq, at))
    manager._maybe_trigger(10, 1199.0)
    assert requests == []
    manager._maybe_trigger(11, 1200.0)
    assert requests == [(11, 1200.0)]
    assert manager.phase == "waiting_portfolio"


@pytest.mark.parametrize("source", ["guided", "ordinary", None])
def test_failed_history_clears_on_any_meaningful_improvement(tmp_path, source):
    existing = [{"task": "previous direction", "outcome": "tie"}]
    rows = ([{"kind": "meaningful_frontier_improvement", "cohort_index": 0,
              "source": source}] if source else [])
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _failed_directions(
        existing=existing, planner=tmp_path / "planner", journal=journal, cohort=0,
    ) == ([] if source else existing)


def test_failed_direction_keeps_its_execution_reference(tmp_path):
    (tmp_path / "portfolio.json").write_text(json.dumps({
        "directions": [{"task": "test schedule", "target_branch_id": "target"}],
    }))
    rows = [
        {"kind": "cohort_bound", "frontier_cycles": 100},
        {"kind": "direction_assigned", "generation": 1, "direction_index": 1},
        {"kind": "guided_attempt_terminal", "generation": 1, "completion_seq": 7,
         "evaluation_valid": True, "cycles": 100},
        {"kind": "guided_cohort_complete"},
    ]
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps({**r, "cohort_index": 0}) + "\n" for r in rows))
    records = _failed_directions(existing=[], planner=tmp_path, journal=journal, cohort=0)
    assert records[0]["completion_seq"] == 7
    assert records[0]["outcome"] == "tie"


@pytest.mark.parametrize("valid", [True, False])
def test_prepared_history_is_compact_without_changing_source_or_dashboard(
    tmp_path, monkeypatch, valid
):
    import prepare_shinka_breadth_view as prep

    code = "print('candidate')\n"
    feedback = "correct" if valid else "[build_error] scratch overflow"
    metadata = {
        "patch_type": "full", "patch_name": "new_schedule",
        "patch_description": "Reorder independent loads.", "stdout_log": feedback,
        "llm_result": {"msg": code, "system_msg": "OLD PROMPT",
                       "thought": "OLD REASONING", "content": code,
                       "new_msg_history": [{"role": "user", "content": code}]},
        "active_proposals_at_start": 32,
    }
    metrics = {"cycles": 1200 if valid else None}
    events = [
        {"event": "attempt_scheduled", "attempt_id": "a1", "parent_id": "root",
         "parent_code": code, "timestamp": 1},
        {"event": "canonical_completion", "attempt_id": "a1", "completion_seq": 1,
         "timestamp": 2, "evaluation_valid": valid, "api_cost_usd": 0.1,
         "failure_stage": None if valid else "build_error", "metadata": metadata,
         "archive_inspiration_ids": ["donor"],
         "evaluation_artifacts": {"correct.json": {"correct": valid}},
         "child": {"id": "child", "parent_id": "root", "correct": valid,
                   "code_location": "search_history/seq_00000001/code.py",
                   "metrics": metrics, "text_feedback": feedback, "metadata": metadata,
                   "embedding": [0.1] * 100, "code_diff": "+" + code}},
        {"event": "program_committed", "program_id": "child", "metadata": metadata,
         "metrics": metrics, "timestamp": 2},
    ]
    raw = "".join(json.dumps(e) + "\n" for e in events)
    source = tmp_path / "attempts.jsonl"
    source.write_text(raw)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "programs").mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"islands": [["root"]]}))
    root_record = json.dumps({"id": "root", "generation": 0, "code": code})
    (checkpoint / "programs/root.json").write_text(root_record)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "schema": "shinka_overlap_planner_request_v1", "normalized_attempts": str(source),
        "checkpoint": str(checkpoint), "cutoff_seq": 1, "cutoff_at": 2,
    }))
    history_path = "search_history/seq_00000001/trajectory.jsonl"
    files = {history_path: raw, "search_history/seq_00000001/code.py": code,
             prep.LLM_DASHBOARD: '{"unchanged_dashboard":true}\n'}

    def build_context(**kwargs):
        assert kwargs["attempts_path"].read_text() == raw
        return {"files": files}

    monkeypatch.setattr(prep, "build_stage_context", build_context)
    monkeypatch.setattr(prep, "materialize_view",
                        lambda **kwargs: (kwargs["output"] / "bundle", "", {}, set()))
    monkeypatch.setattr(prep, "_materialize_engine_profiles", lambda view, scratch:
                        (view / prep.ENGINE_PROFILES).write_text('{"profiles":{}}'))
    output = tmp_path / "view_build"
    view = prep.prepare(request=request, output=output, prompt=tmp_path / "prompt.md")
    private = output / "prepared/controller_private/search_history/seq_00000001"
    compact = (private / "trajectory.jsonl").read_text()
    projected = [json.loads(line) for line in compact.splitlines()]
    assert source.read_text() == raw
    assert (checkpoint / "programs/root.json").read_text() == root_record
    assert (output / "context" / history_path).read_text() == raw
    assert (view / prep.LLM_DASHBOARD).read_text() == files[prep.LLM_DASHBOARD]
    assert (private / "code.py").read_text() == code
    assert not (view / history_path).exists()  # Still read via the registered tool.
    assert len(compact) < len(raw) / 2
    for forbidden in ("llm_result", "parent_code", "code_diff", "embedding",
                      "OLD PROMPT", "OLD REASONING", "active_proposals_at_start"):
        assert forbidden not in compact
    assert compact.count('"metadata"') == 1
    assert projected[0]["parent_id"] == "root"
    assert projected[1]["child"]["metadata"]["patch_description"] == metadata["patch_description"]
    assert projected[1]["child"]["text_feedback"] == feedback
    assert projected[1]["child"]["metrics"] == metrics
    for key in ("evaluation_valid", "failure_stage", "archive_inspiration_ids",
                "evaluation_artifacts", "api_cost_usd", "timestamp"):
        assert projected[1][key] == events[1][key]
