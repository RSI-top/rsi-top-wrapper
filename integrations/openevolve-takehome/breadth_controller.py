"""Dispatch guided portfolios alongside native OpenEvolve work."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _planner_documents(
    planner_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    planner = Path(planner_dir).resolve()
    portfolio = json.loads((planner / "portfolio.json").read_text(encoding="utf-8"))
    catalog = json.loads(
        (planner / "bundle/workspace/meta/branch_catalog.json").read_text(
            encoding="utf-8"
        )
    )
    directions = portfolio.get("directions")
    if not isinstance(directions, list) or not directions:
        raise ValueError("portfolio replay requires at least one direction")
    breadth_config = None
    run_path = planner / "run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("schema") == "oe_breadth_direction_pool_v1":
            if int(run.get("actual_selected_directions", -1)) != len(directions):
                raise ValueError("breadth run/portfolio direction counts disagree")
            breadth_config = {
                key: run.get(key)
                for key in (
                    "candidate_pool_size",
                    "directions_per_planner_call",
                    "planner_call_count",
                    "actual_planner_session_count",
                    "format_retry_count",
                    "planner_workers",
                    "planner_sampling_mode",
                    "planner_sampling_profiles",
                    "planner_routing_mode",
                    "direction_selector",
                    "unfold_selection_mode",
                    "max_selected_directions",
                    "actual_selected_directions",
                    "embedding_backend",
                    "embedding_model",
                    "semantic_duplicate_threshold",
                    "dedup_model",
                    "dedup_base_url",
                    "dedup_api_key_env",
                    "dedup_max_tokens",
                    "dedup_usage",
                )
            }
    return planner, portfolio, catalog, breadth_config


class OverlapPortfolioManager:
    """Repeat a planner-produced breadth burst without draining native work.

    The controller fences only new ordinary dispatches while the external
    supervisor plans from the trigger-time checkpoint.  A delivered portfolio
    is assigned immediately beside the pre-trigger work.  After assignment,
    the enlarged pool decays naturally and ordinary search resumes as soon as
    fewer than the native number of work units remain.
    """

    def __init__(
        self,
        *,
        control_dir: str | Path,
        output_dir: str | Path,
        repeat_flat_seconds: float,
        repeat_flat_min_terminal_attempts: int = 12,
        pending_inflight_path: str | Path | None = None,
        start_ordinary: bool = False,
        minimum_run_age_seconds: float = 0.0,
    ) -> None:
        self.strict_controller_hooks = True
        self.control_dir = Path(control_dir).resolve()
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir).resolve()
        self._journal_path = self.output_dir / "portfolio_cohort.jsonl"
        self._run_started_at = time.time()
        self._minimum_run_age_seconds = float(minimum_run_age_seconds)
        self._repeat_flat_seconds = float(repeat_flat_seconds)
        if self._repeat_flat_seconds <= 0:
            raise ValueError("repeat flat seconds must be positive")
        self._repeat_flat_min_terminal_attempts = int(repeat_flat_min_terminal_attempts)
        if self._repeat_flat_min_terminal_attempts < 1:
            raise ValueError("repeat flat minimum terminal attempts must be positive")
        self._database: Any = None
        self._native_concurrency: int | None = None
        self._portfolio: dict[str, Any] | None = None
        self._catalog: dict[str, Any] | None = None
        self._planner_dir: Path | None = None
        self._breadth_config: dict[str, Any] | None = None
        self._audit_id: str | None = None
        self._cohort_index = -1 if start_ordinary else 0
        self._phase = "ordinary" if start_ordinary else "waiting_portfolio"
        self._by_attempt: dict[str, dict[str, Any]] = {}
        self._assigned: list[str] = []
        self._guided_attempt_ids: set[str] = set()
        self._pretrigger_attempt_ids: set[str] = set()
        self._terminal_attempt_ids: set[str] = set()
        self._active_attempt_ids: set[str] = set()
        self._ordinary_attempt_ids: set[str] = set()
        self._scheduled_at: dict[str, float] = {}
        self._terminal_at: dict[str, float] = {}
        self._completion_payloads: dict[int, dict[str, Any]] = {}
        self._last_terminal_iteration = -1
        self._progress_anchor_cycles: float | None = None
        self._flat_anchor_at: float | None = None
        self._request_pending = False
        self._request_cutoff_at: float | None = None
        self._replay_started_at: float | None = None
        self._replay_pending: list[dict[str, Any]] = []
        self._frozen_programs: dict[str, Any] = {}
        self._frozen_program_islands: dict[str, int] = {}
        if pending_inflight_path is not None:
            document = json.loads(
                Path(pending_inflight_path).read_text(encoding="utf-8")
            )
            if document.get("schema") != "oe_pending_inflight_replay_v1":
                raise ValueError("pending in-flight replay schema mismatch")
            self._replay_pending = [dict(row) for row in document["records"]]

    def _journal(self, kind: str, **payload: Any) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": time.time(),
                        "kind": kind,
                        "audit_id": self._audit_id,
                        "cohort_index": self._cohort_index,
                        **payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def configure_native_concurrency(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError("native concurrency must be positive")
        self._native_concurrency = value

    def bind_database(self, database: Any) -> None:
        self._database = database
        self._capture_frozen_programs()
        best = database.programs.get(getattr(database, "best_program_id", None))
        cycles = (getattr(best, "metrics", None) or {}).get("cycles")
        if isinstance(cycles, (int, float)) and not isinstance(cycles, bool):
            self._progress_anchor_cycles = float(cycles)
        if self._phase == "ordinary":
            self._flat_anchor_at = time.time()
        self._journal(
            "overlap_controller_bound",
            pending_inflight_count=len(self._replay_pending),
            frontier_cycles=self._progress_anchor_cycles,
        )
        if self._phase == "ordinary":
            self._journal("ordinary_search_bootstrapped")

    def _capture_frozen_programs(self) -> None:
        """Keep the exact trigger-time branches alive across overlap planning."""
        self._frozen_programs = dict(self._database.programs)
        self._frozen_program_islands = {
            str(program_id): island
            for island, members in enumerate(getattr(self._database, "islands", ()))
            for program_id in members
        }

    def _restore_frozen_references(self, catalog: dict[str, Any]) -> None:
        """Restore selected cutoff branches evicted by pre-trigger work."""
        frozen_rows = {str(row["id"]): row for row in catalog["programs"]}
        assert self._portfolio is not None
        references = {
            str(branch_id)
            for direction in self._portfolio["directions"]
            for branch_id in (
                direction["target_branch_id"],
                *direction["donor_branch_ids"],
            )
        }
        for branch_id in sorted(references):
            if branch_id in self._database.programs:
                continue
            program = self._frozen_programs.get(branch_id)
            row = frozen_rows.get(branch_id)
            island = self._frozen_program_islands.get(branch_id)
            if (
                program is None
                or row is None
                or island is None
                or _sha256(program.code) != row.get("code_sha256")
            ):
                raise ValueError(
                    f"selected branch cannot be restored from the frozen cutoff: "
                    f"{branch_id}"
                )
            self._database.programs[branch_id] = program
            self._database.islands[island].add(branch_id)
            self._journal(
                "overlap_frozen_branch_restored",
                branch_id=branch_id,
                island=island,
            )

    def initialize_replayed_inflight(self, instance_id: str) -> list[dict[str, Any]]:
        if self._replay_started_at is not None:
            return []
        self._replay_started_at = time.monotonic()
        initialized = []
        for row in self._replay_pending:
            item = dict(row)
            item["attempt_id"] = f"{instance_id}:{int(item['iteration'])}"
            item["replay_due_at"] = self._replay_started_at + float(
                item["remaining_seconds"]
            )
            initialized.append(item)
        self._replay_pending = initialized
        return [dict(row) for row in initialized]

    def take_due_replayed_inflight(self) -> list[dict[str, Any]]:
        if self._replay_started_at is None:
            return []
        now = time.monotonic()
        due = [row for row in self._replay_pending if row["replay_due_at"] <= now]
        if not due:
            return []
        due_ids = {row["attempt_id"] for row in due}
        self._replay_pending = [
            row for row in self._replay_pending if row["attempt_id"] not in due_ids
        ]
        return sorted(due, key=lambda row: float(row["replay_due_at"]))

    def _validate_live_catalog(self, catalog: dict[str, Any]) -> None:
        frozen = {str(row["id"]): row for row in catalog["programs"]}
        assert self._portfolio is not None
        for index, direction in enumerate(self._portfolio["directions"], start=1):
            for role, branch_id in [
                ("target", direction["target_branch_id"]),
                *[("donor", item) for item in direction["donor_branch_ids"]],
            ]:
                program = self._database.programs.get(branch_id)
                row = frozen.get(branch_id)
                if (
                    program is None
                    or row is None
                    or _sha256(program.code) != row.get("code_sha256")
                ):
                    raise ValueError(
                        f"direction {index} {role} is no longer the frozen branch: "
                        f"{branch_id}"
                    )

    def _load_handoff(self, path: Path) -> None:
        handoff = json.loads(path.read_text(encoding="utf-8"))
        if int(handoff.get("cohort_index", -1)) != self._cohort_index:
            raise ValueError("portfolio handoff cohort index mismatch")
        (
            self._planner_dir,
            self._portfolio,
            self._catalog,
            self._breadth_config,
        ) = _planner_documents(handoff["planner_dir"])
        self._restore_frozen_references(self._catalog)
        self._validate_live_catalog(self._catalog)
        self._audit_id = (
            "portfolio-overlap-"
            + _sha256(
                json.dumps(self._portfolio, sort_keys=True, separators=(",", ":"))
            )[:16]
        )
        self._by_attempt = {}
        self._assigned = []
        self._guided_attempt_ids = set()
        self._phase = "assigning_guided"
        self._journal(
            "overlap_portfolio_ready",
            planner_dir=str(self._planner_dir),
            direction_count=len(self._portfolio["directions"]),
            pretrigger_active_count=len(self._active_attempt_ids),
            breadth_config=self._breadth_config,
        )
        self._journal(
            "cohort_bound",
            direction_count=len(self._portfolio["directions"]),
            frontier_cycles=self._progress_anchor_cycles,
            breadth_config=self._breadth_config,
        )

    def reconcile(self) -> None:
        if self._phase != "waiting_portfolio":
            return
        handoff = self.control_dir / f"handoff_{self._cohort_index:03d}.json"
        if handoff.is_file():
            self._load_handoff(handoff)

    def _direction_payload(self, index: int) -> dict[str, Any]:
        from oe_trajectory import (
            BRANCH_PORTFOLIO_GUIDANCE_VERSION,
            diagnosis_guidance_hash,
        )

        assert self._portfolio is not None and self._audit_id is not None
        direction = self._portfolio["directions"][index]
        payload = {
            "plan_version": BRANCH_PORTFOLIO_GUIDANCE_VERSION,
            "audit_id": self._audit_id,
            "exhausted_patterns": list(self._portfolio.get("exhausted_patterns", [])),
            "lane_id": "intervention_direction",
            "lane_task": direction["task"],
            "portfolio_sha256": _sha256(
                json.dumps(self._portfolio, sort_keys=True, separators=(",", ":"))
            ),
            "target_branch_id": direction["target_branch_id"],
            "donor_branch_ids": list(direction["donor_branch_ids"]),
            "diagnosed_bottleneck": direction["mechanism_hypothesis"],
            "recipe": direction["task"],
            "compatibility": list(direction["constraints"]),
        }
        payload["plan_hash"] = diagnosis_guidance_hash(payload)
        return payload

    def active_plan_for_dispatch(
        self, *, attempt_id: str, iteration: int
    ) -> dict[str, Any] | None:
        self.reconcile()
        if attempt_id in self._by_attempt:
            return dict(self._by_attempt[attempt_id])
        if self._phase != "assigning_guided":
            return None
        assert self._portfolio is not None
        if len(self._assigned) >= len(self._portfolio["directions"]):
            return None
        payload = self._direction_payload(len(self._assigned))
        self._assigned.append(attempt_id)
        self._guided_attempt_ids.add(attempt_id)
        self._by_attempt[attempt_id] = payload
        self._journal(
            "direction_assigned",
            attempt_id=attempt_id,
            iteration=iteration,
            direction_index=len(self._assigned),
            target_branch_id=payload["target_branch_id"],
        )
        if len(self._assigned) == len(self._portfolio["directions"]):
            self._phase = "burst_draining_to_native_width"
            self._journal(
                "overlap_burst_fully_assigned",
                assigned_attempt_ids=list(self._assigned),
                total_active_count=len(self._active_attempt_ids) + 1,
            )
        return dict(payload)

    def ordinary_dispatch_fenced(self) -> bool:
        self.reconcile()
        if self._native_concurrency is None:
            raise RuntimeError("native concurrency was not configured")
        if self._phase == "waiting_portfolio":
            return True
        if self._phase == "assigning_guided":
            return False
        return len(self._active_attempt_ids) >= self._native_concurrency

    def on_attempt_scheduled(self, **payload: Any) -> None:
        attempt_id = payload.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        now = float(payload.get("at") or time.time())
        self._active_attempt_ids.add(attempt_id)
        self._scheduled_at[attempt_id] = now
        if payload.get("replayed_inflight") is True:
            self._pretrigger_attempt_ids.add(attempt_id)
        elif attempt_id not in self._guided_attempt_ids:
            self._ordinary_attempt_ids.add(attempt_id)
            if self._phase == "burst_draining_to_native_width":
                self._phase = "ordinary"
                self._flat_anchor_at = now
                self._journal(
                    "ordinary_search_resumed",
                    attempt_id=attempt_id,
                    active_count=len(self._active_attempt_ids),
                )
        self._journal("attempt_scheduled", **payload)

    def on_completion_consumed(self, **payload: Any) -> None:
        now = time.time()
        attempt_id = payload.get("attempt_id")
        if isinstance(attempt_id, str):
            self._active_attempt_ids.discard(attempt_id)
            self._terminal_attempt_ids.add(attempt_id)
            self._terminal_at[attempt_id] = now
        completion_seq = payload.get("completion_seq")
        if isinstance(completion_seq, int):
            self._completion_payloads[completion_seq] = dict(payload)
        iteration = payload.get("iteration")
        if isinstance(iteration, int):
            self._last_terminal_iteration = max(
                self._last_terminal_iteration, iteration
            )
        self._journal("attempt_terminal", **payload)
        if self._guided_attempt_ids and self._guided_attempt_ids.issubset(
            self._terminal_attempt_ids
        ):
            self._journal(
                "guided_cohort_complete",
                guided_attempt_ids=sorted(self._guided_attempt_ids),
            )
            self._guided_attempt_ids = set()
        if (
            isinstance(attempt_id, str)
            and attempt_id in self._ordinary_attempt_ids
            and not isinstance(payload.get("child"), dict)
        ):
            self._maybe_mark_flat_ready(completion_seq, now)

    def on_commit_failed(self, **payload: Any) -> None:
        completion = self._completion_payloads.get(payload.get("completion_seq"), {})
        if completion.get("attempt_id") in self._ordinary_attempt_ids:
            self._maybe_mark_flat_ready(payload.get("completion_seq"), time.time())
        self._journal("commit_failed", **payload)

    def capture_best(self) -> object:
        best = (
            self._database.programs.get(self._database.best_program_id)
            if self._database is not None
            else None
        )
        metrics = getattr(best, "metrics", None) or {}
        cycles = metrics.get("cycles")
        score = metrics.get("combined_score")
        return SimpleNamespace(
            best_id=getattr(best, "id", None),
            best_fitness=float(score) if isinstance(score, (int, float)) else None,
            cycles=float(cycles) if isinstance(cycles, (int, float)) else None,
        )

    def on_program_committed(self, **payload: Any) -> None:
        completion_seq = payload.get("completion_seq")
        completion = self._completion_payloads.get(completion_seq, {})
        child = completion.get("child")
        before, after = payload.get("before"), payload.get("after")
        cycles = (
            (child.get("metrics") or {}).get("cycles")
            if isinstance(child, dict)
            else None
        )
        if (
            completion.get("evaluation_valid") is True
            and getattr(after, "best_id", None) != getattr(before, "best_id", None)
            and isinstance(cycles, (int, float))
            and cycles > 0
        ):
            anchor = self._progress_anchor_cycles
            if anchor is None or (anchor - float(cycles)) / abs(anchor) > 0.002:
                self._progress_anchor_cycles = float(cycles)
                if self._phase == "ordinary":
                    self._flat_anchor_at = time.time()
                self._journal(
                    "meaningful_frontier_improvement",
                    completion_seq=completion_seq,
                    cycles=float(cycles),
                    source=(
                        "guided"
                        if completion.get("attempt_id") in self._by_attempt
                        else "ordinary_or_pretrigger"
                    ),
                )
        if completion.get("attempt_id") in self._ordinary_attempt_ids:
            self._maybe_mark_flat_ready(completion_seq, time.time())

    def _active_search_seconds(self, since_at: float, now_at: float) -> float:
        intervals = []
        for attempt_id in self._ordinary_attempt_ids:
            start = self._scheduled_at.get(attempt_id)
            if start is None:
                continue
            end = self._terminal_at.get(attempt_id, now_at)
            lo, hi = max(start, since_at), min(end, now_at)
            if hi > lo:
                intervals.append((lo, hi))
        intervals.sort()
        total, cursor = 0.0, None
        for lo, hi in intervals:
            if cursor is None or lo > cursor:
                total += hi - lo
                cursor = hi
            elif hi > cursor:
                total += hi - cursor
                cursor = hi
        return total

    def _terminal_attempts_since(self, since_at: float) -> int:
        return sum(
            at >= since_at
            for attempt_id, at in self._terminal_at.items()
            if attempt_id in self._ordinary_attempt_ids
        )

    def _maybe_mark_flat_ready(self, completion_seq: Any, now: float) -> None:
        terminal_attempts = (
            self._terminal_attempts_since(self._flat_anchor_at)
            if self._flat_anchor_at is not None
            else 0
        )
        if (
            self._phase != "ordinary"
            or (
                self._minimum_run_age_seconds > 0
                and now - self._run_started_at < self._minimum_run_age_seconds
            )
            or self._flat_anchor_at is None
            or self._guided_attempt_ids
            or not isinstance(completion_seq, int)
            or self._active_search_seconds(self._flat_anchor_at, now)
            < self._repeat_flat_seconds
            or terminal_attempts < self._repeat_flat_min_terminal_attempts
        ):
            return
        self._phase = "waiting_portfolio"
        self._cohort_index += 1
        self._request_pending = True
        self._request_cutoff_at = now
        self._journal(
            "repeat_flat_ready",
            completion_seq=completion_seq,
            active_search_seconds=self._active_search_seconds(
                self._flat_anchor_at, now
            ),
            terminal_attempts_since_anchor=terminal_attempts,
            minimum_terminal_attempts=self._repeat_flat_min_terminal_attempts,
            pretrigger_attempt_ids=sorted(self._active_attempt_ids),
            progress_anchor_cycles=self._progress_anchor_cycles,
        )

    def on_before_submit(self, iteration: int) -> None:
        self.reconcile()
        if not self._request_pending:
            return
        checkpoint = (
            self.output_dir
            / "checkpoints"
            / f"checkpoint_overlap_trigger_{self._cohort_index:03d}"
        )
        checkpoint.mkdir(parents=True, exist_ok=True)
        self._capture_frozen_programs()
        self._database.save(str(checkpoint), self._last_terminal_iteration)
        request = {
            "schema": "oe_overlap_planner_request_v1",
            "cohort_index": self._cohort_index,
            "cutoff_seq": max(self._completion_payloads, default=-1),
            "cutoff_at": self._request_cutoff_at,
            "checkpoint": str(checkpoint.resolve()),
            "next_iteration_fenced": iteration,
        }
        target = self.control_dir / f"request_{self._cohort_index:03d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
        self._request_pending = False
        self._journal("overlap_planner_requested", **request)

    def controller_heartbeat(self) -> None:
        """Persist a pending trigger only at a controller safe point."""
        if self._request_pending:
            self.on_before_submit(-1)

    def write_checkpoint_sidecar(self, **payload: Any) -> None:
        del payload

    def shutdown(self) -> None:
        self._journal("cohort_shutdown", phase=self._phase)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "portfolio-overlap",
            "minimum_run_age_seconds": self._minimum_run_age_seconds,
            "control_dir": str(self.control_dir),
            "native_concurrency": self._native_concurrency,
            "repeat_flat_seconds": self._repeat_flat_seconds,
            "repeat_flat_min_terminal_attempts": (
                self._repeat_flat_min_terminal_attempts
            ),
            "pending_inflight_count": len(self._replay_pending),
            "initial_phase": self._phase,
        }
