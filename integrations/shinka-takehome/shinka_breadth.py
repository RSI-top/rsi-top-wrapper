"""Opt-in repeated overlap breadth intervention for ShinkaEvolve.

The planner and selector are the OpenEvolve treatment's shared implementation.
This module only adapts its frozen branch directions to Shinka's scheduler and
native proposal/evaluation pipeline.  With the adapter absent, Shinka is
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


MEANINGFUL_RELATIVE_GAIN = 0.002
GUIDANCE_VERSION = "branch_recipe_portfolio_guidance_v2"
BURST_PROPOSAL_CAPACITY = 32


def _cycles(program: Any) -> float | None:
    value = (getattr(program, "public_metrics", None) or {}).get("cycles")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _metrics(program_or_row: Any) -> dict[str, Any]:
    if isinstance(program_or_row, dict):
        public = dict(program_or_row.get("public_metrics") or {})
        combined = program_or_row.get("combined_score")
    else:
        public = dict(getattr(program_or_row, "public_metrics", None) or {})
        combined = getattr(program_or_row, "combined_score", None)
    if isinstance(combined, (int, float)) and not isinstance(combined, bool):
        public["combined_score"] = float(combined)
    return public


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _guidance_hash(payload: dict[str, Any]) -> str:
    fields = (
        "plan_version",
        "audit_id",
        "diagnosis",
        "exhausted_patterns",
        "lane_id",
        "lane_task",
        "portfolio_sha256",
        "target_branch_id",
        "donor_branch_ids",
        "diagnosed_bottleneck",
        "recipe",
        "compatibility",
    )
    document = {key: payload[key] for key in fields if key in payload}
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _render_guidance(
    payload: dict[str, Any], donor_positions: dict[str, int] | None = None
) -> str:
    donors = payload.get("donor_branch_ids") or []
    donor_positions = donor_positions or {}
    donor_text = (
        "; ".join(
            f"{program_id} is Prior program "
            f"{donor_positions.get(program_id, index + 1)}"
            for index, program_id in enumerate(donors)
        )
        if donors
        else "(none)"
    )
    constraints = "\n".join(f"- {condition}" for condition in payload["compatibility"])
    diagnosis = payload.get("diagnosis")
    diagnosis_text = (
        "Independent planner diagnoses — context only, not separate tasks:\n"
        f"{diagnosis}\n\n"
        if diagnosis
        else ""
    )
    return (
        "==== META-INTERVENTION DIAGNOSIS COHORT TASK "
        f"(audit {payload['audit_id']}, {payload['plan_version']}) ====\n"
        "Execute one selected intervention direction for this work unit.\n\n"
        f"{diagnosis_text}"
        f"Target program: {payload['target_branch_id']} is the Current program below\n"
        f"Donor inspirations: {donor_text}\n"
        "Hypothesis:\n"
        f"{payload['diagnosed_bottleneck']}\n"
        "Action:\n"
        f"{payload['lane_task']}\n"
        "Constraints:\n"
        f"{constraints}\n\n"
        "Produce one concrete, bounded candidate for this action using the "
        "native ShinkaEvolve output format below.\n"
        "==== END META-INTERVENTION DIAGNOSIS COHORT TASK ===="
    )


class ShinkaBreadthGuidance:
    """One immutable selected direction bound to one Shinka generation."""

    def __init__(
        self,
        *,
        payload: dict[str, Any],
        frozen_programs: dict[str, Any],
        archive_slots: int,
        top_k_slots: int,
    ) -> None:
        self.payload = payload
        self._frozen_programs = frozen_programs
        self._archive_slots = max(0, int(archive_slots))
        self._top_k_slots = max(0, int(top_k_slots))

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "breadth_guided": True,
            "breadth_audit_id": self.payload["audit_id"],
            "breadth_plan_hash": self.payload["plan_hash"],
            "breadth_target_program_id": self.payload["target_branch_id"],
            "breadth_donor_program_ids": list(
                self.payload.get("donor_branch_ids") or []
            ),
        }

    def note_terminal(self, *, stage: str, reason: str | None = None) -> None:
        # The canonical completion observer records the terminal and closes
        # the cohort.  This method satisfies Shinka's pre-evaluation failure
        # callback without introducing a second lifecycle writer.
        del stage, reason

    async def sample_inputs(
        self,
        *,
        sample_context: Any,
    ) -> tuple[Any, list[Any], list[Any], bool]:
        target_id = self.payload["target_branch_id"]
        target = self._frozen_programs.get(target_id)
        if target is None:
            raise RuntimeError(f"frozen breadth target unavailable: {target_id}")

        archive_programs, top_k_programs = await sample_context(
            target,
            self._archive_slots,
            self._top_k_slots,
        )

        donors: list[Any] = []
        for donor_id in self.payload.get("donor_branch_ids") or []:
            donor = self._frozen_programs.get(donor_id)
            if donor is None:
                raise RuntimeError(f"frozen breadth donor unavailable: {donor_id}")
            donors.append(donor)
        if len(donors) > self._archive_slots + self._top_k_slots:
            raise RuntimeError(
                "breadth donors exceed Shinka's native inspiration capacity"
            )

        # Preserve Shinka's native context selection and channel capacities.
        # Planner donors lead; unused positions use samples selected for the
        # assigned target (never for an unrelated random parent).
        ordered = donors + list(archive_programs) + list(top_k_programs)
        unique: list[Any] = []
        seen = {str(getattr(target, "id", ""))}
        for program in ordered:
            program_id = str(getattr(program, "id", ""))
            if not program_id or program_id in seen:
                continue
            seen.add(program_id)
            unique.append(program)
        archive = unique[: self._archive_slots]
        top_k = unique[self._archive_slots : self._archive_slots + self._top_k_slots]
        # Planner targets are frozen correct programs. Patch type is still
        # chosen by Shinka's native policy from these actual inputs.
        return target, archive, top_k, False

    def build_prompt_override(
        self,
        *,
        prompt_sampler: Any,
        parent_program: Any,
        archive_programs: list[Any],
        top_k_programs: list[Any],
        needs_fix: bool,
        patch_type: str,
    ) -> tuple[str, str, str]:
        ordered = prompt_sampler.context_builder.build_context(
            archive_programs, top_k_programs
        )
        donor_positions = {
            str(program.id): index
            for index, program in enumerate(ordered, start=1)
            if str(program.id) in set(self.payload.get("donor_branch_ids") or [])
        }
        if len(donor_positions) != len(self.payload.get("donor_branch_ids") or []):
            raise RuntimeError(
                "a breadth donor is missing from the native prompt context"
            )
        task_block = _render_guidance(self.payload, donor_positions)
        if needs_fix:
            return prompt_sampler.sample_intervention_fix_task(
                incorrect_program=parent_program,
                ancestor_inspirations=archive_programs,
                audit_id=self.payload["audit_id"],
                task_block=task_block,
            )
        return prompt_sampler.sample_intervention_task(
            parent=parent_program,
            archive_inspirations=archive_programs,
            top_k_inspirations=top_k_programs,
            patch_type=patch_type,
            audit_id=self.payload["audit_id"],
            task_block=task_block,
        )

    def attach(self, prompt: str, *, patch_type: str, mode: str) -> str:
        del prompt, patch_type, mode
        raise RuntimeError("breadth guidance must use its dedicated prompt override")


class ShinkaOverlapBreadthManager:
    """20-minute/12-terminal repeated breadth controller with overlap."""

    def __init__(
        self,
        *,
        results_dir: str | Path,
        control_dir: str | Path,
        flat_seconds: float = 1200.0,
        flat_min_terminal_attempts: int = 12,
        start_ordinary: bool = False,
        stop_after_guided_cohort: bool = False,
        minimum_run_age_seconds: float = 0.0,
    ) -> None:
        if flat_seconds <= 0 or flat_min_terminal_attempts < 1:
            raise ValueError("breadth trigger limits must be positive")
        self.results_dir = Path(results_dir).resolve()
        self.control_dir = Path(control_dir).resolve()
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.flat_seconds = float(flat_seconds)
        self.flat_min_terminal_attempts = int(flat_min_terminal_attempts)
        self.start_ordinary = bool(start_ordinary)
        self._run_started_at = time.time()
        self.minimum_run_age_seconds = float(minimum_run_age_seconds)
        self.stop_after_guided_cohort = bool(stop_after_guided_cohort)
        self.phase = "ordinary" if self.start_ordinary else "waiting_portfolio"
        self.cohort_index = -1 if self.start_ordinary else 0
        self.runner: Any = None
        self.native_concurrency = 4
        self._tail_offset = 0
        self._dashboard_clock_segments: list[tuple[float, float]] = []
        self._normalized_path = self.results_dir / "breadth_normalized_attempts.jsonl"
        self._journal_path = self.results_dir / "breadth_interventions.jsonl"
        self._seen_seqs: set[int] = set()
        self._latest_seq = -1
        self._latest_at: float | None = None
        self._root_program: dict[str, Any] | None = None
        self._program_rows: dict[str, dict[str, Any]] = {}
        self._best_cycles: float | None = None
        self._progress_anchor_cycles: float | None = None
        self._flat_anchor_at: float | None = None
        self._scheduled_at: dict[str, float] = {}
        self._terminal_at: dict[str, float] = {}
        self._generation_by_attempt: dict[str, int] = {}
        self._ordinary_attempts: set[str] = set()
        self._guided_generations: set[int] = set()
        self._guided_terminal: set[int] = set()
        self._directions: list[dict[str, Any]] = []
        self._guidance_by_generation: dict[int, ShinkaBreadthGuidance] = {}
        self._assigned_count = 0
        self._audit_id: str | None = None
        self._frozen_programs: dict[str, Any] = {}

    def _journal(self, kind: str, **payload: Any) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": time.time(),
                        "kind": kind,
                        "cohort_index": self.cohort_index,
                        "audit_id": self._audit_id,
                        **payload,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    @staticmethod
    def _attempt_id(row: dict[str, Any]) -> str:
        return str(row.get("attempt_id") or row.get("attempt_key") or "")

    def _dashboard_time(self, timestamp: float) -> float:
        """Remove offline resume gaps only from the planner's display clock."""
        for start, offset in reversed(self._dashboard_clock_segments):
            if timestamp >= start:
                return timestamp - offset
        return timestamp

    def _normalized_events(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        seq = int(row["completion_seq"])
        child = dict(row.get("child") or {})
        child["metrics"] = _metrics(child)
        attempt_id = self._attempt_id(row) or f"seq-{seq}"
        parent_id = row.get("scheduled_parent_id") or child.get("parent_id")
        parent = self._program_rows.get(str(parent_id)) if parent_id else None
        scheduled_at = row.get("scheduled_at")
        if not isinstance(scheduled_at, (int, float)):
            scheduled_at = row.get("timestamp")
        valid = bool(child.get("correct") and child["metrics"].get("cycles"))
        schedule = {
            "event": "attempt_scheduled",
            "timestamp": self._dashboard_time(float(scheduled_at)),
            "attempt_id": attempt_id,
            "parent_id": parent_id,
            "parent_metrics": _metrics(parent or {}),
            "parent_code": (parent or {}).get("code"),
            "parent_code_sha256": (
                _sha256_text(str(parent["code"]))
                if parent and isinstance(parent.get("code"), str)
                else None
            ),
            "target_island": child.get("island_idx"),
        }
        canonical = dict(row)
        canonical.update(
            {
                "event": "canonical_completion",
                "timestamp": self._dashboard_time(float(row["timestamp"])),
                "scheduled_at": self._dashboard_time(float(scheduled_at)),
                "attempt_id": attempt_id,
                "child": child,
                "evaluation_valid": valid,
                "failure_stage": None
                if valid
                else (row.get("failure_stage") or "incorrect_output"),
                "metadata": dict(child.get("metadata") or {}),
            }
        )
        committed = {
            "event": "program_committed",
            "timestamp": self._dashboard_time(
                float(row.get("timestamp") or time.time())
            ),
            "completion_seq": seq,
            "attempt_id": attempt_id,
            "program_id": child.get("id"),
            "metrics": child["metrics"],
            "metadata": dict(child.get("metadata") or {}),
        }
        return [schedule, canonical, committed]

    def _append_normalized(self, events: list[dict[str, Any]]) -> None:
        with self._normalized_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _observe_completion(self, row: dict[str, Any], *, initialize: bool) -> None:
        seq = row.get("completion_seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq in self._seen_seqs:
            return
        self._seen_seqs.add(seq)
        self._latest_seq = max(self._latest_seq, seq)
        at = float(row.get("timestamp") or time.time())
        self._latest_at = at
        attempt_id = self._attempt_id(row) or f"seq-{seq}"
        generation = int(row.get("generation") or -1)
        scheduled_at = row.get("scheduled_at")
        if isinstance(scheduled_at, (int, float)):
            self._scheduled_at[attempt_id] = float(scheduled_at)
        self._terminal_at[attempt_id] = at
        self._generation_by_attempt[attempt_id] = generation
        if generation not in self._guided_generations:
            self._ordinary_attempts.add(attempt_id)
        else:
            self._guided_terminal.add(generation)

        child = row.get("child") if isinstance(row.get("child"), dict) else {}
        program_id = child.get("id")
        if isinstance(program_id, str) and program_id:
            self._program_rows[program_id] = dict(child)
        events = self._normalized_events(row)
        if not initialize:
            self._append_normalized(events)
            if generation in self._guided_generations:
                self._journal(
                    "guided_attempt_terminal",
                    generation=generation,
                    completion_seq=seq,
                    evaluation_valid=bool(child.get("correct")),
                    child_id=child.get("id"),
                    cycles=_cycles_from_row(child),
                )

        cycles = _cycles_from_row(child)
        if bool(child.get("correct")) and cycles is not None:
            if self._best_cycles is None or cycles < self._best_cycles:
                self._best_cycles = cycles
            anchor = self._progress_anchor_cycles
            if (
                anchor is None
                or (anchor - cycles) / abs(anchor) > MEANINGFUL_RELATIVE_GAIN
            ):
                self._progress_anchor_cycles = cycles
                if self.phase == "ordinary":
                    self._flat_anchor_at = at
                if not initialize:
                    self._journal(
                        "meaningful_frontier_improvement",
                        completion_seq=seq,
                        cycles=cycles,
                        source=(
                            "guided"
                            if generation in self._guided_generations
                            else "ordinary"
                        ),
                    )
        if not initialize:
            self._maybe_trigger(seq, at)
            if (
                self._guided_generations
                and self._guided_generations <= self._guided_terminal
            ):
                self._journal(
                    "guided_cohort_complete",
                    guided_generations=sorted(self._guided_generations),
                )
                self._guided_generations.clear()
                self._guided_terminal.clear()

    def _initialize_trace(self) -> None:
        source = self.results_dir / "attempts.jsonl"
        started: dict[str, Any] | None = None
        roots: list[dict[str, Any]] = []
        completions: list[dict[str, Any]] = []
        last_terminal_at: float | None = None
        clock_offset = 0.0
        if source.is_file():
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    if not any(
                        marker in line
                        for marker in (
                            '"event": "run_started"',
                            '"event": "root_program_committed"',
                            '"event": "canonical_completion"',
                        )
                    ):
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("event") == "run_started":
                        resumed_at = float(row["timestamp"])
                        if (
                            last_terminal_at is not None
                            and resumed_at > last_terminal_at
                        ):
                            clock_offset += resumed_at - last_terminal_at
                            self._dashboard_clock_segments.append(
                                (resumed_at, clock_offset)
                            )
                        if started is None or float(
                            row.get("timestamp", math.inf)
                        ) < float(started.get("timestamp", math.inf)):
                            started = row
                    elif row.get("event") == "root_program_committed":
                        if isinstance(row.get("program"), dict):
                            roots.append(row["program"])
                    elif row.get("event") == "canonical_completion":
                        completions.append(row)
                        last_terminal_at = float(row["timestamp"])
            self._tail_offset = source.stat().st_size

        programs = list(self.runner.db.get_all_programs())
        root_obj = next(
            (
                program
                for program in programs
                if int(getattr(program, "generation", 0) or 0) == 0
                and not getattr(program, "parent_id", None)
            ),
            None,
        )
        root = roots[0] if roots else (root_obj.to_dict() if root_obj else None)
        if root is None:
            raise RuntimeError("Shinka breadth requires a root program")
        self._root_program = dict(root)
        root_id = str(root.get("id") or "root")
        self._program_rows[root_id] = dict(root)
        root_cycles = _cycles_from_row(root)
        if root_cycles is None:
            raise RuntimeError("Shinka breadth root has no finite cycles")
        self._best_cycles = root_cycles
        self._progress_anchor_cycles = root_cycles

        started_at = float(
            (started or {}).get("timestamp") or root.get("timestamp") or time.time()
        )
        self._flat_anchor_at = started_at
        self._normalized_path.write_text(
            json.dumps(
                {
                    "event": "run_started",
                    "timestamp": started_at,
                    "run_instance_id": "shinka-normalized",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for row in sorted(completions, key=lambda item: int(item["completion_seq"])):
            self._observe_completion(row, initialize=True)
            self._append_normalized(self._normalized_events(row))

    def bind(self, runner: Any) -> None:
        self.runner = runner
        self.native_concurrency = int(runner.max_evaluation_jobs)
        # Keep the sealed/raw Shinka configuration at its native value.  The
        # controller widens only this opt-in runtime ceiling so one selected
        # breadth cohort can be assigned; the ordinary pipeline target remains
        # ``native_concurrency`` and therefore returns to width four.
        runner.max_proposal_jobs = max(
            int(runner.max_proposal_jobs), BURST_PROPOSAL_CAPACITY
        )
        self._initialize_trace()
        if self.phase == "ordinary":
            self._flat_anchor_at = time.time()
        self._journal(
            "controller_bound",
            phase=self.phase,
            native_concurrency=self.native_concurrency,
            last_completion_seq=self._latest_seq,
        )
        if self.phase == "waiting_portfolio":
            if self._latest_seq < 0:
                raise RuntimeError(
                    "initial Shinka breadth intervention requires a materialized completion"
                )
            self._request_portfolio(self._latest_seq, self._latest_at or time.time())

    def _poll_trace(self) -> None:
        source = self.results_dir / "attempts.jsonl"
        if not source.is_file():
            return
        with source.open(encoding="utf-8") as handle:
            handle.seek(self._tail_offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(start)
                    break
                self._tail_offset = handle.tell()
                if '"event": "attempt_scheduled"' in line:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    attempt_id = self._attempt_id(row)
                    generation = int(row.get("generation") or -1)
                    if attempt_id:
                        self._scheduled_at[attempt_id] = float(
                            row.get("timestamp") or time.time()
                        )
                        self._generation_by_attempt[attempt_id] = generation
                        if generation not in self._guided_generations:
                            self._ordinary_attempts.add(attempt_id)
                    continue
                if '"event": "canonical_completion"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                self._observe_completion(row, initialize=False)

    def _active_search_seconds(self, since: float, now: float) -> float:
        intervals: list[tuple[float, float]] = []
        for attempt_id in self._ordinary_attempts:
            start = self._scheduled_at.get(attempt_id)
            if start is None:
                continue
            end = self._terminal_at.get(attempt_id, now)
            lo, hi = max(start, since), min(end, now)
            if hi > lo:
                intervals.append((lo, hi))
        intervals.sort()
        total = 0.0
        cursor: float | None = None
        for lo, hi in intervals:
            if cursor is None or lo > cursor:
                total += hi - lo
                cursor = hi
            elif hi > cursor:
                total += hi - cursor
                cursor = hi
        return total

    def _maybe_trigger(self, completion_seq: int, now: float) -> None:
        if (
            self.phase != "ordinary"
            or (
                self.minimum_run_age_seconds > 0
                and now - self._run_started_at < self.minimum_run_age_seconds
            )
            or self._flat_anchor_at is None
            or self._guided_generations
        ):
            return
        terminals = sum(
            at >= self._flat_anchor_at
            for attempt_id, at in self._terminal_at.items()
            if attempt_id in self._ordinary_attempts
        )
        active_seconds = self._active_search_seconds(self._flat_anchor_at, now)
        if (
            active_seconds < self.flat_seconds
            or terminals < self.flat_min_terminal_attempts
        ):
            return
        self.cohort_index += 1
        self.phase = "waiting_portfolio"
        self._journal(
            "repeat_flat_ready",
            completion_seq=completion_seq,
            active_search_seconds=active_seconds,
            terminal_attempts_since_anchor=terminals,
            progress_anchor_cycles=self._progress_anchor_cycles,
        )
        self._request_portfolio(completion_seq, now)

    def _snapshot_database(self, checkpoint: Path) -> None:
        programs = [
            program
            for program in self.runner.db.get_all_programs()
            if bool(getattr(program, "correct", False))
            and isinstance(getattr(program, "code", None), str)
            and getattr(program, "code", "")
        ]
        if not programs:
            raise RuntimeError(
                "no correct Shinka programs available for breadth planning"
            )
        best = min(programs, key=lambda program: _cycles(program) or math.inf)
        islands: list[list[str]] = []
        island_count = int(getattr(self.runner.db_config, "num_islands", 1) or 1)
        for island in range(island_count):
            islands.append(
                sorted(
                    str(program.id)
                    for program in programs
                    if int(getattr(program, "island_idx", 0) or 0) == island
                )
            )
        island_bests: list[str | None] = []
        for members in islands:
            candidates = [program for program in programs if str(program.id) in members]
            island_bests.append(
                str(min(candidates, key=lambda item: _cycles(item) or math.inf).id)
                if candidates
                else None
            )
        checkpoint.mkdir(parents=True, exist_ok=False)
        program_dir = checkpoint / "programs"
        program_dir.mkdir()
        for program in programs:
            document = program.to_dict()
            document["metrics"] = _metrics(program)
            _write_json(program_dir / f"{program.id}.json", document)
        _write_json(
            checkpoint / "metadata.json",
            {
                "best_program_id": str(best.id),
                "islands": islands,
                "island_best_programs": island_bests,
            },
        )
        self._frozen_programs = {str(program.id): program for program in programs}

    def _request_portfolio(self, completion_seq: int, cutoff_at: float) -> None:
        checkpoint = self.control_dir / f"checkpoint_{self.cohort_index:03d}"
        self._snapshot_database(checkpoint)
        normalized = self.control_dir / f"attempts_{self.cohort_index:03d}.jsonl"
        normalized.write_bytes(self._normalized_path.read_bytes())
        request = {
            "schema": "shinka_overlap_planner_request_v1",
            "cohort_index": self.cohort_index,
            "cutoff_seq": int(completion_seq),
            "cutoff_at": self._dashboard_time(float(cutoff_at)),
            "source_cutoff_at": float(cutoff_at),
            "checkpoint": str(checkpoint),
            "normalized_attempts": str(normalized),
        }
        _write_json(self.control_dir / f"request_{self.cohort_index:03d}.json", request)
        self._journal("planner_requested", **request)

    def _load_handoff(self) -> None:
        if self.phase != "waiting_portfolio":
            return
        path = self.control_dir / f"handoff_{self.cohort_index:03d}.json"
        if not path.is_file():
            return
        handoff = json.loads(path.read_text(encoding="utf-8"))
        if int(handoff.get("cohort_index", -1)) != self.cohort_index:
            raise RuntimeError("Shinka breadth handoff cohort mismatch")
        planner = Path(str(handoff["planner_dir"]))
        portfolio = json.loads((planner / "portfolio.json").read_text(encoding="utf-8"))
        catalog = json.loads(
            (planner / "bundle/workspace/meta/branch_catalog.json").read_text(
                encoding="utf-8"
            )
        )
        catalog_by_id = {str(row["id"]): row for row in catalog["programs"]}
        directions = portfolio.get("directions")
        if not isinstance(directions, list) or not directions:
            raise RuntimeError("Shinka breadth planner returned no directions")
        for direction in directions:
            branch_ids = [
                str(direction["target_branch_id"]),
                *[str(item) for item in direction.get("donor_branch_ids") or []],
            ]
            for branch_id in branch_ids:
                program = self._frozen_programs.get(branch_id)
                row = catalog_by_id.get(branch_id)
                if (
                    program is None
                    or row is None
                    or _sha256_text(program.code) != row.get("code_sha256")
                ):
                    raise RuntimeError(
                        f"selected Shinka frozen branch is invalid: {branch_id}"
                    )
        self._audit_id = (
            "shinka-breadth-"
            + hashlib.sha256(
                json.dumps(portfolio, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
        )
        diagnosis = portfolio.get("diagnosis")
        if not diagnosis:
            diagnoses = portfolio.get("diagnoses") or []
            if diagnoses:
                diagnosis = "\n".join(str(item) for item in diagnoses)
        portfolio_sha = hashlib.sha256(
            json.dumps(portfolio, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._directions = []
        for direction in directions:
            payload = {
                "plan_version": GUIDANCE_VERSION,
                "audit_id": self._audit_id,
                "diagnosis": diagnosis,
                "exhausted_patterns": list(portfolio.get("exhausted_patterns") or []),
                "lane_id": "intervention_direction",
                "lane_task": str(direction["task"]),
                "portfolio_sha256": portfolio_sha,
                "target_branch_id": str(direction["target_branch_id"]),
                "donor_branch_ids": [
                    str(item) for item in direction.get("donor_branch_ids") or []
                ],
                "diagnosed_bottleneck": str(direction["mechanism_hypothesis"]),
                "recipe": str(direction["task"]),
                "compatibility": [str(item) for item in direction["constraints"]],
            }
            payload["plan_hash"] = _guidance_hash(payload)
            self._directions.append(payload)
        self._assigned_count = 0
        self.phase = "assigning_guided"
        self._journal(
            "portfolio_ready",
            planner_dir=str(planner),
            direction_count=len(self._directions),
        )
        self._journal(
            "cohort_bound",
            direction_count=len(self._directions),
            frontier_cycles=self._best_cycles,
        )

    def poll(self) -> None:
        self._poll_trace()
        self._load_handoff()

    def pipeline_target(self, native_target: int) -> int:
        self.poll()
        if self.phase == "waiting_portfolio" or (
            self.stop_after_guided_cohort
            and self.phase == "burst_draining_to_native_width"
        ):
            return 0
        if self.phase == "assigning_guided":
            capacity = len(self.runner.running_jobs) + len(
                self.runner.active_proposal_tasks
            )
            return capacity + len(self._directions) - self._assigned_count
        return int(native_target)

    def prepare_start(self, requested: int) -> int:
        self.poll()
        if self.phase == "waiting_portfolio" or (
            self.stop_after_guided_cohort
            and self.phase == "burst_draining_to_native_width"
        ):
            return 0
        if self.phase == "assigning_guided":
            return min(requested, len(self._directions) - self._assigned_count)
        if self.phase == "burst_draining_to_native_width":
            self.phase = "ordinary"
            self._flat_anchor_at = time.time()
            self._ordinary_attempts = {
                attempt_id
                for attempt_id in self._ordinary_attempts
                if self._terminal_at.get(attempt_id, math.inf) >= self._flat_anchor_at
            }
            self._journal("ordinary_search_resumed")
        return requested

    def reserve_generations(self, first_generation: int, count: int) -> None:
        if self.phase != "assigning_guided" or count <= 0:
            return
        archive_slots = int(
            getattr(self.runner.db_config, "num_archive_inspirations", 1) or 0
        )
        top_k_slots = int(
            getattr(self.runner.db_config, "num_top_k_inspirations", 1) or 0
        )
        for offset in range(count):
            generation = first_generation + offset
            payload = self._directions[self._assigned_count]
            self._guidance_by_generation[generation] = ShinkaBreadthGuidance(
                payload=payload,
                frozen_programs=self._frozen_programs,
                archive_slots=archive_slots,
                top_k_slots=top_k_slots,
            )
            self._guided_generations.add(generation)
            self._assigned_count += 1
            self._journal(
                "direction_assigned",
                generation=generation,
                direction_index=self._assigned_count,
                target_branch_id=payload["target_branch_id"],
                donor_branch_ids=payload["donor_branch_ids"],
            )
        if self._assigned_count == len(self._directions):
            self.phase = "burst_draining_to_native_width"
            self._journal(
                "breadth_burst_fully_assigned",
                assigned_count=self._assigned_count,
            )

    def guidance_for_generation(self, generation: int) -> ShinkaBreadthGuidance | None:
        return self._guidance_by_generation.get(int(generation))

    def shutdown(self) -> None:
        self.poll()
        self._journal("controller_shutdown", phase=self.phase)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "repeat-flat-breadth-overlap",
            "flat_seconds": self.flat_seconds,
            "flat_min_terminal_attempts": self.flat_min_terminal_attempts,
            "start_ordinary": self.start_ordinary,
            "stop_after_guided_cohort": self.stop_after_guided_cohort,
            "control_dir": str(self.control_dir),
        }


def _cycles_from_row(row: dict[str, Any]) -> float | None:
    value = (row.get("public_metrics") or row.get("metrics") or {}).get("cycles")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def install_shinka_breadth(manager: ShinkaOverlapBreadthManager) -> None:
    """Install the minimal inert-unless-enabled scheduler hooks."""
    from shinka.core import ShinkaEvolveRunner

    cls = ShinkaEvolveRunner
    if getattr(cls, "_shinka_breadth_installed", False):
        raise RuntimeError("Shinka breadth adapter is already installed")

    original_setup = cls._setup_async

    async def setup(self):
        await original_setup(self)
        manager.bind(self)
        self._meta_plan_provider = manager.guidance_for_generation

    cls._setup_async = setup

    original_target = cls._compute_proposal_pipeline_target

    def target(self):
        return manager.pipeline_target(original_target(self))

    cls._compute_proposal_pipeline_target = target

    original_stuck = cls._is_system_stuck

    def stuck(self):
        # An empty native pipeline is intentional while the external planner
        # owns the next dispatch.  Without this guard Shinka's one-minute
        # deadlock recovery would eventually stop a healthy planner wait.
        if manager.phase == "waiting_portfolio":
            return False
        return original_stuck(self)

    cls._is_system_stuck = stuck

    original_start = cls._start_proposals

    async def start(self, num_proposals):
        count = manager.prepare_start(int(num_proposals))
        if count <= 0:
            return None
        available = min(
            count,
            self.max_proposal_jobs - len(self.active_proposal_tasks),
            self.evo_config.num_generations - self.next_generation_to_submit,
        )
        if available <= 0:
            return None
        first_generation = int(self.next_generation_to_submit)
        manager.reserve_generations(first_generation, available)
        return await original_start(self, available)

    cls._start_proposals = start
    cls._shinka_breadth_installed = True
