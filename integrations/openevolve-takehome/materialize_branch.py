#!/usr/bin/env python3
"""Materialize an exact OpenEvolve resume checkpoint at one completion.

This is state restoration only.  It does not consume an offline audit verdict,
choose an intervention, or change the current online audit policy.  Given a
trace emitted by this codebase and a canonical completion sequence, it restores
the native database at that controller boundary, settles only work that was
already in flight there from its recorded terminal results, and writes a normal
vendor checkpoint plus the causal ledger prefix used by ``runner.py``.

Fixed-system scope: ONLY traces emitted by this exact codebase, current
instrumentation schema (``oe_db_state_v1`` records, complete
``canonical_completion.child`` payloads), and current config. No older-format
fallbacks or experimental settings.

Behavior:

1. ``--completion-seq`` must name a canonical completion with a correlated
   ``program_committed`` record in a current-schema trace; resumed traces may
   contain earlier causal run instances, while the cutoff itself must resolve
   to one unambiguous instance;
2. the causal boundary is that completion's full recorded native
   post-processing block (post-add,
   correlated island increment, and its migration record when present); a
   timestamp is never used to guess the boundary;
3. the exact boundary ProgramDatabase structural state is restored by
   synthesizing a vendor checkpoint from the boundary ``oe_db_state_v1``
   snapshot, resolving every ``state.program_id`` payload only from verified
   current-schema sources (checkpoint program JSON, complete canonical
   child, migration ``new_programs`` + code-digest match, or the verified
   initial program), loading it through the vendor ``database.load``, and
   requiring the recomputed ``database_state_snapshot`` SHA to equal the
   record's; recorded artifacts are reapplied because they enter native
   prompts;
4. attempts dispatched before the boundary but not terminal there are
   settled EXACTLY from their recorded canonical completions in
   ``completion_seq`` order — vendor ``database.add`` with the recorded
   iteration/target island, artifacts reapplied, the recorded island
   incremented exactly once; failed/no-child/commit-failed attempts mutate
   nothing; every later replacement dispatch and unrelated post-boundary
   completion is suppressed;
5. minimal-v1 guard: if settlement reaches ``should_migrate()`` the run
   FAILS CLOSED — no deferred/replayed/counterfactual migration machinery
   (boundary-block migrations recorded before resolution are restored
   normally in step 3);
6. the branch checkpoint's ``last_iteration`` makes the runner resume at the
   first excluded native slot; ``branch_manifest.json`` records source and
   boundary digests, the excluded next slot, handled in-flight ids and
   completion order, original time consumed by settlement, pre/post state
   fingerprints, artifacts restored, and the explicit migration guard; the
   written checkpoint is reloaded and its fingerprint recomputed before
   success. A byte-exact ``historical_prefix.jsonl`` — the branch's causally
   effective ledger history (boundary prefix + settled in-flight terminals,
   rolled-back history excluded under EXACT provenance only), proven through
   ``DispatchLedger.recover`` to end with no in-flight/orphaned dispatch —
   is written for ``runner.py``'s automatic branch bootstrap, giving the
   resumed auditor continuous native completion numbering and bundle/replan
   history and trigger-relevant search history without reapplying anything to
   the database. The
   source trace is never modified (its digest is re-checked at the end).

Fail-closed conditions (all raise :class:`MaterializeError`): missing/corrupt
completion provenance, missing boundary
``db_state`` record, missing/mismatched program/code/artifact payloads,
fingerprint mismatch, a boundary-in-flight attempt with no recorded
terminal, instance mismatch, unexpected vendor add failure, or a migration
required during settlement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dispatch_ledger import fitness_of  # noqa: E402
from source_files import ReplayProofError, _verified_file  # noqa: E402
from oe_trajectory import (  # noqa: E402
    DB_STATE_SCHEMA,
    database_state_sha256,
    database_state_snapshot,
    sha256_text,
)

BRANCH_MANIFEST_SCHEMA = "oe_resume_branch_v2"

_CONTROLLER_EVENTS = (
    "attempt_scheduled",
    "canonical_completion",
    "program_committed",
)

_METADATA_KEYS = (
    "island_feature_maps",
    "islands",
    "archive",
    "best_program_id",
    "island_best_programs",
    "last_iteration",
    "current_island",
    "island_generations",
    "last_migration_generation",
    "feature_stats",
)


class MaterializeError(RuntimeError):
    """A real proof failure under the fixed schema — never guess past it."""


def _dump_json(path: Path, doc: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class _Trace:
    """Read-only line-indexed view of attempts.jsonl (indices == replay's)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.lines = raw.decode("utf-8", errors="replace").splitlines()
        self.rows: list[dict[str, Any] | None] = []
        for line in self.lines:
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            self.rows.append(row if isinstance(row, dict) else None)

    def row(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self.rows) or self.rows[index] is None:
            raise MaterializeError(f"trace line {index} is not a readable record")
        return self.rows[index]

    def verify_unchanged(self) -> None:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise MaterializeError("attempts.jsonl changed during materialization")


class BranchMaterializer:
    def __init__(
        self,
        *,
        trace_root: Path,
        completion_seq: int,
        out_dir: Path,
        preserve_inflight: bool = False,
    ) -> None:
        self.trace_root = Path(trace_root)
        self.cutoff_seq = int(completion_seq)
        self.out = Path(out_dir)
        self.trace = _Trace(self.trace_root / "attempts.jsonl")
        self.boundary_dir = self.out / "boundary_state"
        self.branch_dir = self.out / "branch_checkpoint"
        self.preserve_inflight = bool(preserve_inflight)
        self.pending_inflight: list[dict[str, Any]] = []
        self._payload_sources = {
            "canonical_child": 0,
            "migration_payload": 0,
            "checkpoint_json": 0,
            "verified_initial_program": 0,
        }
        self._artifact_counts = {"boundary_programs": 0, "settled_children": 0}

    # -- step 1: exact completion boundary ----------------------------------

    def _load_sources(self) -> None:
        canonical = [
            (index, row)
            for index, row in enumerate(self.trace.rows)
            if row is not None
            and row.get("event") == "canonical_completion"
            and row.get("completion_seq") == self.cutoff_seq
        ]
        commits = [
            (index, row)
            for index, row in enumerate(self.trace.rows)
            if row is not None
            and row.get("event") == "program_committed"
            and row.get("completion_seq") == self.cutoff_seq
        ]
        if len(canonical) != 1 or len(commits) != 1:
            raise MaterializeError(
                f"completion_seq {self.cutoff_seq} must have exactly one "
                "canonical_completion and one program_committed record"
            )
        canonical_index, canonical_row = canonical[0]
        self.resolution_index, commit_row = commits[0]
        cutoff_instances = {
            row.get("run_instance_id")
            for row in (canonical_row, commit_row)
            if isinstance(row.get("run_instance_id"), str)
        }
        if len(cutoff_instances) != 1:
            raise MaterializeError("cutoff completion spans ambiguous run instances")
        self.instance = next(iter(cutoff_instances))
        starts = [
            row
            for row in self.trace.rows
            if row is not None
            and row.get("event") == "run_started"
            and row.get("run_instance_id") == self.instance
        ]
        if len(starts) != 1:
            raise MaterializeError(
                "cutoff run instance must have exactly one run_started record"
            )
        if self.resolution_index <= canonical_index:
            raise MaterializeError(
                "program_committed precedes its canonical completion"
            )
        if (
            canonical_row.get("run_instance_id") != self.instance
            or commit_row.get("run_instance_id") != self.instance
            or (canonical_row.get("child") or {}).get("id")
            != commit_row.get("program_id")
        ):
            raise MaterializeError("cutoff completion/commit provenance mismatch")
        self.relation = "after_program_committed"
        self.cutoff_at = float(commit_row.get("timestamp") or 0.0)

    # -- step 2: boundary ----------------------------------------------------

    def _require_db_state(self, index: int) -> dict[str, Any]:
        row = self.trace.row(index)
        if (
            row.get("event") != "db_state"
            or row.get("db_state_schema") != DB_STATE_SCHEMA
        ):
            raise MaterializeError(
                f"trace line {index} is not an {DB_STATE_SCHEMA} record"
            )
        if database_state_sha256(row["state"]) != row.get("state_sha256"):
            raise MaterializeError(
                f"db_state record at line {index} fails its own fingerprint"
            )
        return row

    def _locate_boundary(self) -> None:
        if self.relation not in ("after_program_committed", "before_attempt_scheduled"):
            raise MaterializeError(
                f"resolution relation {self.relation!r} is not materializable "
                "(instance-end boundaries are out of minimal-v1 scope)"
            )
        if not isinstance(self.resolution_index, int):
            raise MaterializeError("cutoff lacks a resolution event index")
        boundary_row = self.trace.row(self.resolution_index)
        if boundary_row.get("run_instance_id") != self.instance:
            raise MaterializeError(
                "resolution event belongs to a different run instance"
            )
        block: list[tuple[int, dict[str, Any]]] = []
        if self.relation == "after_program_committed":
            if boundary_row.get("event") != "program_committed":
                raise MaterializeError(
                    "after_program_committed boundary does not point at a "
                    "program_committed record"
                )
            seq = int(boundary_row["completion_seq"])
            # The completion's full native post-processing block: every
            # db_state record until this instance's next controller event.
            for index in range(self.resolution_index + 1, len(self.trace.rows)):
                row = self.trace.rows[index]
                if row is None or row.get("run_instance_id") != self.instance:
                    continue
                event = row.get("event")
                if event == "db_state":
                    capture = row.get("capture")
                    if capture in ("program_committed", "island_generation_increment"):
                        if row.get("completion_seq") != seq:
                            raise MaterializeError(
                                f"boundary block at seq {seq} contains a "
                                "db_state record for a different completion"
                            )
                    block.append((index, self._require_db_state(index)))
                elif event in _CONTROLLER_EVENTS:
                    break
            if not any(row.get("capture") == "program_committed" for _, row in block):
                raise MaterializeError(
                    f"missing post-add db_state record for boundary seq {seq}"
                )
        else:
            if boundary_row.get("event") != "attempt_scheduled":
                raise MaterializeError(
                    "before_attempt_scheduled boundary does not point at an "
                    "attempt_scheduled record"
                )
            for index in range(self.resolution_index - 1, -1, -1):
                row = self.trace.rows[index]
                if (
                    row is not None
                    and row.get("event") == "db_state"
                    and row.get("run_instance_id") == self.instance
                ):
                    block = [(index, self._require_db_state(index))]
                    break
        if not block:
            raise MaterializeError(
                "no boundary db_state record — this trace cannot be "
                "materialized (old traces stay fail-closed)"
            )
        self.boundary_state_index, self.boundary_state_row = block[-1]
        self.boundary_block_lines = [index for index, _ in block]
        self.state = self.boundary_state_row["state"]

        dispatched = [
            row["iteration"]
            for row in self.trace.rows[: self.resolution_index]
            if row is not None
            and row.get("event") == "attempt_scheduled"
            and row.get("run_instance_id") == self.instance
        ]
        if not dispatched:
            raise MaterializeError("no pre-boundary dispatches in this instance")
        # Prove the ACTUAL excluded next native slot: the first same-instance
        # dispatch after the completed boundary block. Never invent a resume
        # iteration — a run that ends with no next slot fails closed.
        if self.relation == "before_attempt_scheduled":
            next_slot_index, next_slot_row = self.resolution_index, boundary_row
        else:
            next_slot_index = next_slot_row = None
            for index in range(self.boundary_state_index + 1, len(self.trace.rows)):
                row = self.trace.rows[index]
                if (
                    row is not None
                    and row.get("event") == "attempt_scheduled"
                    and row.get("run_instance_id") == self.instance
                ):
                    next_slot_index, next_slot_row = index, row
                    break
            if next_slot_row is None:
                raise MaterializeError(
                    "no native next slot: this instance never dispatched again "
                    "after the boundary block — failing closed"
                )
        if next_slot_row.get("iteration") != max(dispatched) + 1:
            raise MaterializeError(
                "excluded next-slot iteration disagrees with the dispatch "
                "history — ambiguous concurrency, failing closed"
            )
        self.excluded_next_iteration = max(dispatched) + 1
        self.excluded_next_slot_index = next_slot_index

    # -- step 3: payload resolution + restore -------------------------------

    def _instance_started(self, instance_id: str) -> dict[str, Any]:
        for row in self.trace.rows:
            if (
                row is not None
                and row.get("event") == "run_started"
                and row.get("run_instance_id") == instance_id
            ):
                return row
        raise MaterializeError(f"no run_started record for instance {instance_id}")

    def _load_verified_config(self):
        started = self._instance_started(self.instance)
        path, _ = _verified_file(
            recorded_path=started.get("config_path"),
            recorded_sha=started.get("config_sha256"),
            fallbacks=[self.trace_root / "config.yaml", HERE / "config.yaml"],
            what="run config",
        )
        from openevolve.config import load_config

        return load_config(path)

    def _build_payload_index(self) -> None:
        limit = self.boundary_state_index + 1
        self.children: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.migrant_payloads: dict[str, dict[str, Any]] = {}
        self.roots: dict[str, dict[str, Any]] = {}
        first_schedule_seen: set[str] = set()
        for row in self.trace.rows[:limit]:
            if row is None:
                continue
            event = row.get("event")
            if event == "canonical_completion" and isinstance(row.get("child"), dict):
                child = row["child"]
                if isinstance(child.get("id"), str):
                    self.children.setdefault(child["id"], (child, row))
            elif event == "db_state" and row.get("capture") == "island_migration":
                for payload in row.get("new_programs") or []:
                    if isinstance(payload, dict) and isinstance(payload.get("id"), str):
                        self.migrant_payloads.setdefault(payload["id"], payload)
            elif event == "attempt_scheduled":
                instance_id = str(row.get("run_instance_id"))
                if instance_id in first_schedule_seen:
                    continue
                first_schedule_seen.add(instance_id)
                started = self._instance_started(instance_id)
                if row.get("parent_code_sha256") != started.get(
                    "initial_program_sha256"
                ):
                    continue
                path, _ = _verified_file(
                    recorded_path=started.get("initial_program_path"),
                    recorded_sha=started.get("initial_program_sha256"),
                    fallbacks=[
                        self.trace_root / "initial_program.py",
                        HERE / "initial_program.py",
                    ],
                    what="initial program",
                )
                self.roots[str(row["parent_id"])] = {
                    "id": row["parent_id"],
                    "code": path.read_text(encoding="utf-8"),
                    "metrics": dict(row.get("parent_metrics") or {}),
                    "generation": 0,
                    "iteration_found": 0,
                    "metadata": {"island": 0},
                    "timestamp": float(started["timestamp"]),
                }

    def _resume_checkpoint_dir(self) -> Path | None:
        """The EXACT checkpoint this instance's run_started records — the only
        legal checkpoint payload source (a fresh instance has none; scanning
        other/later checkpoints would leak non-causal state)."""
        recorded = self._instance_started(self.instance).get("checkpoint")
        if recorded is None:
            return None
        candidates = [
            Path(str(recorded)),
            self.trace_root / "checkpoints" / Path(str(recorded)).name,
        ]
        for candidate in candidates:
            if (candidate / "metadata.json").is_file():
                return candidate
        raise MaterializeError(
            f"recorded resume checkpoint {recorded!r} is not present in this "
            "trace package"
        )

    def _checkpoint_payload(self, program_id: str) -> dict[str, Any] | None:
        if self.resume_checkpoint is None:
            return None
        path = self.resume_checkpoint / "programs" / f"{program_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _code_by_sha(self, digest: str) -> str | None:
        for child, _row in self.children.values():
            if child.get("code_sha256") == digest and isinstance(
                child.get("code"), str
            ):
                return child["code"]
        for root in self.roots.values():
            if sha256_text(root["code"]) == digest:
                return root["code"]
        if self.resume_checkpoint is not None:
            for path in sorted((self.resume_checkpoint / "programs").glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                code = data.get("code")
                if isinstance(code, str) and sha256_text(code) == digest:
                    return code
        return None

    def _resolve_payload(self, program_id: str) -> dict[str, Any]:
        """Full Program payload dict for one boundary-state id, or fail."""
        if program_id in self.children:
            child, _row = self.children[program_id]
            if sha256_text(child.get("code")) != child.get("code_sha256"):
                raise MaterializeError(
                    f"canonical child {program_id} fails its code digest"
                )
            self._payload_sources["canonical_child"] += 1
            return {k: v for k, v in child.items() if k != "code_sha256"}
        if program_id in self.migrant_payloads:
            payload = dict(self.migrant_payloads[program_id])
            digest = payload.pop("code_sha256", None)
            code = self._code_by_sha(digest) if isinstance(digest, str) else None
            if code is None:
                raise MaterializeError(
                    f"migrant {program_id}: no source code matches digest {digest!r}"
                )
            payload["code"] = code
            self._payload_sources["migration_payload"] += 1
            return payload
        checkpoint_payload = self._checkpoint_payload(program_id)
        if checkpoint_payload is not None:
            # Sanitize artifact references: embedded artifacts_json strings
            # are self-contained, but an artifact_dir points at the SOURCE
            # run's disk. At this precedence level no canonical recorded
            # artifacts exist for the id (they would have resolved above), so
            # a referenced artifact dir cannot be proven/copied inside the
            # output — fail closed rather than depend on external paths.
            if checkpoint_payload.get("artifact_dir"):
                raise MaterializeError(
                    f"checkpoint payload for {program_id} references external "
                    f"artifact_dir {checkpoint_payload['artifact_dir']!r} with "
                    "no canonical recorded artifacts to reapply — failing closed"
                )
            self._payload_sources["checkpoint_json"] += 1
            return checkpoint_payload
        if program_id in self.roots:
            self._payload_sources["verified_initial_program"] += 1
            return dict(self.roots[program_id])
        raise MaterializeError(
            f"program {program_id} in the boundary state has no verified payload source"
        )

    @staticmethod
    def _artifact_dict(row: dict[str, Any]) -> dict[str, str] | None:
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return None
        for key, value in artifacts.items():
            if not isinstance(value, str):
                raise MaterializeError(
                    f"recorded artifact {key!r} is not a plain string — "
                    "cannot reapply it exactly"
                )
        return dict(artifacts)

    def _restore_boundary(self) -> None:
        self.config = self._load_verified_config()
        if len(self.state["islands"]) != int(self.config.database.num_islands):
            raise MaterializeError(
                "boundary island count disagrees with the verified config"
            )
        # Any store_artifacts overflow (large artifacts) must land INSIDE the
        # branch checkpoint, never under the process CWD or the source run.
        self.config.database.artifacts_base_path = str(self.branch_dir / "artifacts")
        programs_dir = self.boundary_dir / "programs"
        programs_dir.mkdir(parents=True, exist_ok=True)
        for program_id in self.state["program_ids"]:
            _dump_json(
                programs_dir / f"{program_id}.json", self._resolve_payload(program_id)
            )
        _dump_json(
            self.boundary_dir / "metadata.json",
            {key: self.state[key] for key in _METADATA_KEYS},
        )
        from openevolve.database import ProgramDatabase

        self.database = ProgramDatabase(self.config.database)
        self.database.load(str(self.boundary_dir))
        restored = database_state_snapshot(self.database)
        if (
            restored != self.state
            or database_state_sha256(restored)
            != (self.boundary_state_row["state_sha256"])
        ):
            raise MaterializeError(
                "restored database does not reproduce the boundary state "
                "fingerprint exactly"
            )
        # Recorded artifacts enter native prompts: reapply them for every
        # canonical-child program (checkpoint payloads already embed theirs;
        # migrant copies never carry artifacts natively).
        for program_id in self.state["program_ids"]:
            if program_id not in self.children:
                continue
            artifacts = self._artifact_dict(self.children[program_id][1])
            if artifacts:
                self.database.store_artifacts(program_id, artifacts)
                self._artifact_counts["boundary_programs"] += 1

    # -- step 4/5: selective in-flight settlement ---------------------------

    def _settle_inflight(self) -> None:
        canonical_by_attempt: dict[str, tuple[int, dict[str, Any]]] = {}
        commits_by_seq: dict[int, dict[str, Any]] = {}
        increments_by_seq: dict[int, dict[str, Any]] = {}
        for index, row in enumerate(self.trace.rows):
            if row is None:
                continue
            event = row.get("event")
            if event == "canonical_completion" and isinstance(
                row.get("attempt_id"), str
            ):
                canonical_by_attempt.setdefault(row["attempt_id"], (index, row))
            elif event == "program_committed":
                commits_by_seq.setdefault(int(row["completion_seq"]), row)
            elif (
                event == "db_state"
                and row.get("capture") == "island_generation_increment"
                and isinstance(row.get("completion_seq"), int)
            ):
                increments_by_seq.setdefault(row["completion_seq"], row)

        inflight: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for index, row in enumerate(self.trace.rows[: self.resolution_index]):
            if (
                row is None
                or row.get("event") != "attempt_scheduled"
                or row.get("run_instance_id") != self.instance
            ):
                continue
            attempt_id = str(row.get("attempt_id"))
            terminal = canonical_by_attempt.get(attempt_id)
            if terminal is None:
                raise MaterializeError(
                    f"boundary-in-flight attempt {attempt_id} has no recorded "
                    "terminal in the frozen trace"
                )
            terminal_index, terminal_row = terminal
            if terminal_index < self.resolution_index:
                continue  # already terminal at the boundary
            inflight.append((int(terminal_row["completion_seq"]), row, terminal_row))
        inflight.sort(key=lambda entry: entry[0])

        self.handled: list[dict[str, Any]] = []
        settlement_end = self.cutoff_at
        if self.preserve_inflight:
            for seq, schedule_row, completion_row in inflight:
                completion_index = canonical_by_attempt[
                    str(schedule_row.get("attempt_id"))
                ][0]
                next_completion_index = next(
                    (
                        index
                        for index in range(completion_index + 1, len(self.trace.rows))
                        if self.trace.rows[index] is not None
                        and self.trace.rows[index].get("event")
                        == "canonical_completion"
                    ),
                    len(self.trace.rows),
                )
                if any(
                    row is not None
                    and row.get("event") == "db_state"
                    and row.get("capture") == "island_migration"
                    for row in self.trace.rows[completion_index:next_completion_index]
                ):
                    raise MaterializeError(
                        f"preserved seq {seq} crosses an island migration; "
                        "exact overlap replay is not available at this cutoff"
                    )
                commit_row = commits_by_seq.get(seq)
                increment_row = increments_by_seq.get(seq)
                child = completion_row.get("child")
                if commit_row is not None:
                    if not isinstance(child, dict) or sha256_text(
                        child.get("code")
                    ) != child.get("code_sha256"):
                        raise MaterializeError(
                            f"preserved completion seq {seq} lacks an exact child payload"
                        )
                    if child.get("id") != commit_row.get("program_id"):
                        raise MaterializeError(
                            f"preserved completion seq {seq}: child/commit identity mismatch"
                        )
                    if increment_row is None:
                        raise MaterializeError(
                            f"preserved seq {seq}: missing correlated island increment"
                        )
                terminal_at = float(
                    (commit_row or completion_row).get("timestamp") or 0.0
                )
                self.pending_inflight.append(
                    {
                        "source_attempt_id": schedule_row.get("attempt_id"),
                        "iteration": int(completion_row["iteration"]),
                        "completion_seq": seq,
                        "scheduled_at": schedule_row.get("timestamp"),
                        "terminal_at": terminal_at,
                        "remaining_seconds": max(0.0, terminal_at - self.cutoff_at),
                        "target_island": completion_row.get("scheduled_target_island"),
                        "parent_id": completion_row.get("scheduled_parent_id"),
                        "terminal": completion_row.get("terminal"),
                        "status": completion_row.get("status"),
                        "error": completion_row.get("error"),
                        "failure_stage": completion_row.get("failure_stage"),
                        "child": child,
                        "artifacts": self._artifact_dict(completion_row),
                        "committed": commit_row is not None,
                        "increment_island": (
                            increment_row.get("island_idx")
                            if increment_row is not None
                            else None
                        ),
                    }
                )
            _dump_json(
                self.out / "pending_inflight.json",
                {
                    "schema": "oe_pending_inflight_replay_v1",
                    "cutoff_seq": self.cutoff_seq,
                    "cutoff_at": self.cutoff_at,
                    "records": self.pending_inflight,
                },
            )
            self.settlement_end_at = self.cutoff_at
            return
        for seq, schedule_row, completion_row in inflight:
            commit_row = commits_by_seq.get(seq)
            committed = commit_row is not None
            if committed:
                child = completion_row.get("child")
                if not isinstance(child, dict) or sha256_text(
                    child.get("code")
                ) != child.get("code_sha256"):
                    raise MaterializeError(
                        f"settled completion seq {seq} lacks an exact child payload"
                    )
                if child.get("id") != commit_row.get("program_id"):
                    raise MaterializeError(
                        f"settled completion seq {seq}: child/commit identity mismatch"
                    )
                from openevolve.database import Program

                try:
                    program = Program(
                        **{k: v for k, v in child.items() if k != "code_sha256"}
                    )
                    self.database.add(
                        program,
                        iteration=int(completion_row["iteration"]),
                        target_island=completion_row.get("scheduled_target_island"),
                    )
                except Exception as exc:
                    raise MaterializeError(
                        f"vendor add failed for settled seq {seq} where the "
                        f"original committed: {type(exc).__name__}: {exc}"
                    ) from exc
                artifacts = self._artifact_dict(completion_row)
                if artifacts:
                    self.database.store_artifacts(program.id, artifacts)
                    self._artifact_counts["settled_children"] += 1
                # Vendor end-of-completion order: increment the committed
                # child's island exactly once, then the migration check. The
                # correlated increment record is REQUIRED provenance — a
                # missing one is a proof failure, never silently inferred.
                island = program.metadata.get("island", self.database.current_island)
                increment_row = increments_by_seq.get(seq)
                if increment_row is None:
                    raise MaterializeError(
                        f"settled seq {seq}: missing correlated "
                        "island_generation_increment record — the "
                        "end-of-completion increment cannot be proven"
                    )
                if increment_row.get("island_idx") != island:
                    raise MaterializeError(
                        f"settled seq {seq}: recorded increment island "
                        f"{increment_row.get('island_idx')} != child island {island}"
                    )
                self.database.increment_island_generation(island_idx=island)
                if self.database.should_migrate():
                    raise MaterializeError(
                        "selective settlement reached should_migrate(): "
                        "counterfactual migration is out of minimal-v1 scope — "
                        "failing closed (choose an earlier decisive boundary "
                        "or extend the materializer)"
                    )
                settlement_end = max(
                    settlement_end, float(commit_row.get("timestamp") or 0.0)
                )
            else:
                settlement_end = max(
                    settlement_end, float(completion_row.get("timestamp") or 0.0)
                )
            self.handled.append(
                {
                    "attempt_id": schedule_row.get("attempt_id"),
                    "iteration": completion_row.get("iteration"),
                    "completion_seq": seq,
                    "terminal": completion_row.get("terminal"),
                    "committed": committed,
                    "island": (
                        commit_row.get("metadata", {}).get("island")
                        if committed
                        else None
                    ),
                }
            )
        self.settlement_end_at = settlement_end

    # -- historical prefix (auditor context for the resumed branch) ----------

    _PREFIX_EVENTS = (
        "run_started",
        "attempt_scheduled",
        "canonical_completion",
        "program_committed",
        "program_commit_failed",
        "meta_eligible_completion",
    )

    def _write_historical_prefix(self) -> dict[str, Any]:
        """Project the branch's causally effective ledger history.

        Original trace LINES (byte-exact) for the source ``run_started`` and
        exactly the completions that exist in this branch — everything through
        the proven boundary plus the settled in-flight terminals — restricted
        to the compact event set the ledger and v4 history fold consume. The
        excluded next slot, every
        unrelated later event, and rolled-back prior-instance history are
        omitted; rollback provenance must be EXACT (reject older-format
        heuristics). Selection anchors on completions, so the projection
        provably ends with no in-flight or orphaned dispatch — the branch
        recorder continues at max seq + 1 and the recovered rows are auditor
        context only.
        """
        settled_seqs = {entry["completion_seq"] for entry in self.handled}
        selected_seqs: set[int] = set(settled_seqs)
        selected_attempts: dict[str, int] = {}
        for index, row in enumerate(self.trace.rows):
            if row is None or row.get("event") != "canonical_completion":
                continue
            seq = row.get("completion_seq")
            attempt_id = str(row.get("attempt_id"))
            included = index < self.resolution_index or (
                index == self.resolution_index
                and self.relation == "after_program_committed"
            )
            if included or seq in settled_seqs:
                selected_seqs.add(seq)
                selected_attempts[attempt_id] = seq

        selected_lines: list[str] = []
        selected_instances = {
            str(row.get("run_instance_id"))
            for row in self.trace.rows
            if row is not None
            and row.get("event") == "attempt_scheduled"
            and str(row.get("attempt_id")) in selected_attempts
        }
        for index, row in enumerate(self.trace.rows):
            if row is None:
                continue
            event = row.get("event")
            if event not in self._PREFIX_EVENTS:
                continue
            if event == "run_started":
                take = str(row.get("run_instance_id")) in selected_instances
            elif event == "attempt_scheduled":
                take = str(row.get("attempt_id")) in selected_attempts
            else:
                take = row.get("completion_seq") in selected_seqs
            if take:
                selected_lines.append(self.trace.lines[index])
        prefix_path = self.out / "historical_prefix.jsonl"
        prefix_bytes = ("\n".join(selected_lines) + "\n").encode("utf-8")
        prefix_path.write_bytes(prefix_bytes)

        # Re-open and prove the projection through the production recovery:
        # exact max seq, every dispatch consumed, nothing inflight/orphaned.
        from dispatch_ledger import DispatchLedger

        ledger = DispatchLedger.recover(prefix_path, "oe-branch-prefix-validation")
        max_seq = max(selected_seqs)
        if (
            ledger.last_completion_seq != max_seq
            or set(ledger.completions) != selected_seqs
        ):
            raise MaterializeError(
                "historical prefix recovery does not reproduce the selected "
                "completion sequence exactly"
            )
        for dispatch in ledger.dispatches.values():
            if dispatch.status != "consumed" or dispatch.completion_seq is None:
                raise MaterializeError(
                    f"historical prefix leaves dispatch {dispatch.attempt_id} "
                    f"{dispatch.status} — the projection must end settled"
                )
        return {
            "path": str(prefix_path.resolve()),
            "sha256": hashlib.sha256(prefix_bytes).hexdigest(),
            "bytes": len(prefix_bytes),
            "completion_seqs": sorted(selected_seqs),
            "max_completion_seq": max_seq,
            "attempt_ids": sorted(selected_attempts),
            "settled_completion_seqs": sorted(settled_seqs),
            "source_attempts_sha256": self.trace.sha256,
            "causal_cutoff": {
                "resolution_event_index": self.resolution_index,
                "resolution_relation": self.relation,
            },
        }

    # -- step 6: finalize ----------------------------------------------------

    def _global_best_fitness(self) -> float | None:
        """Strict global-best fitness under the ADAPTER's native semantics
        (``fitness_of`` over the database's best program) — the same formula
        the online causal token / CONFIRM cancellation observes."""
        best_id = self.database.best_program_id
        best = self.database.programs.get(best_id) if best_id else None
        if best is None:
            return None
        return fitness_of(best.metrics, list(self.config.database.feature_dimensions))

    def run(self) -> dict[str, Any]:
        if self.out.exists() and any(self.out.iterdir()):
            raise MaterializeError(f"--output {self.out} is not empty")
        self.out.mkdir(parents=True, exist_ok=True)
        self._load_sources()
        self._locate_boundary()
        self.resume_checkpoint = self._resume_checkpoint_dir()
        self._build_payload_index()
        self._restore_boundary()
        best_before_settlement = self._global_best_fitness()
        self._settle_inflight()
        best_after_settlement = self._global_best_fitness()
        settlement_frontier_improved = best_after_settlement is not None and (
            best_before_settlement is None
            or best_after_settlement > best_before_settlement
        )

        last_iteration = self.excluded_next_iteration - 1
        self.database.last_iteration = last_iteration
        post_snapshot = database_state_snapshot(self.database)
        post_sha = database_state_sha256(post_snapshot)
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        self.database.save(str(self.branch_dir), iteration=last_iteration)

        from openevolve.database import ProgramDatabase

        reloaded = ProgramDatabase(self.config.database)
        reloaded.load(str(self.branch_dir))
        if database_state_sha256(database_state_snapshot(reloaded)) != post_sha:
            raise MaterializeError(
                "written branch checkpoint does not reload to the settled "
                "state fingerprint"
            )
        historical_prefix = self._write_historical_prefix()
        self.trace.verify_unchanged()

        resolved_at = self.cutoff_at
        manifest = {
            "schema": BRANCH_MANIFEST_SCHEMA,
            "source": {
                "trace_root": str(self.trace_root.resolve()),
                "attempts_sha256": self.trace.sha256,
                "cutoff_seq": self.cutoff_seq,
                "selection": "canonical_completion_seq",
            },
            "boundary": {
                "resolution_event_index": self.resolution_index,
                "resolution_relation": self.relation,
                "run_instance_id": self.instance,
                "resolved_at": resolved_at,
                "db_state_line": self.boundary_state_index,
                "db_state_capture": self.boundary_state_row.get("capture"),
                "block_lines": self.boundary_block_lines,
                "state_sha256": self.boundary_state_row["state_sha256"],
                "excluded_next_slot_line": self.excluded_next_slot_index,
            },
            "excluded_next_iteration": self.excluded_next_iteration,
            "resume": {
                "checkpoint_last_iteration": last_iteration,
                "resumes_at_iteration": self.excluded_next_iteration,
            },
            "inflight": {
                "handled": self.handled,
                "pending": [
                    {
                        key: row.get(key)
                        for key in (
                            "source_attempt_id",
                            "iteration",
                            "completion_seq",
                            "remaining_seconds",
                            "target_island",
                            "committed",
                        )
                    }
                    for row in self.pending_inflight
                ],
                "count": len(self.handled) + len(self.pending_inflight),
                "policy": (
                    "preserve_for_overlap"
                    if self.preserve_inflight
                    else "settle_before_resume"
                ),
                "pending_path": (
                    str((self.out / "pending_inflight.json").resolve())
                    if self.preserve_inflight
                    else None
                ),
                "settlement_original_end_at": self.settlement_end_at,
                "settlement_original_seconds": max(
                    0.0, self.settlement_end_at - resolved_at
                ),
                "settlement_frontier": {
                    "best_fitness_before": best_before_settlement,
                    "best_fitness_after": best_after_settlement,
                    "frontier_improved": settlement_frontier_improved,
                },
            },
            "fingerprints": {
                "pre_settlement_state_sha256": self.boundary_state_row["state_sha256"],
                "post_settlement_state_sha256": post_sha,
                "reloaded_checkpoint_verified": True,
            },
            "artifacts_restored": dict(self._artifact_counts),
            "payload_sources": dict(self._payload_sources),
            "migration_during_settlement": {
                "policy": "fail_closed",
                "triggered": False,
            },
            "historical_prefix": historical_prefix,
            "branch_checkpoint": str(self.branch_dir.resolve()),
            "boundary_state_dir": str(self.boundary_dir.resolve()),
        }
        _dump_json(self.out / "branch_manifest.json", manifest)
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize an exact OpenEvolve resume checkpoint at one canonical "
            "completion (fixed current schema only)."
        )
    )
    parser.add_argument(
        "--trace-root", required=True, help="frozen source run directory"
    )
    parser.add_argument(
        "--completion-seq",
        required=True,
        type=int,
        help="canonical completion_seq to restore",
    )
    parser.add_argument("--output", required=True, help="fresh output directory")
    parser.add_argument(
        "--preserve-inflight",
        action="store_true",
        help=(
            "write the exact boundary checkpoint plus pending_inflight.json "
            "instead of settling cutoff-time work before resume"
        ),
    )
    args = parser.parse_args(argv)
    trace_root = Path(args.trace_root)
    if not (trace_root / "attempts.jsonl").is_file():
        sys.exit(f"no attempts.jsonl under --trace-root {trace_root}")
    try:
        manifest = BranchMaterializer(
            trace_root=trace_root,
            completion_seq=args.completion_seq,
            out_dir=Path(args.output),
            preserve_inflight=args.preserve_inflight,
        ).run()
    except (MaterializeError, ReplayProofError) as exc:
        sys.exit(f"materialize fail-closed: {exc}")
    print(
        json.dumps(
            {
                "branch_checkpoint": manifest["branch_checkpoint"],
                "resumes_at_iteration": manifest["resume"]["resumes_at_iteration"],
                "inflight_handled": manifest["inflight"]["count"],
                "post_settlement_state_sha256": (
                    manifest["fingerprints"]["post_settlement_state_sha256"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
