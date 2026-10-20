"""Repeat a breadth intervention after 20 flat minutes and 12 terminal attempts.

This is the default repeated-intervention treatment supervisor, not an audit
policy. It starts from
one exact materialized historical cutoff, generates the existing breadth
portfolio, runs that one-shot burst, returns immediately to native ordinary
search, and repeats after the next active flat-frontier interval.  No
PASS/INTERVENE model call is made.
"""

from __future__ import annotations
from release_paths import release_resource
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent.parent / "shared" / "meta-agent-audit"
VENDOR = HERE / "vendor" / "openevolve"
for import_root in (HERE, SHARED, VENDOR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
MEANINGFUL_RELATIVE_GAIN = 0.002
METRIC_SUBSPACE_EPOCH_STRIDE = 1000003
PLANNER_PROMPT_FILES = {"production": release_resource("planner.md")}
ACTIVE_TIME_AXIS_SCHEMA = "oe_supervisor_active_time_axis_v1"


def _epoch_planner_settings(
    *,
    candidate_pool_size: int,
    directions_per_call: int,
    metric_subspace_seed: int,
    epoch: int,
) -> dict[str, int]:
    """Resolve one reproducible planner shape and metric seed per intervention."""
    if (candidate_pool_size, directions_per_call) != (32, 8):
        raise ValueError(
            "the released planner uses four calls with eight directions each"
        )
    return {
        "directions_per_call": directions_per_call,
        "planner_calls": math.ceil(candidate_pool_size / directions_per_call),
        "metric_subspace_seed": metric_subspace_seed
        + epoch * METRIC_SUBSPACE_EPOCH_STRIDE,
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _active_seconds(
    schedules: dict[str, float],
    terminals: dict[str, float],
    *,
    since: float,
    now: float,
) -> float:
    intervals = sorted(
        (
            (max(start, since), min(terminals.get(attempt_id, now), now))
            for attempt_id, start in schedules.items()
            if min(terminals.get(attempt_id, now), now) > max(start, since)
        )
    )
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


def first_flat_cutoff(
    attempts_path: Path,
    *,
    minimum_elapsed_seconds: float,
    flat_seconds: float,
    flat_min_terminal_attempts: int,
) -> dict[str, Any]:
    """First committed cutoff after the run-age gate with 20 active flat minutes."""
    rows = _rows(attempts_path)
    starts = [
        float(row["timestamp"])
        for row in rows
        if row.get("event") == "run_started"
        and isinstance(row.get("timestamp"), (int, float))
    ]
    if not starts:
        raise RuntimeError("trace has no run_started timestamp")
    started_at = min(starts)
    schedules: dict[str, float] = {}
    terminals: dict[str, float] = {}
    canonical: dict[int, dict[str, Any]] = {}
    best_after: dict[int, str] = {}
    commits: list[dict[str, Any]] = []
    root_cycles: float | None = None
    for row in rows:
        event = row.get("event")
        if event == "attempt_scheduled":
            attempt_id = row.get("attempt_id")
            at = row.get("timestamp")
            if isinstance(attempt_id, str) and isinstance(at, (int, float)):
                schedules[attempt_id] = float(at)
            if root_cycles is None:
                cycles = (row.get("parent_metrics") or {}).get("cycles")
                if isinstance(cycles, (int, float)) and (not isinstance(cycles, bool)):
                    root_cycles = float(cycles)
        elif event == "canonical_completion":
            seq, attempt_id, at = (
                row.get("completion_seq"),
                row.get("attempt_id"),
                row.get("timestamp"),
            )
            if isinstance(seq, int):
                canonical[seq] = row
            if isinstance(attempt_id, str) and isinstance(at, (int, float)):
                terminals[attempt_id] = float(at)
        elif event == "program_committed" and isinstance(
            row.get("completion_seq"), int
        ):
            commits.append(row)
        elif event == "db_state" and isinstance(row.get("completion_seq"), int):
            best_id = (row.get("state") or {}).get("best_program_id")
            if isinstance(best_id, str):
                best_after[int(row["completion_seq"])] = best_id
    if root_cycles is None:
        raise RuntimeError("trace has no finite root cycles")
    anchor_cycles = root_cycles
    anchor_at = started_at
    for commit in sorted(commits, key=lambda row: int(row["completion_seq"])):
        seq = int(commit["completion_seq"])
        completion = canonical.get(seq)
        at = commit.get("timestamp")
        if completion is None or not isinstance(at, (int, float)):
            continue
        cycles = (commit.get("metrics") or {}).get("cycles")
        frontier_moved = (
            completion.get("evaluation_valid") is True
            and best_after.get(seq) == commit.get("program_id")
            and isinstance(cycles, (int, float))
            and (not isinstance(cycles, bool))
            and (cycles > 0)
        )
        if frontier_moved:
            gain = (anchor_cycles - float(cycles)) / abs(anchor_cycles)
            if gain > MEANINGFUL_RELATIVE_GAIN:
                anchor_cycles = float(cycles)
                anchor_at = float(at)
        elapsed = float(at) - started_at
        active = _active_seconds(schedules, terminals, since=anchor_at, now=float(at))
        terminal_attempts = sum(
            (
                anchor_at <= terminal_at <= float(at)
                for terminal_at in terminals.values()
            )
        )
        if (
            elapsed >= minimum_elapsed_seconds
            and active >= flat_seconds
            and (terminal_attempts >= flat_min_terminal_attempts)
        ):
            return {
                "completion_seq": seq,
                "cutoff_at": float(at),
                "run_elapsed_seconds": elapsed,
                "active_flat_seconds": active,
                "terminal_attempts_since_anchor": terminal_attempts,
                "minimum_terminal_attempts": flat_min_terminal_attempts,
                "progress_anchor_cycles": anchor_cycles,
                "progress_anchor_at": anchor_at,
            }
    raise RuntimeError("trace contains no eligible flat-frontier cutoff")


def _write_json(path: Path, document: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _event_time_axis(
    *,
    output: Path,
    historical_prefix: Path,
    supervisor_started_at: float,
    active_started_seconds: float,
    resume_existing: bool,
) -> dict[str, Any]:
    """Persist the wall-clock segments that count toward this experiment."""
    source_timestamps = [
        float(row["timestamp"])
        for row in _rows(historical_prefix)
        if isinstance(row.get("timestamp"), (int, float))
        and (not isinstance(row.get("timestamp"), bool))
    ]
    if not source_timestamps:
        raise RuntimeError("historical prefix has no timestamp")
    source_ended_at = max(source_timestamps)
    path = output / "active_time_axis.json"
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != ACTIVE_TIME_AXIS_SCHEMA:
            raise RuntimeError("active time axis schema mismatch")
        if not math.isclose(float(document.get("source_ended_at")), source_ended_at):
            raise RuntimeError("active time axis source boundary mismatch")
    else:
        manifest = json.loads(
            (output / "supervisor_manifest.json").read_text(encoding="utf-8")
        )
        document = {
            "schema": ACTIVE_TIME_AXIS_SCHEMA,
            "source_ended_at": source_ended_at,
            "segments": [
                {
                    "raw_started_at": float(manifest["created_at"]),
                    "active_started_seconds": 0.0,
                }
            ],
        }
        source_root = Path(manifest["source_trace_root"])
        for source_axis_path in (
            source_root / "active_time_axis.json",
            source_root.parent / "active_time_axis.json",
        ):
            if source_axis_path.is_file():
                document["source_time_axis"] = json.loads(
                    source_axis_path.read_text(encoding="utf-8")
                )
                break
    segments = list(document["segments"])
    if resume_existing:
        segment = {
            "raw_started_at": float(supervisor_started_at),
            "active_started_seconds": float(active_started_seconds),
        }
        if len(segments) > 1 and math.isclose(
            float(segments[-1]["active_started_seconds"]), float(active_started_seconds)
        ):
            segments[-1] = segment
        else:
            if float(active_started_seconds) < float(
                segments[-1]["active_started_seconds"]
            ):
                raise RuntimeError("active time axis moved backwards")
            segments.append(segment)
    document["segments"] = segments
    _write_json(path, document)
    return document


def _events_on_active_time_axis(
    events: list[dict[str, Any]], time_axis: dict[str, Any]
) -> list[dict[str, Any]]:
    """Remove offline resume gaps while retaining time spent by live supervisors."""
    source_ended_at = float(time_axis["source_ended_at"])
    source_axis = time_axis.get("source_time_axis")
    normalized_source_end = (
        _events_on_active_time_axis([{"timestamp": source_ended_at}], source_axis)[0][
            "timestamp"
        ]
        if source_axis
        else source_ended_at
    )
    segments = sorted(
        time_axis["segments"], key=lambda item: float(item["raw_started_at"])
    )
    normalized: list[dict[str, Any]] = []
    for event in events:
        timestamp = event.get("timestamp")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or float(timestamp) <= source_ended_at
        ):
            normalized.append(
                _events_on_active_time_axis([event], source_axis)[0]
                if source_axis
                else event
            )
            continue
        segment = next(
            (
                item
                for item in reversed(segments)
                if float(item["raw_started_at"]) <= float(timestamp)
            ),
            None,
        )
        if segment is None:
            raise RuntimeError("event precedes the first active time segment")
        shifted = dict(event)
        shifted["timestamp"] = (
            normalized_source_end
            + float(segment["active_started_seconds"])
            + float(timestamp)
            - float(segment["raw_started_at"])
        )
        normalized.append(shifted)
    return normalized


class _IncrementalContextCache:
    """Fold the latest causal prefix while ordinary workers keep running."""

    def __init__(
        self,
        *,
        attempts_path: Path,
        time_axis: dict[str, Any],
        status_path: Path,
        checkpoint_roots: tuple[Path, ...] = (),
    ) -> None:
        self.attempts_path = attempts_path
        self.time_axis = time_axis
        self.status_path = status_path
        self.checkpoint_roots = checkpoint_roots
        self._offset = 0
        self._events: list[dict[str, Any]] = []
        self._contexts: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="oe-dashboard-cache", daemon=True
        )

    @staticmethod
    def _key(cutoff_seq: int, live_ids: set[str]) -> tuple[int, tuple[str, ...]]:
        return (int(cutoff_seq), tuple(sorted(live_ids)))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _poll(self) -> bool:
        if not self.attempts_path.is_file():
            return False
        if self.attempts_path.stat().st_size < self._offset:
            with self._lock:
                self._events.clear()
                self._contexts.clear()
            self._offset = 0
        rows: list[dict[str, Any]] = []
        with self.attempts_path.open("rb") as handle:
            handle.seek(self._offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.seek(start)
                    break
                self._offset = handle.tell()
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, ValueError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        if not rows:
            return False
        normalized = _events_on_active_time_axis(rows, self.time_axis)
        with self._lock:
            self._events.extend(normalized)
        return True

    def _snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @staticmethod
    def _cutoff_at(events: list[dict[str, Any]], cutoff_seq: int) -> float:
        rows = [
            row
            for row in events
            if row.get("event") == "canonical_completion"
            and row.get("completion_seq") == cutoff_seq
            and isinstance(row.get("timestamp"), (int, float))
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"frozen cutoff completion {cutoff_seq} is not uniquely available"
            )
        return float(rows[0]["timestamp"])

    def _engine_profiles(
        self, context: dict[str, Any], live_ids: set[str]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        from planner_context import (
            ROUTING_DASHBOARD,
            ROUTING_SUMMARY,
            profile_source_code,
        )

        files = context["files"]
        routing_rows = json.loads(str(files[ROUTING_DASHBOARD]))
        summary = json.loads(str(files[ROUTING_SUMMARY]))
        selected_ids = set(live_ids)
        for record in (summary.get("records") or {}).values():
            for field in ("program_id", "parent_id"):
                value = record.get(field)
                if isinstance(value, str) and value:
                    selected_ids.add(value)
        sources: dict[str, str] = {}
        for row in routing_rows:
            location = row.get("code_location")
            program_id = row.get("program_id")
            if (
                program_id in selected_ids
                and isinstance(program_id, str)
                and isinstance(location, str)
                and (location in files)
            ):
                sources[program_id] = str(files[location])
        for root in self.checkpoint_roots:
            checkpoints = (
                [root]
                if (root / "programs").is_dir()
                else list(root.glob("checkpoint_*") if root.is_dir() else ())
            )
            if not checkpoints:
                continue
            checkpoint = max(
                checkpoints,
                key=lambda path: (
                    (path / "metadata.json").stat().st_mtime
                    if (path / "metadata.json").is_file()
                    else -1.0
                ),
            )
            for path in (checkpoint / "programs").glob("*.json"):
                try:
                    source = json.loads(path.read_text(encoding="utf-8")).get("code")
                except (OSError, ValueError):
                    continue
                program_id = path.stem
                if program_id in selected_ids and isinstance(source, str):
                    sources.setdefault(program_id, source)
        profiles: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        by_hash: dict[str, tuple[str, list[str]]] = {}
        for program_id, source in sources.items():
            code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            if code_hash not in by_hash:
                by_hash[code_hash] = (source, [])
            by_hash[code_hash][1].append(program_id)
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(by_hash)))) as pool:
            futures = {
                pool.submit(profile_source_code, source): (code_hash, program_ids)
                for code_hash, (source, program_ids) in by_hash.items()
            }
            for future, (_code_hash, program_ids) in futures.items():
                try:
                    profile = future.result()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failures.update({program_id: error for program_id in program_ids})
                else:
                    profiles.update({program_id: profile for program_id in program_ids})
        return (profiles, failures)

    def context_for(self, *, cutoff_seq: int, live_ids: set[str]) -> dict[str, Any]:
        """Return an exact cutoff snapshot, building it once if not yet warm."""
        self._poll()
        key = self._key(cutoff_seq, live_ids)
        with self._lock:
            cached = self._contexts.get(key)
        if cached is not None:
            return cached
        with self._build_lock:
            with self._lock:
                cached = self._contexts.get(key)
            if cached is not None:
                return cached
            events = self._snapshot()
            try:
                cutoff_at = self._cutoff_at(events, cutoff_seq)
            except RuntimeError:
                events = _events_on_active_time_axis(
                    _rows(self.attempts_path), self.time_axis
                )
                cutoff_at = self._cutoff_at(events, cutoff_seq)
            prefix = [
                row
                for row in events
                if not isinstance(row.get("timestamp"), (int, float))
                or float(row["timestamp"]) <= cutoff_at
            ]
            from planner_context import build_stage_context

            build_kwargs = {
                "attempts_path": self.attempts_path,
                "cutoff_seq": cutoff_seq,
                "cutoff_at": cutoff_at,
                "root_code": (HERE / "initial_program.py").read_text(encoding="utf-8"),
                "live_program_ids": live_ids,
                "preloaded_events": prefix,
            }
            context = build_stage_context(**build_kwargs)
            profiles, failures = self._engine_profiles(context, live_ids)
            context = build_stage_context(
                **build_kwargs, engine_profiles=profiles, engine_failures=failures
            )
            with self._lock:
                self._contexts[key] = context
                if len(self._contexts) > 2:
                    oldest = min(self._contexts, key=lambda item: item[0])
                    del self._contexts[oldest]
            return context

    def _write_status(self, **fields: Any) -> None:
        try:
            _write_json(
                self.status_path,
                {
                    "schema": "oe_incremental_dashboard_cache_v1",
                    "updated_at": time.time(),
                    **fields,
                },
            )
        except OSError:
            pass

    def _run(self) -> None:
        from planner_context import live_program_ids_at_cutoff

        last_warmed = -1
        quiet_since = time.monotonic()
        retry_after = 0.0
        while not self._stop.wait(0.25):
            if self._poll():
                quiet_since = time.monotonic()
            events = self._snapshot()
            completions = [
                row
                for row in events
                if row.get("event") == "canonical_completion"
                and isinstance(row.get("completion_seq"), int)
            ]
            if not completions or time.monotonic() - quiet_since < 0.5:
                continue
            cutoff_seq = max((int(row["completion_seq"]) for row in completions))
            if cutoff_seq <= last_warmed or time.monotonic() < retry_after:
                continue
            started = time.monotonic()
            try:
                recorded_live_ids = live_program_ids_at_cutoff(events, cutoff_seq)
                if recorded_live_ids is None:
                    retry_after = time.monotonic() + 5.0
                    self._write_status(
                        state="waiting_for_db_state", cutoff_seq=cutoff_seq
                    )
                    continue
                live_ids = set(recorded_live_ids)
                self.context_for(cutoff_seq=cutoff_seq, live_ids=live_ids)
                last_warmed = cutoff_seq
                self._write_status(
                    last_warmed_completion_seq=cutoff_seq,
                    build_seconds=time.monotonic() - started,
                    state="ready",
                )
            except Exception as exc:
                retry_after = time.monotonic() + 5.0
                self._write_status(
                    last_warmed_completion_seq=last_warmed,
                    failed_completion_seq=cutoff_seq,
                    state="retrying",
                    error=f"{type(exc).__name__}: {exc}",
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(
    argv: list[str], *, log_path: Path, env: dict[str, str], timeout: float
) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=HERE,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(f"command timed out; see {log_path}")
    if returncode != 0:
        raise RuntimeError(f"command failed ({returncode}); see {log_path}")


def _prepare_view(
    *,
    attempts_path: Path,
    checkpoint: Path,
    output: Path,
    prompt_path: Path,
    time_axis: dict[str, Any],
    cutoff_seq: int | None = None,
    cutoff_at: float | None = None,
    context_cache: _IncrementalContextCache | None = None,
) -> Path:
    from planner_context import (
        ENGINE_PROFILES,
        LLM_DASHBOARD,
        _materialize_engine_profiles,
        build_stage_context,
    )
    from planner_runtime import materialize_view, sanitize_planner_view

    events = _events_on_active_time_axis(_rows(attempts_path), time_axis)
    if cutoff_seq is not None:
        cutoff_rows = [
            row
            for row in events
            if row.get("event") == "canonical_completion"
            and row.get("completion_seq") == cutoff_seq
            and isinstance(row.get("timestamp"), (int, float))
        ]
        if len(cutoff_rows) != 1:
            raise RuntimeError(
                f"frozen cutoff completion {cutoff_seq} is not uniquely available"
            )
        cutoff_at = float(cutoff_rows[0]["timestamp"])
    if cutoff_at is not None:
        events = [
            row
            for row in events
            if not isinstance(row.get("timestamp"), (int, float))
            or float(row["timestamp"]) <= float(cutoff_at)
        ]
    completions = [
        row
        for row in events
        if row.get("event") == "canonical_completion"
        and isinstance(row.get("completion_seq"), int)
        and isinstance(row.get("timestamp"), (int, float))
    ]
    if not completions:
        raise RuntimeError("causal trace has no canonical completion")
    if cutoff_seq is None:
        cutoff = max(completions, key=lambda item: int(item["completion_seq"]))
    else:
        matches = [
            row for row in completions if int(row["completion_seq"]) == int(cutoff_seq)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"frozen cutoff completion {cutoff_seq} is not uniquely available"
            )
        cutoff = matches[0]
    cutoff_seq, cutoff_at = (int(cutoff["completion_seq"]), float(cutoff["timestamp"]))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    live_ids = {
        str(program_id)
        for members in metadata.get("islands") or ()
        for program_id in members
    }
    context = (
        context_cache.context_for(cutoff_seq=cutoff_seq, live_ids=live_ids)
        if context_cache is not None
        else build_stage_context(
            attempts_path=attempts_path,
            cutoff_seq=cutoff_seq,
            cutoff_at=cutoff_at,
            root_code=(HERE / "initial_program.py").read_text(encoding="utf-8"),
            live_program_ids=live_ids,
            preloaded_events=events,
        )
    )
    raw = output / "context"
    for relative, content in context["files"].items():
        target = raw / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    build = output / "prepared"
    view, _, _, _ = materialize_view(
        context_root=raw,
        dashboard_root=None,
        checkpoint=checkpoint,
        output=build,
        prompt_path=prompt_path,
        num_directions=4,
    )
    for relative, content in context["files"].items():
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    profile_scratch = output / "profile_scratch"
    profile_scratch.mkdir()
    _materialize_engine_profiles(view, profile_scratch)
    if context_cache is None:
        profile_document = json.loads(
            (view / ENGINE_PROFILES).read_text(encoding="utf-8")
        )
        enriched_context = build_stage_context(
            attempts_path=attempts_path,
            cutoff_seq=cutoff_seq,
            cutoff_at=cutoff_at,
            root_code=(HERE / "initial_program.py").read_text(encoding="utf-8"),
            live_program_ids=live_ids,
            preloaded_events=events,
            engine_profiles=profile_document.get("profiles") or {},
            engine_failures={
                str(program_id): str(record.get("error") or "profile_failed")
                for program_id, record in (
                    profile_document.get("failures") or {}
                ).items()
            },
        )
        (view / LLM_DASHBOARD).write_text(
            enriched_context["files"][LLM_DASHBOARD], encoding="utf-8"
        )
    sanitize_planner_view(view)
    return view


def _journal_has(path: Path, *, offset: int, kind: str) -> bool:
    if not path.is_file() or path.stat().st_size <= offset:
        return False
    with path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("kind") == kind:
                return True
    return False


def _journal_rows_since(path: Path, *, offset: int) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= offset:
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _update_failed_interventions(
    existing: list[dict[str, Any]], *, planner: Path, journal_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep failed directions only within the current flat-frontier episode."""
    if any(
        (row.get("kind") == "meaningful_frontier_improvement" for row in journal_rows)
    ):
        return []
    completed = {
        str(row["audit_id"])
        for row in journal_rows
        if row.get("kind") == "guided_cohort_complete"
        and isinstance(row.get("audit_id"), str)
    }
    if not completed:
        return list(existing)
    if len(completed) != 1:
        raise RuntimeError("one search segment must complete exactly one guided cohort")
    audit_id = completed.pop()
    directions = json.loads(
        (planner / "portfolio.json").read_text(encoding="utf-8")
    ).get("directions")
    if not isinstance(directions, list):
        raise RuntimeError("planner portfolio has no directions")
    assignments = sorted(
        (
            row
            for row in journal_rows
            if row.get("kind") == "direction_assigned"
            and row.get("audit_id") == audit_id
        ),
        key=lambda row: int(row.get("direction_index", -1)),
    )
    terminals = {
        str(row["attempt_id"]): row
        for row in journal_rows
        if row.get("kind") == "attempt_terminal"
        and row.get("audit_id") == audit_id
        and isinstance(row.get("attempt_id"), str)
    }
    if len(assignments) != len(directions):
        raise RuntimeError("completed cohort does not match its selected directions")
    bound = [
        row
        for row in journal_rows
        if row.get("kind") == "cohort_bound" and row.get("audit_id") == audit_id
    ]
    cohort_frontier = bound[-1].get("frontier_cycles") if bound else None
    failed = list(existing)
    for assignment in assignments:
        index = int(assignment["direction_index"]) - 1
        if not 0 <= index < len(directions):
            raise RuntimeError("cohort direction index is outside the portfolio")
        direction = directions[index]
        terminal = terminals.get(str(assignment.get("attempt_id")))
        frontier = cohort_frontier
        if frontier is None:
            frontier = (direction.get("frontier_case") or {}).get("frontier_cycles")
        cycles = (((terminal or {}).get("child") or {}).get("metrics") or {}).get(
            "cycles"
        )
        finite_frontier = (
            isinstance(frontier, (int, float))
            and (not isinstance(frontier, bool))
            and math.isfinite(float(frontier))
            and (float(frontier) > 0)
        )
        finite_cycles = (
            isinstance(cycles, (int, float))
            and (not isinstance(cycles, bool))
            and math.isfinite(float(cycles))
        )
        if (
            terminal is None
            or terminal.get("evaluation_valid") is not True
            or (not finite_frontier)
            or (not finite_cycles)
        ):
            outcome = "invalid"
            frontier_delta = None
        else:
            frontier_value = float(frontier)
            cycles_value = float(cycles)
            frontier_delta = frontier_value - cycles_value
            if frontier_delta / frontier_value > MEANINGFUL_RELATIVE_GAIN:
                outcome = "improve"
            elif frontier_delta >= 0:
                outcome = "tie"
            else:
                outcome = "regress"
        failed.append(
            {
                "type": direction.get("type"),
                "mechanism_hypothesis": direction.get("mechanism_hypothesis"),
                "task": direction.get("task"),
                "target_branch_id": direction.get("target_branch_id"),
                "outcome": outcome,
                "frontier_delta_cycles": frontier_delta,
                "completion_seq": (terminal or {}).get("completion_seq"),
            }
        )
    return failed


def _overlap_guided_pending(journal: Path) -> bool:
    rows = _journal_rows_since(journal, offset=0)
    assigned = {
        int(row["cohort_index"])
        for row in rows
        if row.get("kind") == "direction_assigned"
        and isinstance(row.get("cohort_index"), int)
    }
    completed = {
        int(row["cohort_index"])
        for row in rows
        if row.get("kind") == "guided_cohort_complete"
        and isinstance(row.get("cohort_index"), int)
    }
    return bool(assigned - completed)


def _overlap_cohort_complete(journal: Path, cohort_index: int) -> bool:
    return any(
        (
            row.get("kind") == "guided_cohort_complete"
            and row.get("cohort_index") == cohort_index
            for row in _journal_rows_since(journal, offset=0)
        )
    )


def _run_overlap_planner(
    *,
    args: argparse.Namespace,
    cohort_index: int,
    request: dict[str, Any],
    attempts_path: Path,
    time_axis: dict[str, Any],
    failed_interventions: list[dict[str, Any]],
    env: dict[str, str],
    deadline: float,
    settings: dict[str, int],
    context_cache: _IncrementalContextCache | None = None,
) -> Path:
    epoch_root = args.output / "epochs" / f"epoch_{cohort_index:03d}"
    epoch_root.mkdir(exist_ok=True)
    wait_until = time.monotonic() + 30.0
    while not attempts_path.is_file() and time.monotonic() < wait_until:
        time.sleep(0.1)
    if not attempts_path.is_file():
        raise RuntimeError(f"overlap runner did not create {attempts_path}")
    checkpoint = Path(str(request["checkpoint"])).resolve()
    prepared_view = _prepare_view(
        attempts_path=attempts_path,
        checkpoint=checkpoint,
        output=epoch_root,
        prompt_path=PLANNER_PROMPT_FILES["production"],
        time_axis=time_axis,
        cutoff_seq=int(request["cutoff_seq"]),
        context_cache=context_cache,
    )
    planner = epoch_root / "planner"
    planner_env = dict(env)
    planner_env.update(
        {
            "AUDIT_PI_PROVIDER": args.planner_provider,
            "AUDIT_PI_MODEL": args.planner_model,
            "AUDIT_PI_REASONING": args.planner_reasoning,
            "AUDIT_PI_API_KEY_ENV": args.planner_api_key_env,
        }
    )
    failed_path = epoch_root / "prior_failed_interventions.json"
    _write_json(failed_path, failed_interventions)
    _run_checked(
        [
            sys.executable,
            str(HERE / "run_branch_portfolio_breadth.py"),
            "--prepared-view",
            str(prepared_view),
            "--output",
            str(planner),
            "--prompt",
            str(PLANNER_PROMPT_FILES["production"]),
            *[],
            "--candidate-pool-size",
            str(args.candidate_pool_size),
            "--directions-per-call",
            str(settings["directions_per_call"]),
            *[],
            "--planner-workers",
            str(args.planner_workers),
            "--planner-random-metric-count",
            str(args.planner_random_metric_count),
            "--metric-subspace-seed",
            str(settings["metric_subspace_seed"]),
            "--dedup-model",
            args.dedup_model,
            "--dedup-base-url",
            args.dedup_base_url,
            "--dedup-api-key-env",
            args.dedup_api_key_env,
            "--dedup-max-tokens",
            str(args.dedup_max_tokens),
            "--prior-failed-interventions",
            str(failed_path),
            "--model",
            args.planner_model,
            "--provider",
            args.planner_provider,
            "--base-url",
            args.planner_base_url,
            "--api-key-env",
            args.planner_api_key_env,
            *(
                ["--planner-backends-json", args.planner_backends_json]
                if args.planner_backends_json
                else []
            ),
        ],
        log_path=epoch_root / "planner.log",
        env=planner_env,
        timeout=max(6 * 3600.0, deadline - time.monotonic()),
    )
    return planner


def _run_overlap_supervisor(
    *,
    args: argparse.Namespace,
    branch: Path | None,
    run_output: Path,
    initial: dict[str, Any],
    time_axis: dict[str, Any],
    env: dict[str, str],
    deadline: float,
) -> list[dict[str, Any]]:
    """Run one controller while planner handoffs arrive asynchronously."""
    control = args.output / "overlap_control"
    control.mkdir()
    if not args.initial_ordinary and branch is not None:
        request0 = {
            "schema": "oe_overlap_planner_request_v1",
            "cohort_index": 0,
            "cutoff_seq": int(initial["completion_seq"]),
            "checkpoint": str((branch / "branch_checkpoint").resolve()),
            "next_iteration_fenced": None,
        }
        _write_json(control / "request_000.json", request0)
    command = [
        sys.executable,
        str(HERE / "runner.py"),
        "--config",
        str(args.config),
        "--takehome",
        str(args.takehome),
        "--output",
        str(run_output),
        *(["--checkpoint", str(branch / "branch_checkpoint")] if branch else []),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--budget",
        str(args.budget),
        "--llm-provider",
        args.worker_provider,
        "--portfolio-overlap-control-dir",
        str(control),
        *(
            ["--portfolio-pending-inflight", str(branch / "pending_inflight.json")]
            if branch
            else []
        ),
        "--minimum-run-age-seconds",
        str(args.minimum_run_age_seconds if branch is None else 0),
        "--portfolio-repeat-flat-seconds",
        str(args.flat_seconds),
        "--portfolio-repeat-flat-min-terminal-attempts",
        str(args.flat_min_terminal_attempts),
        *(["--portfolio-overlap-start-ordinary"] if args.initial_ordinary else []),
    ]
    if args.worker_provider == "gemini":
        command += [
            "--gemini-model",
            args.worker_model,
            "--gemini-reasoning-effort",
            args.worker_reasoning,
        ]
    else:
        command += [
            "--deepseek-model",
            args.worker_model,
            "--deepseek-reasoning-effort",
            args.worker_reasoning,
        ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    log_path = args.output / "overlap_runner.log"
    summaries: list[dict[str, Any]] = []
    failed_interventions: list[dict[str, Any]] = []
    planners: dict[int, Path] = {}
    processed_failures: set[int] = set()
    latest_handoff_cohort: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=HERE,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        context_cache = _IncrementalContextCache(
            attempts_path=run_output / "attempts.jsonl",
            time_axis=time_axis,
            status_path=args.output / "incremental_dashboard_cache.json",
            checkpoint_roots=(
                *((branch / "branch_checkpoint",) if branch else ()),
                run_output / "checkpoints",
            ),
        )
        context_cache.start()
        try:
            cohort_index = 0
            while process.poll() is None:
                request_path = control / f"request_{cohort_index:03d}.json"
                handoff_path = control / f"handoff_{cohort_index:03d}.json"
                if request_path.is_file() and (not handoff_path.is_file()):
                    if cohort_index > 0 and cohort_index - 1 not in processed_failures:
                        previous = cohort_index - 1
                        cohort_rows = [
                            row
                            for row in _journal_rows_since(
                                run_output / "portfolio_cohort.jsonl", offset=0
                            )
                            if row.get("cohort_index") == previous
                        ]
                        failed_interventions = _update_failed_interventions(
                            failed_interventions,
                            planner=planners[previous],
                            journal_rows=cohort_rows,
                        )
                        processed_failures.add(previous)
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    active_elapsed_seconds = args.wall_time_seconds - max(
                        0.0, deadline - time.monotonic()
                    )
                    settings = _epoch_planner_settings(
                        candidate_pool_size=args.candidate_pool_size,
                        directions_per_call=args.directions_per_call,
                        metric_subspace_seed=args.metric_subspace_seed,
                        epoch=cohort_index,
                    )
                    planner = _run_overlap_planner(
                        args=args,
                        cohort_index=cohort_index,
                        request=request,
                        attempts_path=run_output / "attempts.jsonl",
                        time_axis=time_axis,
                        failed_interventions=failed_interventions,
                        env=env,
                        deadline=deadline,
                        settings=settings,
                        context_cache=context_cache,
                    )
                    planners[cohort_index] = planner
                    _write_json(
                        handoff_path,
                        {
                            "schema": "oe_overlap_portfolio_handoff_v1",
                            "cohort_index": cohort_index,
                            "planner_dir": str(planner.resolve()),
                        },
                    )
                    latest_handoff_cohort = cohort_index
                    selected = json.loads(
                        (planner / "run.json").read_text(encoding="utf-8")
                    )["actual_selected_directions"]
                    summaries.append(
                        {
                            "epoch": cohort_index,
                            "planner": str(planner),
                            "selected_directions": selected,
                            "trigger_completion_seq": request["cutoff_seq"],
                            "active_elapsed_seconds": active_elapsed_seconds,
                            **settings,
                        }
                    )
                    _write_json(
                        args.output / "supervisor_state.json",
                        {
                            "epochs": summaries,
                            "failed_interventions": failed_interventions,
                        },
                    )
                    cohort_index += 1
                journal = run_output / "portfolio_cohort.jsonl"
                if args.stop_after_guided_cohort and _journal_has(
                    journal, offset=0, kind="guided_cohort_complete"
                ):
                    process.send_signal(signal.SIGTERM)
                    break
                if time.monotonic() >= deadline:
                    if (
                        latest_handoff_cohort is not None
                        and (
                            not _overlap_cohort_complete(journal, latest_handoff_cohort)
                        )
                        or _overlap_guided_pending(journal)
                    ):
                        time.sleep(2.0)
                        continue
                    process.send_signal(signal.SIGTERM)
                    break
                time.sleep(1.0)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise
        finally:
            context_cache.stop()
            try:
                returncode = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"overlap runner exited with {returncode}; see {log_path}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-trace-root",
        type=Path,
        help="optional recorded baseline; omit for fresh ordinary search",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--initial-completion-seq", type=int)
    parser.add_argument("--minimum-run-age-seconds", type=float, default=3600.0)
    parser.add_argument("--flat-seconds", type=float, default=1200.0)
    parser.add_argument("--flat-min-terminal-attempts", type=int, default=12)
    parser.add_argument("--wall-time-seconds", type=float, default=32400.0)
    parser.add_argument(
        "--stop-after-guided-cohort",
        action="store_true",
        help="finish the current guided cohort, then stop before another epoch",
    )
    parser.add_argument(
        "--initial-ordinary",
        action="store_true",
        help="resume overlap mode with ordinary search before the first flat trigger",
    )
    parser.add_argument("--budget", type=float, default=400.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--worker-provider",
        choices=("deepseek", "gemini"),
        default="deepseek",
    )
    parser.add_argument("--worker-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--worker-reasoning", choices=("low", "high", "max"), default="high"
    )
    parser.add_argument("--planner-model", default="deepseek-v4-flash")
    parser.add_argument("--planner-reasoning", default="high")
    parser.add_argument("--planner-provider", default="deepseek")
    parser.add_argument("--planner-base-url", default="https://api.deepseek.com")
    parser.add_argument("--planner-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--planner-backends-json", help="JSON list of per-call planner backend objects"
    )
    parser.add_argument("--candidate-pool-size", type=int, default=32)
    parser.add_argument("--directions-per-call", type=int, default=8)
    parser.add_argument("--planner-workers", type=int, default=4)
    parser.add_argument(
        "--planner-random-metric-count",
        type=int,
        default=15,
        help="number of reproducibly sampled metrics visible to each planner",
    )
    parser.add_argument("--metric-subspace-seed", type=int, default=20260830)
    parser.add_argument("--dedup-model", default="deepseek-v4-flash")
    parser.add_argument("--dedup-base-url", default="https://api.deepseek.com")
    parser.add_argument("--dedup-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--dedup-max-tokens", type=int, default=128000)
    parser.add_argument("--takehome", type=Path, default=HERE / "vendor" / "takehome")
    return parser.parse_args()


def main() -> int:
    supervisor_started_at = time.time()
    args = parse_args()
    if args.planner_random_metric_count < 0:
        raise RuntimeError("--planner-random-metric-count must be non-negative")
    if args.flat_min_terminal_attempts < 1:
        raise RuntimeError("--flat-min-terminal-attempts must be positive")
    args.output = args.output.resolve()
    incompatible = []
    sampling_shape = (
        args.candidate_pool_size // args.directions_per_call
        if args.directions_per_call > 0
        else 0
    )
    if args.candidate_pool_size != 32 or (
        sampling_shape,
        args.directions_per_call,
    ) not in {(2, 16), (4, 8)}:
        incompatible.append("planner sampling must be 4x8")
    if args.planner_workers < sampling_shape:
        incompatible.append("planner workers must cover every planner call")
    if incompatible:
        raise RuntimeError(
            "verbalized-distribution configuration is invalid: "
            + "; ".join(incompatible)
        )
    if args.output.exists() and (not False):
        raise RuntimeError(f"output already exists: {args.output}")
    if args.wall_time_seconds <= 0:
        raise RuntimeError("--wall-time-seconds must be positive")
    required_keys = {args.planner_api_key_env}
    required_keys.add(
        {
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[args.worker_provider]
    )
    required_keys.add(args.dedup_api_key_env)
    for name in sorted(required_keys):
        if not os.environ.get(name):
            raise RuntimeError(f"{name} is not set")
    if args.source_trace_root is None:
        if False or "overlap" != "overlap" or args.initial_completion_seq is not None:
            raise RuntimeError(
                "fresh runs require overlap mode without resume/cutoff options"
            )
        args.initial_ordinary = True
        args.output.mkdir(parents=True)
        (args.output / "epochs").mkdir()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(SHARED), str(HERE), str(VENDOR)])
        time_axis = {
            "schema": ACTIVE_TIME_AXIS_SCHEMA,
            "source_ended_at": supervisor_started_at,
            "segments": [
                {"raw_started_at": supervisor_started_at, "active_started_seconds": 0.0}
            ],
        }
        _write_json(args.output / "active_time_axis.json", time_axis)
        _write_json(
            args.output / "supervisor_manifest.json",
            {
                "schema": "oe_repeat_flat_breadth_manifest_v1",
                "argv": sys.argv,
                "created_at": supervisor_started_at,
                "source_trace_root": None,
                "dispatch_mode": "overlap",
                "fresh_start": True,
                "minimum_run_age_seconds": args.minimum_run_age_seconds,
            },
        )
        summaries = _run_overlap_supervisor(
            args=args,
            branch=None,
            run_output=args.output / "run",
            initial={},
            time_axis=time_axis,
            env=env,
            deadline=time.monotonic() + args.wall_time_seconds,
        )
        _write_json(
            args.output / "run_summary.json",
            {
                "schema": "oe_repeat_flat_breadth_v1",
                "fresh_start": True,
                "epochs": summaries,
                "finish_guided_cohort_after_wall_time": True,
            },
        )
        return 0
    attempts = args.source_trace_root.resolve() / "attempts.jsonl"
    if not attempts.is_file():
        raise RuntimeError(f"source trace is missing {attempts}")
    args.output.mkdir(parents=True)
    (args.output / "epochs").mkdir()
    initial = (
        {"completion_seq": args.initial_completion_seq, "selection": "explicit"}
        if args.initial_completion_seq is not None
        else first_flat_cutoff(
            attempts,
            minimum_elapsed_seconds=args.minimum_run_age_seconds,
            flat_seconds=args.flat_seconds,
            flat_min_terminal_attempts=args.flat_min_terminal_attempts,
        )
    )
    _write_json(args.output / "initial_cutoff.json", initial)
    _write_json(
        args.output / "supervisor_manifest.json",
        {
            "schema": "oe_repeat_flat_breadth_manifest_v1",
            "argv": sys.argv,
            "created_at": time.time(),
            "dispatch_mode": "overlap",
            "source_trace_root": str(args.source_trace_root.resolve()),
            "source_attempts_sha256": _sha256_file(attempts),
            "source_code_sha256": {
                name: _sha256_file(release_resource(name))
                for name in (
                    "run_repeat_flat_breadth.py",
                    "planner_context.py",
                    "code_geometry.py",
                    "metric_scaled_dashboard.py",
                    "run_branch_portfolio_breadth.py",
                    "planner_runtime.py",
                    "pi_cli.py",
                    "pi_unfold_selection_extension.ts",
                    "run_direction_llm_dedup.py",
                    "deduplicate.md",
                    "breadth_controller.py",
                    "materialize_branch.py",
                    "runner.py",
                    "oe_trajectory.py",
                    "planner.md",
                    "metrics.json",
                )
            },
            "credential_values_recorded": False,
        },
    )
    branch = args.output / "initial_branch"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SHARED), str(HERE), str(VENDOR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    run_output = args.output / "run"
    deadline = time.monotonic() + (args.wall_time_seconds - 0.0)
    _run_checked(
        [
            sys.executable,
            str(HERE / "materialize_branch.py"),
            "--trace-root",
            str(args.source_trace_root.resolve()),
            "--completion-seq",
            str(initial["completion_seq"]),
            "--output",
            str(branch),
            *["--preserve-inflight"],
        ],
        log_path=args.output / "initial_materialize.log",
        env=env,
        timeout=min(1800.0, deadline - time.monotonic()),
    )
    summaries = []
    time_axis = _event_time_axis(
        output=args.output,
        historical_prefix=branch / "historical_prefix.jsonl",
        supervisor_started_at=supervisor_started_at,
        active_started_seconds=0.0,
        resume_existing=False,
    )
    summaries = _run_overlap_supervisor(
        args=args,
        branch=branch,
        run_output=run_output,
        initial=initial,
        time_axis=time_axis,
        env=env,
        deadline=deadline,
    )
    _write_json(
        args.output / "run_summary.json",
        {
            "schema": "oe_repeat_flat_breadth_v1",
            "dispatch_mode": "overlap",
            "source_trace_root": str(args.source_trace_root.resolve()),
            "initial_cutoff": initial,
            "initial_ordinary": args.initial_ordinary,
            "flat_seconds": args.flat_seconds,
            "flat_min_terminal_attempts": args.flat_min_terminal_attempts,
            "wall_time_seconds": args.wall_time_seconds,
            "candidate_pool_size": args.candidate_pool_size,
            "planner_coordination_mode": "independent",
            "planner_sampling_mode": "verbalized-distribution",
            "planner_routing_mode": "shared",
            "planner_random_metric_count": args.planner_random_metric_count,
            "metric_subspace_seed": args.metric_subspace_seed,
            "direction_selector": "llm",
            "unfold_selection_mode": "registered",
            "worker_model": args.worker_model,
            "worker_provider": args.worker_provider,
            "worker_reasoning": args.worker_reasoning,
            "planner_model": args.planner_model,
            "planner_backends": json.loads(args.planner_backends_json)
            if args.planner_backends_json
            else None,
            "planner_reasoning": args.planner_reasoning,
            "planner_prompt_mode": "production",
            "finish_guided_cohort_after_wall_time": True,
            "epochs": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
