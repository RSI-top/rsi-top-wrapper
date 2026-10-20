"""Run Shinka with the validated OpenEvolve-1108 breadth treatment."""

from __future__ import annotations
from release_paths import release_resource
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OE = HERE.parent / "openevolve-takehome"
PROMPT = release_resource("planner.md")


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run(command: list[str], *, log: Path, env: dict[str, str], timeout: float) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=HERE,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _cohort_complete(journal: Path, cohort: int) -> bool:
    return any(
        (
            row.get("kind") == "guided_cohort_complete"
            and row.get("cohort_index") == cohort
            for row in _rows(journal)
        )
    )


def _guided_pending(journal: Path) -> bool:
    rows = _rows(journal)
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


def _failed_directions(
    *, existing: list[dict[str, Any]], planner: Path, journal: Path, cohort: int
) -> list[dict[str, Any]]:
    rows = [row for row in _rows(journal) if row.get("cohort_index") == cohort]
    if any((row.get("kind") == "meaningful_frontier_improvement" for row in rows)):
        return []
    if not _cohort_complete(journal, cohort):
        return list(existing)
    directions = json.loads((planner / "portfolio.json").read_text(encoding="utf-8"))[
        "directions"
    ]
    assignments = {
        int(row["generation"]): int(row["direction_index"]) - 1
        for row in rows
        if row.get("kind") == "direction_assigned"
    }
    terminals = {
        int(row["generation"]): row
        for row in rows
        if row.get("kind") == "guided_attempt_terminal"
    }
    bound = next((row for row in rows if row.get("kind") == "cohort_bound"), {})
    frontier = bound.get("frontier_cycles")
    result = list(existing)
    for generation, index in sorted(assignments.items(), key=lambda item: item[1]):
        direction = directions[index]
        terminal = terminals.get(generation, {})
        cycles = terminal.get("cycles")
        valid = terminal.get("evaluation_valid") is True
        if not valid or not isinstance(cycles, (int, float)):
            outcome, delta = ("invalid", None)
        elif not isinstance(frontier, (int, float)):
            outcome, delta = ("unknown", None)
        else:
            delta = float(frontier) - float(cycles)
            outcome = "tie" if delta >= 0 else "regress"
        result.append(
            {
                "type": direction.get("type"),
                "mechanism_hypothesis": direction.get("mechanism_hypothesis"),
                "task": direction.get("task"),
                "target_branch_id": direction.get("target_branch_id"),
                "outcome": outcome,
                "frontier_delta_cycles": delta,
                "completion_seq": terminal.get("completion_seq"),
            }
        )
    return result


def _source_results(output: Path) -> Path | None:
    manifest = output / "branch_manifest.json"
    if not manifest.is_file():
        return None
    document = json.loads(manifest.read_text(encoding="utf-8"))
    source = document.get("source")
    return Path(source).resolve() if isinstance(source, str) and source else None


def _latest_config(root: Path | None) -> Path | None:
    if root is None:
        return None
    snapshots = sorted((root / "run_configs").glob("*.json"))
    return snapshots[-1] if snapshots else None


def _runner_python(output: Path) -> Path:
    identity = output / "evaluator_provenance" / "interpreter.json"
    if not identity.is_file():
        return Path(sys.executable)
    executable = Path(json.loads(identity.read_text(encoding="utf-8"))["executable"])
    if not executable.is_file():
        raise RuntimeError(f"sealed Shinka interpreter is unavailable: {executable}")
    return executable


def _resume_planner_inputs(
    source: Path,
) -> tuple[int, list[dict[str, Any]], Path, Path | None]:
    """Reuse the pending frozen planner input, preserving earlier outcomes."""
    epochs = json.loads((source / "breadth_supervisor_state.json").read_text())[
        "epochs"
    ]
    offset = int(epochs[-1]["cohort_index"]) + 1 if epochs else 0
    control = source / "breadth_control"
    prepared = (
        source / "breadth_epochs" / f"epoch_{offset:03d}" / "view_build/prepared/bundle"
    )
    if (
        not (control / f"request_{offset:03d}.json").is_file()
        or (control / f"handoff_{offset:03d}.json").exists()
        or (not prepared.is_dir())
    ):
        raise RuntimeError(
            "resume source must have an unhanded, prepared planner request"
        )
    failed: list[dict[str, Any]] = []
    for epoch in epochs:
        failed = _failed_directions(
            existing=failed,
            planner=Path(epoch["planner"]),
            journal=source / "breadth_interventions.jsonl",
            cohort=int(epoch["cohort_index"]),
        )
    completed_planner = source / "breadth_epochs" / f"epoch_{offset:03d}" / "planner"
    if (completed_planner / "run.json").is_file():
        run = json.loads((completed_planner / "run.json").read_text())
        portfolio = json.loads((completed_planner / "portfolio.json").read_text())
        if len(portfolio["directions"]) != run["actual_selected_directions"]:
            raise RuntimeError("completed pending planner direction count mismatch")
    else:
        completed_planner = None
    manifest_path = source / "breadth_supervisor_manifest.json"
    source_offset = (
        int(json.loads(manifest_path.read_text()).get("cohort_index_offset", 0))
        if manifest_path.is_file()
        else 0
    )
    return (source_offset + offset, failed, prepared, completed_planner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="materialized Shinka branch/results directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Shinka config snapshot; default: the materialized branch's latest run_configs snapshot",
    )
    parser.add_argument("--budget", type=float, default=400.0)
    parser.add_argument("--wall-time-seconds", type=float, default=32400.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-model", default="deepseek-v4-flash")
    parser.add_argument("--worker-reasoning", default="high")
    parser.add_argument("--runner-entrypoint", type=Path, default=HERE / "run_evo.py")
    parser.add_argument("--planner-model", default="deepseek-v4-flash")
    parser.add_argument("--planner-provider", default="deepseek")
    parser.add_argument("--planner-reasoning", default="high")
    parser.add_argument("--planner-base-url", default="https://api.deepseek.com")
    parser.add_argument("--planner-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--metric-subspace-seed", type=int, default=20260830)
    parser.add_argument("--flat-seconds", type=float, default=1200.0)
    parser.add_argument("--minimum-run-age-seconds", type=float, default=0.0)
    parser.add_argument("--flat-min-terminal-attempts", type=int, default=12)
    parser.add_argument("--initial-ordinary", action="store_true")
    parser.add_argument("--stop-after-guided-cohort", action="store_true")
    parser.add_argument(
        "--resume-interventions-from",
        type=Path,
        help="reuse a stopped run's pending frozen planner input and prior outcomes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.config is None:
        config = _latest_config(_source_results(output)) or _latest_config(output)
        if config is None:
            raise SystemExit(
                "--config is required for a new run; a materialized branch automatically reuses its source run config"
            )
    else:
        config = args.config.resolve()
    if not config.is_file():
        raise SystemExit(f"Shinka config not found: {config}")
    cohort_offset, failed, initial_prepared, initial_planner = (
        _resume_planner_inputs(args.resume_interventions_from.resolve())
        if args.resume_interventions_from
        else (0, [], None, None)
    )
    control = output / "breadth_control"
    if control.exists() and any(control.iterdir()):
        raise SystemExit(f"breadth control directory is not empty: {control}")
    control.mkdir(exist_ok=True)
    epochs = output / "breadth_epochs"
    epochs.mkdir(exist_ok=True)
    journal = output / "breadth_interventions.jsonl"
    env = dict(os.environ)
    env.update(
        {
            "AUDIT_PI_PROVIDER": args.planner_provider,
            "AUDIT_PI_MODEL": args.planner_model,
            "AUDIT_PI_REASONING": args.planner_reasoning,
            "AUDIT_PI_API_KEY_ENV": args.planner_api_key_env,
        }
    )
    if not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    if not env.get(args.planner_api_key_env):
        raise SystemExit(f"{args.planner_api_key_env} is not set")
    if not args.runner_entrypoint.is_file():
        raise SystemExit(f"runner entry point not found: {args.runner_entrypoint}")
    manifest = {
        "schema": "shinka_repeat_flat_breadth_1108_v1",
        "created_at": time.time(),
        "output": str(output),
        "worker": {"model": args.worker_model, "reasoning": args.worker_reasoning},
        "shinka_config": str(config),
        "runner_python": str(_runner_python(output)),
        "runner_entrypoint": str(args.runner_entrypoint.resolve()),
        "trigger": {
            "flat_seconds": args.flat_seconds,
            "terminal_attempts": args.flat_min_terminal_attempts,
        },
        "dispatch_mode": "overlap",
        "stop_after_guided_cohort": args.stop_after_guided_cohort,
        "planner": {
            "model": args.planner_model,
            "provider": args.planner_provider,
            "base_url": args.planner_base_url,
            "api_key_env": args.planner_api_key_env,
            "reasoning": args.planner_reasoning,
            "shape": "4x8",
            "candidate_pool_size": 32,
            "sampling_mode": "verbalized-distribution",
            "routing_mode": "shared",
            "unfold_selection_mode": "registered",
            "random_metric_count": 15,
            "metric_subspace_seed": args.metric_subspace_seed + cohort_offset * 1000003,
            "direction_selector": "llm",
            "dedup_model": "deepseek-v4-flash",
            "dedup_max_tokens": 128000,
            "prompt": str(PROMPT),
            "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
        },
        "credential_values_recorded": False,
        "resume_interventions_from": str(args.resume_interventions_from)
        if args.resume_interventions_from
        else None,
        "cohort_index_offset": cohort_offset,
    }
    _write_json(output / "breadth_supervisor_manifest.json", manifest)
    runner_command = [
        str(_runner_python(output)),
        str(args.runner_entrypoint.resolve()),
        "--output",
        str(output),
        "--config",
        str(config),
        "--budget",
        str(args.budget),
        "--seed",
        str(args.seed),
        "--minimum-run-age-seconds",
        str(args.minimum_run_age_seconds),
        "--llm-model",
        args.worker_model,
        "--effort",
        args.worker_reasoning,
        "--resume-capture",
        "--breadth-control-dir",
        str(control),
        "--breadth-flat-seconds",
        str(args.flat_seconds),
        "--breadth-flat-min-terminal-attempts",
        str(args.flat_min_terminal_attempts),
        *(["--breadth-start-ordinary"] if args.initial_ordinary else []),
        *(
            ["--breadth-stop-after-guided-cohort"]
            if args.stop_after_guided_cohort
            else []
        ),
    ]
    log_path = output / "breadth_runner.log"
    log = log_path.open("w", encoding="utf-8")
    runner = subprocess.Popen(
        runner_command,
        cwd=HERE,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + args.wall_time_seconds
    cohort = 0
    planners: dict[int, Path] = {}
    summaries: list[dict[str, Any]] = []
    try:
        while runner.poll() is None:
            request = control / f"request_{cohort:03d}.json"
            handoff = control / f"handoff_{cohort:03d}.json"
            if request.is_file() and (not handoff.is_file()):
                if cohort > 0:
                    failed = _failed_directions(
                        existing=failed,
                        planner=planners[cohort - 1],
                        journal=journal,
                        cohort=cohort - 1,
                    )
                epoch = epochs / f"epoch_{cohort:03d}"
                epoch.mkdir(exist_ok=False)
                prepared = epoch / "view_build"
                if cohort == 0 and initial_prepared is not None:
                    prepared_view = initial_prepared
                    _write_json(
                        epoch / "reused_frozen_input.json",
                        {"prepared_view": str(prepared_view)},
                    )
                else:
                    _run(
                        [
                            sys.executable,
                            str(HERE / "prepare_shinka_breadth_view.py"),
                            "--request",
                            str(request),
                            "--output",
                            str(prepared),
                            "--prompt",
                            str(PROMPT),
                        ],
                        log=epoch / "prepare.log",
                        env=env,
                        timeout=3600.0,
                    )
                    prepared_view = prepared / "prepared" / "bundle"
                failed_path = epoch / "prior_failed_interventions.json"
                _write_json(failed_path, failed)
                planner = (
                    initial_planner
                    if cohort == 0 and initial_planner
                    else epoch / "planner"
                )
                planners[cohort] = planner
                if cohort == 0 and initial_planner is not None:
                    _write_json(
                        epoch / "reused_planner.json",
                        {
                            "planner_dir": str(planner),
                            "run_sha256": hashlib.sha256(
                                (planner / "run.json").read_bytes()
                            ).hexdigest(),
                            "portfolio_sha256": hashlib.sha256(
                                (planner / "portfolio.json").read_bytes()
                            ).hexdigest(),
                        },
                    )
                else:
                    _run(
                        [
                            sys.executable,
                            str(OE / "run_branch_portfolio_breadth.py"),
                            "--prepared-view",
                            str(prepared_view),
                            "--output",
                            str(planner),
                            "--prompt",
                            str(PROMPT),
                            "--candidate-pool-size",
                            "32",
                            "--directions-per-call",
                            "8",
                            "--planner-workers",
                            "4",
                            "--planner-random-metric-count",
                            "15",
                            "--metric-subspace-seed",
                            str(
                                args.metric_subspace_seed
                                + (cohort_offset + cohort) * 1000003
                            ),
                            "--dedup-model",
                            "deepseek-v4-flash",
                            "--dedup-base-url",
                            "https://api.deepseek.com",
                            "--dedup-api-key-env",
                            "DEEPSEEK_API_KEY",
                            "--dedup-max-tokens",
                            "128000",
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
                        ],
                        log=epoch / "planner.log",
                        env=env,
                        timeout=6 * 3600.0,
                    )
                run = json.loads((planner / "run.json").read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "cohort_index": cohort,
                        "trigger_completion_seq": json.loads(
                            request.read_text(encoding="utf-8")
                        )["cutoff_seq"],
                        "selected_directions": run["actual_selected_directions"],
                        "planner": str(planner),
                    }
                )
                _write_json(
                    output / "breadth_supervisor_state.json", {"epochs": summaries}
                )
                _write_json(
                    handoff,
                    {
                        "schema": "shinka_overlap_portfolio_handoff_v1",
                        "cohort_index": cohort,
                        "planner_dir": str(planner),
                    },
                )
                cohort += 1
            if (
                args.stop_after_guided_cohort
                and cohort
                and _cohort_complete(journal, cohort - 1)
            ):
                os.killpg(runner.pid, signal.SIGINT)
                break
            cohort_in_progress = cohort > 0 and (
                not _cohort_complete(journal, cohort - 1)
            )
            if (
                time.monotonic() >= deadline
                and (not cohort_in_progress)
                and (not _guided_pending(journal))
            ):
                os.killpg(runner.pid, signal.SIGINT)
                break
            time.sleep(1.0)
    except BaseException:
        if runner.poll() is None:
            os.killpg(runner.pid, signal.SIGINT)
        raise
    finally:
        if runner.poll() is None:
            try:
                runner.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(runner.pid, signal.SIGKILL)
                runner.wait()
        log.close()
    if runner.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(f"Shinka runner exited {runner.returncode}; see {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
