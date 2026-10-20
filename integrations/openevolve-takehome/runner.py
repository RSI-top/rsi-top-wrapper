"""
Run OpenEvolve on the Anthropic performance takehome until a hard USD budget
or wall-clock limit is reached.

Stopping is budget-driven, not iteration-driven:
  1. Every worker's LLM call checks the shared cost ledger first and refuses
     to spend past the budget (gemini_llm.BudgetExhausted).
  2. A watchdog thread in this process sends SIGTERM to itself once the
     ledger crosses the budget; OpenEvolve's signal handler then performs a
     graceful shutdown, saving a final checkpoint and the best program to
     <output>/best/best_program.py.

Resume after any stop (crash, Ctrl+C, budget bump) with:
  python runner.py --checkpoint <output>/checkpoints/checkpoint_<N>

Usage:
  export DEEPSEEK_API_KEY=...
  python runner.py --budget 400
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "shared/meta-agent-audit"))
from meta_audit.source_snapshot import capture_sources, write_source_snapshot
logger = logging.getLogger("runner")


def bootstrap_branch_prefix(checkpoint, attempts_path) -> dict | None:
    """Seed an exact materialized branch's causal ledger prefix once."""
    if checkpoint is None:
        return None
    checkpoint_path = Path(checkpoint).resolve()
    manifest_path = checkpoint_path.parent / "branch_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        sys.exit(f"branch manifest unreadable at {manifest_path}: {exc}")
    if manifest.get("schema") not in ("oe_resume_branch_v2", "oe_branch_manifest_v1"):
        return None
    if Path(str(manifest.get("branch_checkpoint", ""))).resolve() != checkpoint_path:
        sys.exit(
            f"branch manifest {manifest_path} does not describe --checkpoint {checkpoint_path}"
        )
    prefix_info = manifest.get("historical_prefix")
    if not isinstance(prefix_info, dict):
        sys.exit("branch manifest lacks historical_prefix")
    prefix_path = manifest_path.parent / "historical_prefix.jsonl"
    try:
        prefix_bytes = prefix_path.read_bytes()
    except OSError as exc:
        sys.exit(f"branch historical prefix unreadable at {prefix_path}: {exc}")
    if hashlib.sha256(prefix_bytes).hexdigest() != prefix_info.get("sha256"):
        sys.exit("branch historical prefix digest mismatch")
    if prefix_info.get("source_attempts_sha256") != (manifest.get("source") or {}).get(
        "attempts_sha256"
    ):
        sys.exit("branch source/prefix trace digests disagree")
    seed_bytes = prefix_bytes
    has_started = any(
        (
            isinstance(row, dict) and row.get("event") == "run_started"
            for row in (
                json.loads(line)
                for line in prefix_bytes.decode("utf-8", errors="replace").splitlines()
                if line.strip()
            )
        )
    )
    if not has_started:
        sys.exit("branch historical prefix must include its run_started record")
    attempts = Path(attempts_path)
    seeded = False
    if attempts.exists():
        with attempts.open("rb") as handle:
            if handle.read(len(seed_bytes)) != seed_bytes:
                sys.exit(
                    f"existing {attempts} does not start with the exact branch prefix"
                )
    else:
        attempts.parent.mkdir(parents=True, exist_ok=True)
        attempts.write_bytes(seed_bytes)
        seeded = True
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "seeded": seeded,
        "prefix_bytes": len(seed_bytes),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=float, default=400.0, help="hard budget in USD")
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--output", default=str(HERE / "output"))
    p.add_argument("--checkpoint", default=None, help="checkpoint dir to resume from")
    p.add_argument(
        "--iterations", type=int, default=None, help="optional iteration cap"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random-seed override for the loaded config",
    )
    p.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="optional native checkpoint cadence override",
    )
    p.add_argument(
        "--minimum-run-age-seconds",
        type=float,
        default=0.0,
        help="minimum age before a fresh run's first breadth intervention",
    )
    p.add_argument(
        "--takehome",
        default=str(HERE / "vendor" / "takehome"),
        help="path to a pristine original_performance_takehome clone",
    )
    p.add_argument(
        "--llm-provider",
        choices=("deepseek", "gemini"),
        default="deepseek",
        help="worker/evaluator model provider; deepseek requires DEEPSEEK_API_KEY and gemini requires GEMINI_API_KEY",
    )
    p.add_argument(
        "--gemini-model",
        default="gemini-3.1-pro-preview",
        help="Gemini model name used when --llm-provider gemini",
    )
    p.add_argument(
        "--gemini-reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="optional explicit Gemini worker/evaluator reasoning effort",
    )
    p.add_argument(
        "--deepseek-model",
        default="deepseek-v4-flash",
        help="DeepSeek model name used when --llm-provider deepseek",
    )
    p.add_argument(
        "--deepseek-reasoning-effort",
        choices=("high", "max"),
        default=None,
        help="optional explicit DeepSeek worker/evaluator reasoning effort",
    )
    p.add_argument(
        "--wall-time-seconds",
        type=float,
        default=None,
        help="optional wall-clock limit for the whole run (audit + generation + evaluation); a graceful SIGTERM is requested when it elapses",
    )
    p.add_argument(
        "--portfolio-repeat-flat-seconds",
        type=float,
        default=None,
        help="while continuing ordinary search after a portfolio burst, fence and drain again after this much active flat-frontier time (the repeat supervisor then starts the next burst)",
    )
    p.add_argument(
        "--portfolio-repeat-flat-min-terminal-attempts",
        type=int,
        default=12,
        help="also require this many terminal attempts since the last meaningful frontier improvement before repeating a portfolio burst",
    )
    p.add_argument(
        "--portfolio-overlap-control-dir",
        default=None,
        help="dynamic planner handoff directory for overlap breadth dispatch",
    )
    p.add_argument(
        "--portfolio-pending-inflight",
        default=None,
        help="pending_inflight.json produced by --preserve-inflight",
    )
    p.add_argument(
        "--portfolio-overlap-start-ordinary",
        action="store_true",
        help="resume overlap mode in ordinary search and start a fresh flat timer",
    )
    args = p.parse_args()
    if args.gemini_reasoning_effort is not None and args.llm_provider != "gemini":
        p.error("--gemini-reasoning-effort requires --llm-provider gemini")
    if args.deepseek_reasoning_effort is not None and args.llm_provider != "deepseek":
        p.error("--deepseek-reasoning-effort requires --llm-provider deepseek")
    if args.portfolio_overlap_control_dir is not None:
        if args.checkpoint is None and (not args.portfolio_overlap_start_ordinary):
            p.error("fresh overlap runs require --portfolio-overlap-start-ordinary")
    elif args.portfolio_pending_inflight is not None:
        p.error("--portfolio-pending-inflight requires --portfolio-overlap-control-dir")
    if (
        args.portfolio_overlap_start_ordinary
        and args.portfolio_overlap_control_dir is None
    ):
        p.error(
            "--portfolio-overlap-start-ordinary requires --portfolio-overlap-control-dir"
        )
    if (
        args.portfolio_repeat_flat_seconds is not None
        and args.portfolio_overlap_control_dir is None
    ):
        p.error(
            "--portfolio-repeat-flat-seconds requires --portfolio-overlap-control-dir"
        )
    if args.portfolio_repeat_flat_min_terminal_attempts < 1:
        p.error("--portfolio-repeat-flat-min-terminal-attempts must be positive")
    return args


def make_provider_ledger(provider: str):
    """The selected provider's cost ledger, applying that provider's prices."""
    if provider == "gemini":
        from gemini_llm import GeminiCostLedger

        return GeminiCostLedger()
    from deepseek_llm import DeepSeekCostLedger

    return DeepSeekCostLedger()


def budget_watchdog(budget: float, poll_s: float = 20.0, provider: str = "deepseek"):
    ledger = make_provider_ledger(provider)
    while True:
        time.sleep(poll_s)
        spent = ledger.total()
        if spent >= budget:
            logger.warning(
                f"BUDGET REACHED: ${spent:.2f} >= ${budget:.2f} — requesting graceful shutdown (best program + checkpoint will be saved)"
            )
            os.kill(os.getpid(), signal.SIGTERM)
            return


def wall_time_watchdog(limit_seconds: float, started_monotonic: float):
    """Graceful SIGTERM once total wall time (audit + generation + evaluation)
    crosses the limit — same shutdown path as the budget watchdog."""
    remaining = limit_seconds - (time.monotonic() - started_monotonic)
    if remaining > 0:
        time.sleep(remaining)
    logger.warning(
        f"WALL TIME REACHED: {limit_seconds:.0f}s elapsed — requesting graceful shutdown (best program + checkpoint will be saved)"
    )
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    args = parse_args()
    wall_started_monotonic = time.monotonic()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    takehome = Path(args.takehome).resolve()
    if (
        not (takehome / "problem.py").exists()
        or not (takehome / "frozen_problem.py").exists()
    ):
        sys.exit(
            f"Takehome repo not found/complete at {takehome} — run ./setup.sh first"
        )
    key_env = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}[args.llm_provider]
    if not os.environ.get(key_env):
        sys.exit(f"--llm-provider {args.llm_provider} requires {key_env} to be set")
    effort_env = f"OE_{args.llm_provider.upper()}_REASONING_EFFORT"
    effort = getattr(args, f"{args.llm_provider}_reasoning_effort")
    if effort is None:
        os.environ.pop(effort_env, None)
    else:
        os.environ[effort_env] = effort
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OE_BUDGET_USD"] = str(args.budget)
    os.environ["OE_COST_LEDGER"] = str(out_dir / "cost_ledger.json")
    os.environ["OE_TAKEHOME_DIR"] = str(takehome)
    os.environ["OE_RUN_INSTANCE_ID"] = str(uuid.uuid4())
    os.environ["OE_ATTEMPT_TRACE"] = str(out_dir / "attempts.jsonl")
    branch_resume = bootstrap_branch_prefix(
        args.checkpoint, os.environ["OE_ATTEMPT_TRACE"]
    )
    import openevolve.controller as controller_module
    from openevolve.config import load_config
    import deepseek_llm
    import gemini_llm
    from oe_trajectory import (
        emit_event,
        install_checkpoint_sidecar_trace,
        install_evaluator_trace,
        install_log_trace,
        install_prompt_trace,
        make_traced_parallel_controller,
        sha256_text,
        write_json_atomic,
    )

    install_log_trace()
    install_prompt_trace()
    install_evaluator_trace()
    install_checkpoint_sidecar_trace()
    meta_manager = None
    meta_manifest = {"mode": "off"}
    if args.portfolio_overlap_control_dir is not None:
        from breadth_controller import OverlapPortfolioManager

        if args.portfolio_repeat_flat_seconds is None:
            sys.exit("portfolio overlap requires --portfolio-repeat-flat-seconds")
        meta_manager = OverlapPortfolioManager(
            control_dir=args.portfolio_overlap_control_dir,
            output_dir=out_dir,
            repeat_flat_seconds=args.portfolio_repeat_flat_seconds,
            repeat_flat_min_terminal_attempts=args.portfolio_repeat_flat_min_terminal_attempts,
            pending_inflight_path=args.portfolio_pending_inflight,
            start_ordinary=args.portfolio_overlap_start_ordinary,
            minimum_run_age_seconds=args.minimum_run_age_seconds,
        )
        meta_manifest = meta_manager.describe()
    controller_module.ProcessParallelController = make_traced_parallel_controller(
        meta_audit=meta_manager
    )
    OpenEvolve = controller_module.OpenEvolve
    config = load_config(args.config)
    if not config.prompt.system_message:
        config.prompt.system_message = (
            HERE.parents[1] / "prompts" / "worker_openevolve.md"
        ).read_text(encoding="utf-8")
    if args.portfolio_overlap_control_dir is not None:
        native_concurrency = int(config.evaluator.parallel_evaluations)
        meta_manager.configure_native_concurrency(native_concurrency)
        config.evaluator.parallel_evaluations = max(native_concurrency, 32)
        meta_manifest = meta_manager.describe()
    if args.seed is not None:
        config.random_seed = args.seed
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval < 1:
            sys.exit("--checkpoint-interval must be >= 1")
        config.checkpoint_interval = args.checkpoint_interval

    def git_snapshot(path: Path):

        def git(*parts):
            return subprocess.check_output(
                ["git", "-C", str(path), *parts], text=True, stderr=subprocess.DEVNULL
            ).strip()

        try:
            return {
                "path": str(path),
                "commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "status_porcelain": git(
                    "status", "--porcelain=v1", "--untracked-files=all"
                ),
            }
        except (OSError, subprocess.CalledProcessError):
            return {"path": str(path), "git_metadata_error": True}

    sources = capture_sources(
        HERE.parents[1],
        extra_files=[Path(args.config), takehome / "problem.py", takehome / "frozen_problem.py"],
    )
    source_provenance = write_source_snapshot(out_dir, sources)
    # Preserve the existing absolute-path/content interface for trace consumers.
    source_snapshot = {entry["path"]: entry for entry in sources.values()}
    write_json_atomic(
        out_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "created_at": time.time(),
            "run_instance_id": os.environ["OE_RUN_INSTANCE_ID"],
            "argv": sys.argv,
            "budget_usd": args.budget,
            "llm_provider": args.llm_provider,
            "model": getattr(args, f"{args.llm_provider}_model"),
            "reasoning_effort": getattr(args, f"{args.llm_provider}_reasoning_effort"),
            "wall_time_seconds": args.wall_time_seconds,
            "iterations": args.iterations,
            "random_seed": config.random_seed,
            "checkpoint_interval": config.checkpoint_interval,
            "checkpoint": args.checkpoint,
            "materialized_branch_resume": {
                "manifest_path": branch_resume["manifest_path"],
                "schema": branch_resume["manifest"].get("schema"),
                "prefix_seeded": branch_resume["seeded"],
            }
            if branch_resume is not None
            else None,
            "output_dir": str(out_dir),
            "takehome_dir": str(takehome),
            "python": sys.version,
            "platform": platform.platform(),
            "openevolve": git_snapshot(HERE / "vendor" / "openevolve"),
            "takehome": git_snapshot(takehome),
            "meta_audit": meta_manifest,
            "source_snapshot": source_snapshot,
            "source_provenance": source_provenance,
            "credential_values_recorded": False,
        },
    )
    emit_event(
        "run_started",
        source_provenance=source_provenance,
        budget_usd=args.budget,
        iterations=args.iterations,
        random_seed=config.random_seed,
        checkpoint_interval=config.checkpoint_interval,
        checkpoint=args.checkpoint,
        config_path=str(Path(args.config).resolve()),
        config_sha256=sha256_text(Path(args.config).read_text()),
        initial_program_path=str(HERE / "initial_program.py"),
        initial_program_sha256=sha256_text((HERE / "initial_program.py").read_text()),
        evaluator_sha256=sha256_text((HERE / "evaluator.py").read_text()),
        takehome_dir=str(takehome),
        output_dir=str(out_dir),
        python=sys.version,
    )
    problem_src = (takehome / "problem.py").read_text()
    config.prompt.system_message = (
        config.prompt.system_message
        + "\n\n# ==== problem.py (simulator + ISA, read-only reference) ====\n"
        + "```python\n"
        + problem_src
        + "\n```\n"
    )
    llm_module = {"deepseek": deepseek_llm, "gemini": gemini_llm}[args.llm_provider]
    model_name = getattr(args, f"{args.llm_provider}_model")
    init_client = getattr(llm_module, f"init_{args.llm_provider}_client")
    for m in config.llm.models + config.llm.evaluator_models:
        m.name = model_name
        m.init_client = init_client
    controller = OpenEvolve(
        initial_program_path=str(HERE / "initial_program.py"),
        evaluation_file=str(HERE / "evaluator.py"),
        config=config,
        output_dir=str(out_dir),
    )
    threading.Thread(
        target=budget_watchdog,
        args=(args.budget,),
        kwargs={"provider": args.llm_provider},
        daemon=True,
    ).start()
    if args.wall_time_seconds is not None:
        threading.Thread(
            target=wall_time_watchdog,
            args=(args.wall_time_seconds, wall_started_monotonic),
            daemon=True,
        ).start()
    ledger = make_provider_ledger(args.llm_provider)
    logger.info(
        f"Starting evolution: budget=${args.budget:.2f} (already spent ${ledger.total():.2f}), output={out_dir}"
    )
    try:
        best = asyncio.run(
            controller.run(iterations=args.iterations, checkpoint_path=args.checkpoint)
        )
    except BaseException as exc:
        emit_event("run_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if meta_manager is not None:
            meta_manager.shutdown()
    spent = ledger.total()
    emit_event(
        "run_completed",
        spent_usd=spent,
        budget_usd=args.budget,
        best_program_id=best.id if best is not None else None,
        best_metrics=best.metrics if best is not None else None,
    )
    print("\n" + "=" * 70)
    print(f"Run finished. Total spend: ${spent:.2f} of ${args.budget:.2f}")
    if best is not None:
        cycles = best.metrics.get("cycles")
        print(f"Best program: {out_dir / 'best' / 'best_program.py'}")
        print(f"Best metrics: {best.metrics}")
        if cycles:
            print(
                f"Cycles: {int(cycles)}  (baseline 147734, speedup {147734 / cycles:.2f}x)"
            )
        print(
            f"\nRe-score with: python {HERE.parents[1] / 'evaluate.py'} {out_dir / 'best/best_program.py'}"
        )
    else:
        print("No valid program produced (check logs).")
    print(
        f"Resume with more budget:  python runner.py --budget <new_total> --checkpoint {out_dir}/checkpoints/checkpoint_<latest>"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
