"""
Run ShinkaEvolve on the performance takehome until a hard USD budget is spent.
The worker model comes from the YAML config and can be replaced explicitly with
``--llm-model``; provider credentials are selected from that effective model.

Stopping is budget-driven: ShinkaEvolve natively enforces
evo_config.max_api_costs (actual + in-flight cost) and shuts down gracefully.
num_generations is set effectively unlimited, so the budget is the only stop.

Resume: just re-run the same command. If results/programs.sqlite exists the
runner resumes automatically AND reloads prior API spend, so --budget is a
LIFETIME total for the results dir (pass a larger value to add budget).

Usage:
  export GEMINI_API_KEY=...
  python run_evo.py --budget 400 --llm-model gemini-3.1-pro-preview
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parents[1] / "shared" / "meta-agent-audit"
VENDOR = HERE / "vendor" / "shinkaevolve"
for path in (HERE, SHARED, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from meta_audit.source_snapshot import capture_sources, write_source_snapshot

os.environ.setdefault("SHINKA_LLM_TIMEOUT", "3600")
os.environ.setdefault("SHINKA_LLM_BACKOFF_MAX_TRIES", "5")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit(path):
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_dirty(path):
    proc = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def _write_json_atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tmp.open("r+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _record_manifest(results_dir, run_id, args, config, takehome):
    results_dir = Path(results_dir)
    config_dir = results_dir / "run_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_id}.json"
    _write_json_atomic(config_path, config)
    manifest_path = results_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = {}
    sources = capture_sources(
        HERE.parents[1],
        extra_files=[Path(args.config), Path(takehome) / "problem.py", Path(takehome) / "frozen_problem.py"],
    )
    source_provenance = write_source_snapshot(results_dir, sources)
    local_prefix = HERE.relative_to(HERE.parents[1]).as_posix() + "/"
    shinka_vendor = HERE / "vendor" / "shinkaevolve"
    vendor_runtime_names = (
        "shinka/core/async_runner.py",
        "shinka/embed/client.py",
        "shinka/embed/embedding.py",
        "shinka/google_genai.py",
        "shinka/llm/client.py",
        "shinka/llm/query.py",
        "shinka/llm/providers/gemini.py",
        "shinka/llm/providers/model_resolver.py",
        "shinka/llm/providers/pricing.csv",
    )
    gemini_judge_source = (
        HERE.parent / "openevolve-takehome" / "gemini_llm.py"
    ).resolve()
    manifest.update(
        {
            "schema_version": "shinka-lossless-run-v1",
            "project_dir": str(HERE),
            "results_dir": str(results_dir),
            "source_files": {
                name.removeprefix(local_prefix): entry["sha256"]
                for name, entry in sources.items()
                if name.startswith(local_prefix)
            },
            "source_provenance": source_provenance,
            "worker_prompt_sha256": _sha256(
                HERE.parents[1] / "prompts" / "worker_shinka.md"
            ),
            "vendor_shinkaevolve_commit": _git_commit(shinka_vendor),
            "vendor_shinkaevolve_dirty": _git_dirty(shinka_vendor),
            "vendor_shinkaevolve_runtime_files": {
                name: _sha256(shinka_vendor / name)
                for name in vendor_runtime_names
                if (shinka_vendor / name).is_file()
            },
            "gemini_judge_backend_source": {
                "path": str(gemini_judge_source),
                "sha256": _sha256(gemini_judge_source)
                if gemini_judge_source.is_file()
                else None,
                "shared_with": "openevolve-takehome",
            },
            "vendor_takehome_commit": _git_commit(takehome),
            "trajectory_authority": "attempts.jsonl",
            "program_authority": "programs.sqlite after successful transaction",
            "completion_order": "fsync append immediately after SQLite commit, interleaved with durable terminal proposal failures",
        }
    )
    instances = manifest.setdefault("run_instances", [])
    instances.append(
        {
            "run_instance_id": run_id,
            "source_provenance": source_provenance,
            "started_at": time.time(),
            "budget_usd_lifetime": args.budget,
            "effort": args.effort,
            "random_seed": args.seed,
            "llm_model_override": args.llm_model,
            "breadth_control_dir": args.breadth_control_dir,
            "num_generations": config["evo_config"].get("num_generations"),
            "config_snapshot": str(config_path),
            "config_sha256": _sha256(config_path),
            "anthropic_api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "gemini_api_key_present": bool(os.environ.get("GEMINI_API_KEY")),
            "openrouter_api_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
        }
    )
    _write_json_atomic(manifest_path, manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--budget",
        type=float,
        default=400.0,
        help="lifetime USD budget for this results dir",
    )
    p.add_argument("--config", default=str(HERE / "shinka.yaml"))
    p.add_argument(
        "--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"]
    )
    p.add_argument("--takehome", default=str(HERE / "vendor" / "takehome"))
    p.add_argument(
        "--output",
        default=None,
        help="results directory (default: config results_dir under this project)",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="optional finite generation cap, useful for controlled tests",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="runner sampling seed (matches the OpenEvolve control)",
    )
    p.add_argument("--minimum-run-age-seconds", type=float, default=0.0)
    p.add_argument(
        "--llm-model",
        default=None,
        help="optional worker model override; replaces evo_config.llm_models without changing any search setting",
    )
    p.add_argument(
        "--resume-capture",
        action="store_true",
        help="after every committed completion, capture SQLite/controller/RNG state without fencing or draining native dispatch",
    )
    p.add_argument(
        "--breadth-control-dir",
        default=None,
        help="internal opt-in control directory for the repeated OpenEvolve-1108 breadth treatment",
    )
    p.add_argument(
        "--breadth-flat-seconds",
        type=float,
        default=1200.0,
        help="active flat-search seconds before a repeated breadth trigger",
    )
    p.add_argument(
        "--breadth-flat-min-terminal-attempts",
        type=int,
        default=12,
        help="terminal ordinary attempts required with the breadth time gate",
    )
    p.add_argument(
        "--breadth-start-ordinary",
        action="store_true",
        help="start ordinary search before the first breadth trigger; default is an immediate intervention at a materialized cutoff",
    )
    p.add_argument(
        "--breadth-stop-after-guided-cohort",
        action="store_true",
        help="do not refill ordinary proposals while the one-shot cohort finishes",
    )
    p.add_argument(
        "--causal-checkpoint-cadence",
        type=int,
        default=5,
        metavar="N",
        help="request a barrier every N completed generations (default 5; only meaningful with --causal-checkpoint)",
    )
    p.add_argument(
        "--causal-drain-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for quiescence at a barrier before disabling checkpointing (the run continues natively)",
    )
    args = p.parse_args()
    if args.causal_checkpoint_cadence < 1:
        p.error("--causal-checkpoint-cadence must be >= 1")
    if args.breadth_control_dir is not None:
        if args.breadth_flat_seconds <= 0:
            p.error("--breadth-flat-seconds must be positive")
        if args.breadth_flat_min_terminal_attempts < 1:
            p.error("--breadth-flat-min-terminal-attempts must be positive")
    takehome = Path(args.takehome).resolve()
    if (
        not (takehome / "problem.py").exists()
        or not (takehome / "frozen_problem.py").exists()
    ):
        sys.exit(f"Takehome repo incomplete at {takehome} — run ./setup.sh first")
    os.environ["SHINKA_TAKEHOME_DIR"] = str(takehome)
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)
    import yaml
    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.llm_model is not None:
        config["evo_config"]["llm_models"] = [args.llm_model]
    configured_models = tuple(config["evo_config"].get("llm_models") or ())
    if any((str(model).startswith("gemini-") for model in configured_models)):
        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY is not set for the configured Gemini worker")
    import gemini31_patch

    gemini31_patch.apply()
    task_sys_msg = (
        (HERE.parents[1] / "prompts" / "worker_shinka.md")
        .read_text(encoding="utf-8")
        .replace(
            "{simulator_source}", (takehome / "problem.py").read_text(encoding="utf-8")
        )
    )
    config["evo_config"]["task_sys_msg"] = task_sys_msg
    config["evo_config"]["max_api_costs"] = args.budget
    config["evo_config"]["init_program_path"] = str(HERE / "initial.py")
    configured_output = (
        args.output or config["evo_config"].get("results_dir") or "results"
    )
    output = Path(configured_output)
    if not output.is_absolute():
        output = HERE / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["evo_config"]["results_dir"] = str(output)
    if args.generations is not None:
        if args.generations < 1:
            p.error("--generations must be at least 1")
        config["evo_config"]["num_generations"] = args.generations
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path=str(HERE / "evaluate.py"), time="00:15:00"
    )
    db_config = DatabaseConfig(**config["db_config"])
    checkpointer = None
    if False or args.resume_capture:
        import shinka_causal_checkpoint

        checkpointer = shinka_causal_checkpoint.ShinkaCausalCheckpointer(
            results=output,
            here=HERE,
            takehome=takehome,
            evo_config=evo_config,
            db_config=db_config,
            raw_config=config,
            task_sys_msg=task_sys_msg,
            cli={
                "budget": args.budget,
                "causal_checkpoint_cadence": args.causal_checkpoint_cadence,
                "config": str(Path(args.config).resolve()),
                "effort": args.effort,
                "generations": args.generations,
                "llm_model": args.llm_model,
                "output": str(output),
                "seed": args.seed,
                "state_capture_mode": "completion",
                "takehome": str(takehome),
            },
            cadence_generations=args.causal_checkpoint_cadence,
            drain_timeout=args.causal_drain_timeout,
            allowed_runtime_source_drift=frozenset(
                {"run_evo.py", "vendor_shinka_tree", "trajectory.py"}
            )
            if args.breadth_control_dir is not None
            else frozenset(),
        )
        causal_bound = checkpointer.bind_job_config(job_config)
    run_id = str(uuid.uuid4())
    os.environ["SHINKA_RUN_INSTANCE_ID"] = run_id
    os.environ["SHINKA_ATTEMPT_TRACE"] = str(output / "attempts.jsonl")
    os.environ["SHINKA_SQLITE_PATH"] = str(output / "programs.sqlite")
    import trajectory

    trajectory.emit_event(
        "run_started",
        budget_usd_lifetime=args.budget,
        effort=args.effort,
        random_seed=args.seed,
        output_dir=str(output),
        config_path=str(Path(args.config).resolve()),
        llm_models=list(config["evo_config"].get("llm_models") or ()),
    )
    trajectory.install_runner_trace()
    if checkpointer is not None:
        checkpointer.recorder = lambda event, **fields: trajectory.emit_event(
            event, **fields
        )
        trajectory.emit_event(
            "causal_checkpoint_enabled",
            mode=checkpointer.mode,
            evaluator_closure_digest=checkpointer.closure["digest"],
            config_digest=checkpointer.config["digest"],
            cadence_generations=checkpointer.cadence,
            barrier="none_completion_level_capture",
            resume_capture=bool(args.resume_capture),
            **causal_bound,
        )
    meta_audit_manager = None
    breadth_manager = None
    if args.breadth_control_dir is not None:
        import shinka_breadth

        breadth_manager = shinka_breadth.ShinkaOverlapBreadthManager(
            results_dir=output,
            control_dir=args.breadth_control_dir,
            flat_seconds=args.breadth_flat_seconds,
            flat_min_terminal_attempts=args.breadth_flat_min_terminal_attempts,
            start_ordinary=args.breadth_start_ordinary,
            minimum_run_age_seconds=args.minimum_run_age_seconds,
            stop_after_guided_cohort=args.breadth_stop_after_guided_cohort,
        )
        shinka_breadth.install_shinka_breadth(breadth_manager)
        trajectory.emit_event(
            "breadth_intervention_enabled", **breadth_manager.describe()
        )
    _record_manifest(output, run_id, args, config, takehome)
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    if checkpointer is not None:
        checkpointer.install(runner, periodic=False)
    if args.resume_capture:
        import shinka_resume_capture

        shinka_resume_capture.install_resume_capture(
            runner,
            results=output,
            checkpointer=checkpointer,
            recorder=lambda event, **fields: trajectory.emit_event(event, **fields),
        )
    try:
        runner.run()
    finally:
        trajectory.finalize_open_attempts("runner stopped before durable terminal")
        total_cost = getattr(runner, "total_api_cost", None)
        trajectory.emit_event(
            "run_finished",
            total_api_cost_usd=total_cost,
            total_cost_usd=total_cost,
            total_cost_components=["llm", "embedding", "novelty", "meta"],
            completed_generations=getattr(runner, "completed_generations", None),
        )
        if meta_audit_manager is not None:
            meta_audit_manager.shutdown()
            trajectory.emit_event(
                "meta_audit_finished", **meta_audit_manager.describe()
            )
        if breadth_manager is not None:
            breadth_manager.shutdown()
            trajectory.emit_event(
                "breadth_intervention_finished", **breadth_manager.describe()
            )
        import export_best

        export_best.main(results_dir=str(output))


if __name__ == "__main__":
    main()
