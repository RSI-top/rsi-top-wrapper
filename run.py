#!/usr/bin/env python3
"""Run the Anthropic performance takehome with a baseline or RSI-top."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
INTEGRATIONS = {
    "openevolve": ROOT / "integrations/openevolve-takehome",
    "shinka": ROOT / "integrations/shinka-takehome",
}
MODELS = {"deepseek": "deepseek-v4-flash", "gemini": "gemini-3.1-pro-preview"}
KEYS = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for method in ("baseline", "rsi-top"):
        q = sub.add_parser(method)
        q.add_argument("--framework", choices=INTEGRATIONS, default="openevolve")
        q.add_argument("--provider", choices=MODELS, default="deepseek")
        q.add_argument("--model", help="worker model ID; defaults to the local experiment configuration")
        q.add_argument("--output", type=Path, required=True, help="new output directory")
        q.add_argument("--hours", type=positive, default=12.0)
        q.add_argument("--budget", type=positive, default=400.0,
                       help="worker API budget estimate in USD; planner/dedup spend is additional")
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--dry-run", action="store_true", help="print commands without writing or calling APIs")
        if method == "baseline":
            q.add_argument("--iterations", type=int, help="optional finite iteration/generation cap")
        else:
            q.add_argument("--source", type=Path, help="optional stopped OpenEvolve baseline from this release")
            q.add_argument("--cutoff", type=int, help="explicit completion sequence in --source")
    return p


def environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "shared/meta-agent-audit")
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AUDIT_PI_CLI"] = str(ROOT / "node_modules/.bin/pi")
    return env


def plan(args: argparse.Namespace) -> dict:
    folder = INTEGRATIONS[args.framework]
    output = args.output.expanduser().resolve()
    model = args.model or MODELS[args.provider]
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.command == "baseline" and args.iterations is not None and args.iterations < 1:
        raise ValueError("--iterations must be positive")
    common = ["--output", str(output / "search"), "--budget", str(args.budget), "--seed", str(args.seed)]
    required = {KEYS[args.provider]}
    # The recorded Shinka profile stores OpenRouter embeddings for each candidate.
    if args.framework == "shinka":
        required.add("OPENROUTER_API_KEY")
    if args.command == "baseline":
        if args.framework == "openevolve":
            command = [sys.executable, str(folder / "runner.py"), *common,
                       "--llm-provider", args.provider, f"--{args.provider}-model", model,
                       f"--{args.provider}-reasoning-effort", "high",
                       "--wall-time-seconds", str(args.hours * 3600)]
            if args.iterations is not None:
                command += ["--iterations", str(args.iterations)]
        else:
            command = [sys.executable, str(folder / "run_evo.py"), *common,
                       "--llm-model", model, "--effort", "high", "--resume-capture"]
            if args.iterations is not None:
                command += ["--generations", str(args.iterations)]
    else:
        required.add("DEEPSEEK_API_KEY")
        if args.source and args.framework != "openevolve":
            raise ValueError("--source is supported by the OpenEvolve launcher; Shinka starts from scratch")
        if args.cutoff is not None and (not args.source or args.cutoff < 1):
            raise ValueError("a positive --cutoff requires --source")
        command = [sys.executable, str(folder / "run_repeat_flat_breadth.py"), *common,
                   "--worker-model", model, "--worker-reasoning", "high",
                   "--wall-time-seconds", str(args.hours * 3600),
                   "--minimum-run-age-seconds", "3600"]
        if args.framework == "openevolve":
            command += ["--worker-provider", args.provider]
            if args.source:
                source = args.source.expanduser().resolve()
                public_manifest = source / "launch.json"
                if public_manifest.is_file():
                    original = json.loads(public_manifest.read_text())
                    for key, expected in {"method": "baseline", "framework": args.framework,
                                          "provider": args.provider, "model": model}.items():
                        if original.get(key) != expected:
                            raise ValueError(f"source {key} differs from selected configuration")
                    source /= "search"
                command += ["--source-trace-root", str(source)]
                if args.cutoff is not None:
                    command += ["--initial-completion-seq", str(args.cutoff)]
        else:
            command += ["--config", str(folder / "shinka.yaml"), "--initial-ordinary"]
    return {
        "method": args.command, "framework": args.framework, "provider": args.provider,
        "model": model, "seed": args.seed, "hours": args.hours,
        "worker_budget_usd": args.budget, "output": str(output),
        "required_environment": sorted(required), "command": command,
        "planner": ({"provider": "deepseek", "model": "deepseek-v4-flash",
                     "calls": 4, "directions_per_call": 8, "metrics_per_call": 15,
                     "routing": "shared", "selector": "llm"} if args.command == "rsi-top" else None),
    }


def execute(spec: dict, env: dict[str, str]) -> int:
    output = Path(spec["output"])
    if output.exists():
        raise ValueError(f"output already exists; choose a new directory: {output}")
    missing = [name for name in spec["required_environment"] if not env.get(name)]
    if missing:
        raise ValueError("set " + ", ".join(missing))
    if spec["method"] == "rsi-top":
        if not Path(env["AUDIT_PI_CLI"]).is_file():
            raise ValueError("Pi runtime missing; run ./setup.sh")
        check_node()
    output.mkdir(parents=True)
    (output / "launch.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("Run:", output, flush=True)
    print("Log:", output / "launcher.log", flush=True)
    started = time.monotonic()
    timed_out = False
    with (output / "launcher.log").open("w") as log:
        process = subprocess.Popen(spec["command"], cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            # RSI-top's own supervisor finishes a triggered cohort after its deadline.
            timeout = spec["hours"] * 3600 if spec["method"] == "baseline" and spec["framework"] == "shinka" else None
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGINT)
                code = process.wait(timeout=180)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (output / "result.json").write_text(json.dumps({"status": "interrupted", "returncode": process.returncode}, indent=2) + "\n")
            raise
    success = code == 0 or (timed_out and code in (-signal.SIGINT, 130))
    (output / "result.json").write_text(json.dumps({
        "status": "completed" if success else "failed", "returncode": code,
        "wall_limit_requested": timed_out, "elapsed_seconds": time.monotonic() - started,
    }, indent=2) + "\n")
    if not success:
        print(f"Run failed ({code}); see {output / 'launcher.log'}", file=sys.stderr)
    return 0 if success else 1


def check_node() -> None:
    executable = shutil.which("node")
    if not executable:
        raise ValueError("Node.js >=22.19.0 is required for the planner")
    version = subprocess.check_output([executable, "-p", "process.versions.node"], text=True).strip()
    if tuple(int(v) for v in version.split(".")[:2]) < (22, 19):
        raise ValueError(f"Node.js >=22.19.0 is required; PATH selects {version}")


def main() -> int:
    args = parser().parse_args()
    try:
        spec = plan(args)
        if args.dry_run:
            print(json.dumps(spec, indent=2))
            print("\n" + shlex.join(spec["command"]))
            return 0
        return execute(spec, environment())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
