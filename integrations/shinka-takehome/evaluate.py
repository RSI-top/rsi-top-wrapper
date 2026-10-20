"""
ShinkaEvolve evaluator for the Anthropic performance takehome.

Invoked by the scheduler as:
  python evaluate.py --program_path <candidate.py> --results_dir <dir>

Two-phase, cheat-resistant scoring using the original benchmark rules:
  Phase A (eval_build.py, UNTRUSTED): runs the evolved program in a throwaway
    subprocess and extracts kb.instrs as plain JSON.
  Phase B (eval_score.py, TRUSTED): replays those instructions on the pristine
    frozen simulator (vendor/takehome/frozen_problem.py) against reference_kernel2 on
    freshly generated random inputs.

Writes metrics.json / correct.json in the schema ShinkaEvolve ingests.
combined_score is a log-scaled speedup over the 147734-cycle baseline,
saturating at 1000 cycles. Failure stderr goes into text_feedback so the
LLM can debug its next mutation.
"""

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAKEHOME_DIR = os.environ.get(
    "SHINKA_TAKEHOME_DIR", os.path.join(HERE, "vendor", "takehome")
)
BASELINE = 147734
TARGET_CYCLES = 1000.0
BUILD_TIMEOUT = int(os.environ.get("SHINKA_BUILD_TIMEOUT", "180"))
SCORE_TIMEOUT = int(os.environ.get("SHINKA_SCORE_TIMEOUT", "600"))
N_TRIALS = int(os.environ.get("SHINKA_SCORE_TRIALS", "2"))


def _tail(s: str, n: int = 4000) -> str:
    return (s or "")[-n:]


def _write(results_dir, metrics, correct, error):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=4)


def _fail(results_dir, stage, feedback):
    metrics = {
        "combined_score": 0.0,
        "public": {"cycles": float(BASELINE * 10), "speedup": 0.0},
        "private": {},
        "extra_data": {"failure_stage": stage},
        "text_feedback": f"[{stage}] {feedback}",
    }
    _write(results_dir, metrics, False, stage)


def _score_from_cycles(cycles: int) -> float:
    if cycles <= 0:
        return 0.0
    raw = math.log(BASELINE / cycles) / math.log(BASELINE / TARGET_CYCLES)
    # Correct-but-slower-than-baseline still beats broken (score floor 0.001)
    return max(0.001, min(1.0, raw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    program_path = os.path.abspath(args.program_path)
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    instrs_json = os.path.join(results_dir, "instrs.json")
    result_json = os.path.join(results_dir, "sim_result.json")

    # Phase A: untrusted build
    try:
        build = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "eval_build.py"),
                program_path,
                TAKEHOME_DIR,
                instrs_json,
            ],
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
            cwd=results_dir,
        )
    except subprocess.TimeoutExpired:
        return _fail(
            results_dir, "build_timeout", f"Kernel build exceeded {BUILD_TIMEOUT}s"
        )
    if build.returncode != 0 or not os.path.exists(instrs_json):
        return _fail(
            results_dir,
            "build_error",
            f"stderr:\n{_tail(build.stderr)}\nstdout:\n{_tail(build.stdout)}",
        )

    # Phase B: trusted scoring on the frozen simulator
    try:
        score = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "eval_score.py"),
                instrs_json,
                TAKEHOME_DIR,
                str(N_TRIALS),
                result_json,
            ],
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT,
            cwd=results_dir,
        )
    except subprocess.TimeoutExpired:
        return _fail(
            results_dir,
            "simulation_timeout",
            f"Simulation exceeded {SCORE_TIMEOUT}s "
            f"(kernel far too slow or non-terminating)",
        )
    if score.returncode != 0 or not os.path.exists(result_json):
        return _fail(
            results_dir,
            "simulator_error",
            f"stderr:\n{_tail(score.stderr)}\nstdout:\n{_tail(score.stdout)}",
        )

    with open(result_json) as f:
        result = json.load(f)

    if not result.get("ok"):
        return _fail(
            results_dir,
            "incorrect_output",
            f"{result.get('error', 'unknown')}\ntrials: {result.get('trials', [])}",
        )

    cycles = int(result["cycles"])
    speedup = BASELINE / cycles
    metrics = {
        "combined_score": _score_from_cycles(cycles),
        "public": {"cycles": float(cycles), "speedup": speedup},
        "private": {},
        "extra_data": {"trials": result.get("trials", [])},
        "text_feedback": (
            f"Correct on {N_TRIALS} random input sets. "
            f"CYCLES: {cycles} (baseline {BASELINE}, speedup {speedup:.2f}x). "
            f"Lower cycles -> higher score."
        ),
    }
    _write(results_dir, metrics, True, None)
    print(f"CYCLES: {cycles}  speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
