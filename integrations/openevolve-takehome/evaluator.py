"""
OpenEvolve evaluator for the Anthropic performance takehome.

Two-phase, cheat-resistant evaluation:
  Phase A (eval_build.py, untrusted): executes the evolved program in a
    throwaway subprocess and extracts the built instruction stream as JSON.
  Phase B (eval_score.py, trusted): runs those instructions on the pristine
    frozen simulator against reference_kernel2 on fresh random inputs.

Score: combined_score in [0, 1], log-scaled speedup over the 147734-cycle
baseline, saturating at 1000 cycles (better than the best published Claude
result). Failures score 0.0 and return stderr as artifacts so the LLM can
debug the next attempt.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

from openevolve.evaluation_result import EvaluationResult

HERE = os.path.dirname(os.path.abspath(__file__))
TAKEHOME_DIR = os.environ.get(
    "OE_TAKEHOME_DIR", os.path.join(HERE, "vendor", "takehome")
)
BASELINE = 147734
TARGET_CYCLES = 1000.0
BUILD_TIMEOUT = int(os.environ.get("OE_BUILD_TIMEOUT", "180"))
SCORE_TIMEOUT = int(os.environ.get("OE_SCORE_TIMEOUT", "600"))
N_TRIALS = int(os.environ.get("OE_SCORE_TRIALS", "2"))


def _tail(s: str, n: int = 4000) -> str:
    s = s or ""
    return s[-n:]


def _fail(reason: str, artifacts: dict) -> EvaluationResult:
    return EvaluationResult(
        metrics={"combined_score": 0.0, "cycles": float(BASELINE * 10), "speedup": 0.0},
        artifacts={"failure_stage": reason, **artifacts},
    )


def _score_from_cycles(cycles: int) -> float:
    if cycles <= 0:
        return 0.0
    raw = math.log(BASELINE / cycles) / math.log(BASELINE / TARGET_CYCLES)
    # Correct-but-slower-than-baseline still beats broken (score floor 0.001)
    return max(0.001, min(1.0, raw))


def evaluate(program_path: str):
    program_path = os.path.abspath(program_path)
    with tempfile.TemporaryDirectory(prefix="oe_takehome_") as tmp:
        instrs_json = os.path.join(tmp, "instrs.json")
        result_json = os.path.join(tmp, "result.json")

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
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return _fail(
                "build_timeout", {"stderr": f"Kernel build exceeded {BUILD_TIMEOUT}s"}
            )
        if build.returncode != 0 or not os.path.exists(instrs_json):
            return _fail(
                "build_error",
                {
                    "stderr": _tail(build.stderr),
                    "stdout": _tail(build.stdout),
                },
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
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return _fail(
                "simulation_timeout",
                {
                    "stderr": f"Simulation exceeded {SCORE_TIMEOUT}s "
                    f"(kernel likely far too slow or non-terminating)"
                },
            )
        if score.returncode != 0 or not os.path.exists(result_json):
            return _fail(
                "simulator_error",
                {
                    "stderr": _tail(score.stderr),
                    "stdout": _tail(score.stdout),
                },
            )

        with open(result_json) as f:
            result = json.load(f)

    if not result.get("ok"):
        return _fail(
            "incorrect_output",
            {
                "detail": result.get("error", "unknown"),
                "trials": json.dumps(result.get("trials", [])),
            },
        )

    cycles = int(result["cycles"])
    return EvaluationResult(
        metrics={
            "combined_score": _score_from_cycles(cycles),
            "cycles": float(cycles),
            "speedup": BASELINE / cycles,
        },
        artifacts={
            "cycles": str(cycles),
            "speedup_over_baseline": f"{BASELINE / cycles:.2f}x",
        },
    )
