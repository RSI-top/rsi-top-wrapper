#!/usr/bin/env python3
"""Verify a candidate using a pristine, pinned official submission harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


OFFICIAL_REPOSITORY = "https://github.com/anthropics/original_performance_takehome.git"
OFFICIAL_COMMIT = "5452f74bd977807ac2e74f3d29432b9df6f25197"


def _git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True,
        timeout=180, check=True,
    )
    return result.stdout.strip()


def _harness_hashes(directory: Path) -> dict[str, str]:
    paths = [directory / "problem.py", *(directory / "tests").rglob("*")]
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths) if p.is_file() and "__pycache__" not in p.parts
    }


def verify(
    program: Path, *, local_repo: Path | None = None,
    correctness_only: bool = False, timeout: float = 600,
) -> dict:
    """Fetch only the pinned commit; local_repo is an offline Git object cache.

    Local working-tree changes are never copied. The candidate runs as ordinary
    Python, just as in the official harness; this is not an execution sandbox.
    """
    candidate = Path(program).resolve().read_bytes()
    with tempfile.TemporaryDirectory(prefix="rsi_official_") as temporary:
        checkout = Path(temporary)
        _git(checkout, "init", "--quiet")
        source = str(Path(local_repo).resolve()) if local_repo is not None else OFFICIAL_REPOSITORY
        _git(checkout, "fetch", "--quiet", "--depth=1", "--", source, OFFICIAL_COMMIT)
        _git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        commit = _git(checkout, "rev-parse", "HEAD")
        if commit != OFFICIAL_COMMIT:
            raise ValueError("official harness commit mismatch")
        for name in ("tests/submission_tests.py", "tests/frozen_problem.py", "problem.py"):
            if not (checkout / name).is_file():
                raise ValueError(f"official harness is missing {name}")
        before = _harness_hashes(checkout)
        (checkout / "perf_takehome.py").write_bytes(candidate)
        command = [sys.executable, "-E", "-s", "-B", "tests/submission_tests.py"]
        if correctness_only:
            command.append("CorrectnessTests")
        # Ignore PYTHONPATH/PYTHONOPTIMIZE, and do not pass model credentials to
        # an offline candidate verification process.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PYTHON") and not k.endswith(("_API_KEY", "_AUTH_TOKEN"))}
        timed_out = False
        try:
            result = subprocess.run(command, cwd=checkout, env=env, capture_output=True,
                                    text=True, timeout=timeout, check=False)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            code = None
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        after = _harness_hashes(checkout)
        pristine = before == after and not _git(
            checkout, "status", "--porcelain", "--untracked-files=all", "--", "tests", "problem.py"
        )
        return {
            "schema": "rsi_official_verification_v1",
            "repository": OFFICIAL_REPOSITORY,
            "commit": commit,
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
            "mode": "correctness_only" if correctness_only else "full_submission",
            "harness_sha256": before,
            "harness_unchanged": bool(pristine),
            "returncode": code,
            "timed_out": timed_out,
            "passed": code == 0 and bool(pristine),
            "stdout": stdout,
            "stderr": stderr,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--local-repo", type=Path, help="offline Git clone containing the pinned official commit")
    parser.add_argument("--correctness-only", action="store_true", help="run the official 8 random-input checks without performance thresholds")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--report", type=Path, help="also save the verification JSON and test output")
    args = parser.parse_args()
    if not args.program.is_file():
        parser.error(f"program not found: {args.program}")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    try:
        report = verify(args.program, local_repo=args.local_repo,
                        correctness_only=args.correctness_only, timeout=args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 2
    document = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(document, encoding="utf-8")
    print(document, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
