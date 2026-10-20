"""Verify source file digests before reconstructing a checkpoint branch."""

from __future__ import annotations

import hashlib

import sys

from pathlib import Path

from typing import Any

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class ReplayProofError(RuntimeError):
    """Required code or state could not be proven from the frozen trace."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_file(
    *,
    recorded_path: Any,
    recorded_sha: Any,
    fallbacks: list[Path],
    what: str,
) -> tuple[Path, str]:
    """Locate a file ONLY by SHA-256 match against the run_started record."""
    if not (isinstance(recorded_sha, str) and recorded_sha):
        raise ReplayProofError(f"run_started records no SHA-256 for the {what}")
    candidates: list[Path] = []
    if isinstance(recorded_path, str) and recorded_path:
        candidates.append(Path(recorded_path))
    candidates.extend(fallbacks)
    tried: list[str] = []
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            tried.append(f"{candidate} (unreadable)")
            continue
        if _sha256_text(text) == recorded_sha:
            return candidate, recorded_sha
        tried.append(f"{candidate} (sha mismatch)")
    raise ReplayProofError(
        f"cannot prove the {what}: no candidate matches the recorded "
        f"SHA-256 {recorded_sha[:12]}…; tried: {'; '.join(tried)}"
    )
