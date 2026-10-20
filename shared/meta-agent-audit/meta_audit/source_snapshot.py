"""Capture the release's runtime sources, including shared and vendored code."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


TEXT_SUFFIXES = {".py", ".ts", ".json", ".yaml", ".yml", ".toml", ".lock", ".md", ".txt", ".csv", ".sh"}
IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", "build", "dist", "tests", ".pytest_cache"}
REQUIRED_ROOT_FILES = ("run.py", "evaluate.py", "verify.py", "setup.sh", "pyproject.toml", "uv.lock", "package.json", "package-lock.json")
RUNTIME_TREES = (
    "prompts",
    "shared/meta-agent-audit/meta_audit",
    "integrations/openevolve-takehome/vendor",
    "integrations/shinka-takehome/vendor",
)


def _runtime_paths(root: Path) -> set[Path]:
    paths = {root / name for name in REQUIRED_ROOT_FILES}
    # Only direct integration files: run outputs can be arbitrarily large and
    # contain credentials or generated programs, and are not release sources.
    for relative in (".", "shared/meta-agent-audit", "integrations/openevolve-takehome", "integrations/shinka-takehome"):
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.update(p for p in directory.iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES)
    for relative in RUNTIME_TREES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for current, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not d.endswith(".egg-info"))
            for name in files:
                path = Path(current) / name
                if path.suffix in TEXT_SUFFIXES:
                    paths.add(path)
    return paths


def capture_sources(
    root: Path, *, extra_files: Iterable[Path] = (), environ: Mapping[str, str] | None = None
) -> dict[str, dict]:
    """Map repository-relative names to original hashes and redacted text.

    Explicit inputs (custom configs / simulators) are required, even outside
    the release. Missing files raise instead of silently weakening provenance.
    """
    root = Path(root).resolve()
    env = os.environ if environ is None else environ
    credentials = sorted(
        ((name, value) for name, value in env.items()
         if value and (name.endswith("_API_KEY") or name.endswith("_AUTH_TOKEN"))),
        key=lambda item: (-len(item[1]), item[0]),
    )
    paths = _runtime_paths(root) | {Path(p).resolve() for p in extra_files}
    sources = {}
    for path in sorted(paths):
        resolved = path.resolve()
        data = path.read_bytes()
        content = data.decode("utf-8")
        redactions = 0
        for name, value in credentials:
            if value in content:
                redactions += content.count(value)
                content = content.replace(value, f"<REDACTED:{name}>")
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        sources[name] = {
            "path": str(resolved),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "content": content,
            "captured_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "credential_redactions": redactions,
        }
    return sources


def write_source_snapshot(output: Path, sources: dict[str, dict]) -> dict:
    """Store a content-addressed snapshot; earlier resume instances keep theirs."""
    data = (json.dumps({"schema": "rsi_source_snapshot_v1", "files": sources},
                       indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    relative = Path("source_snapshots") / f"{digest}.json"
    target = Path(output) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"source snapshot digest mismatch: {target}")
    else:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    return {"path": relative.as_posix(), "sha256": digest, "file_count": len(sources)}
