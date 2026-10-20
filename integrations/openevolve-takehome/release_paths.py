"""Locate release resources from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = {
    "planner.md",
    "deduplicate.md",
    "metrics.json",
    "verbalized_sampling.md",
    "worker_openevolve.md",
    "worker_shinka.md",
}


def release_resource(name: str) -> Path:
    if name in PROMPTS:
        return ROOT / "prompts" / name
    return Path(__file__).resolve().parent / name
