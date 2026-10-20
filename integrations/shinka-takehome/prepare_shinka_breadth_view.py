#!/usr/bin/env python3
"""Convert one frozen Shinka cutoff into the shared breadth-planner view."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OE = HERE.parent / "openevolve-takehome"
if str(OE) not in sys.path:
    sys.path.insert(0, str(OE))

from planner_context import (  # noqa: E402
    ENGINE_PROFILES,
    LLM_DASHBOARD,
    _materialize_engine_profiles,
    build_stage_context,
)
from planner_runtime import (  # noqa: E402
    ENGINE_PROFILES_PATH,
    materialize_view,
    sanitize_planner_view,
)


def _write_files(root: Path, files: dict[str, Any]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")


def _compact_planner_history(view: Path) -> None:
    """Keep attempt evidence, not embedded copies of old LLM sessions.

    Run only on the planner view, after metrics/profiles have been computed
    from the full records. Source logs and separately readable code stay intact.
    """
    child_fields = {
        "id",
        "parent_id",
        "generation",
        "timestamp",
        "island_idx",
        "archive_inspiration_ids",
        "top_k_inspiration_ids",
        "code_location",
        "code_sha256",
        "correct",
        "metrics",
        "text_feedback",
    }
    metadata_fields = {
        "changes",
        "patch_name",
        "patch_description",
        "patch_type",
        "diff_summary",
        "stderr_log",
        "stdout_log",
        "results_missing",
    }
    for path in view.glob("*/seq_*/trajectory.jsonl"):
        events = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for event in events:
            for key in (
                "metadata",
                "parent_code",
                "pid",
                "event_id",
                "run_instance_id",
                "attempt_key",
                "source_job_id",
                "task_id",
            ):
                event.pop(key, None)
            child = event.get("child")
            if isinstance(child, dict):
                compact = {
                    key: value for key, value in child.items() if key in child_fields
                }
                compact["metadata"] = {
                    key: value
                    for key, value in (child.get("metadata") or {}).items()
                    if key in metadata_fields
                }
                event["child"] = compact
        path.write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                for event in events
            ),
            encoding="utf-8",
        )


def prepare(*, request: Path, output: Path, prompt: Path) -> Path:
    document = json.loads(request.read_text(encoding="utf-8"))
    if document.get("schema") != "shinka_overlap_planner_request_v1":
        raise RuntimeError("invalid Shinka breadth request")
    attempts = Path(document["normalized_attempts"]).resolve()
    checkpoint = Path(document["checkpoint"]).resolve()
    cutoff_seq = int(document["cutoff_seq"])
    cutoff_at = float(document["cutoff_at"])
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    live_ids = {
        str(program_id)
        for members in metadata.get("islands") or []
        for program_id in members
    }
    programs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((checkpoint / "programs").glob("*.json"))
    ]
    root = next(
        (
            program
            for program in programs
            if int(program.get("generation", 0) or 0) == 0
            and not program.get("parent_id")
        ),
        None,
    )
    if root is None or not isinstance(root.get("code"), str):
        raise RuntimeError("frozen Shinka checkpoint has no root program")

    context = build_stage_context(
        attempts_path=attempts,
        cutoff_seq=cutoff_seq,
        cutoff_at=cutoff_at,
        root_code=root["code"],
        live_program_ids=live_ids,
    )
    context_root = output / "context"
    _write_files(context_root, context["files"])
    prepared_root = output / "prepared"
    view, _, _, _ = materialize_view(
        context_root=context_root,
        dashboard_root=None,
        checkpoint=checkpoint,
        output=prepared_root,
        prompt_path=prompt,
        num_directions=4,
        isolate_raw_history=False,
    )

    _write_files(view, context["files"])
    profile_scratch = output / "profile_scratch"
    profile_scratch.mkdir()
    _materialize_engine_profiles(view, profile_scratch)
    profile_document = json.loads((view / ENGINE_PROFILES).read_text(encoding="utf-8"))
    enriched = build_stage_context(
        attempts_path=attempts,
        cutoff_seq=cutoff_seq,
        cutoff_at=cutoff_at,
        root_code=root["code"],
        live_program_ids=live_ids,
        engine_profiles=profile_document.get("profiles") or {},
        engine_failures={
            str(program_id): str(
                record.get("error") if isinstance(record, dict) else record
            )
            for program_id, record in (profile_document.get("failures") or {}).items()
        },
    )
    _write_files(context_root, enriched["files"])
    _write_files(view, enriched["files"])
    _compact_planner_history(view)
    sanitize_planner_view(view, isolate_raw_history=True)
    shutil.rmtree(profile_scratch)
    if not (view / LLM_DASHBOARD).is_file():
        raise RuntimeError("prepared Shinka view has no LLM dashboard")
    if (view / ENGINE_PROFILES_PATH).exists():
        raise RuntimeError("engine profiles leaked into the visible planner view")
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    view = prepare(request=args.request, output=args.output, prompt=args.prompt)
    print(view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
