"""Capture native Shinka state for causal history and checkpoints."""

from __future__ import annotations

import hashlib

import json

from typing import Any


def frontier_fingerprint(global_best_id: str | None) -> str:
    """Digest of the state an intervention actually depends on.

    STRICT GLOBAL-frontier semantics, matching the other baseline adapters:
    the digest is a function of the global best identity only. Deliberately
    NOT a function of the completion count or of island bests. A buffered
    non-improving completion — and an island-only best move — leaves it
    identical, so a verdict formed against it stays fresh: island bests remain
    fully visible in the database, the frozen bundle, and island metadata, but
    they are not shared-frontier moves and must never stale an audit or
    intervention. A change of the strict global best *does* change this, and
    correctly stales an in-flight verdict.
    """
    payload = json.dumps(
        {"global_best": global_best_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _program_row(program: Any) -> dict[str, Any]:
    """Deterministic, bounded, plain-value projection of one Program row."""
    metadata = getattr(program, "metadata", None) or {}
    return {
        "program_id": str(getattr(program, "id", "") or ""),
        "generation": int(getattr(program, "generation", 0) or 0),
        "island_idx": getattr(program, "island_idx", None),
        "parent_id": getattr(program, "parent_id", None),
        "archive_inspiration_ids": sorted(
            getattr(program, "archive_inspiration_ids", None) or []
        ),
        "top_k_inspiration_ids": sorted(
            getattr(program, "top_k_inspiration_ids", None) or []
        ),
        "correct": bool(getattr(program, "correct", False)),
        "combined_score": float(getattr(program, "combined_score", 0.0) or 0.0),
        "public_metrics": _plain(getattr(program, "public_metrics", None) or {}),
        "text_feedback": _truncate(
            getattr(program, "text_feedback", ""), MAX_FEEDBACK_CHARS
        ),
        "complexity": float(getattr(program, "complexity", 0.0) or 0.0),
        "in_archive": bool(getattr(program, "in_archive", False)),
        "patch_type": metadata.get("patch_type"),
        "code_diff_summary": _truncate(
            getattr(program, "code_diff", None), MAX_DIFF_SUMMARY_CHARS
        ),
        "code_diff_chars": len(getattr(program, "code_diff", None) or ""),
        "source_job_id": metadata.get("source_job_id"),
        "is_island_copy": bool(metadata.get("_is_island_copy")),
        "original_program_id": metadata.get("_original_program_id"),
        "timestamp": float(getattr(program, "timestamp", 0.0) or 0.0),
        "side_effects_applied": bool(metadata.get("postprocess_side_effects_applied")),
        "api_costs": metadata.get("api_costs"),
        "model_name": metadata.get("model_name"),
        "stderr_tail": _truncate(metadata.get("stderr_log"), 1000),
    }


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Total order that never depends on dict/SQLite iteration order."""
    return sorted(
        rows,
        key=lambda row: (
            row.get("generation", 0),
            row.get("timestamp", 0.0),
            str(row.get("program_id") or ""),
        ),
    )


def _plain(value: Any) -> Any:
    """Deep-copy into plain JSON-safe values; no live object ever escapes."""
    if isinstance(value, dict):
        return {str(k): _plain(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def frontier_from_db(
    db, *, exclude_program_id: str | None = None
) -> tuple[tuple[float, str] | None, dict[str, tuple[float, str]]]:
    """Compute global / per-island frontier identity from the DATABASE.

    Shinka already holds correct persisted rows before the first evaluated
    completion — the initial program and its island copies (bindings §3) —
    and those genuinely are the frontier at that point. An eligible-ledger
    baseline would start empty and mislabel a first completion *below* the
    initial program as an improvement.

    The database is also the only place island copies appear: they are
    written by ``copy_program_to_islands`` and never pass through the
    persistence hook, so an incremental cache fed only by the hook would
    drift from what a resume reconstructs. Deriving from the DB every time
    keeps the live token and the resumed token identical by construction.

    Ties never displace an incumbent (strict ``>``), and rows are visited in
    a deterministic order, so the result is arrival-order independent.
    """
    best: tuple[float, str] | None = None
    islands: dict[str, tuple[float, str]] = {}
    programs = list(db.get_all_programs()) if db is not None else []
    for program in sorted(
        programs,
        key=lambda p: (
            float(getattr(p, "timestamp", 0.0) or 0.0),
            int(getattr(p, "generation", 0) or 0),
            str(getattr(p, "id", "")),
        ),
    ):
        if not bool(getattr(program, "correct", False)):
            continue
        program_id = str(getattr(program, "id", ""))
        if exclude_program_id is not None:
            if program_id == exclude_program_id:
                continue
            # Island copies are the SAME logical program committed in the
            # same transaction (islands.py tags them `_is_island_copy` /
            # `_original_program_id`); a completion must not read as a tie
            # against its own copy when classified.
            metadata = getattr(program, "metadata", None) or {}
            if str(metadata.get("_original_program_id") or "") == exclude_program_id:
                continue
        score = float(getattr(program, "combined_score", 0.0) or 0.0)
        key = str(getattr(program, "island_idx", None))
        if best is None or score > best[0]:
            best = (score, program_id)
        current = islands.get(key)
        if current is None or score > current[0]:
            islands[key] = (score, program_id)
    return best, islands


def budget_snapshot(controller) -> dict[str, Any]:
    """Remaining budget from the run's real accounting source (§10)."""
    evo = getattr(controller, "evo_config", None)
    total = getattr(controller, "total_api_cost", None)
    cap = getattr(evo, "max_api_costs", None)
    committed = None
    getter = getattr(controller, "_get_committed_cost", None)
    if callable(getter):
        try:
            committed = float(getter())
        except Exception:  # noqa: BLE001 — budget is advisory, never fatal
            committed = None
    remaining = None
    if cap is not None and committed is not None:
        remaining = float(cap) - committed
    elif cap is not None and total is not None:
        remaining = float(cap) - float(total)
    return _plain(
        {
            "source": "ShinkaEvolveRunner.total_api_cost + _get_committed_cost()",
            "max_api_costs": cap,
            "total_api_cost": total,
            "committed_cost": committed,
            "avg_proposal_cost": getattr(controller, "avg_proposal_cost", None),
            "remaining_usd": remaining,
        }
    )


def inflight_snapshot(controller) -> dict[str, Any]:
    """Queued proposals, active proposal tasks, running evals (§7)."""
    active_tasks = dict(getattr(controller, "active_proposal_tasks", {}) or {})
    running_jobs = list(getattr(controller, "running_jobs", []) or [])
    assigned = set(getattr(controller, "assigned_generations", set()) or set())
    running_gens = {int(getattr(j, "generation", -1)) for j in running_jobs}
    active_gens = set()
    for task in active_tasks.values():
        name = getattr(task, "get_name", lambda: "")()
        if isinstance(name, str) and name.startswith("proposal_"):
            try:
                active_gens.add(int(name.split("_", 1)[1]))
            except ValueError:
                pass
    launches = []
    for job in sorted(running_jobs, key=lambda j: str(getattr(j, "job_id", ""))):
        launches.append(
            {
                "launch_id": str(getattr(job, "job_id", "")),
                "submitted_seq": 0,
                "lineage_id": getattr(job, "parent_id", None),
                "kind": "running_evaluation",
                "generation": getattr(job, "generation", None),
                "parent_id": getattr(job, "parent_id", None),
                "archive_insp_ids": sorted(getattr(job, "archive_insp_ids", []) or []),
                "top_k_insp_ids": sorted(getattr(job, "top_k_insp_ids", []) or []),
            }
        )
    for task_id in sorted(active_tasks):
        launches.append(
            {
                "launch_id": f"proposal:{task_id}",
                "submitted_seq": 0,
                "lineage_id": None,
                "kind": "active_proposal_task",
            }
        )
    return _plain(
        {
            "launches": launches,
            "active_proposal_tasks": len(active_tasks),
            "running_evaluations": len(running_jobs),
            "queued_proposal_generations": sorted(
                assigned - running_gens - active_gens
            ),
            "next_generation_to_submit": getattr(
                controller, "next_generation_to_submit", None
            ),
            "completed_job_persistence_backlog": _safe_int(
                controller, "_get_completed_job_work_count"
            ),
            "background_side_effect_backlog": _safe_int(
                controller, "_get_background_side_effect_work_count"
            ),
            "failed_db_jobs_pending": len(
                getattr(controller, "failed_jobs_for_retry", []) or []
            ),
        }
    )


def _safe_int(controller: Any, method_name: str) -> int | None:
    method = getattr(controller, method_name, None)
    if not callable(method):
        return None
    try:
        return int(method())
    except Exception:  # noqa: BLE001 — optional introspection
        return None


MAX_DIFF_SUMMARY_CHARS = 2000

MAX_FEEDBACK_CHARS = 4000


def _truncate(text: Any, limit: int) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
