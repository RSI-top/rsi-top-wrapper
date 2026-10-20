"""Loss-minimizing, append-only tracing around an unmodified OpenEvolve.

OpenEvolve's native ``evolution_trace.jsonl`` describes accepted parent/child
transitions.  This module supplies the other half of a post-analysis record:
every scheduled attempt, built prompt, physical Anthropic call, worker result,
and log event, including attempts that fail before a child program exists.

The integrations are runtime wrappers.  No file under vendor/openevolve or
vendor/takehome needs to be patched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
_WRITE_LOCK = threading.Lock()
_CONTEXT_LOCK = threading.Lock()
_CURRENT_ATTEMPT: dict[str, Any] = {}


def _trace_path() -> str | None:
    return os.environ.get("OE_ATTEMPT_TRACE")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex()}
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return repr(value)


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_sha256(system_message: str | None, messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {"system": system_message or "", "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return sha256_text(canonical) or ""


# ---------------------------------------------------------------------------
# Exact database-state capture (Phase 3A — observation only)
# ---------------------------------------------------------------------------

#: Versioned schema of the ``db_state`` trajectory records: a full canonical
#: STRUCTURAL post-state snapshot (no program code — code is already durably
#: recorded per commit and in checkpoints) captured at the controller safe
#: points after each committed ``database.add``, after the committed child's
#: ``increment_island_generation`` (the true end-of-completion state,
#: correlated with the same completion — the post-add record alone is one
#: completion behind on ``island_generations`` and cannot prove
#: ``should_migrate`` timing), and after each island migration. A full
#: snapshot is deliberately preferred over a hand-written delta taxonomy: it
#: is self-healing (each record re-establishes complete truth) and directly
#: proves programs, global/island best ids, island membership, archive, MAP
#: cell owners, island generations, last migration generation, and the next
#: iteration for a future branch materializer. Traces WITHOUT these records
#: stay readable but cannot be materialized across unrepresented mutations
#: (they keep failing closed).
DB_STATE_SCHEMA = "oe_db_state_v1"


def db_state_capture_enabled() -> bool:
    """Capture is active iff tracing is active (OE_ATTEMPT_TRACE set).

    Without a trace the capture path is fully inert: no O(population)
    snapshot or payload work may happen just to be discarded by emit_event.
    """
    return bool(_trace_path())


def database_state_snapshot(database: Any) -> dict[str, Any]:
    """Full canonical structural state of a live ProgramDatabase (read-only).

    ``feature_stats`` uses the database's OWN checkpoint serializer, so this
    record and ``checkpoints/*/metadata.json`` stay format-compatible; it is
    part of the snapshot because it drives MAP feature calculation for later
    adds. Program attribute payloads are NOT duplicated here: committed
    children live in their ``canonical_completion`` rows and migrant copies in
    the migration record's ``new_programs``.
    """
    return {
        "schema": DB_STATE_SCHEMA,
        "program_ids": sorted(database.programs.keys()),
        "best_program_id": database.best_program_id,
        "island_best_programs": list(database.island_best_programs),
        "islands": [sorted(island) for island in database.islands],
        "archive": sorted(database.archive),
        "island_feature_maps": [
            {str(key): value for key, value in sorted(feature_map.items())}
            for feature_map in database.island_feature_maps
        ],
        "island_generations": list(database.island_generations),
        "last_migration_generation": database.last_migration_generation,
        "last_iteration": int(getattr(database, "last_iteration", 0) or 0),
        "current_island": database.current_island,
        "feature_stats": database._serialize_feature_stats(),
    }


def database_state_sha256(snapshot: dict[str, Any]) -> str:
    """Deterministic fingerprint of one canonical snapshot document."""
    canonical = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def program_payload_sans_code(program: Any) -> dict[str, Any]:
    """Complete Program attribute payload with code replaced by its SHA-256.

    Used for programs created OUTSIDE the commit path (migration copies whose
    uuid4 ids are otherwise unrecoverable); their code bytes equal an
    already-recorded program's code, so the digest suffices to resolve it.
    """
    data = dict(program.to_dict())
    code = data.pop("code", None)
    data["code_sha256"] = sha256_text(code) if isinstance(code, str) else None
    return data


def emit_db_state(capture: str, database: Any, **payload: Any) -> str | None:
    """Append one exact post-mutation ``db_state`` record; never raises.

    Observation only: a capture failure logs and returns None — evolution is
    never altered, and the resulting record GAP is exactly what a downstream
    materializer must fail closed on.
    """
    if not db_state_capture_enabled():
        return None
    try:
        snapshot = database_state_snapshot(database)
        return emit_event(
            "db_state",
            db_state_schema=DB_STATE_SCHEMA,
            capture=capture,
            state=snapshot,
            state_sha256=database_state_sha256(snapshot),
            **payload,
        )
    except Exception:
        logging.getLogger("trajectory").exception(
            "db-state capture failed (evolution continues)"
        )
        return None


def emit_event(event: str, **payload: Any) -> str | None:
    """Append one complete JSON object under both thread and process locks."""
    path = _trace_path()
    if not path:
        return None

    event_id = str(uuid.uuid4())
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "event": event,
        "timestamp": time.time(),
        "run_instance_id": os.environ.get("OE_RUN_INSTANCE_ID"),
        "pid": os.getpid(),
        **payload,
    }
    line = json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return event_id


def _last_canonical_completion_seq(path: str | None) -> int:
    """Return the last durable completion sequence, tolerating old traces."""
    if not path or not os.path.isfile(path):
        return -1
    last = -1
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if '"event": "canonical_completion"' not in line:
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                value = record.get("completion_seq")
                if (
                    record.get("event") == "canonical_completion"
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    last = max(last, value)
    except OSError:
        return -1
    return last


class CanonicalCompletionRecorder:
    """Assign one durable, monotonic controller-consumption order.

    OpenEvolve workers finish concurrently, while the controller consumes one
    future at a time.  This sequence records that controller order, including
    failures and shutdown abandonment.  It resumes from an existing trace so
    checkpoint continuations retain one global completion coordinate.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = _last_canonical_completion_seq(_trace_path()) + 1

    @property
    def last_assigned_seq(self) -> int:
        """Highest completion_seq assigned so far (-1 before the first)."""
        with self._lock:
            return self._next - 1

    def record(
        self,
        *,
        iteration: int,
        status: str,
        terminal: str,
        scheduled_parent_id: str | None,
        scheduled_target_island: int | None,
        scheduled_at: float | None = None,
        **payload: Any,
    ) -> int:
        with self._lock:
            completion_seq = self._next
            emit_event(
                "canonical_completion",
                completion_seq=completion_seq,
                commit_index=completion_seq,
                attempt_id=(f"{os.environ.get('OE_RUN_INSTANCE_ID')}:{iteration}"),
                iteration=iteration,
                status=status,
                terminal=terminal,
                scheduled_parent_id=scheduled_parent_id,
                scheduled_target_island=scheduled_target_island,
                seconds_since_scheduled=(
                    time.time() - scheduled_at if scheduled_at is not None else None
                ),
                **payload,
            )
            self._next += 1
            return completion_seq


def set_current_attempt(context: dict[str, Any]) -> None:
    # Each ProcessPool worker runs one evolution task at a time.  The LLM call
    # itself uses a helper thread, so a process-global copy is intentional.
    with _CONTEXT_LOCK:
        _CURRENT_ATTEMPT.clear()
        _CURRENT_ATTEMPT.update(context)


def current_attempt() -> dict[str, Any]:
    with _CONTEXT_LOCK:
        return dict(_CURRENT_ATTEMPT)


def _program_ref(program: Any) -> dict[str, Any]:
    if not isinstance(program, dict):
        return {"repr": repr(program)}
    code = program.get("code")
    return {
        "id": program.get("id"),
        "code_sha256": sha256_text(code) if isinstance(code, str) else None,
        "metrics": program.get("metrics"),
        "generation": program.get("generation"),
        "iteration_found": program.get("iteration_found"),
        "metadata": program.get("metadata"),
    }


def install_prompt_trace() -> None:
    """Wrap PromptSampler without changing the rendered model prompt."""
    from openevolve.prompt.sampler import PromptSampler

    original = PromptSampler.build_prompt
    if getattr(original, "_oe_complete_trace", False):
        return

    def traced_build_prompt(self, *args, **kwargs):
        prompt = original(self, *args, **kwargs)
        evolution_round = kwargs.get("evolution_round", 0)
        current_program = kwargs.get("current_program", "")
        parent_program = kwargs.get("parent_program", "")
        messages = [{"role": "user", "content": prompt.get("user", "")}]
        context = {
            "iteration": evolution_round,
            "prompt_sha256": prompt_sha256(prompt.get("system"), messages),
            "current_program_sha256": sha256_text(current_program),
            "parent_program_sha256": sha256_text(parent_program),
            "program_metrics": kwargs.get("program_metrics"),
            "language": kwargs.get("language"),
            "diff_based_evolution": kwargs.get("diff_based_evolution"),
            "feature_dimensions": kwargs.get("feature_dimensions"),
            "previous_programs": [
                _program_ref(p) for p in kwargs.get("previous_programs", [])
            ],
            "top_programs": [_program_ref(p) for p in kwargs.get("top_programs", [])],
            "inspirations": [_program_ref(p) for p in kwargs.get("inspirations", [])],
        }
        set_current_attempt(context)
        emit_event("prompt_built", **context)
        return prompt

    traced_build_prompt._oe_complete_trace = True
    PromptSampler.build_prompt = traced_build_prompt


def install_evaluator_trace() -> None:
    """Record every evaluation, including the initial baseline evaluation."""
    from openevolve.evaluator import Evaluator

    original = Evaluator.evaluate_program
    if getattr(original, "_oe_complete_trace", False):
        return

    async def traced_evaluate_program(self, program_code: str, program_id: str = ""):
        started = time.time()
        code_hash = sha256_text(program_code)
        emit_event(
            "evaluation_started",
            program_id=program_id,
            program_code_sha256=code_hash,
            program_code=program_code,
            evaluation_file=getattr(self, "evaluation_file", None),
            evaluator_timeout=getattr(getattr(self, "config", None), "timeout", None),
        )
        try:
            metrics = await original(self, program_code, program_id)
        except BaseException as exc:
            emit_event(
                "evaluation_failed",
                program_id=program_id,
                program_code_sha256=code_hash,
                duration_seconds=time.time() - started,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            raise

        # Do not call get_pending_artifacts(): the upstream controller/worker
        # must still consume and store them.  A snapshot is safe here.
        artifacts = None
        pending = getattr(self, "_pending_artifacts", None)
        if isinstance(pending, dict) and program_id in pending:
            artifacts = pending[program_id]
        emit_event(
            "evaluation_completed",
            program_id=program_id,
            program_code_sha256=code_hash,
            duration_seconds=time.time() - started,
            metrics=metrics,
            artifacts=artifacts,
        )
        return metrics

    traced_evaluate_program._oe_complete_trace = True
    Evaluator.evaluate_program = traced_evaluate_program


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Durably replace a JSON manifest without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, ensure_ascii=False, indent=2, default=_json_default
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class JsonlLogHandler(logging.Handler):
    """Mirror all Python log records into the structured attempt stream."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            exc = None
            if record.exc_info:
                exc = "".join(traceback.format_exception(*record.exc_info))
            emit_event(
                "log",
                logger=record.name,
                level=record.levelname,
                message=record.getMessage(),
                created=record.created,
                process=record.process,
                process_name=record.processName,
                thread=record.thread,
                thread_name=record.threadName,
                exception=exc,
            )
        except Exception:
            # Tracing must never alter or abort evolution behavior.
            pass


def install_log_trace() -> None:
    path = _trace_path()
    if not path:
        return
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, JsonlLogHandler):
            return
    root.addHandler(JsonlLogHandler())


def _result_summary(result: Any, *, include_child_code: bool = False) -> dict[str, Any]:
    child = getattr(result, "child_program_dict", None)
    prompt = getattr(result, "prompt", None)
    response = getattr(result, "llm_response", None)
    summary: dict[str, Any] = {
        "reported_iteration": getattr(result, "iteration", None),
        "error": getattr(result, "error", None),
        "parent_id": getattr(result, "parent_id", None),
        "target_island": getattr(result, "target_island", None),
        "iteration_time": getattr(result, "iteration_time", None),
        "prompt_sha256": (
            prompt_sha256(
                prompt.get("system"),
                [{"role": "user", "content": prompt.get("user", "")}],
            )
            if isinstance(prompt, dict)
            else None
        ),
        "llm_response_sha256": sha256_text(response)
        if isinstance(response, str)
        else None,
        "artifacts": getattr(result, "artifacts", None),
    }
    if isinstance(child, dict):
        # Preserve the COMPLETE Program payload from the worker's
        # child_program_dict (timestamp, changes_description, language,
        # complexity/diversity, prompt/artifact refs, embedding, ...) so a
        # branch materializer can reconstruct the exact Program object.
        # Additive trace data only: the source dict is never mutated, and the
        # code-inclusion behavior is unchanged (code only on canonical rows,
        # code_sha256 always).
        summary["child"] = dict(child)
        summary["child"]["code_sha256"] = sha256_text(child.get("code"))
        if not include_child_code:
            summary["child"].pop("code", None)
    return summary


# The native high-level action objective the canonical block REPLACES: the
# "# Task" paragraph of the vendor default templates
# (vendor/openevolve/openevolve/prompts/defaults/{diff,full_rewrite}_user.txt).
# Its middle line is dynamic ("...diversity across these dimensions:
# {feature_dimensions}"), so the paragraph is located by its exact FIRST and
# LAST literal lines and replaced as a span. Everything before it is native
# problem/context; everything after it is native output-format instruction.
# Both stay untouched.
ORDINARY_OBJECTIVE_SPANS = (
    (
        "diff_user",
        "Suggest improvements to the program that will improve its FITNESS SCORE.",
        "Different solutions with similar fitness but different features are valuable.",
    ),
    (
        "full_rewrite_user",
        "Rewrite the program to improve its FITNESS SCORE.",
        "Provide the complete new program code.",
    ),
)


def replace_ordinary_objective(user_prompt: str, block: str) -> tuple[str, str] | None:
    """Swap the ordinary high-level action objective for the canonical block.

    The unique-primary-objective contract: the intervention dispatch must not
    contain BOTH the ordinary proposal objective and the canonical task, so
    the whole objective paragraph — located by its exact first and last
    template lines, both required to occur exactly once and in order, with no
    blank line inside the span (a paragraph, never a swallowed section) — is
    substituted in place. Native problem/context stays above, native
    output-format instructions stay below. Returns
    ``(new_user_prompt, objective_kind)``, or ``None`` when no known
    objective paragraph is uniquely present (unknown custom template, or an
    ambiguous echo of an anchor elsewhere) — the caller then fails closed
    and delivers the untouched ordinary prompt rather than guessing.
    """
    for kind, first_line, last_line in ORDINARY_OBJECTIVE_SPANS:
        if user_prompt.count(first_line) != 1 or user_prompt.count(last_line) != 1:
            continue
        start = user_prompt.index(first_line)
        end = user_prompt.index(last_line)
        if end < start:
            continue
        span_end = end + len(last_line)
        if "\n\n" in user_prompt[start:span_end]:
            continue
        return user_prompt[:start] + block + user_prompt[span_end:], kind
    return None


#: quiescent-branch-recipe-portfolio-v2: each guided dispatch is sampled
#: FROM its lane's exact live target branch (donors prepended to the native
#: inspirations) and carries ONLY its own branch-aware action.
BRANCH_PORTFOLIO_GUIDANCE_VERSION = "branch_recipe_portfolio_guidance_v2"
#: Canonical labels for the four ordered v2 portfolio slots. The first slot
#: may fall back from repair/restore according to the locked search regime:
#: integration/merge for wasteful wandering, representation exploration for
#: premature convergence.
BRANCH_PORTFOLIO_LANE_IDS = (
    "intervention_direction",
    "repair_or_restore",
    "integration_merge",
    "bottleneck_refinement",
    "representation_exploration",
)

DIAGNOSIS_BEGIN = "==== META-INTERVENTION DIAGNOSIS COHORT TASK"
DIAGNOSIS_END = "==== END META-INTERVENTION DIAGNOSIS COHORT TASK ===="

#: Content-hash coverage for one v2 lane envelope. start_completion and
#: plan_hash are dispatch metadata and are excluded.
_DIAGNOSIS_HASH_FIELDS = (
    "plan_version",
    "audit_id",
    "diagnosis",
    "quantitative_evidence",
    "exhausted_patterns",
    "lane_id",
    "lane_task",
    "portfolio_sha256",
    "target_branch_id",
    "donor_branch_ids",
    "diagnosed_bottleneck",
    "recipe",
    "compatibility",
)


class DiagnosisGuidanceDeliveryError(RuntimeError):
    """A STRICT diagnosis-cohort dispatch could not deliver its guidance.

    Raised inside the guided prompt-build wrapper so the vendor worker's own
    exception handling returns a normal ``SerializableResult(error=...)``
    BEFORE any LLM call — the assigned cohort slot terminates failed instead
    of masquerading as an ordinary proposal. There is no fail-soft fallback.
    """


def diagnosis_guidance_hash(payload: dict[str, Any]) -> str:
    doc = {k: payload[k] for k in _DIAGNOSIS_HASH_FIELDS if k in payload}
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_diagnosis_guidance(payload: Any) -> dict[str, Any]:
    """Strict integrity check of a diagnosis guidance payload (worker side).

    Version-pinned, optional mapping-only quantitative payload, and a matching
    content hash — a tampered or foreign payload never renders. Raises
    ``ValueError`` with the exact reason.
    """
    if not isinstance(payload, dict):
        raise ValueError("guidance payload is not a mapping")
    version = payload.get("plan_version")
    if version != BRANCH_PORTFOLIO_GUIDANCE_VERSION:
        raise ValueError(
            f"plan_version {version!r} is not {BRANCH_PORTFOLIO_GUIDANCE_VERSION!r}"
        )
    audit_id = payload.get("audit_id")
    if not isinstance(audit_id, str) or not audit_id.strip():
        raise ValueError("audit_id is not a non-empty string")
    diagnosis = payload.get("diagnosis")
    if diagnosis is not None and (
        not isinstance(diagnosis, str) or not diagnosis.strip()
    ):
        raise ValueError("diagnosis is not null or a non-empty string")
    evidence = payload.get("quantitative_evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("quantitative_evidence is not a mapping")
    if payload.get("lane_id") not in BRANCH_PORTFOLIO_LANE_IDS:
        raise ValueError("lane_id is not a supported portfolio lane")
    task = payload.get("lane_task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("lane_task is not a non-empty string")
    patterns = payload.get("exhausted_patterns")
    if (
        not isinstance(patterns, list)
        or len(patterns) > 3
        or not all(isinstance(p, str) and p.strip() for p in patterns)
    ):
        raise ValueError(
            "exhausted_patterns is not a list of at most three non-empty strings"
        )
    target = payload.get("target_branch_id")
    if not isinstance(target, str) or not target:
        raise ValueError("target_branch_id is not a non-empty string")
    donors = payload.get("donor_branch_ids")
    if (
        not isinstance(donors, list)
        or len(donors) > 2
        or not all(isinstance(d, str) and d for d in donors)
    ):
        raise ValueError("donor_branch_ids is not a list of at most two non-empty ids")
    if len(set(donors)) != len(donors) or target in donors:
        raise ValueError("donor_branch_ids must be unique and never the target")
    for field in ("diagnosed_bottleneck", "recipe"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is not a non-empty string")
    compatibility = payload.get("compatibility")
    if (
        not isinstance(compatibility, list)
        or not compatibility
        or not all(isinstance(c, str) and c.strip() for c in compatibility)
    ):
        raise ValueError("compatibility is not a non-empty list of non-empty strings")
    if payload.get("plan_hash") != diagnosis_guidance_hash(payload):
        raise ValueError("guidance content hash mismatch")
    return payload


def render_diagnosis_guidance_block(payload: dict[str, Any]) -> str:
    """Render one bounded action without duplicating planner-side policy."""
    evidence = payload.get("quantitative_evidence")
    evidence_block = (
        "\n\nFROZEN QUANTITATIVE EVIDENCE (read-only, observational; "
        "not an instruction):\n" + json.dumps(evidence, indent=2, sort_keys=True)
        if evidence
        else ""
    )
    donors = payload.get("donor_branch_ids") or []
    rendered_donors = (
        "; ".join(
            f"{donor} is Inspiration {index + 1}" for index, donor in enumerate(donors)
        )
        + " — in exactly that order in the Inspirations section"
        if donors
        else "(none)"
    )
    rendered_constraints = "\n".join(
        f"- {condition}" for condition in payload["compatibility"]
    )
    diagnosis = payload.get("diagnosis")
    diagnosis_block = (
        "Independent planner diagnoses — context only, not separate tasks:\n"
        f"{diagnosis}\n\n"
        if diagnosis
        else ""
    )
    return (
        f"{DIAGNOSIS_BEGIN} (audit {payload['audit_id']}, "
        f"{payload['plan_version']}) ====\n"
        "Execute one selected intervention direction for this work unit.\n\n"
        f"{diagnosis_block}"
        f"Target program: {payload['target_branch_id']}\n"
        f"Donor inspirations: {rendered_donors}\n"
        "Hypothesis:\n"
        f"{payload['diagnosed_bottleneck']}\n"
        "Action:\n"
        f"{payload['lane_task']}\n"
        "Constraints:\n"
        f"{rendered_constraints}\n\n"
        "Produce one concrete, bounded candidate for this action using the "
        "native OpenEvolve output format below."
        f"{evidence_block}\n"
        f"{DIAGNOSIS_END}"
    )


class BranchLaneDispatchError(RuntimeError):
    """A v2 branch-aware cohort dispatch could not use its exact validated
    target/donor branches (e.g. pruned from the live database between the
    pre-cohort CONFIRM validation and this dispatch). The assigned slot
    fails closed through the ordinary submission-failure path — zero LLM
    calls, never a random or global-frontier fallback parent."""


def _branch_aware_sample(database, plan: dict[str, Any], num_inspirations):
    """quiescent-branch-recipe-portfolio-v2 ONLY (plan §7): pre-sampling
    branch-aware selection for ONE reserved lane dispatch.

    The lane's exact validated live target branch is the parent and its
    CURRENT island is the dispatch island; validated live donors are
    prepended as inspirations; the remaining slots are filled through the
    EXISTING native island inspiration sampler; parent/donors/native fills
    are deterministically deduplicated (order-preserving, donors first).
    A missing target/donor raises (fail closed) — this function never
    substitutes a random or frontier parent. Ordinary dispatch never calls
    it.
    """

    def _live(branch_id: str, role: str):
        program = database.programs.get(branch_id)
        if program is None:
            emit_event(
                "branch_lane_dispatch_failed",
                audit_id=plan.get("audit_id"),
                lane_id=plan.get("lane_id"),
                plan_hash=plan.get("plan_hash"),
                reason=f"{role}_not_live:{branch_id}",
            )
            raise BranchLaneDispatchError(
                f"branch lane {plan.get('lane_id')}: {role} branch "
                f"{branch_id} is no longer live — failing closed"
            )
        return program

    parent = _live(plan["target_branch_id"], "target")
    donors = [_live(d, "donor") for d in plan.get("donor_branch_ids") or ()]
    target_island = next(
        (
            index
            for index, members in enumerate(database.islands)
            if parent.id in members
        ),
        None,
    )
    if target_island is None:
        emit_event(
            "branch_lane_dispatch_failed",
            audit_id=plan.get("audit_id"),
            lane_id=plan.get("lane_id"),
            plan_hash=plan.get("plan_hash"),
            reason=f"target_not_on_any_island:{parent.id}",
        )
        raise BranchLaneDispatchError(
            f"branch lane {plan.get('lane_id')}: target branch {parent.id} "
            "is not on any island — failing closed"
        )
    capacity = int(num_inspirations or 0)
    seen = {parent.id}
    inspirations = []
    for donor in donors:
        if donor.id not in seen:
            seen.add(donor.id)
            inspirations.append(donor)
    if len(inspirations) > capacity:
        # Explicitly assigned donors are never silently dropped and the
        # configured capacity is never exceeded: an assignment that cannot
        # fit its own donors fails closed.
        emit_event(
            "branch_lane_dispatch_failed",
            audit_id=plan.get("audit_id"),
            lane_id=plan.get("lane_id"),
            plan_hash=plan.get("plan_hash"),
            reason=(
                f"donors_exceed_inspiration_capacity:{len(inspirations)}>{capacity}"
            ),
        )
        raise BranchLaneDispatchError(
            f"branch lane {plan.get('lane_id')}: {len(inspirations)} donors "
            f"do not fit the inspiration capacity {capacity} — failing closed"
        )
    if len(inspirations) < capacity:
        # Request FULL capacity from the native sampler: a native candidate
        # that duplicates a same-island donor is skipped and a later
        # distinct candidate still fills the slot (no avoidable holes).
        # Donor-first order is preserved and the loop stops exactly at
        # capacity; an island with too few distinct programs still
        # legitimately yields fewer.
        for program in database._sample_inspirations(
            parent, n=capacity, island_id=target_island
        ):
            if len(inspirations) >= capacity:
                break
            if program.id not in seen:
                seen.add(program.id)
                inspirations.append(program)
    return target_island, parent, inspirations


def _run_iteration_worker_with_diagnosis_guidance(
    iteration: int, db_snapshot: dict, parent_id: str, inspiration_ids: list
):
    """Module-level worker entrypoint (fork- AND spawn-safe by reference).

    branch-aware portfolio v2: pops the lane guidance payload from THIS
    dispatch's snapshot and, for exactly the dispatch's
    evolution-proposal prompt, REPLACES the ordinary high-level action
    objective with the versioned guidance block (diagnosis + canonical
    reflection task) exactly once. Native sampling, parent, inspirations,
    patch type, RNG state, and output-format instructions are untouched.
    STRICT delivery: every fault terminates the assigned cohort slot as a
    failed work unit BEFORE any model call — never an ordinary fallback.
    """
    from openevolve.process_parallel import _run_iteration_worker

    payload = db_snapshot.pop("diagnosis_guidance", None)
    if not payload:
        return _run_iteration_worker(iteration, db_snapshot, parent_id, inspiration_ids)

    def _fail_event(reason: str) -> None:
        emit_event(
            "diagnosis_guidance_delivery_failed",
            iteration=iteration,
            audit_id=(payload or {}).get("audit_id"),
            plan_version=(payload or {}).get("plan_version"),
            plan_hash=(payload or {}).get("plan_hash"),
            reason=reason,
        )

    try:
        verified = verify_diagnosis_guidance(payload)
    except ValueError as exc:
        _fail_event(f"guidance_integrity:{exc}")
        from openevolve.process_parallel import SerializableResult

        return SerializableResult(
            error=f"diagnosis_guidance_delivery_failed: guidance_integrity:{exc}",
            iteration=iteration,
        )

    from openevolve.prompt.sampler import PromptSampler

    original_build = PromptSampler.build_prompt
    parent_record = (db_snapshot.get("programs") or {}).get(parent_id) or {}
    parent_code = parent_record.get("code")
    state = {"attached": False}

    def _fail_delivery(reason: str) -> DiagnosisGuidanceDeliveryError:
        _fail_event(reason)
        return DiagnosisGuidanceDeliveryError(
            f"diagnosis_guidance_delivery_failed: {reason}"
        )

    def guided_build_prompt(sampler_self, *args, **kwargs):
        if state["attached"]:
            return original_build(sampler_self, *args, **kwargs)
        # Only the dispatch's own evolution-proposal call is decorated;
        # evaluator/template builds pass through untouched and the wrapper
        # stays armed for the real proposal call.
        if (
            kwargs.get("template_key") is not None
            or kwargs.get("evolution_round") != iteration
            or (
                parent_code is not None and kwargs.get("current_program") != parent_code
            )
        ):
            return original_build(sampler_self, *args, **kwargs)
        state["attached"] = True
        PromptSampler.build_prompt = original_build
        donors = list(verified.get("donor_branch_ids") or [])
        if donors:
            # Deterministic donor -> Inspiration mapping WITHOUT vendor
            # edits: the native builder drops inspirations whose ids also
            # appear in the top/diverse sections (built from top_programs),
            # which could displace an assigned donor. For exactly THIS
            # guided prompt, assigned donors are removed from top_programs
            # so the donor-first inspiration order survives dedup and
            # renders as Inspiration 1 (first donor) and Inspiration 2
            # (second donor). A mapping that cannot be maintained fails the
            # slot closed — never a silently reordered prompt.
            inspiration_ids_in_prompt = [
                program.get("id") for program in (kwargs.get("inspirations") or [])
            ]
            if inspiration_ids_in_prompt[: len(donors)] != donors:
                raise _fail_delivery(
                    "donor_inspiration_mapping_broken:"
                    f"{inspiration_ids_in_prompt[: len(donors)]}!={donors}"
                )
            kwargs = dict(kwargs)
            kwargs["top_programs"] = [
                program
                for program in (kwargs.get("top_programs") or [])
                if program.get("id") not in set(donors)
            ]
        try:
            prompt = original_build(sampler_self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — strict cohort slot
            raise _fail_delivery(
                f"original_prompt_construction_failed:{type(exc).__name__}: {exc}"
            ) from exc
        try:
            user_before = prompt.get("user", "")
            block = render_diagnosis_guidance_block(verified)
            replaced = replace_ordinary_objective(user_before, block)
            if replaced is None:
                # Unknown custom template or ambiguous objective echo: never
                # deliver two competing objectives and never guess.
                raise _fail_delivery("ordinary_objective_not_uniquely_found")
            new_user, objective_kind = replaced
            if new_user.count(DIAGNOSIS_BEGIN) != 1:
                raise _fail_delivery("guidance_block_not_exactly_once")
            attached = dict(prompt)
            attached["user"] = new_user
            emit_event(
                "diagnosis_guidance_attached_to_prompt",
                iteration=iteration,
                audit_id=verified["audit_id"],
                plan_version=verified["plan_version"],
                plan_hash=verified["plan_hash"],
                diagnosis_sha256=sha256_text(verified.get("diagnosis")),
                has_quantitative_evidence=bool(verified.get("quantitative_evidence")),
                objective_kind=objective_kind,
                block_sha256=sha256_text(block),
                prompt_user_sha256_before=sha256_text(user_before),
                prompt_user_sha256_after=sha256_text(new_user),
                system_sha256=sha256_text(prompt.get("system")),
            )
            return attached
        except DiagnosisGuidanceDeliveryError:
            raise  # already durably recorded — never double-log the miss
        except Exception as exc:  # noqa: BLE001 — strict cohort slot
            raise _fail_delivery(f"attach_error:{type(exc).__name__}: {exc}") from exc

    PromptSampler.build_prompt = guided_build_prompt
    try:
        return _run_iteration_worker(iteration, db_snapshot, parent_id, inspiration_ids)
    finally:
        PromptSampler.build_prompt = original_build
        if not state["attached"]:
            # The dispatch never built its matching proposal prompt: an
            # attach that never happened is recorded honestly, never
            # silently claimed as success.
            _fail_event("no_matching_proposal_prompt_in_task")


def install_checkpoint_sidecar_trace() -> None:
    """After every successful upstream checkpoint save, write the meta-audit
    causal sidecar binding that checkpoint to the exact canonical completion
    cutoff.

    ``database.last_iteration`` cannot define the rollback boundary exactly:
    the upstream loop consumes whichever future finishes first and passes that
    ``completed_iteration`` to ``checkpoint_callback``, so the saved snapshot
    may contain higher-numbered completions and lack lower-numbered attempts
    that finish later. The sidecar records the last canonical
    ``completion_seq`` consumed at save time — the precise causal cutoff.

    Wraps ``OpenEvolve._save_checkpoint`` (runtime wrapper, no vendor edit),
    which covers BOTH the interval checkpoints (via ``checkpoint_callback``)
    and the final checkpoint in ``_run_evolution_with_checkpoints``. With
    meta-audit off (no manager on the controller) this is a no-op and no file
    is written.
    """
    from openevolve.controller import OpenEvolve

    original = OpenEvolve._save_checkpoint
    if getattr(original, "_oe_meta_sidecar", False):
        return

    def traced_save_checkpoint(self, iteration):
        result = original(self, iteration)  # only a SUCCESSFUL save gets a sidecar
        try:
            parallel = getattr(self, "parallel_controller", None)
            meta = getattr(parallel, "_meta_audit", None)
            if meta is not None:
                recorder = getattr(parallel, "_canonical_completions", None)
                cutoff_seq = (
                    recorder.last_assigned_seq
                    if recorder is not None
                    else meta.ledger.last_completion_seq
                )
                checkpoint_path = os.path.join(
                    self.output_dir, "checkpoints", f"checkpoint_{iteration}"
                )
                meta.write_checkpoint_sidecar(
                    checkpoint_path=checkpoint_path,
                    checkpoint_iteration=iteration,
                    cutoff_seq=cutoff_seq,
                    database=self.database,
                )
        except Exception as exc:
            emit_event(
                "meta_audit_hook_error",
                hook="write_checkpoint_sidecar",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            logging.getLogger("meta_audit_adapter").exception(
                "checkpoint sidecar write failed (evolution continues)"
            )
        return result

    traced_save_checkpoint._oe_meta_sidecar = True
    OpenEvolve._save_checkpoint = traced_save_checkpoint


class _ParkedDispatch:
    """Future facade for a dispatch deferred by the quiescent fence.

    Branch-aware portfolio v2 only. Keeps the vendor loop's pending-future
    bookkeeping intact while NO sampling, scheduling, or worker submission
    happens (the vendor never regrows its pipeline after a dip, so fenced
    slots must stay visibly pending). On release the REAL dispatch runs —
    fresh native sampling at release time — and this facade delegates to its
    CanonicalFuture. Cancel-while-parked (vendor shutdown) abandons the
    never-dispatched slot without emitting anything.
    """

    def __init__(self, controller, iteration, island_id):
        self._controller = controller
        self.iteration = iteration
        self.island_id = island_id
        self._inner = None
        self._cancelled = False
        self._failed = False

    def release(self) -> None:
        if self._cancelled or self._inner is not None:
            return
        inner = self._controller._dispatch_iteration(
            self.iteration, self.island_id, time.time()
        )
        if inner is None:
            # submission_failed: already recorded through the canonical path.
            self._failed = True
        else:
            self._inner = inner

    def done(self):
        if self._inner is None and not self._cancelled and not self._failed:
            # Real-loop heartbeat (controller thread): run_evolution only
            # polls pending future.done() every ~10ms — once EVERY slot is
            # parked there is no future _submit_iteration call and no
            # completion, so this poll is the ONLY controller-thread
            # opportunity to apply a finished background audit outcome and
            # release the fence (PASS resumes ordinary dispatch; a
            # validated INTERVENE opens the guided lanes). While the fence
            # is unchanged this stays a silent no-op: reconcile drains an
            # empty queue and the release helper re-checks the fence before
            # any sampling — no resampling, no repeated journaling.
            self._controller._parked_heartbeat()
        if self._inner is not None:
            return self._inner.done()
        return self._cancelled or self._failed

    def result(self, *args, **kwargs):
        if self._inner is not None:
            return self._inner.result(*args, **kwargs)
        if self._cancelled:
            raise CancelledError()
        raise RuntimeError("parked dispatch failed to submit")

    def cancel(self):
        if self._inner is not None:
            return self._inner.cancel()
        self._cancelled = True
        return True

    def __getattr__(self, name):
        inner = object.__getattribute__(self, "_inner")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)


def make_traced_parallel_controller(meta_audit=None):
    """Return an upstream-compatible controller with schedule/result events.

    ``meta_audit`` (optional): the v2 online audit manager. When None — the
    default, i.e. --meta-audit-mode off — every
    meta hook below is skipped entirely and behavior is byte-identical to the
    pre-meta controller. Meta hooks are additive and guarded: a hook failure
    is journaled and must never alter or abort evolution.
    """
    from openevolve.process_parallel import (
        ProcessParallelController,
        _run_iteration_worker,
    )

    class TracedProcessParallelController(ProcessParallelController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._canonical_completions = CanonicalCompletionRecorder()
            self._pending_program_commits: dict[str, int] = {}
            #: v2: submissions deferred by the dispatch
            #: fence, released oldest-first when the fence lifts.
            self._parked_dispatches: list[_ParkedDispatch] = []
            self._meta_audit = meta_audit
            if self._meta_audit is not None:
                self._meta_guard(
                    "bind_database", self._meta_audit.bind_database, self.database
                )
                scheduler_num_islands = getattr(
                    self._meta_audit, "scheduler_num_islands", None
                )
                if scheduler_num_islands is not None:
                    self.num_islands = int(scheduler_num_islands)
            self._db_state_commit_correlation = None
            self._install_database_commit_trace()
            self._install_database_generation_trace()
            self._install_database_migration_trace()
            self._initialize_replayed_inflight()

        def _initialize_replayed_inflight(self) -> None:
            """Register exact cutoff-time work restored by the branch materializer."""
            self._replayed_inflight: dict[str, dict[str, Any]] = {}
            if self._meta_audit is None:
                return
            initializer = getattr(
                self._meta_audit, "initialize_replayed_inflight", None
            )
            if not callable(initializer):
                return
            instance_id = str(os.environ.get("OE_RUN_INSTANCE_ID"))
            for source in initializer(instance_id):
                row = dict(source)
                attempt_id = str(row["attempt_id"])
                iteration = int(row["iteration"])
                scheduled_at = time.time()
                row["replay_scheduled_at"] = scheduled_at
                self._replayed_inflight[attempt_id] = row
                parent = self.database.programs.get(row.get("parent_id"))
                emit_event(
                    "attempt_scheduled",
                    attempt_id=attempt_id,
                    iteration=iteration,
                    target_island=row.get("target_island"),
                    parent_id=row.get("parent_id"),
                    parent_code_sha256=sha256_text(getattr(parent, "code", None)),
                    parent_metrics=getattr(parent, "metrics", None),
                    inspiration_ids=[],
                    inspiration_code_sha256=[],
                    database_program_count=len(self.database.programs),
                    meta_intervention=False,
                    replayed_inflight=True,
                    replay_source_attempt_id=row.get("source_attempt_id"),
                    replay_source_completion_seq=row.get("completion_seq"),
                    replay_remaining_seconds=row.get("remaining_seconds"),
                )
                # Exact replay is a controller invariant, not an optional
                # observation.  Registration faults must stop this branch.
                self._meta_audit.on_attempt_scheduled(
                    attempt_id=attempt_id,
                    iteration=iteration,
                    island=row.get("target_island"),
                    parent_id=row.get("parent_id"),
                    inspiration_ids=(),
                    at=scheduled_at,
                    replayed_inflight=True,
                    replay_source_attempt_id=row.get("source_attempt_id"),
                )

        def _consume_due_replayed_inflight(self) -> None:
            if self._meta_audit is None or not self._replayed_inflight:
                return
            take_due = getattr(self._meta_audit, "take_due_replayed_inflight", None)
            if not callable(take_due):
                return
            from openevolve.database import Program
            from openevolve.process_parallel import SerializableResult

            for source in take_due():
                row = self._replayed_inflight.pop(str(source["attempt_id"]))
                iteration = int(row["iteration"])
                target_island = row.get("target_island")
                child = row.get("child")
                result = SerializableResult(
                    child_program_dict=(
                        dict(child) if isinstance(child, dict) else None
                    ),
                    parent_id=row.get("parent_id"),
                    iteration_time=max(
                        0.0,
                        float(row.get("terminal_at") or 0.0)
                        - float(row.get("scheduled_at") or 0.0),
                    ),
                    artifacts=(
                        dict(row["artifacts"])
                        if isinstance(row.get("artifacts"), dict)
                        else None
                    ),
                    iteration=iteration,
                    error=row.get("error"),
                    target_island=target_island,
                )
                status = str(
                    row.get("status") or ("error" if result.error else "completed")
                )
                terminal = str(
                    row.get("terminal")
                    or ("worker_error" if result.error else "child_ready")
                )
                emit_event(
                    "worker_result",
                    iteration=iteration,
                    scheduled_target_island=target_island,
                    scheduled_parent_id=row.get("parent_id"),
                    status=status,
                    replayed_inflight=True,
                    replay_source_attempt_id=row.get("source_attempt_id"),
                    replay_source_completion_seq=row.get("completion_seq"),
                    **_result_summary(result),
                )
                completion_seq = self._record_consumed_result(
                    iteration=iteration,
                    target_island=target_island,
                    parent_id=row.get("parent_id"),
                    scheduled_at=float(row["replay_scheduled_at"]),
                    result=result,
                    status=status,
                    terminal=terminal,
                    replayed_inflight=True,
                    replay_source_attempt_id=row.get("source_attempt_id"),
                    replay_source_completion_seq=row.get("completion_seq"),
                )
                if row.get("committed") is True:
                    if not isinstance(child, dict):
                        raise RuntimeError(
                            "committed in-flight replay has no child payload"
                        )
                    program = Program(
                        **{
                            key: value
                            for key, value in child.items()
                            if key != "code_sha256"
                        }
                    )
                    self.database.add(
                        program,
                        iteration=iteration,
                        target_island=target_island,
                    )
                    if result.artifacts:
                        self.database.store_artifacts(program.id, result.artifacts)
                    increment_island = row.get("increment_island")
                    if not isinstance(increment_island, int):
                        raise RuntimeError(
                            "committed in-flight replay has no island increment"
                        )
                    self.database.increment_island_generation(
                        island_idx=increment_island
                    )
                    if self.database.should_migrate():
                        raise RuntimeError(
                            "in-flight replay reached an unrecorded migration boundary"
                        )
                elif isinstance(child, dict) and child.get("id"):
                    self._pending_program_commits.pop(child["id"], None)
                    self._meta_audit.on_commit_failed(
                        completion_seq=completion_seq,
                        program_id=child["id"],
                        error="source completion was not committed",
                    )
                emit_event(
                    "replayed_inflight_consumed",
                    attempt_id=row["attempt_id"],
                    completion_seq=completion_seq,
                    source_attempt_id=row.get("source_attempt_id"),
                    source_completion_seq=row.get("completion_seq"),
                )

        def _meta_guard(self, hook_name, fn, *args, **kwargs):
            """Run one hook; exact overlap managers may opt into fail-closed."""
            if self._meta_audit is None:
                return None
            if getattr(self._meta_audit, "strict_controller_hooks", False):
                return fn(*args, **kwargs)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                emit_event(
                    "meta_audit_hook_error",
                    hook=hook_name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
                logging.getLogger("meta_audit_adapter").exception(
                    "meta-audit hook %s failed (evolution continues)", hook_name
                )
                return None

        def _install_database_commit_trace(self) -> None:
            """Confirm a child was added, after the upstream add succeeds."""
            original_add = self.database.add

            def traced_add(program, *args, **kwargs):
                meta = self._meta_audit
                program_id = getattr(program, "id", "")
                pending_seq = self._pending_program_commits.get(program_id)
                best_before = None
                if meta is not None and pending_seq is not None:
                    best_before = self._meta_guard("capture_best", meta.capture_best)
                try:
                    result = original_add(program, *args, **kwargs)
                except Exception as exc:
                    # A consumed valid child failed to commit. Upstream
                    # tolerates this (run_evolution catches and continues), so
                    # the attempt must settle as terminal-but-uncommitted here
                    # rather than wait for a commit that will never come.
                    failed_seq = self._pending_program_commits.pop(program_id, None)
                    if failed_seq is not None:
                        emit_event(
                            "program_commit_failed",
                            completion_seq=failed_seq,
                            program_id=program_id,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            traceback="".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        )
                        if meta is not None:
                            self._meta_guard(
                                "on_commit_failed",
                                meta.on_commit_failed,
                                completion_seq=failed_seq,
                                program_id=program_id,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                    raise  # upstream behavior unchanged

                completion_seq = self._pending_program_commits.pop(
                    getattr(program, "id", ""), None
                )
                if completion_seq is not None:
                    emit_event(
                        "program_committed",
                        completion_seq=completion_seq,
                        commit_index=completion_seq,
                        program_id=getattr(program, "id", None),
                        parent_id=getattr(program, "parent_id", None),
                        generation=getattr(program, "generation", None),
                        iteration_found=getattr(program, "iteration_found", None),
                        metrics=getattr(program, "metrics", None),
                        metadata=getattr(program, "metadata", None),
                        code_sha256=sha256_text(getattr(program, "code", None)),
                        database_program_count=len(self.database.programs),
                    )
                    # Exact post-commit state (add-internal population
                    # cleanup / zombie removal included): same safe point,
                    # observation only. NOTE: the controller increments the
                    # child's island generation only AFTER add returns, so
                    # this record is one completion behind on
                    # island_generations; the correlated
                    # island_generation_increment record below establishes
                    # the final end-of-completion state.
                    emit_db_state(
                        "program_committed",
                        self.database,
                        completion_seq=completion_seq,
                        program_id=getattr(program, "id", None),
                    )
                    self._db_state_commit_correlation = {
                        "completion_seq": completion_seq,
                        "program_id": getattr(program, "id", None),
                    }
                    if meta is not None and best_before is not None:
                        best_after = self._meta_guard("capture_best", meta.capture_best)
                        if best_after is not None:
                            self._meta_guard(
                                "on_program_committed",
                                meta.on_program_committed,
                                completion_seq=completion_seq,
                                program_id=getattr(program, "id", None),
                                before=best_before,
                                after=best_after,
                            )
                return result

            self.database.add = traced_add

        def _install_database_generation_trace(self) -> None:
            """Capture the end-of-completion island-generation increment.

            ProcessParallelController increments the committed child's
            island generation only AFTER ``database.add`` returned, so the
            post-add record's ``island_generations`` is one completion
            behind the true end-of-completion safe point and cannot prove
            ``should_migrate`` timing. This record — correlated with the
            SAME committed ``completion_seq``/``program_id`` — establishes
            the final structural state after the increment. The post-add
            record is kept: add may succeed while later controller work
            fails. Observation only; the upstream call is unchanged.
            """
            original_increment = self.database.increment_island_generation

            def traced_increment(*args, **kwargs):
                result = original_increment(*args, **kwargs)
                correlation = self._db_state_commit_correlation
                self._db_state_commit_correlation = None
                island_idx = kwargs.get("island_idx", args[0] if args else None)
                emit_db_state(
                    "island_generation_increment",
                    self.database,
                    completion_seq=(correlation or {}).get("completion_seq"),
                    program_id=(correlation or {}).get("program_id"),
                    island_idx=island_idx,
                )
                return result

            self.database.increment_island_generation = traced_increment

        def _install_database_migration_trace(self) -> None:
            """Capture exact post-state after each real island migration.

            The controller gates migration via ``should_migrate()`` before
            calling, so every intercepted call is a real mutation. Migrant
            copies carry fresh uuid4 ids that exist nowhere else in the
            trace, so the record includes their full attribute payloads
            (code by SHA-256 — the bytes equal the source program's).
            Observation only: the upstream call is returned unchanged and a
            capture failure never propagates.
            """
            original_migrate = self.database.migrate_programs

            def traced_migrate(*args, **kwargs):
                if not db_state_capture_enabled():
                    # Fully inert: no id-set copies or payload work either.
                    return original_migrate(*args, **kwargs)
                before_ids = set(self.database.programs.keys())
                result = original_migrate(*args, **kwargs)
                try:
                    new_ids = sorted(set(self.database.programs.keys()) - before_ids)
                    emit_db_state(
                        "island_migration",
                        self.database,
                        new_programs=[
                            program_payload_sans_code(self.database.programs[pid])
                            for pid in new_ids
                            if pid in self.database.programs
                        ],
                    )
                except Exception:
                    logging.getLogger("trajectory").exception(
                        "db-state migration capture failed (evolution continues)"
                    )
                return result

            self.database.migrate_programs = traced_migrate

        def _record_consumed_result(
            self,
            *,
            iteration: int,
            target_island: int | None,
            parent_id: str | None,
            scheduled_at: float,
            result: Any = None,
            status: str,
            terminal: str,
            **payload: Any,
        ) -> int:
            summary = (
                _result_summary(result, include_child_code=True)
                if result is not None
                else {}
            )
            child = getattr(result, "child_program_dict", None)
            artifacts = getattr(result, "artifacts", None)
            failure_stage = (
                artifacts.get("failure_stage") if isinstance(artifacts, dict) else None
            )
            evaluation_valid = (
                isinstance(child, dict)
                and not getattr(result, "error", None)
                and failure_stage is None
            )
            completion_seq = self._canonical_completions.record(
                iteration=iteration,
                status=status,
                terminal=terminal,
                scheduled_parent_id=parent_id,
                scheduled_target_island=target_island,
                scheduled_at=scheduled_at,
                evaluation_valid=evaluation_valid,
                failure_stage=failure_stage,
                **summary,
                **payload,
            )
            if isinstance(child, dict) and child.get("id"):
                self._pending_program_commits[child["id"]] = completion_seq
            if self._meta_audit is not None:
                self._meta_guard(
                    "on_completion_consumed",
                    self._meta_audit.on_completion_consumed,
                    attempt_id=f"{os.environ.get('OE_RUN_INSTANCE_ID')}:{iteration}",
                    completion_seq=completion_seq,
                    iteration=iteration,
                    island=target_island,
                    parent_id=parent_id,
                    status=status,
                    terminal=terminal,
                    evaluation_valid=evaluation_valid,
                    error=(
                        getattr(result, "error", None)
                        if result is not None
                        else payload.get("error")
                    ),
                    failure_stage=failure_stage,
                    child=child if isinstance(child, dict) else None,
                )
            return completion_seq

        def _submit_iteration(self, iteration: int, island_id: int | None = None):
            scheduled_at = time.time()
            if self._meta_audit is not None:
                # Pre-submission safe point: reconcile finished audits and
                # stamp next_ordinary_dispatch_at BEFORE any sampling for the
                # new attempt.
                self._meta_guard(
                    "on_before_submit", self._meta_audit.on_before_submit, iteration
                )
                self._release_parked_dispatches()
                if self._meta_guard(
                    "ordinary_dispatch_fenced",
                    self._meta_audit.ordinary_dispatch_fenced,
                ):
                    parked = _ParkedDispatch(self, iteration, island_id)
                    self._parked_dispatches.append(parked)
                    return parked
            return self._dispatch_iteration(iteration, island_id, scheduled_at)

        def _parked_heartbeat(self) -> None:
            """Controller-thread heartbeat from a parked ``done()`` poll:
            apply any finished audit outcome and release parked dispatches
            once the fence drops (see ``_ParkedDispatch.done``)."""
            if self._meta_audit is not None:
                self._consume_due_replayed_inflight()
                self._meta_guard("reconcile", self._meta_audit.reconcile)
                heartbeat = getattr(self._meta_audit, "controller_heartbeat", None)
                if callable(heartbeat):
                    heartbeat()
                self._release_parked_dispatches()

        def _release_parked_dispatches(self) -> None:
            """Dispatch parked submissions oldest-first while unfenced.

            Controller thread only. Guided cohort dispatches consume the
            selected lanes first; later released slots are ordinary once the
            executor has fallen back to native concurrency.
            """
            while self._parked_dispatches:
                if self._meta_guard(
                    "ordinary_dispatch_fenced",
                    self._meta_audit.ordinary_dispatch_fenced,
                ):
                    return
                self._parked_dispatches.pop(0).release()

        def _dispatch_iteration(
            self, iteration: int, island_id: int | None, scheduled_at: float
        ):
            try:
                target_island = (
                    island_id if island_id is not None else self.database.current_island
                )
                # Globally unambiguous launch identity: iteration numbers are
                # REUSED across checkpoint rollback/resume (observed in real
                # traces), so every attempt is identified by instance+iteration
                # — the same composite canonical_completion always recorded.
                attempt_id = f"{os.environ.get('OE_RUN_INSTANCE_ID')}:{iteration}"

                # quiescent-branch-recipe-portfolio-v2 ONLY (plan §7): the
                # lane assignment is reserved BEFORE any sampling; the exact
                # validated live target branch becomes the parent (and sets
                # the dispatch island), donors are prepended to the
                # inspirations, and the remaining slots come from the
                # existing native island sampler with deterministic dedup.
                # A missing/stale branch fails this slot closed (raises
                # into the submission-failure path below) — never a random
                # or global-frontier fallback. Ordinary dispatch takes the
                # exact pre-existing native path.
                branch_plan = None
                if self._meta_audit is not None:
                    try:
                        # Deliberately NOT _meta_guard: a missing getter or a
                        # reservation-getter fault must fail THIS slot closed
                        # before any parent sampling — never be swallowed
                        # into a silent native fallback. A legitimate None
                        # (no active cohort assignment) still takes the
                        # native path below.
                        plan_getter = getattr(
                            self._meta_audit, "active_plan_for_dispatch", None
                        )
                        if not callable(plan_getter):
                            emit_event(
                                "branch_lane_dispatch_failed",
                                audit_id=None,
                                lane_id=None,
                                plan_hash=None,
                                reason="assignment_getter_missing",
                            )
                            raise BranchLaneDispatchError(
                                "branch-aware assignment getter is missing "
                                "— failing closed"
                            )
                        try:
                            branch_plan = plan_getter(
                                attempt_id=attempt_id, iteration=iteration
                            )
                        except Exception as exc:
                            emit_event(
                                "branch_lane_dispatch_failed",
                                audit_id=None,
                                lane_id=None,
                                plan_hash=None,
                                reason=(
                                    "assignment_getter_failed:"
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            )
                            raise BranchLaneDispatchError(
                                "branch-aware assignment getter raised "
                                f"{type(exc).__name__} — failing closed"
                            ) from exc
                        if branch_plan is not None:
                            if not branch_plan.get("target_branch_id"):
                                emit_event(
                                    "branch_lane_dispatch_failed",
                                    audit_id=branch_plan.get("audit_id"),
                                    lane_id=branch_plan.get("lane_id"),
                                    plan_hash=branch_plan.get("plan_hash"),
                                    reason="assignment_missing_target",
                                )
                                raise BranchLaneDispatchError(
                                    "branch-aware cohort assignment carries "
                                    "no target_branch_id — failing closed"
                                )
                            target_island, parent, inspirations = _branch_aware_sample(
                                self.database,
                                branch_plan,
                                self.config.prompt.num_diverse_programs,
                            )
                    except BranchLaneDispatchError:
                        # This attempt id is consumed (when a lane was
                        # reserved, no replacement slot exists). Record the
                        # launch — no parent was ever sampled — so the
                        # ordinary submission-failure completion below can
                        # pair with it and settle the cohort slot.
                        emit_event(
                            "attempt_scheduled",
                            attempt_id=attempt_id,
                            iteration=iteration,
                            target_island=target_island,
                            parent_id=None,
                            parent_code_sha256=None,
                            parent_metrics=None,
                            inspiration_ids=[],
                            inspiration_code_sha256=[],
                            database_program_count=len(self.database.programs),
                            meta_intervention=False,
                            meta_audit_id=None,
                            meta_prompt_version=None,
                            branch_lane_failed_closed=True,
                        )
                        self._meta_guard(
                            "on_attempt_scheduled",
                            self._meta_audit.on_attempt_scheduled,
                            attempt_id=attempt_id,
                            iteration=iteration,
                            island=target_island,
                            parent_id=None,
                            inspiration_ids=(),
                            at=scheduled_at,
                        )
                        raise
                if branch_plan is None:
                    parent, inspirations = self.database.sample_from_island(
                        island_id=target_island,
                        num_inspirations=self.config.prompt.num_diverse_programs,
                    )
                db_snapshot = self._create_database_snapshot()
                db_snapshot["sampling_island"] = target_island
                emit_event(
                    "attempt_scheduled",
                    attempt_id=attempt_id,
                    iteration=iteration,
                    target_island=target_island,
                    parent_id=parent.id,
                    parent_code_sha256=sha256_text(parent.code),
                    parent_metrics=parent.metrics,
                    inspiration_ids=[p.id for p in inspirations],
                    inspiration_code_sha256=[sha256_text(p.code) for p in inspirations],
                    database_program_count=len(self.database.programs),
                    meta_intervention=bool(branch_plan),
                    meta_audit_id=(branch_plan or {}).get("audit_id"),
                    meta_prompt_version=(branch_plan or {}).get("plan_version"),
                )
                if self._meta_audit is not None:
                    self._meta_guard(
                        "on_attempt_scheduled",
                        self._meta_audit.on_attempt_scheduled,
                        attempt_id=attempt_id,
                        iteration=iteration,
                        island=target_island,
                        parent_id=parent.id,
                        inspiration_ids=tuple(p.id for p in inspirations),
                        at=scheduled_at,
                    )

                worker_fn = _run_iteration_worker
                if branch_plan:
                    db_snapshot["diagnosis_guidance"] = dict(branch_plan)
                    worker_fn = _run_iteration_worker_with_diagnosis_guidance

                raw_future = self.executor.submit(
                    worker_fn,
                    iteration,
                    db_snapshot,
                    parent.id,
                    [p.id for p in inspirations],
                )

                def record_result(done_future):
                    try:
                        result = done_future.result()
                    except CancelledError:
                        emit_event(
                            "worker_result",
                            iteration=iteration,
                            target_island=target_island,
                            parent_id=parent.id,
                            status="cancelled",
                        )
                    except BaseException as exc:
                        emit_event(
                            "worker_result",
                            iteration=iteration,
                            target_island=target_island,
                            parent_id=parent.id,
                            status="exception",
                            error_type=type(exc).__name__,
                            error=str(exc),
                            traceback="".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        )
                    else:
                        emit_event(
                            "worker_result",
                            iteration=iteration,
                            scheduled_target_island=target_island,
                            scheduled_parent_id=parent.id,
                            status=("error" if result.error else "completed"),
                            **_result_summary(result),
                        )

                raw_future.add_done_callback(record_result)

                controller = self

                class CanonicalFuture:
                    """Future facade that records the controller's consume order."""

                    def __init__(self, wrapped):
                        self._wrapped = wrapped
                        self._record_lock = threading.Lock()
                        self._recorded = False

                    def _record_once(self, *, result=None, status, terminal, **payload):
                        with self._record_lock:
                            if self._recorded:
                                return
                            controller._record_consumed_result(
                                iteration=iteration,
                                target_island=target_island,
                                parent_id=parent.id,
                                scheduled_at=scheduled_at,
                                result=result,
                                status=status,
                                terminal=terminal,
                                **payload,
                            )
                            self._recorded = True

                    def result(self, *args, **kwargs):
                        try:
                            result = self._wrapped.result(*args, **kwargs)
                        except CancelledError:
                            self._record_once(
                                status="cancelled",
                                terminal="future_cancelled",
                            )
                            raise
                        except BaseException as exc:
                            self._record_once(
                                status="exception",
                                terminal="future_exception",
                                error_type=type(exc).__name__,
                                error=str(exc),
                                traceback="".join(
                                    traceback.format_exception(
                                        type(exc), exc, exc.__traceback__
                                    )
                                ),
                            )
                            raise
                        self._record_once(
                            result=result,
                            status=("error" if result.error else "completed"),
                            terminal=(
                                "worker_error" if result.error else "child_ready"
                            ),
                        )
                        return result

                    def done(self):
                        controller._parked_heartbeat()
                        return self._wrapped.done()

                    def cancel(self):
                        cancelled = self._wrapped.cancel()
                        self._record_once(
                            status=("cancelled" if cancelled else "abandoned"),
                            terminal=(
                                "cancelled_before_start"
                                if cancelled
                                else "abandoned_at_shutdown"
                            ),
                            cancel_succeeded=bool(cancelled),
                        )
                        return cancelled

                    def __getattr__(self, name):
                        return getattr(self._wrapped, name)

                return CanonicalFuture(raw_future)
            except Exception as exc:
                emit_event(
                    "attempt_submission_failed",
                    iteration=iteration,
                    target_island=island_id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
                # Route through the SAME consumed-result path as every other
                # terminal so the canonical record and the meta-audit hook fire
                # exactly once: the attempt settles immediately (never eligible,
                # never in-flight in a later-frozen bundle).
                self._record_consumed_result(
                    iteration=iteration,
                    target_island=locals().get("target_island", island_id),
                    parent_id=getattr(locals().get("parent"), "id", None),
                    scheduled_at=scheduled_at,
                    result=None,
                    status="exception",
                    terminal="submission_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback="".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                )
                logging.getLogger(__name__).error(
                    "Error submitting iteration %s: %s", iteration, exc
                )
                return None

    return TracedProcessParallelController
