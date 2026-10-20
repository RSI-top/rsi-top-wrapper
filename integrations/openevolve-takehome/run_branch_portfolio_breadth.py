"""Generate a direction pool in parallel and remove semantic duplicates."""

from __future__ import annotations
from release_paths import release_resource
import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from run_direction_llm_dedup import (
    DEFAULT_BASE_URL as DEFAULT_DEDUP_BASE_URL,
    DEFAULT_MAX_TOKENS as DEFAULT_DEDUP_MAX_TOKENS,
    DEFAULT_MODEL as DEFAULT_DEDUP_MODEL,
    deduplicate_directions,
)
from metric_scaled_dashboard import validate_dashboard_jsonl, write_llm_dashboard_jsonl
from planner_runtime import (
    DIRECTION_KEYS,
    _private_engine_profiles_path,
    _private_history_path,
    materialize_view,
    merge_pi_evidence_metadata,
    normalize_portfolio_direction_fields,
    normalize_portfolio_evidence_ids,
    run_pi,
    sanitize_planner_view,
    validate_portfolio,
    validate_reads,
)

RUN_SCHEMA = "oe_breadth_direction_pool_v1"
DEFAULT_CANDIDATE_POOL_SIZE = 32
DEFAULT_DIRECTIONS_PER_CALL = 8
DEFAULT_FORMAT_ATTEMPTS = 20
PRIOR_FAILED_INTERVENTIONS_HEADING = (
    "## Failed interventions since the last meaningful frontier improvement"
)
PLANNER_PROMPT = release_resource("planner.md")


def _read_llm_dashboard(path: Path) -> tuple[dict[str, Any], list[list[Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or not isinstance(records[0], dict):
        raise ValueError("planner dashboard is missing its header")
    if not all((isinstance(row, list) for row in records[1:])):
        raise ValueError("planner dashboard contains an invalid observation row")
    return (records[0], records[1:])


def _balanced_metric_subspaces(
    *, metric_order: list[str], planner_calls: int, metric_count: int, seed: int
) -> list[dict[str, Any]]:
    """Assign reproducible metric subsets while minimizing repeated coverage."""
    if metric_count < 1:
        return []
    if metric_count > len(metric_order):
        raise ValueError("--planner-random-metric-count exceeds the available metrics")
    rng = random.Random(seed)
    use_count = {metric_id: 0 for metric_id in metric_order}
    seen: set[frozenset[str]] = set()
    assignments: list[dict[str, Any]] = []
    for batch_index in range(planner_calls):
        selected: list[str] | None = None
        for _ in range(1000):
            shuffled = list(metric_order)
            rng.shuffle(shuffled)
            candidate = sorted(shuffled, key=use_count.__getitem__)[:metric_count]
            key = frozenset(candidate)
            if key not in seen:
                selected = candidate
                seen.add(key)
                break
        if selected is None:
            raise ValueError("could not construct distinct planner metric subspaces")
        for metric_id in selected:
            use_count[metric_id] += 1
        selected_set = set(selected)
        assignments.append(
            {
                "batch_index": batch_index,
                "metric_ids": [
                    metric_id for metric_id in metric_order if metric_id in selected_set
                ],
                "seed": seed,
            }
        )
    return assignments


def _filter_metric_dashboard(
    *,
    header: dict[str, Any],
    rows: list[list[Any]],
    metric_ids: list[str],
    definitions: dict[str, Any],
) -> tuple[dict[str, Any], list[list[Any]], str]:
    metric_order = [str(item) for item in header.get("metric_order") or []]
    selected = set(metric_ids)
    if len(selected) != len(metric_ids) or not selected <= set(metric_order):
        raise ValueError("planner metric subspace contains invalid metric IDs")
    filtered_order = [metric_id for metric_id in metric_order if metric_id in selected]
    if filtered_order != metric_ids:
        raise ValueError("planner metric subspace is not in dashboard order")
    filtered_definitions = copy.deepcopy(definitions)
    source_metrics = definitions.get("metrics")
    if not isinstance(source_metrics, dict) or not selected <= set(source_metrics):
        raise ValueError("metric definitions do not cover the planner subspace")
    filtered_definitions["metrics"] = {
        metric_id: copy.deepcopy(source_metrics[metric_id])
        for metric_id in filtered_order
    }
    definitions_text = json.dumps(filtered_definitions, indent=2) + "\n"
    definitions_sha256 = hashlib.sha256(definitions_text.encode("utf-8")).hexdigest()
    filtered_header = copy.deepcopy(header)
    filtered_header["metric_order"] = filtered_order
    filtered_header["definitions_sha256"] = definitions_sha256
    for field in ("metric_window_source", "metric_reference_role"):
        values = header.get(field) or {}
        filtered_header[field] = {
            metric_id: values[metric_id]
            for metric_id in filtered_order
            if metric_id in values
        }
    row_columns = filtered_header.get("row_columns") or []
    try:
        metric_columns_index = row_columns.index("metric_columns")
        availability_index = row_columns.index("availability_details")
    except ValueError as exc:
        raise ValueError("planner dashboard lacks metric row columns") from exc
    selected_indices = [metric_order.index(metric_id) for metric_id in filtered_order]
    index_remap = {
        old_index: new_index for new_index, old_index in enumerate(selected_indices)
    }
    filtered_rows: list[list[Any]] = []
    for row in rows:
        if len(row) != len(row_columns):
            raise ValueError("planner dashboard row does not match row_columns")
        projected = copy.deepcopy(row)
        metric_columns = projected[metric_columns_index]
        if not isinstance(metric_columns, list) or len(metric_columns) != 5:
            raise ValueError("planner dashboard metric_columns are invalid")
        projected_columns: list[Any] = []
        for column in metric_columns:
            if isinstance(column, str):
                if len(column) != len(metric_order):
                    raise ValueError("planner dashboard status column is invalid")
                projected_columns.append(
                    "".join((column[index] for index in selected_indices))
                )
            elif isinstance(column, list) and len(column) == len(metric_order):
                projected_columns.append([column[index] for index in selected_indices])
            else:
                raise ValueError("planner dashboard metric column length is invalid")
        projected[metric_columns_index] = projected_columns
        details = projected[availability_index]
        if not isinstance(details, list):
            raise ValueError("planner dashboard availability details are invalid")
        projected_details = []
        for detail in details:
            if not isinstance(detail, list) or not detail:
                raise ValueError("planner dashboard availability detail is invalid")
            old_index = int(detail[0])
            if old_index not in index_remap:
                continue
            remapped = list(detail)
            remapped[0] = index_remap[old_index]
            projected_details.append(remapped)
        projected[availability_index] = projected_details
        filtered_rows.append(projected)
    return (filtered_header, filtered_rows, definitions_text)


def _materialize_planner_metric_view(
    *, source_view: Path, batch_output: Path, assignment: dict[str, Any]
) -> Path:
    """Clone one frozen view and project only its compact metric files."""
    view = batch_output / "metric_view"
    shutil.copytree(source_view, view, copy_function=os.link)
    source_private_root = _private_history_path(source_view).parent
    if source_private_root.is_dir():
        shutil.copytree(
            source_private_root,
            _private_history_path(view).parent,
            copy_function=os.link,
        )
    history = view / "search_history"
    dashboard_path = history / "dashboard_llm.jsonl"
    definitions_path = history / "metric_definitions.json"
    header, rows = _read_llm_dashboard(dashboard_path)
    definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
    filtered_header, filtered_rows, definitions_text = _filter_metric_dashboard(
        header=header,
        rows=rows,
        metric_ids=list(assignment["metric_ids"]),
        definitions=definitions,
    )
    definitions_path.unlink()
    definitions_path.write_text(definitions_text, encoding="utf-8")
    dashboard_path.unlink()
    dashboard_path.write_text(
        "".join(
            (
                json.dumps(
                    record, sort_keys=isinstance(record, dict), separators=(",", ":")
                )
                + "\n"
                for record in [filtered_header, *filtered_rows]
            )
        ),
        encoding="utf-8",
    )
    (batch_output / "metric_subset.json").write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return view


def _planner_waves(mode: str, planner_calls: int) -> list[list[int]]:
    if mode != "independent":
        raise ValueError("planners run independently")
    return [list(range(planner_calls))]


def _verbalized_distribution_profiles(
    planner_calls: int, directions_per_call: int, *, mixed: bool = False
) -> list[dict[str, Any]]:
    if mixed or (planner_calls, directions_per_call) != (4, 8):
        raise ValueError(
            "the released planner uses four calls with eight directions each"
        )
    return [{"mode": "verbalized-distribution"} for _ in range(planner_calls)]


def _sampling_prompt_block(profile: dict[str, Any]) -> str:
    if profile != {"mode": "verbalized-distribution"}:
        raise ValueError("unsupported sampling profile")
    return (
        release_resource("verbalized_sampling.md")
        .read_text(encoding="utf-8")
        .rstrip("\n")
        .replace("{mixture}", "")
    )


def _target_sampling_shape(mode: str, *, directions_per_call: int) -> None:
    if mode != "none":
        raise ValueError("separate target sampling is not supported")
    return None


def _validate_distribution_sampling_metadata(
    document: Any, *, expected_count: int
) -> list[dict[str, Any]]:
    directions = document.get("directions") if isinstance(document, dict) else None
    if not isinstance(directions, list) or len(directions) != expected_count:
        raise ValueError("verbalized-distribution returned an invalid direction count")
    metadata = []
    for index, direction in enumerate(directions):
        if not isinstance(direction, dict):
            raise ValueError(f"direction {index} is not an object")
        probability = direction.get("verbalized_probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or (not math.isfinite(float(probability)))
            or (not 0.0 <= float(probability) <= 1.0)
        ):
            raise ValueError(f"direction {index} has invalid verbalized_probability")
        metadata.append({"verbalized_probability": float(probability)})
    return metadata


def _execution_directions(directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: direction[key] for key in DIRECTION_KEYS} for direction in directions]


def _prompt_for_count(
    template: str,
    count: int,
    *,
    objective_block: str | None = None,
    target_sampling_shape: dict[str, int] | None = None,
    causal_diversity: bool = False,
) -> str:
    if (
        objective_block is not None
        or target_sampling_shape is not None
        or causal_diversity
    ):
        raise ValueError("the released planner uses its shared production objective")
    prompt = template.replace("{{num_directions}}", str(count))
    if "{{num_directions}}" in prompt:
        raise RuntimeError("direction-count placeholder was not fully resolved")
    return prompt


def _load_prior_failed_interventions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or not all(
        (isinstance(item, dict) for item in document)
    ):
        raise ValueError("prior failed interventions must be a JSON list of objects")
    return document


def _planner_backends(
    *,
    raw: str | None,
    planner_calls: int,
    model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
) -> list[dict[str, str]]:
    default = {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
    }
    if raw is None:
        return [dict(default) for _ in range(planner_calls)]
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--planner-backends-json is not valid JSON") from exc
    if not isinstance(document, list) or len(document) != planner_calls:
        raise ValueError(
            "--planner-backends-json must contain one object per planner call"
        )
    required = set(default)
    backends: list[dict[str, str]] = []
    for index, item in enumerate(document):
        if not isinstance(item, dict) or not required <= set(item):
            raise ValueError(f"planner backend {index} must define {sorted(required)}")
        backend = {key: item[key] for key in default}
        if not all((isinstance(value, str) and value for value in backend.values())):
            raise ValueError(f"planner backend {index} contains an empty value")
        backends.append(backend)
    return backends


def _run_batch(
    *,
    batch_index: int,
    count: int,
    planning_objective_name: str,
    planning_objective_block: str | None,
    view: Path,
    prompt_template: str,
    output: Path,
    executable: Path,
    model: str,
    provider: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: float,
    format_attempts: int,
    catalog: dict[str, Any],
    evidence_ids: set[str],
    require_dashboard: bool,
    max_history_unfolds: int | None,
    unfold_selection_mode: str,
    prior_failed_interventions: list[dict[str, Any]],
    sampling_profile: dict[str, Any] | None = None,
    target_sampling_shape: dict[str, int] | None = None,
    causal_diversity: bool = False,
    coordination_wave: int = 1,
    metric_subspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    batch_output = output / f"planner_{batch_index:02d}"
    batch_output.mkdir()
    if metric_subspace is not None:
        view = _materialize_planner_metric_view(
            source_view=view, batch_output=batch_output, assignment=metric_subspace
        )
    base_prompt = _prompt_for_count(
        prompt_template,
        count,
        objective_block=planning_objective_block,
        target_sampling_shape=target_sampling_shape,
        causal_diversity=causal_diversity,
    )
    if prior_failed_interventions:
        base_prompt += (
            "\n\n"
            + PRIOR_FAILED_INTERVENTIONS_HEADING
            + "\n\nThese directions were executed during the current plateau and did not produce a meaningful frontier improvement. Treat them as measured negative evidence. Do not repeat the same causal mechanism unless a concrete, testable difference could change the prior outcome.\n\n"
            + json.dumps(prior_failed_interventions, indent=2, sort_keys=True)
            + "\n"
        )
    if sampling_profile is not None:
        base_prompt += "\n\n" + _sampling_prompt_block(sampling_profile) + "\n"
    previous: dict[str, Any] | None = None
    errors: list[str] = []
    attempt_metadata: list[dict[str, Any]] = []
    session_attempt_metadata: list[dict[str, Any]] = []
    session_id = str(uuid.uuid4())
    session_home = batch_output / "pi-session-home"
    session_tmp = batch_output / "pi-session-tmp"
    for attempt in range(1, format_attempts + 1):
        attempt_output = batch_output / f"attempt_{attempt}"
        attempt_output.mkdir()
        user_message = None
        if errors:
            user_message = f"The previous portfolio failed controller validation: {errors[-1]}. Correct that exact failure in this same session. Reuse the evidence and tool context already gathered, perform any missing read or compare calls, and return the complete corrected JSON object only."
            if previous is not None:
                user_message += " Keep the previous portfolio unchanged except where the validation failure or newly read evidence requires a correction."
        candidate: dict[str, Any] | None = None
        sampling_metadata: list[dict[str, Any]] = []
        target_sampling_metadata: list[dict[str, Any]] = []
        try:
            candidate, metadata = run_pi(
                view=view,
                prompt=base_prompt,
                output=attempt_output,
                executable=executable,
                model=model,
                provider=provider,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_seconds=timeout_seconds,
                num_directions=count,
                unfold_selection_mode=unfold_selection_mode,
                require_dashboard=require_dashboard,
                session_id=session_id,
                session_home=session_home,
                session_tmp=session_tmp,
                user_message=user_message,
            )
            metadata["direction_field_normalizations"] = (
                normalize_portfolio_direction_fields(candidate)
            )
            metadata["evidence_id_normalizations"] = normalize_portfolio_evidence_ids(
                candidate, evidence_ids
            )
            attempt_metadata.append(dict(metadata))
            session_attempt_metadata.append(dict(metadata))
            cumulative = merge_pi_evidence_metadata(session_attempt_metadata)
            validation_candidate = copy.deepcopy(candidate)
            if sampling_profile is not None:
                if sampling_profile.get("mode") == "verbalized-distribution":
                    sampling_metadata = _validate_distribution_sampling_metadata(
                        candidate, expected_count=count
                    )
                else:
                    raise ValueError("unsupported sampling profile")
                for direction in validation_candidate["directions"]:
                    direction.pop("verbalized_probability", None)
            candidate = validate_portfolio(
                validation_candidate,
                catalog=catalog,
                evidence_ids=evidence_ids,
                num_directions=count,
            )
            reads = validate_reads(
                candidate,
                cumulative["successful_read_paths"],
                view=view,
                require_dashboard=require_dashboard,
                max_history_unfolds=max_history_unfolds,
                tool_names=cumulative["tool_names"],
                direct_read_paths=cumulative["direct_read_paths"],
                tool_internal_read_paths=cumulative["tool_internal_read_paths"],
                unfold_selection_mode=unfold_selection_mode,
                read_events=cumulative["read_events"],
                compare_events=cumulative["compare_events"],
                unfold_selection_events=cumulative["unfold_selection_events"],
            )
            attempt_metadata[-1].update(reads)
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            failed_metadata = getattr(exc, "pi_metadata", None)
            if failed_metadata is not None:
                attempt_metadata.append(dict(failed_metadata))
                session_attempt_metadata.append(dict(failed_metadata))
            previous = candidate if isinstance(candidate, dict) else None
            errors.append(f"attempt_{attempt}: {exc}")
            continue
        result = {
            "batch_index": batch_index,
            "coordination_wave": coordination_wave,
            "planning_objective": planning_objective_name,
            "sampling_profile": sampling_profile,
            "target_sampling_shape": target_sampling_shape,
            "target_sampling_metadata": target_sampling_metadata,
            "metric_subspace": metric_subspace,
            "planner_backend": {
                "model": model,
                "provider": provider,
                "base_url": base_url,
                "api_key_env": api_key_env,
            },
            "direction_sampling_metadata": sampling_metadata,
            "requested_directions": count,
            "portfolio": candidate,
            "metadata": {**metadata, **cumulative, **reads},
            "attempt_metadata": attempt_metadata,
            "format_attempt": attempt,
            "format_errors": errors,
        }
        (batch_output / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    raise RuntimeError(
        f"planner batch {batch_index} remained invalid after {format_attempts} attempts: {'; '.join(errors)}"
    )


def _usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "catalog_cost_estimate_usd",
        "read_count",
        "history_unfold_count",
    )
    return {
        field: sum(
            (
                float(metadata.get(field) or 0)
                for result in results
                for metadata in result["attempt_metadata"]
            )
        )
        for field in fields
    }


def _copy_prepared_view(
    *,
    source: Path,
    output: Path,
    prompt_path: Path,
    num_directions: int,
    isolate_raw_history: bool = False,
    dashboard_root: Path | None = None,
) -> tuple[Path, dict[str, Any], set[str]]:
    """Copy one frozen view, optionally replacing only its metric dashboard."""
    view = output / "bundle"
    shutil.copytree(source, view)
    history_dirs = sorted((view / "search_history").glob("seq_*"))
    source_private_history = _private_history_path(source)
    if source_private_history.is_dir():
        history_dirs.extend(sorted(source_private_history.glob("seq_*")))
    history_sequences: set[int] = set()
    for path in history_dirs:
        if not path.is_dir():
            continue
        try:
            history_sequences.add(int(path.name.removeprefix("seq_")))
        except ValueError as exc:
            raise RuntimeError(f"invalid prepared history directory: {path}") from exc
    sanitize_planner_view(view, isolate_raw_history=isolate_raw_history)
    if isolate_raw_history and source_private_history.is_dir():
        target_private_history = _private_history_path(view)
        target_private_history.mkdir(parents=True, exist_ok=True)
        for history_dir in sorted(source_private_history.glob("seq_*")):
            target = target_private_history / history_dir.name
            if not target.exists():
                shutil.copytree(history_dir, target)
    source_private_profiles = _private_engine_profiles_path(source)
    if source_private_profiles.is_file():
        target_private_profiles = _private_engine_profiles_path(view)
        target_private_profiles.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_private_profiles, target_private_profiles)
    catalog_path = view / "workspace" / "meta" / "branch_catalog.json"
    if not catalog_path.is_file():
        raise RuntimeError("prepared view lacks branch_catalog.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cutoff_seq = int(catalog["cutoff_seq"])
    if dashboard_root is None:
        evidence_ids = {
            f"attempt:seq:{seq:08d}" for seq in history_sequences if seq <= cutoff_seq
        }
    else:
        dashboard_path = dashboard_root / "dashboard.jsonl"
        definitions_path = dashboard_root / "metric_definitions.json"
        header, observations = validate_dashboard_jsonl(
            dashboard_path,
            definitions_path=definitions_path,
            expected_cutoff_seq=cutoff_seq,
        )
        evidence_ids = {
            f"attempt:seq:{int(observation['cutoff_seq']):08d}"
            for observation in observations
        }
        for evidence_id in evidence_ids:
            seq = int(evidence_id.rsplit(":", 1)[1])
            if seq not in history_sequences:
                raise RuntimeError(
                    f"replacement dashboard lacks prepared evidence: {evidence_id}"
                )
        history = view / "search_history"
        shutil.copy2(definitions_path, history / "metric_definitions.json")
        write_llm_dashboard_jsonl(history / "dashboard_llm.jsonl", header, observations)
    required = (
        view / "search_history" / "dashboard_llm.jsonl",
        view / "search_history" / "metric_definitions.json",
        view / "workspace" / "reference" / "problem.py",
        view / "workspace" / "reference" / "evaluator.py",
    )
    missing = [str(path.relative_to(view)) for path in required if not path.is_file()]
    if missing or not evidence_ids:
        raise RuntimeError(
            "prepared view is incomplete: "
            + (", ".join(missing) if missing else "no causal history")
        )
    prompt = _prompt_for_count(prompt_path.read_text(encoding="utf-8"), num_directions)
    (output / "prompt.md").write_text(prompt, encoding="utf-8")
    (output / "materialization.json").write_text(
        json.dumps(
            {
                "schema": "oe_prepared_stage2_view_v1",
                "cutoff_seq": cutoff_seq,
                "history_mode": "prepared_dashboard",
                "live_branch_count": len(catalog.get("programs") or ()),
                "num_directions": num_directions,
                "raw_history_visibility": "controller_private"
                if isolate_raw_history
                else "planner",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (view, catalog, evidence_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path)
    parser.add_argument("--dashboard-root", type=Path)
    parser.add_argument(
        "--prepared-view",
        type=Path,
        help="already-frozen bundle containing dashboard, raw evidence, live branch catalog, and references",
    )
    parser.add_argument("--prior-failed-interventions", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=release_resource("planner.md"))
    parser.add_argument(
        "--candidate-pool-size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE
    )
    parser.add_argument(
        "--directions-per-call", type=int, default=DEFAULT_DIRECTIONS_PER_CALL
    )
    parser.add_argument("--dedup-model", default=DEFAULT_DEDUP_MODEL)
    parser.add_argument("--dedup-base-url", default=DEFAULT_DEDUP_BASE_URL)
    parser.add_argument("--dedup-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--dedup-max-tokens", type=int, default=DEFAULT_DEDUP_MAX_TOKENS
    )
    parser.add_argument("--planner-workers", type=int, default=None)
    parser.add_argument(
        "--planner-random-metric-count",
        type=int,
        default=0,
        help="number of reproducibly sampled metrics visible to each planner; 0 keeps the shared dashboard unchanged",
    )
    parser.add_argument("--metric-subspace-seed", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--planner-backends-json",
        help="JSON list of per-call model/provider/base_url/api_key_env objects",
    )
    parser.add_argument("--pi", default=os.environ.get("AUDIT_PI_CLI", "pi"))
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--format-attempts", type=int, default=DEFAULT_FORMAT_ATTEMPTS)
    args = parser.parse_args()
    if args.prepared_view is None:
        if args.context_root is None or args.checkpoint is None:
            parser.error("--context-root and --checkpoint are required")
        if False == (args.dashboard_root is not None):
            parser.error("provide exactly one of --dashboard-root or --no-dashboard")
    elif (
        any((value is not None for value in (args.context_root, args.checkpoint)))
        or False
    ):
        parser.error(
            "--prepared-view cannot be combined with context/checkpoint/no-dashboard inputs"
        )
    for name in ("candidate_pool_size", "directions_per_call", "format_attempts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.dedup_max_tokens < 1:
        parser.error("--dedup-max-tokens must be positive")
    if args.planner_random_metric_count < 0:
        parser.error("--planner-random-metric-count must be non-negative")
    planner_calls = math.ceil(args.candidate_pool_size / args.directions_per_call)
    try:
        target_sampling_shape = _target_sampling_shape(
            "none", directions_per_call=args.directions_per_call
        )
    except ValueError as exc:
        parser.error(str(exc))
    if target_sampling_shape is not None:
        if args.candidate_pool_size % args.directions_per_call:
            parser.error("verbalized target sampling requires full planner batches")
    try:
        planner_backends = _planner_backends(
            raw=args.planner_backends_json,
            planner_calls=planner_calls,
            model=args.model,
            provider=args.provider,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.candidate_pool_size != 32:
        parser.error("verbalized-distribution requires --candidate-pool-size 32")
    try:
        sampling_profiles = _verbalized_distribution_profiles(
            planner_calls,
            args.directions_per_call,
            mixed="verbalized-distribution" == "verbalized-distribution-mix",
        )
    except ValueError as exc:
        parser.error(str(exc))
    planning_objectives = [("shared", None)] * planner_calls
    planner_workers = args.planner_workers or planner_calls
    if planner_workers < 1:
        parser.error("--planner-workers must be positive")
    if (
        "verbalized-distribution"
        in {"verbalized-distribution", "verbalized-distribution-mix"}
        and planner_workers < 2
    ):
        parser.error("verbalized-distribution requires at least 2 planner workers")
    try:
        prompt_template = args.prompt.read_text(encoding="utf-8")
        _prompt_for_count(
            prompt_template,
            args.directions_per_call,
            target_sampling_shape=target_sampling_shape,
            causal_diversity=False,
        )
        if "verbalized-distribution" in {
            "verbalized-distribution",
            "verbalized-distribution-mix",
        } and prompt_template != PLANNER_PROMPT.read_text(encoding="utf-8"):
            parser.error("verbalized-distribution requires the production prompt")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    executable = shutil.which(args.pi)
    if executable is None:
        parser.error(f"Pi CLI not found: {args.pi}")
    prior_failed_interventions = _load_prior_failed_interventions(
        args.prior_failed_interventions
    )
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepared_view is not None:
        view, catalog, evidence_ids = _copy_prepared_view(
            source=args.prepared_view,
            output=args.output,
            prompt_path=args.prompt,
            num_directions=args.directions_per_call,
            isolate_raw_history="registered" != "direct",
            dashboard_root=args.dashboard_root,
        )
    else:
        view, _, catalog, evidence_ids = materialize_view(
            context_root=args.context_root,
            dashboard_root=args.dashboard_root,
            checkpoint=args.checkpoint,
            output=args.output,
            prompt_path=args.prompt,
            num_directions=args.directions_per_call,
            isolate_raw_history="registered" != "direct",
        )
    metric_subspaces: list[dict[str, Any]] = []
    if args.planner_random_metric_count:
        dashboard_header, _ = _read_llm_dashboard(
            view / "search_history" / "dashboard_llm.jsonl"
        )
        metric_subspaces = _balanced_metric_subspaces(
            metric_order=[str(item) for item in dashboard_header["metric_order"]],
            planner_calls=planner_calls,
            metric_count=args.planner_random_metric_count,
            seed=args.metric_subspace_seed,
        )
    started = time.monotonic()
    counts = [
        min(
            args.directions_per_call,
            args.candidate_pool_size - index * args.directions_per_call,
        )
        for index in range(planner_calls)
    ]
    results: list[dict[str, Any]] = []
    planner_waves = _planner_waves("independent", planner_calls)
    for wave_number, batch_indices in enumerate(planner_waves, start=1):
        wave_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=min(planner_workers, len(batch_indices))
        ) as executor:
            futures = {
                executor.submit(
                    _run_batch,
                    batch_index=index,
                    count=counts[index],
                    planning_objective_name=planning_objectives[index][0],
                    planning_objective_block=planning_objectives[index][1],
                    sampling_profile=sampling_profiles[index],
                    target_sampling_shape=target_sampling_shape,
                    causal_diversity=False,
                    coordination_wave=wave_number,
                    view=view,
                    prompt_template=prompt_template,
                    output=args.output,
                    executable=Path(executable).absolute(),
                    model=planner_backends[index]["model"],
                    provider=planner_backends[index]["provider"],
                    base_url=planner_backends[index]["base_url"],
                    api_key_env=planner_backends[index]["api_key_env"],
                    timeout_seconds=args.timeout_seconds,
                    format_attempts=args.format_attempts,
                    catalog=catalog,
                    evidence_ids=evidence_ids,
                    require_dashboard=not False,
                    max_history_unfolds=None,
                    unfold_selection_mode="registered",
                    prior_failed_interventions=prior_failed_interventions,
                    metric_subspace=metric_subspaces[index]
                    if metric_subspaces
                    else None,
                ): index
                for index in batch_indices
            }
            for future in as_completed(futures):
                wave_results.append(future.result())
        wave_results.sort(key=lambda result: int(result["batch_index"]))
        results.extend(wave_results)
    results.sort(key=lambda result: int(result["batch_index"]))
    directions: list[dict[str, Any]] = []
    for result in results:
        batch_index = int(result["batch_index"])
        for local_index, direction in enumerate(result["portfolio"]["directions"]):
            sampling = (
                result["direction_sampling_metadata"][local_index]
                if result["direction_sampling_metadata"]
                else {}
            )
            directions.append(
                {
                    **direction,
                    **{
                        key: sampling[key]
                        for key in ("type", "verbalized_probability")
                        if key in sampling
                    },
                    "direction_id": f"batch_{batch_index:02d}/direction_{local_index:02d}",
                }
            )
    if len(directions) != args.candidate_pool_size:
        raise RuntimeError(
            f"planner pool has {len(directions)} directions, expected {args.candidate_pool_size}"
        )
    all_exhausted: list[str] = []
    for result in results:
        for pattern in result["portfolio"]["exhausted_patterns"]:
            if pattern not in all_exhausted:
                all_exhausted.append(pattern)
    candidate_pool_path = args.output / "candidate_pool.json"
    candidate_pool_path.write_text(
        json.dumps(
            {
                "exhausted_patterns": all_exhausted,
                "planner_coordination_mode": "independent",
                "planner_objectives": [
                    {
                        "batch_index": int(result["batch_index"]),
                        "name": result["planning_objective"],
                    }
                    for result in results
                ],
                "prior_failed_interventions": prior_failed_interventions,
                "directions": directions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    deduplication = deduplicate_directions(
        directions,
        output=args.output / "llm_dedup",
        model=args.dedup_model,
        base_url=args.dedup_base_url,
        api_key_env=args.dedup_api_key_env,
        max_tokens=args.dedup_max_tokens,
        timeout_seconds=args.timeout_seconds,
        candidate_pool=candidate_pool_path,
        exhausted_patterns=all_exhausted,
        prior_failed_interventions=prior_failed_interventions,
    )
    by_id = {direction["direction_id"]: direction for direction in directions}
    selected_directions = [
        by_id[direction_id] for direction_id in deduplication["kept_direction_ids"]
    ]
    selection = {
        "selection_method": "llm",
        **deduplication,
        "selected_count": len(selected_directions),
        "selected_direction_ids": deduplication["kept_direction_ids"],
        "selected_directions": selected_directions,
    }
    selected = _execution_directions(selection["selected_directions"])
    exhausted = all_exhausted[:3]
    portfolio = {"exhausted_patterns": exhausted, "directions": selected}
    run = {
        "schema": RUN_SCHEMA,
        "candidate_pool_size": args.candidate_pool_size,
        "directions_per_planner_call": args.directions_per_call,
        "planner_call_count": planner_calls,
        "planner_coordination_mode": "independent",
        "planner_waves": planner_waves,
        "planner_sampling_mode": "verbalized-distribution",
        "planner_sampling_profiles": sampling_profiles,
        "planner_target_sampling": "none",
        "planner_target_sampling_shape": target_sampling_shape,
        "planner_causal_diversity": False,
        "planner_routing_mode": "shared",
        "planner_objective_mix": {
            "shared": sum(
                (result["planning_objective"] == "shared" for result in results)
            ),
            "frontier_escape": sum(
                (
                    result["planning_objective"] == "frontier_escape"
                    for result in results
                )
            ),
            "breadth_expansion": sum(
                (
                    result["planning_objective"] == "breadth_expansion"
                    for result in results
                )
            ),
        },
        "actual_planner_session_count": sum(
            (len(result["attempt_metadata"]) for result in results)
        ),
        "format_retry_count": sum(
            (len(result["attempt_metadata"]) - 1 for result in results)
        ),
        "planner_workers": planner_workers,
        "planner_backends": planner_backends,
        "direction_selector": "llm",
        "unfold_selection_mode": "registered",
        "max_selected_directions": None,
        "actual_selected_directions": len(selected),
        "prior_failed_intervention_count": len(prior_failed_interventions),
        "embedding_backend": selection.get("embedding_backend"),
        "embedding_model": None,
        "semantic_duplicate_threshold": None,
        "dedup_model": args.dedup_model,
        "dedup_base_url": args.dedup_base_url,
        "dedup_api_key_env": args.dedup_api_key_env,
        "dedup_max_tokens": args.dedup_max_tokens,
        "dedup_usage": selection.get("usage"),
        "model": args.model,
        "provider": args.provider,
        "duration_seconds": time.monotonic() - started,
        "usage": _usage(results),
        "planner_results": [
            {
                "batch_index": result["batch_index"],
                "coordination_wave": result["coordination_wave"],
                "planning_objective": result["planning_objective"],
                "sampling_profile": result["sampling_profile"],
                "target_sampling_shape": result["target_sampling_shape"],
                "target_sampling_metadata": result["target_sampling_metadata"],
                "direction_sampling_metadata": result["direction_sampling_metadata"],
                "requested_directions": result["requested_directions"],
                "format_attempt": result["format_attempt"],
                "format_errors": result["format_errors"],
            }
            for result in results
        ],
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "portfolio.json").write_text(
        json.dumps(portfolio, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
