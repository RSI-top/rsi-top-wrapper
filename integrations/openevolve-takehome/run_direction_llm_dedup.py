#!/usr/bin/env python3
"""Select one frozen direction pool with one listwise LLM call."""

from __future__ import annotations

from release_paths import release_resource

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_TOKENS = 128000
DEFAULT_VALIDATION_ATTEMPTS = 3
DEFAULT_PROMPT = release_resource("deduplicate.md")


def _plain_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    document = json.loads(stripped)
    if not isinstance(document, dict):
        raise ValueError("deduplicator output must be a JSON object")
    return document


def _compact_direction(direction: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "direction_id",
        "target_branch_id",
        "donor_branch_ids",
        "mechanism_hypothesis",
        "task",
        "constraints",
        "evidence_attempt_ids",
    )
    return {field: direction.get(field) for field in fields}


def _render_prompt(
    template: str,
    directions: list[dict[str, Any]],
    exhausted_patterns: list[str],
    prior_failed_interventions: list[dict[str, Any]] | None = None,
) -> str:
    replacements = {
        "{{candidate_directions_json}}": json.dumps(
            [_compact_direction(direction) for direction in directions],
            indent=2,
            sort_keys=True,
        ),
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    unresolved = [placeholder for placeholder in replacements if placeholder in prompt]
    if unresolved:
        raise ValueError(f"unresolved selector prompt placeholders: {unresolved}")
    return prompt


def validate_deduplication(
    document: dict[str, Any], directions: list[dict[str, Any]]
) -> dict[str, Any]:
    if set(document) != {"kept_direction_ids", "removed_directions"}:
        raise ValueError(
            "output must contain exactly kept_direction_ids/removed_directions"
        )
    known = [direction.get("direction_id") for direction in directions]
    if not all(isinstance(item, str) and item for item in known):
        raise ValueError("every candidate direction requires a direction_id")
    if len(set(known)) != len(known):
        raise ValueError("candidate direction IDs must be unique")

    kept = document["kept_direction_ids"]
    removed = document["removed_directions"]
    if not isinstance(kept, list) or not kept:
        raise ValueError("kept_direction_ids must be a non-empty list")
    if not all(isinstance(item, str) and item in known for item in kept):
        raise ValueError("kept_direction_ids contains an unknown ID")
    if len(set(kept)) != len(kept):
        raise ValueError("kept_direction_ids contains duplicates")
    if not isinstance(removed, list):
        raise ValueError("removed_directions must be a list")

    removed_ids: list[str] = []
    normalized_removed: list[dict[str, Any]] = []
    for item in removed:
        if not isinstance(item, dict) or set(item) != {
            "direction_id",
            "duplicate_of",
            "reason",
        }:
            raise ValueError(
                "each removed direction requires direction_id/duplicate_of/reason"
            )
        direction_id = item["direction_id"]
        duplicate_of = item["duplicate_of"]
        reason = item["reason"]
        if direction_id not in known or direction_id in kept:
            raise ValueError("removed direction is unknown or also kept")
        if not isinstance(duplicate_of, str) or duplicate_of not in kept:
            raise ValueError("duplicate_of must identify a kept direction")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("removed direction reason must be non-empty")
        removed_ids.append(direction_id)
        normalized_removed.append(
            {
                "direction_id": direction_id,
                "duplicate_of": duplicate_of,
                "reason": reason.strip(),
            }
        )
    if len(set(removed_ids)) != len(removed_ids):
        raise ValueError("removed_directions contains duplicate IDs")
    if set(kept) | set(removed_ids) != set(known):
        raise ValueError("kept and removed IDs do not partition the candidate pool")
    return {
        "kept_direction_ids": list(kept),
        "removed_directions": normalized_removed,
    }


def deduplicate_directions(
    directions: list[dict[str, Any]],
    *,
    output: Path,
    prompt_path: Path = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = "DEEPSEEK_API_KEY",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: float = 1800.0,
    candidate_pool: Path | None = None,
    exhausted_patterns: list[str] | None = None,
    prior_failed_interventions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select a frozen pool, retrying malformed responses without changing input."""
    if not directions:
        raise ValueError("candidate pool requires a non-empty directions list")
    exhausted_patterns = list(exhausted_patterns or [])
    prior_failed_interventions = list(prior_failed_interventions or [])
    prompt = _render_prompt(
        prompt_path.read_text(encoding="utf-8"),
        directions,
        exhausted_patterns,
        prior_failed_interventions,
    )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=2,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    started = time.monotonic()
    validation_errors: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(1, DEFAULT_VALIDATION_ATTEMPTS + 1):
        attempt_output = output / f"attempt_{attempt}"
        attempt_output.mkdir()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        raw = (
            json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n"
        )
        (attempt_output / "raw_response.json").write_text(raw, encoding="utf-8")
        (output / "raw_response.json").write_text(raw, encoding="utf-8")
        usage["input_tokens"] += int(getattr(response.usage, "prompt_tokens", 0) or 0)
        usage["output_tokens"] += int(
            getattr(response.usage, "completion_tokens", 0) or 0
        )
        try:
            if not response.choices:
                raise ValueError(
                    f"deduplicator returned no choices: {getattr(response, 'error', None)!r}"
                )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError(
                    "deduplicator returned empty content "
                    f"(finish_reason={response.choices[0].finish_reason!r})"
                )
            result = validate_deduplication(_plain_json(content), directions)
        except ValueError as exc:
            validation_errors.append(str(exc))
            (attempt_output / "validation_error.json").write_text(
                json.dumps({"attempt": attempt, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            if attempt == DEFAULT_VALIDATION_ATTEMPTS:
                raise ValueError(
                    f"deduplicator remained invalid after {attempt} attempts: {exc}"
                ) from exc
            time.sleep(5)
        else:
            break
    artifact = {
        "schema": "oe_direction_llm_dedup_v1",
        "model": model,
        "candidate_count": len(directions),
        "kept_count": len(result["kept_direction_ids"]),
        "removed_count": len(result["removed_directions"]),
        "duration_seconds": time.monotonic() - started,
        "usage": usage,
        "validation_attempt": attempt,
        "validation_errors": validation_errors,
        **result,
    }
    if candidate_pool is not None:
        artifact["candidate_pool"] = str(candidate_pool.resolve())
    (output / "result.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    pool = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
    directions = pool.get("directions") if isinstance(pool, dict) else None
    if not isinstance(directions, list) or not directions:
        raise ValueError("candidate pool requires a non-empty directions list")
    artifact = deduplicate_directions(
        directions,
        output=args.output,
        prompt_path=args.prompt,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        candidate_pool=args.candidate_pool,
        exhausted_patterns=pool.get("exhausted_patterns") or [],
        prior_failed_interventions=pool.get("prior_failed_interventions") or [],
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
