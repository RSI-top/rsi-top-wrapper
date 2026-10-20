"""Implementation fingerprints and distances used by the M25 dashboard."""

from __future__ import annotations

import difflib

import io

import math

import re

import tokenize

from collections import Counter

from decimal import Decimal, localcontext

from functools import lru_cache

from typing import Any, Mapping, Sequence

STATE_WINDOW = 12

_CANONICAL_PREC = 40

_WS_RUN_RE = re.compile(r"\s+")

_SYNTAX_ONLY_RE = re.compile(r"[^\w]+\Z")


class QuantitativeStateError(RuntimeError):
    """Malformed state, non-monotonic completion input, or root mismatch."""


def _comment_starts(text: str) -> dict[int, int]:
    starts: dict[int, int] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                row, column = token.start
                if row not in starts or column < starts[row]:
                    starts[row] = column
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return starts
    return starts


@lru_cache(maxsize=4096)
def _normalized_source_lines(text: str) -> tuple[str, ...]:
    """DeSearch normalizer-v1, byte-for-byte semantic port."""
    starts = _comment_starts(text)
    output: list[str] = []
    for row, line in enumerate(text.splitlines(), 1):
        if row in starts:
            line = line[: starts[row]]
        line = _WS_RUN_RE.sub(" ", line.strip())
        if not line or line.startswith("#"):
            continue
        if _SYNTAX_ONLY_RE.match(line):
            continue
        output.append(line)
    return tuple(output)


Fingerprint = Counter[tuple[str, str, str]]


@lru_cache(maxsize=8192)
def _signed_fingerprint_cached(root_code: str, program_code: str) -> Fingerprint:
    """Signed cumulative normalized diff from the run root."""
    base = _normalized_source_lines(root_code)
    node = _normalized_source_lines(program_code)
    output: Fingerprint = Counter()
    matcher = difflib.SequenceMatcher(None, base, node, autojunk=False)
    path = "initial_program.py"
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            for line in base[i1:i2]:
                output[("-", path, line)] += 1
        if tag in ("replace", "insert"):
            for line in node[j1:j2]:
                output[("+", path, line)] += 1
    return output


def signed_fingerprint(root_code: str, program_code: str) -> Fingerprint:
    """Return the process-cached immutable-by-convention fingerprint."""
    return _signed_fingerprint_cached(root_code, program_code)


def _fingerprint_key(fingerprint: Mapping[Any, int]) -> tuple[tuple[Any, int], ...]:
    key = getattr(fingerprint, "_desearch_cache_key", None)
    if key is None:
        key = tuple(sorted((item, int(count)) for item, count in fingerprint.items()))
        # Counter instances produced above are process-cache residents. Keeping
        # their canonical key beside them avoids sorting it for every overlapping
        # W12 and nearest-neighbour comparison.
        try:
            setattr(fingerprint, "_desearch_cache_key", key)
        except (AttributeError, TypeError):
            pass
    return key


@lru_cache(maxsize=262144)
def _weighted_multiset_jaccard_cached(
    left: tuple[tuple[Any, int], ...],
    right: tuple[tuple[Any, int], ...],
) -> float:
    left_map = dict(left)
    right_map = dict(right)
    keys = set(left_map) | set(right_map)
    if not keys:
        return 1.0
    intersection = sum(
        min(int(left_map.get(key, 0)), int(right_map.get(key, 0))) for key in keys
    )
    union = sum(
        max(int(left_map.get(key, 0)), int(right_map.get(key, 0))) for key in keys
    )
    return intersection / union if union else 1.0


def weighted_multiset_jaccard(
    left: Mapping[Any, int], right: Mapping[Any, int]
) -> float:
    left_key = _fingerprint_key(left)
    right_key = _fingerprint_key(right)
    if right_key < left_key:
        left_key, right_key = right_key, left_key
    return _weighted_multiset_jaccard_cached(left_key, right_key)


def _effective_similarity_count(
    similarities: Sequence[Sequence[float]],
) -> float:
    n = len(similarities)
    if n <= 0:
        raise QuantitativeStateError("Formula V2 requires a non-empty window")
    for i, row in enumerate(similarities):
        if len(row) != n:
            raise QuantitativeStateError("Formula V2 similarity matrix is not square")
        for j, value in enumerate(row):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise QuantitativeStateError("Formula V2 similarity is outside [0,1]")
            if i == j and not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise QuantitativeStateError("Formula V2 diagonal is not one")
            if j < i and not math.isclose(
                value, similarities[j][i], rel_tol=0.0, abs_tol=1e-12
            ):
                raise QuantitativeStateError("Formula V2 matrix is not symmetric")
    with localcontext() as context:
        context.prec = _CANONICAL_PREC
        probability = Decimal(1) / Decimal(n)
        log_mean_similarity = Decimal(0)
        for row in similarities:
            density = sum(
                (probability * Decimal(float(value)) for value in row),
                Decimal(0),
            )
            if density <= 0:
                raise QuantitativeStateError("Formula V2 density is non-positive")
            log_mean_similarity += probability * density.ln()
        return float((-log_mean_similarity).exp())


def implementation_spread(fingerprints: Sequence[Fingerprint]) -> float:
    n = len(fingerprints)
    if n <= 1:
        return 0.0
    similarities = [
        [weighted_multiset_jaccard(left, right) for right in fingerprints]
        for left in fingerprints
    ]
    effective = _effective_similarity_count(similarities)
    return min(1.0, max(0.0, (effective - 1.0) / (n - 1.0)))
