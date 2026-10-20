"""Canonical JSON serialization and hashing helpers.

Every JSON artifact inside an audit bundle is serialized with
:func:`canonical_json` so that identical logical content always produces
identical bytes (and therefore identical digests). Rules:

- object keys sorted, compact separators, UTF-8, ``allow_nan=False``;
- values that cannot be serialized deterministically raise
  :class:`CanonicalizationError` (fail loud at freeze time rather than
  silently produce a nondeterministic bundle).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "CanonicalizationError",
    "canonical_json",
    "canonical_json_bytes",
    "ceil_div",
    "sha256_hex",
]


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonically serialized as JSON."""


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to the canonical JSON text form used across bundles."""
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(
            f"value is not canonically JSON-serializable: {exc}"
        ) from exc


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON encoded as UTF-8 bytes."""
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division."""
    return -(-numerator // denominator)
