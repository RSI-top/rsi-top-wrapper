"""Numeric validation shared by the history and dashboard builders."""

from __future__ import annotations

import math

import sys

from pathlib import Path

from typing import Any

HERE = Path(__file__).resolve().parent

SHARED = HERE.parents[1] / "shared" / "meta-agent-audit"

for entry in (str(HERE), str(SHARED)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

STATE_WINDOW = 12

TOOL_CALL_LIMIT = 100


class ExperimentError(RuntimeError):
    pass


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
