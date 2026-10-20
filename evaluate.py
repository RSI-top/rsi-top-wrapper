#!/usr/bin/env python3
"""Score a KernelBuilder program with the bundled frozen simulator; no API calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    args = parser.parse_args()
    program = args.program.expanduser().resolve()
    if not program.is_file():
        parser.error(f"program not found: {program}")
    sys.path.insert(0, str(ROOT / "integrations/openevolve-takehome"))
    from evaluator import evaluate
    result = evaluate(str(program))
    valid = "failure_stage" not in result.artifacts
    print(json.dumps({"valid": valid, **result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
