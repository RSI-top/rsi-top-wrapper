# Anthropic performance takehome snapshot

Source: [anthropics/original_performance_takehome](https://github.com/anthropics/original_performance_takehome).

This release retains the starter kernel and simulator for the Anthropic
performance takehome. Copyright and permission notices remain in the source.
Standalone upstream test suites are omitted. The unchanged frozen simulator is
stored in `frozen_problem.py`; the release evaluator uses it to check correctness
and measure cycles independently of candidate code.

From the release root, score a candidate with:

```bash
uv run python evaluate.py path/to/candidate.py
```

See the root README for installation and experiment commands.
