# RSI-top

Code and prompts for the Anthropic kernel optimization part of
[RSI-top: Escaping Exploration Collapse in Long-Horizon Autoresearch](https://rsi-top.github.io/).

RSI-top monitors an evolutionary search with a compact metric dashboard. When
progress stalls, four planners selectively inspect the history, propose new
directions, and return a parallel cohort of candidates to the native search.
This release includes **OpenEvolve** and **ShinkaEvolve**, their baseline runners,
the M25 dashboard, intervention prompts, and the frozen Anthropic evaluator.

## Installation

Use **Linux**, **Python 3.11**, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and **Node.js 22.19+ with npm**. No GPU is required: the benchmark runs a simulated
VLIW/SIMD machine on the CPU. Allow enough CPU and RAM for a burst of up to 32
candidate evaluations; ordinary search uses four workers.

From this directory:

```bash
./setup.sh
```

Setup installs the bundled, modified framework versions, the locked Python
dependencies, and Pi 0.84.1 locally. Pi is the planner's tool-calling runtime;
there is no plugin installation or interactive login.

## Quickstart

Supply your own API key, then launch a new run:

```bash
export DEEPSEEK_API_KEY='your-key'

# OpenEvolve + RSI-top, starting from the bundled starter program
uv run python run.py rsi-top \
  --framework openevolve --provider deepseek \
  --hours 12 --budget 400 --seed 42 --output runs/oe-rsi
```

To run the corresponding baseline:

```bash
uv run python run.py baseline \
  --framework openevolve --provider deepseek \
  --hours 12 --budget 400 --seed 42 --output runs/oe-baseline
```

For ShinkaEvolve, its original search profile also uses OpenRouter embeddings:

```bash
export OPENROUTER_API_KEY='your-key'

uv run python run.py rsi-top \
  --framework shinka --provider deepseek \
  --hours 12 --budget 400 --seed 42 --output runs/shinka-rsi

uv run python run.py baseline \
  --framework shinka --provider deepseek \
  --hours 12 --budget 400 --seed 42 --output runs/shinka-baseline
```

Commands run in the foreground and print the output and log paths. Each command
requires a **new output directory**, so rerunning an example cannot overwrite an
earlier experiment. Use `--dry-run` to inspect the model, required environment
variables, and exact command without credentials, output files, or API calls.

```bash
uv run python run.py rsi-top --framework shinka --output runs/preview --dry-run
```

## Method and prompts

The released path uses these local experiment defaults:

| Setting | Value |
| --- | --- |
| Ordinary concurrency | 4 |
| First intervention | After one hour, 20 active flat minutes, and 12 completed attempts |
| Meaningful progress | More than 0.2% fewer cycles relative to the progress anchor |
| Dashboard | 25 fixed metrics; recent windows contain 12 attempts/states |
| Planner cohort | 4 independent calls × 8 proposed directions |
| Metric views | 15 metrics sampled per planner with balanced coverage |
| Proposal sampling | Joint portfolios with verbalized probabilities |
| Evidence access | Registered, metric-attributed selective unfolding |
| Planner / dedup model | DeepSeek v4 Flash, high reasoning |
| Direction selection | Semantic dedup; execute every retained direction |
| Handoff | Existing work overlaps planning; native search resumes as the cohort drains |


| File | Purpose |
| --- | --- |
| [prompts/planner.md](prompts/planner.md) | Complete-objective intervention planner |
| [prompts/verbalized_sampling.md](prompts/verbalized_sampling.md) | Verbalized proposal-distribution instructions |
| [prompts/deduplicate.md](prompts/deduplicate.md) | Listwise semantic deduplication |
| [prompts/worker_openevolve.md](prompts/worker_openevolve.md) | OpenEvolve task instructions |
| [prompts/worker_shinka.md](prompts/worker_shinka.md) | Shinka task instructions |
| [prompts/metrics.json](prompts/metrics.json) | M25 metric definitions |
| [run_branch_portfolio_breadth.py](integrations/openevolve-takehome/run_branch_portfolio_breadth.py) | Portfolio, verbalized-sampling, and metric-subset instructions added at runtime |

## Inspect and score a result

Each run writes `launch.json`, `launcher.log`, and `result.json` at its root.
`result.json` reports process completion or failure, not benchmark correctness.

| Run | Best candidate |
| --- | --- |
| OpenEvolve baseline | `runs/oe-baseline/search/best/best_program.py` |
| OpenEvolve + RSI-top | `runs/oe-rsi/search/run/best/best_program.py` |
| Shinka baseline or RSI-top | `runs/<name>/search/best/best_program.py` |

Re-score any candidate without an API key:

```bash
uv run python evaluate.py runs/oe-rsi/search/run/best/best_program.py
```

The evaluator builds the instruction stream in one subprocess and replays its
plain JSON on the bundled `vendor/takehome/frozen_problem.py` in another. It checks output values for
`forest_height=10`, `rounds=16`, and `batch_size=256`, using seeds 1000 and 1001 by
default. **Lower cycles is better.** The framework selection score is a logarithmic
speedup score capped at 1000 cycles. The blog's `1363 / cycles` normalization is a
presentation metric, not the native search objective. An invalid candidate exits
with a nonzero status and reports the failure stage.

For independent final validation with the official submission harness:

```bash
uv run python verify.py runs/oe-rsi/search/run/best/best_program.py \
  --report runs/oe-rsi/official-verification.json
```

This fetches official takehome commit `5452f74bd977807ac2e74f3d29432b9df6f25197`
into a temporary clean checkout, copies only the candidate, and runs the unmodified
`tests/submission_tests.py`, including the official performance thresholds. Use
`--correctness-only` to skip the thresholds, or `--local-repo PATH` to verify
offline against a local clone of that commit. The harness executes candidate
Python directly; use trusted candidates or a separate sandbox.

## Tests

```bash
uv run --frozen --group test python -m pytest -q
```

The offline regression suite needs no API keys, model calls, or Node/Pi runtime.

## License

This repository is released under the [MIT License](LICENSE). Bundled upstream
components keep their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).
