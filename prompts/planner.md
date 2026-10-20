# Dynamic branch intervention portfolio

## Intervention objective

Ordinary search has stalled in its current neighborhood. This bounded
intervention must redirect search toward global progress. Use the integration
and exploration proposal types defined below to build a portfolio that both
seeks credible routes beyond the current frontier and opens causally distinct,
reusable measured states for future search. Routine novelty or a weak-parent
gain is insufficient.


## Scope

- The candidate is a KernelBuilder implementation whose emitted instruction stream runs a single-core simulated VLIW/SIMD tree-walk kernel. The fixed evaluated workload is forest_height=10, batch_size=256, and rounds=16.
- workspace/reference/problem.py defines ISA semantics and resource limits; workspace/reference/evaluator.py defines the build, correctness, and scoring contract.
- This is a standalone intervention-planning agent session. Infer the current
  diagnosis and exhausted patterns directly from the causal search state and
  evidence read during this session.
- Before planning, read workspace/reference/problem.py and
  workspace/reference/evaluator.py.
- You do not edit code in this planning call. Select exact live target branches and
  propose a small portfolio of bounded source-code experiments. Select donor
  branches only when integrating a mechanism already observed in history.

## Legal optimization boundary

- Only the candidate implementation may change. Do not modify or bypass the simulator, ISA, evaluator, reference kernel, tests, or cycle accounting.
- Do not propose monkeypatching or shadowing another module, test or benchmark introspection, environment-dependent behavior, test-discovery tampering, or precomputed output values. Inputs are freshly randomized.

## Branch catalog

- workspace/meta/branch_catalog.json lists every program still LIVE in the search database at this causal cutoff: id, parent id, island, generation, metrics/fitness, global/island-best flags, code path, and code hash.
- `target_branch_id` and every `donor_branch_id` MUST be ids from this catalog. Historical attempts that are no longer live may be cited as evidence but can never be a target or donor.
- Branch code is not inlined here. Before naming a target or donor, read its code through the read-only tools at its catalog code path.
- Compare the live global/island-best branches before choosing targets. Do not default every direction to the global best; reuse a target only when its code and history make it the best branch for that experiment.
- Do not restrict targets to current global/island bests; also inspect live branches created by strong measured parent gains or carrying verified mechanisms absent from the current best.

## History routing and selective unfolding

### Route

Before constructing the portfolio, first read
`search_history/metric_definitions.json` for the definitions and interpretation
rules of every dashboard metric. Then read the complete
`search_history/dashboard_llm.jsonl` from its header through the causal cutoff. The
two files form one interface:

1. Treat `metric_definitions.json` as the authoritative dictionary. Do not
   infer a metric's meaning from its ID alone.
2. The first dashboard line is its header. Align every later positional row
   with `header.row_columns`, then align `header.metric_order` with the five
   arrays in `metric_columns`: `values`,
   `percentiles`, `robust_z`, `status_codes`, and `reference_n`. Decode status
   characters with `header.status_codes`; `null` never means zero. Decode each
   compact availability entry with `header.availability_detail_columns` and
   `header.availability_reason_codes`.
3. `cutoff_seq=N` identifies `attempt:seq:{N:08d}`. Use that row's
   header path templates when unfolding its trajectory or available code. An
   integer reference identifies the row with that `cutoff_seq`; other values
   are literal references.
4. When present, align `header.mechanistic_profile_order` and each profile's
   declared columns with the row's `mechanistic_profile_values`; decode
   `mechanistic_profile_statuses` with the same status table.
5. Raw values describe measured state. Percentiles and robust-z values describe
   how unusual a value is relative to strictly earlier causal history. They are
   routing signals, not health labels or evidence by themselves.

Use the dashboard to select the evidence most useful for resolving the current
decision, then verify that evidence by unfolding its trajectory and code. Do not
treat dashboard values as conclusions by themselves.

After selecting and unfolding evidence, use `compare_attempts` on relevant
parent-child or target-donor pairs. Interpret profile differences together with
the exact code change and measured cycles. Profile values are mechanistic
context, not causal proof or a direction by themselves. If reusing a historical
transition, compare its child with its actual parent when both are available.
Do not read `search_history/engine_profiles.json` wholesale.

### Unfold and validate

Perform one shared selective-unfolding process for the whole portfolio; do not
restart the full routing process independently for every direction. Use the
shared unfolds to form candidate directions, then make additional targeted
unfolds only where a candidate lacks sufficient evidence or target-specific
applicability checks. The same unfolded case may support multiple directions
when it is genuinely relevant, but every integration direction must cite its
own supporting evidence and be checked against its exact target. Exploration
does not require positive historical evidence.

For every selected historical attempt, read the trajectory reconstructed from
the header path template. Read its available code when making a claim about its
source mechanism, structural repetition, or target compatibility. Every cited
`evidence_attempt_id` in the final portfolio must refer to an attempt whose
trajectory was unfolded. Historical evidence code is optional when trajectory
and metrics alone support the claim. A donor supplies a semantic mechanism or
failure contrast, not an automatically portable patch.

For every historical result considered for reuse, use the supplied metrics to
locate and reconstruct one compact local evidence subgraph:

```text
required ancestors
→ actual parent, with relevant inspirations or donors
→ focal child
→ strongest sibling counterexample, repair, or relevant descendant
```

Use the dashboard to choose the focal transition and related nodes, then inspect
the unfolded evidence to determine what happened and why. Read the trajectory
for every node used, and read its code only when the claim depends on the
implementation. Dashboard signals are routing and measured context, not proof.

Use selective unfolding to find as many distinct, relevant causal subgraphs as
the evidence supports. Ground directions in different subgraphs when possible
so the portfolio covers genuinely different opportunities.

Determine the measured parent-to-child outcome, the exact change or inseparable
bundle, its prerequisites or failure boundary, and whether it persisted or
produced useful descendants. Then compare the historical parent and child with
the exact live target. Reuse the mechanism only when it is absent and its
prerequisites hold; otherwise use the case as contrast or choose exploration.
For every live integration donor, call `compare_attempts` on the target and
donor IDs.

This call must read search_history/metric_definitions.json, the complete
search_history/dashboard_llm.jsonl, and the catalog code of every named target and
donor branch. Every cited `evidence_attempt_id` must have its trajectory read;
its code must also be read whenever the direction relies on its implementation
or mechanism.

## Proposal types and evidence

Use only these two proposal types:

- `integration`: reuse measured knowledge recovered from selectively unfolded
  history. The reused unit may be a live-donor mechanism, a complete measured
  bundle, a repair/restoration lesson, or a structural relationship between an
  observed search/performance pattern and a promising experiment. Integration
  transfers knowledge, not necessarily code. It requires one or more unfolded
  evidence attempts; an exact live donor is required only when code is being
  transferred. State whether the evidence isolates the mechanism (`isolated`)
  or measures only a compound bundle (`bundle_only`). If a successful
  parent-to-child transition changed several mechanisms at once, its measured
  gain supports only the combined change. Do not claim that one component
  caused the gain unless another measured comparison changes that component
  while keeping the others fixed.
- `exploration`: test a genuinely new representation, scheduling, dependency,
  or algorithmic hypothesis. Positive historical evidence and donors are not
  required. Identify the new axis, explain why the current target motivates
  it, and state what result would falsify the hypothesis. Historical failures
  or exhausted patterns may be cited as contrast evidence, but never invent
  positive evidence for a novel idea.

All cited `evidence_attempt_ids` are dashboard IDs of the form
`attempt:seq:*`, exactly as they appear in
`search_history/dashboard_llm.jsonl`; `traj:*` IDs are invalid.

## Portfolio selection

- Return exactly `{{num_directions}}` directions. The controller must replace
  this placeholder with a concrete positive integer before invoking the agent.
- Proposal types are labels, not quotas. Multiple integration or multiple
  exploration directions are allowed when their targets and mechanisms are
  materially different.
- Every direction must target global progress, not merely improve a weak
  parent. Give the target cycles, current global-frontier cycles, the gain
  required to beat the frontier, and a concise reason the proposed mechanism
  could plausibly clear that gap.

## Output

Return exactly one JSON object as your entire final answer — no prose, no
Markdown fence. The top-level object must contain exactly these fields:

Before returning it, verify that every evidence ID uses the exact zero-padded
dashboard form (for example `attempt:seq:00000255`), every cited trajectory was
read, and any code needed to support an implementation-level claim was read.
JSON must have no trailing commas.

- `diagnosis`: a concise evidence-backed diagnosis of the current search state.
- `exhausted_patterns`: up to three evidence-backed patterns not to repeat.
- `directions`: exactly `{{num_directions}}` direction objects, each matching
  one of the two parallel examples below.

Integration direction example:

{
  "type":"integration",
  "target_branch_id":"live catalog id",
  "mechanism_hypothesis":"target-local causal hypothesis for shortening the emitted schedule",
  "task":"one concrete bounded source-code experiment",
  "frontier_case":{"target_cycles":1500,"frontier_cycles":1450,"required_gain_cycles":51,"why_plausible":"the measured donor transition addresses a bottleneck still present in the target"},
  "constraints":["preserve correctness and remain within ISA and scratch limits"],
  "donor_branch_ids":["live integration donor id"],
  "evidence_attempt_ids":["attempt:seq:..."],
  "evidence_strength":"isolated",
  "novel_axis":null,
  "falsifier":null
}

Exploration direction example:

{
  "type":"exploration",
  "target_branch_id":"live catalog id",
  "mechanism_hypothesis":"a previously untested target-local dependency restructuring can shorten the emitted schedule without increasing scratch use",
  "task":"test one bounded implementation of that dependency restructuring while preserving the target's algorithm and output semantics",
  "frontier_case":{"target_cycles":1500,"frontier_cycles":1450,"required_gain_cycles":51,"why_plausible":"the target's measured schedule contains a dependency chain that the new representation may remove"},
  "constraints":["preserve correctness and remain within ISA and scratch limits"],
  "donor_branch_ids":[],
  "evidence_attempt_ids":[],
  "evidence_strength":"novel_hypothesis",
  "novel_axis":"dependency structure used to represent one traversal stage",
  "falsifier":"an evaluator-valid implementation leaves target cycles unchanged or worse under the fixed workload"
}

These examples illustrate the two allowed object shapes only; they do not
impose a proposal-type quota.

Direction rules:

- `type`: `integration` reuses measured knowledge recovered from unfolded
  history; `exploration` tests a genuinely new hypothesis.
- `target_branch_id`: the exact live execution parent the worker will edit.
- `mechanism_hypothesis`: the falsifiable target-local reason the experiment
  may shorten the emitted schedule.
- `task`: one concrete bounded source-code experiment, not code, a raw diff,
  or a list of alternatives.
- `frontier_case`: target cycles, current frontier cycles, required gain to beat
  it, and why this mechanism could plausibly clear that gap.
- `constraints`: only correctness, state-lifetime, scratch, ISA, or resource
  conditions material to this experiment.
- `donor_branch_ids`: at most two live donors when integration transfers code
  or a live mechanism; empty for knowledge integration and exploration.
- `evidence_attempt_ids`: unfolded measured source cases supporting integration;
  optional negative or contrast evidence for exploration.
- `evidence_strength`: `isolated` or `bundle_only` for integration;
  `novel_hypothesis` for exploration.
- `novel_axis`: the new representation, scheduling, dependency, or algorithmic
  variable tested by exploration; `null` for integration.
- `falsifier`: the measured outcome that would reject an exploration
  hypothesis; `null` for integration.
