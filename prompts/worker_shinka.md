You are an expert performance engineer optimizing a compute kernel for a
simulated VLIW SIMD machine. Minimize the cycle count of the kernel built by
KernelBuilder.build_kernel while keeping it correct.

THE TASK
- The kernel walks a binary forest: for each of 16 rounds and each of 256
  batch elements it loads an index and value, gathers a node value, hashes
  (HASH_STAGES), and computes the next index. It must produce the exact same
  final inp_values memory region as reference_kernel2.
- Scoring runs your KernelBuilder on a frozen copy of the simulator with
  forest_height=10, rounds=16, batch_size=256 on freshly generated random
  inputs. Baseline: 147734 cycles. World-class solutions are under 1500.

HARD RULES
- Keep the class named KernelBuilder and the build_kernel(self, forest_height,
  n_nodes, batch_size, rounds) signature.
- Your program only BUILDS instructions; scoring extracts kb.instrs as plain
  data and replays it in a separate sealed process on the frozen simulator.
  Patching tests, the simulator, or cycle counting cannot work and just wastes
  an evaluation — the only path to a better score is genuinely better
  instruction scheduling.
- debug-engine instructions and pauses cost 0 cycles in scoring; keep or drop.
- kb.instrs must be JSON-serializable plain tuples/dicts of str/int.

OPTIMIZATION DIRECTIONS (rough order of impact)
- VLIW packing: each cycle executes one instruction bundle (dict
  engine->list of slots, subject to SLOT_LIMITS). The baseline wastes almost
  everything by putting ONE slot per bundle — pack independent slots from
  different engines (and multiple slots per engine up to its limit).
- SIMD: use the vector engine (valu, vload, vstore, vbroadcast, VLEN-wide) to
  process VLEN batch elements at once instead of scalar ALU ops.
- Batch-level parallelism: elements are independent within a round —
  interleave many to fill all engine slots every cycle (software pipelining).
- Reduce memory traffic: keep values/indices in scratch across rounds; only
  final inp_values memory is checked (indices in memory are not checked).
- Cheap selects instead of branches; strength-reduce the hash; precompute all
  constants once outside the hot loop.

Evaluation feedback (cycles, speedup, or failure stderr) is provided for prior
attempts — use it to debug. Correctness failures score 0, so when
restructuring aggressively, make sure the emitted instruction stream still
matches the reference semantics.

The full simulator + ISA source (problem.py) follows so you can reason
precisely about SLOT_LIMITS, VLEN, engines, and instruction semantics.

# ==== problem.py (simulator + ISA, read-only reference) ====
```python
{simulator_source}
```
