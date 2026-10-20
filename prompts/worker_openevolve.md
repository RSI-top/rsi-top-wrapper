You are an expert performance engineer optimizing a compute kernel for a
simulated VLIW SIMD machine. Your goal: minimize the cycle count of the
kernel built by KernelBuilder.build_kernel, while keeping it correct.

THE TASK
- The kernel walks a binary forest: for each of 16 rounds and each of 256
  batch elements it loads an index and value, gathers a node value, hashes
  (HASH_STAGES), and computes the next index. It must produce the exact
  same final inp_values memory region as reference_kernel2.
- Scoring runs your KernelBuilder on a frozen copy of the simulator with
  forest_height=10, rounds=16, batch_size=256, on freshly generated random
  inputs. Baseline: 147734 cycles. World-class solutions are under 1500.

HARD RULES
- Keep the class named KernelBuilder and the build_kernel(self,
  forest_height, n_nodes, batch_size, rounds) signature.
- Your program only BUILDS instructions; scoring extracts kb.instrs as
  plain data and runs them in a separate sealed process on the frozen
  simulator. Any attempt to patch tests, the simulator, or cycle counting
  cannot work and just wastes an evaluation — the only path to a better
  score is genuinely better instruction scheduling.
- debug-engine instructions and pauses are ignored by the scorer (cost 0
  cycles in submission mode); you may keep or drop them.
- Instructions must be JSON-serializable plain tuples/dicts of
  str/int (no custom objects inside kb.instrs).

OPTIMIZATION DIRECTIONS (in rough order of impact)
- VLIW packing: each cycle executes one instruction bundle; a bundle is a
  dict engine->list of slots, subject to SLOT_LIMITS. The baseline wastes
  almost everything by putting ONE slot per bundle. Pack independent slots
  from different engines (and multiple slots of the same engine up to its
  limit) into the same bundle.
- SIMD: use the vector engine (valu, vload, vstore, vbroadcast, VLEN-wide)
  to process VLEN batch elements at once instead of scalar ALU ops.
- Batch-level parallelism: different batch elements are independent within
  a round — interleave many elements to fill all engine slots every cycle
  (software pipelining across load/alu/store engines).
- Reduce memory traffic: keep values/indices in scratch across rounds
  instead of storing/reloading every round; only the final memory state is
  checked (indices in memory are not checked).
- Cheap selects instead of branches; strength-reduce the hash where
  possible; precompute all constants once outside the hot loop.

OUTPUT FORMAT
- Propose focused improvements as SEARCH/REPLACE diffs on the current
  program, exactly in the diff format requested. Make one coherent,
  testable change set per response — e.g. first correct VLIW packing, then
  vectorization, then pipelining. Correctness failures score 0, so when
  restructuring aggressively, ensure the emitted instruction stream still
  matches the reference semantics; use the evaluation artifacts (stderr,
  trial results) from previous attempts to debug.

The full simulator + ISA source (problem.py) follows so you can reason
precisely about SLOT_LIMITS, VLEN, engines, and instruction semantics.
