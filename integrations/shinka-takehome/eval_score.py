"""
Phase B (TRUSTED): score kernel instructions on the frozen simulator.

Loads only plain JSON produced by eval_build.py, sanitizes it, then runs the
pristine frozen_problem.Machine against reference_kernel2 on freshly
generated random inputs using the original benchmark scoring rules.

Usage: python eval_score.py <instrs_json> <takehome_dir> <n_trials> <out_json>
"""

import json
import random
import sys

ALLOWED_LEAF = (str, int, float, bool, type(None))


def to_slot(x):
    """Recursively convert JSON lists back to tuples; verify leaf types."""
    if isinstance(x, list):
        return tuple(to_slot(e) for e in x)
    if not isinstance(x, ALLOWED_LEAF):
        raise ValueError(f"Illegal value in instruction stream: {type(x)}")
    return x


def main():
    instrs_path, takehome_dir, n_trials, out_path = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
    )
    sys.path.insert(0, takehome_dir)

    from frozen_problem import (
        N_CORES,
        DebugInfo,
        Input,
        Machine,
        Tree,
        build_mem_image,
        reference_kernel2,
    )

    with open(instrs_path) as f:
        raw = json.load(f)

    instrs = []
    for bundle in raw["instrs"]:
        if not isinstance(bundle, dict):
            raise ValueError("Instruction bundle is not a dict")
        instrs.append(
            {
                str(engine): [to_slot(s) for s in slots]
                for engine, slots in bundle.items()
            }
        )

    result = {"ok": False, "cycles": None, "trials": []}
    cycles_seen = []
    for t in range(n_trials):
        random.seed(1000 + t)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)

        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for ref_mem in reference_kernel2(mem):
            pass

        inp_values_p = ref_mem[6]
        correct = (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        )
        result["trials"].append({"correct": bool(correct), "cycles": machine.cycle})
        if not correct:
            result["error"] = f"Incorrect output values on trial {t}"
            with open(out_path, "w") as f:
                json.dump(result, f)
            return
        cycles_seen.append(machine.cycle)

    result["ok"] = True
    result["cycles"] = max(cycles_seen)
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
