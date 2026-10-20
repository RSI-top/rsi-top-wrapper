"""
Phase A (UNTRUSTED): run the evolved program to build kernel instructions.

The evolved code executes only in this throwaway subprocess. Its sole output
is a JSON file of plain instruction data — it never touches the scoring
process or the frozen simulator, which makes test-tampering cheats
structurally impossible.

Usage: python eval_build.py <program_path> <takehome_dir> <out_json>
"""

import importlib.util
import json
import random
import sys


def main():
    program_path, takehome_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, takehome_dir)

    from problem import Tree

    random.seed(0)
    forest = Tree.generate(10)
    n_nodes = len(forest.values)

    spec = importlib.util.spec_from_file_location("evolved_kernel", program_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kb = mod.KernelBuilder()
    kb.build_kernel(10, n_nodes, 256, 16)

    # json.dump rejects anything that isn't plain data — that is intentional.
    with open(out_path, "w") as f:
        json.dump({"instrs": kb.instrs, "scratch_words": kb.scratch_ptr}, f)


if __name__ == "__main__":
    main()
