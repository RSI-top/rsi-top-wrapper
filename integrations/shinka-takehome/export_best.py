#!/usr/bin/env python3
"""
Export the best correct program from a ShinkaEvolve results DB.

Writes <results_dir>/best/best_program.py (+ best_program_info.json).
Usage: python export_best.py [results_dir]
"""

import json
import os
import sqlite3
import sys


def main(results_dir="results"):
    db_path = os.path.join(results_dir, "programs.sqlite")
    if not os.path.exists(db_path):
        print(f"No database at {db_path}")
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, code, generation, combined_score, public_metrics "
        "FROM programs WHERE correct = 1 AND combined_score IS NOT NULL "
        "ORDER BY combined_score DESC, timestamp ASC LIMIT 1"
    ).fetchone()
    conn.close()

    if row is None:
        print("No correct program in database yet")
        return None

    best_dir = os.path.join(results_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    code_path = os.path.join(best_dir, "best_program.py")
    with open(code_path, "w") as f:
        f.write(row["code"])

    public = json.loads(row["public_metrics"] or "{}")
    info = {
        "id": row["id"],
        "generation": row["generation"],
        "combined_score": row["combined_score"],
        "public_metrics": public,
    }
    with open(os.path.join(best_dir, "best_program_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    cycles = public.get("cycles")
    print(f"Best program -> {code_path}")
    print(
        f"  score={row['combined_score']:.4f} generation={row['generation']}"
        + (f" cycles={int(cycles)} (speedup {147734 / cycles:.2f}x)" if cycles else "")
    )
    return code_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
