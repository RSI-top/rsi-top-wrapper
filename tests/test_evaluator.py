import json
import subprocess
import sys

from tests.helpers import ROOT


def test_starter_is_correct_with_original_cycle_count():
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"),
         str(ROOT / "integrations/openevolve-takehome/initial_program.py")],
        capture_output=True, text=True, check=True, timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )
    score = json.loads(result.stdout)
    assert score["valid"] is True and score["cycles"] == 147734


def test_invalid_candidate_reports_failure_and_nonzero_exit(tmp_path):
    program = tmp_path / "broken.py"
    program.write_text("raise RuntimeError('intentional build failure')\n")
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate.py"), str(program)],
        capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )
    score = json.loads(result.stdout)
    assert result.returncode == 1 and score["valid"] is False
    assert score["artifacts"]["failure_stage"] == "build_error"
    assert "intentional build failure" in score["artifacts"]["stderr"]
