"""Exercise the verifier offline with a tiny Git-backed harness fixture."""
import subprocess

import pytest

import verify


@pytest.fixture
def harness(tmp_path, monkeypatch):
    repo = tmp_path / "official"
    (repo / "tests").mkdir(parents=True)
    (repo / "problem.py").write_text("# simulator\n")
    (repo / "tests/frozen_problem.py").write_text("# frozen simulator\n")
    (repo / "tests/submission_tests.py").write_text('''import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_takehome import VALUE
class CorrectnessTests(unittest.TestCase):
    def test_correct(self):
        assert VALUE >= 1
class SpeedTests(unittest.TestCase):
    def test_fast(self):
        assert VALUE > 1
if __name__ == "__main__":
    unittest.main()
''')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "fixture"], check=True)
    pin = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(verify, "OFFICIAL_COMMIT", pin)
    return repo


def test_fresh_checkout_ignores_dirty_cache_and_separates_speed(tmp_path, harness):
    (harness / "tests/submission_tests.py").write_text("raise RuntimeError('dirty cache')")
    program = tmp_path / "candidate.py"
    program.write_text("VALUE = 1\n")
    correctness = verify.verify(program, local_repo=harness, correctness_only=True)
    full = verify.verify(program, local_repo=harness)
    assert correctness["passed"] and correctness["mode"] == "correctness_only"
    assert not full["passed"] and full["returncode"] == 1
    assert full["harness_unchanged"] and full["mode"] == "full_submission"
    assert "AssertionError" in full["stderr"]
    assert (harness / "tests/submission_tests.py").read_text().startswith("raise RuntimeError")


def test_incorrect_candidate_cannot_pass_with_pythonoptimize(tmp_path, harness, monkeypatch):
    program = tmp_path / "candidate.py"
    program.write_text("VALUE = 0\n")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    result = verify.verify(program, local_repo=harness, correctness_only=True)
    assert not result["passed"] and result["returncode"] == 1


def test_modified_tests_are_rejected_even_if_child_exits_zero(tmp_path, harness):
    program = tmp_path / "candidate.py"
    program.write_text("from pathlib import Path\nPath('tests/frozen_problem.py').write_text('changed')\nVALUE = 2\n")
    result = verify.verify(program, local_repo=harness)
    assert result["returncode"] == 0
    assert not result["harness_unchanged"] and not result["passed"]


def test_timeout_is_a_failed_verification(tmp_path, harness):
    program = tmp_path / "candidate.py"
    program.write_text("import time\ntime.sleep(10)\nVALUE = 2\n")
    result = verify.verify(program, local_repo=harness, timeout=0.1)
    assert result["timed_out"] and not result["passed"]


def test_missing_pinned_commit_fails_closed(tmp_path, harness, monkeypatch):
    program = tmp_path / "candidate.py"
    program.write_text("VALUE = 2\n")
    monkeypatch.setattr(verify, "OFFICIAL_COMMIT", "0" * 40)
    with pytest.raises(subprocess.CalledProcessError):
        verify.verify(program, local_repo=harness)
