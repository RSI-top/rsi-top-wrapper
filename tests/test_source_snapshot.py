import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_audit.source_snapshot import capture_sources, write_source_snapshot
from tests.helpers import ROOT, load_integration


def test_runtime_closure_covers_extracted_modules_and_dependencies():
    sources = capture_sources(ROOT, environ={})
    required = {
        "integrations/openevolve-takehome/code_geometry.py",
        "integrations/openevolve-takehome/search_history.py",
        "integrations/openevolve-takehome/history_utils.py",
        "integrations/openevolve-takehome/source_files.py",
        "integrations/openevolve-takehome/pi_cli.py",
        "integrations/openevolve-takehome/deepseek_llm.py",
        "integrations/openevolve-takehome/pi_unfold_selection_extension.ts",
        "integrations/shinka-takehome/state_snapshot.py",
        "integrations/shinka-takehome/shinka_resume_capture.py",
        "integrations/shinka-takehome/shinka_causal_checkpoint.py",
        "integrations/shinka-takehome/vendor/shinkaevolve/shinka/core/sampler.py",
        "integrations/shinka-takehome/vendor/shinkaevolve/shinka/launch/scheduler.py",
        "integrations/openevolve-takehome/vendor/openevolve/openevolve/process_parallel.py",
        "shared/meta-agent-audit/meta_audit/source_snapshot.py",
        "shared/meta-agent-audit/meta_audit/causal_checkpoint.py",
        "prompts/metrics.json", "prompts/planner.md", "prompts/worker_shinka.md",
        "uv.lock", "package-lock.json", "verify.py",
    }
    assert required <= sources.keys()
    for relative in required:
        data = (ROOT / relative).read_bytes()
        assert sources[relative]["content"].encode() == data
        assert sources[relative]["sha256"] == hashlib.sha256(data).hexdigest()
    assert not any(set(Path(name).parts) & {".venv", "node_modules", "runs", "tests"}
                   for name in sources)


def test_redaction_preserves_original_digest_and_resume_snapshots(tmp_path):
    config = tmp_path / "custom.yaml"
    config.write_text("credential: test-secret-value\n")
    original = capture_sources(ROOT, extra_files=[config], environ={"DEEPSEEK_API_KEY": "test-secret-value"})
    entry = original[str(config)]
    assert "test-secret-value" not in entry["content"]
    assert entry["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert entry["captured_sha256"] == hashlib.sha256(entry["content"].encode()).hexdigest()
    old = write_source_snapshot(tmp_path, original)
    assert write_source_snapshot(tmp_path, original) == old
    config.write_text("changed: true\n")
    new = write_source_snapshot(tmp_path, capture_sources(ROOT, extra_files=[config], environ={}))
    assert new["path"] != old["path"]
    assert (tmp_path / old["path"]).is_file()
    assert hashlib.sha256((tmp_path / old["path"]).read_bytes()).hexdigest() == old["sha256"]
    (tmp_path / new["path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        write_source_snapshot(tmp_path, capture_sources(ROOT, extra_files=[config], environ={}))


def test_missing_custom_input_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        capture_sources(ROOT, extra_files=[tmp_path / "missing.yaml"])


def test_shinka_manifest_preserves_each_instances_shared_source_snapshot(tmp_path):
    runner = load_integration("release_shinka_runner", "integrations/shinka-takehome/run_evo.py")
    config = tmp_path / "custom.yaml"
    config.write_text("{}"); takehome = ROOT / "integrations/shinka-takehome/vendor/takehome"
    args = SimpleNamespace(config=str(config), budget=1, effort="high", seed=42,
                           llm_model="deepseek-v4-flash", breadth_control_dir=None)
    runner._record_manifest(tmp_path, "first", args, {"evo_config": {}}, takehome)
    config.write_text("changed: true\n")
    runner._record_manifest(tmp_path, "second", args, {"evo_config": {}}, takehome)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    references = [row["source_provenance"] for row in manifest["run_instances"]]
    assert len(references) == 2 and references[0] != references[1]
    for reference in references:
        snapshot = json.loads((tmp_path / reference["path"]).read_text())
        assert "integrations/openevolve-takehome/code_geometry.py" in snapshot["files"]
    assert "code_geometry.py" not in manifest["source_files"]  # no nonexistent local file


def test_openevolve_startup_records_snapshot_in_manifest_and_trace(tmp_path, monkeypatch):
    import sys
    import openevolve.controller as controller
    import oe_trajectory

    runner = load_integration("release_oe_runner", "integrations/openevolve-takehome/runner.py")

    class StopBeforeSearch(Exception):
        pass

    def stop_before_search(**kwargs):
        raise StopBeforeSearch

    monkeypatch.setattr(controller, "OpenEvolve", stop_before_search)
    # Keep this metadata-only test from installing global runtime wrappers.
    monkeypatch.setattr(controller, "ProcessParallelController", controller.ProcessParallelController)
    monkeypatch.setattr(oe_trajectory, "make_traced_parallel_controller",
                        lambda **kwargs: controller.ProcessParallelController)
    for name in ("install_log_trace", "install_prompt_trace", "install_evaluator_trace",
                 "install_checkpoint_sidecar_trace"):
        monkeypatch.setattr(oe_trajectory, name, lambda: None)
    monkeypatch.setattr(os, "environ", {"PATH": os.environ["PATH"],
                                      "DEEPSEEK_API_KEY": "offline-key-no-model-calls"})
    monkeypatch.setattr(sys, "argv", ["runner.py", "--output", str(tmp_path),
                                     "--llm-provider", "deepseek", "--iterations", "1"])
    with pytest.raises(StopBeforeSearch):
        runner.main()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    reference = manifest["source_provenance"]
    snapshot = json.loads((tmp_path / reference["path"]).read_text())
    relative = "integrations/openevolve-takehome/code_geometry.py"
    assert snapshot["files"][relative]["sha256"] == manifest["source_snapshot"][str(ROOT / relative)]["sha256"]
    events = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    started = next(event for event in events if event["event"] == "run_started")
    assert started["source_provenance"] == reference
