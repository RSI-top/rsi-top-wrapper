"""A complete checkpoint must survive a round trip and reject lost/tampered state."""
from dataclasses import replace

import pytest

from meta_audit.causal_checkpoint import (
    CausalCheckpointError, CausalCheckpointManifest, digest_file,
    read_checkpoint, verify_checkpoint, write_checkpoint,
)
from meta_audit.checkpoint_contract import REQUIRED_STATE_COMPONENTS


@pytest.fixture(params=["openevolve", "shinka"])
def checkpoint(tmp_path, request):
    state = tmp_path / "state.json"
    state.write_text('{"population": ["root"], "pending": []}\n')
    digest = digest_file(state)["sha256"]
    manifest = CausalCheckpointManifest(
        baseline=request.param,
        boundary={"completion_identity": "run:1", "completion_seq": 1},
        state_components={name: digest for name in REQUIRED_STATE_COMPONENTS[request.param]},
        files={"state.json": digest_file(state),
               "evaluator_provenance/scorer.py": digest_file(state)}, documents={},
        evaluator_closure={**{name: digest for name in (
            "entrypoint", "phase_code", "frozen_reference", "interpreter", "dependencies", "digest")},
            "provenance_dir": "evaluator_provenance"},
        prompt_provenance={"sha256": digest}, budget_state={"spent_usd": 0},
        concurrency_state={"pending_sessions": []}, settings={"seed": 42},
    )
    return tmp_path / "checkpoint", state, manifest


def test_checkpoint_roundtrip_and_payload_tampering(checkpoint):
    directory, state, manifest = checkpoint
    payload = write_checkpoint(directory, manifest, payload_files={"state.json": state, "evaluator_provenance/scorer.py": state})
    assert read_checkpoint(directory) == payload
    verify_checkpoint(directory)
    (directory / "state.json").write_text("tampered")
    with pytest.raises(CausalCheckpointError):
        verify_checkpoint(directory)


def test_incomplete_checkpoint_cannot_resume(checkpoint):
    directory, state, manifest = checkpoint
    write_checkpoint(directory, manifest, payload_files={"state.json": state, "evaluator_provenance/scorer.py": state})
    (directory / "COMPLETE").unlink()
    with pytest.raises(CausalCheckpointError, match="never.*completed"):
        read_checkpoint(directory)


def test_missing_search_state_is_rejected_before_checkpoint_is_written(checkpoint):
    directory, state, manifest = checkpoint
    components = dict(manifest.state_components)
    del components["search_population"]
    with pytest.raises(CausalCheckpointError, match="search_population"):
        write_checkpoint(directory, replace(manifest, state_components=components),
                         payload_files={"state.json": state, "evaluator_provenance/scorer.py": state})
    assert not (directory / "COMPLETE").exists()
