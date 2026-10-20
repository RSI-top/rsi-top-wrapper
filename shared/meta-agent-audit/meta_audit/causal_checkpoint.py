"""Forward exact-causal-checkpoint contract for future runs. Zero calls.

Step 12 Phase A established that none of the four historical baselines can be
forked at exact ``t0``: each keeps some part of its causal state only as a single
mutable final-state file, or not at all. This module is the forward fix — the
contract a *future* run writes so that a later fork is exact by construction
rather than by archaeology.

A :class:`CausalCheckpointManifest` is:

* **baseline-global.** It binds every name in
  ``REQUIRED_STATE_COMPONENTS[baseline]``. One missing component and the manifest
  is not restorable — there is no "mostly complete" state;
* **content-addressed.** Every referenced file carries a sha256 and a byte count.
  Nothing is identified by path alone, because a path can be rewritten;
* **atomically completed.** A checkpoint is only real once a separate ``COMPLETE``
  marker holds the manifest digest. A crash mid-write leaves a directory that
  every API here refuses, rather than a manifest that half-describes a state;
* **fail closed.** :func:`validate_manifest`, :func:`verify_checkpoint` and
  :func:`verify_restored` raise on anything unexpected. Nothing degrades to a
  warning, and no default fills a gap.

**No shell recipe is trusted.** The restore plan is a list of typed operations,
each carrying the digest of what it must produce. A rendered shell preview exists
for humans and is explicitly marked non-authoritative; executing it without
running :func:`verify_restored` afterwards proves nothing.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._canonical import canonical_json, sha256_hex
from .checkpoint_contract import EXACT_T0_RULE, REQUIRED_STATE_COMPONENTS

__all__ = [
    "COMPLETE_MARKER_NAME",
    "CausalCheckpointError",
    "CausalCheckpointManifest",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "RestoreOp",
    "digest_file",
    "read_checkpoint",
    "restore_plan",
    "validate_manifest",
    "verify_checkpoint",
    "verify_restored",
    "write_checkpoint",
]

MANIFEST_VERSION = "causal_checkpoint_manifest_v1"
MANIFEST_NAME = "manifest.json"
#: Written last, and only after the manifest is durable. Its content is the
#: manifest digest, so a truncated or stale marker fails the same way a missing
#: one does.
COMPLETE_MARKER_NAME = "COMPLETE"

#: Restore-plan operation kinds. Deliberately tiny: anything a checkpoint needs
#: to reproduce is either a file copied from the checkpoint or a JSON document
#: the manifest already pins by digest.
RESTORE_OPS = ("assert_empty", "mkdir", "copy_file", "write_json", "extract_archive")

#: Every part of the scorer closure a future run must record at campaign start.
#: Same standard the Phase A audit applied, now written by the producer instead of
#: reconstructed by the consumer.
EVALUATOR_CLOSURE_PARTS = ("entrypoint", "phase_code", "frozen_reference", "interpreter",
                           "dependencies")


class CausalCheckpointError(RuntimeError):
    """A checkpoint is incomplete, drifted, or otherwise not restorable."""


def digest_file(path: str | os.PathLike) -> dict[str, Any]:
    """Content address for one file: sha256 plus byte count."""
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": sha256_hex(data)}


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalCheckpointManifest:
    """One exact causal checkpoint of a baseline-global state.

    ``boundary`` names the completion this checkpoint sits immediately after —
    the exact point at which a fork would resume. ``state_components`` maps every
    required component to a digest. ``files`` content-addresses each file the
    checkpoint carries, relative to the checkpoint directory. ``documents`` holds
    JSON state the restore writes verbatim, each with its own digest.
    """

    baseline: str
    boundary: Mapping[str, Any]
    state_components: Mapping[str, str]
    files: Mapping[str, Mapping[str, Any]]
    documents: Mapping[str, Any]
    evaluator_closure: Mapping[str, Any]
    prompt_provenance: Mapping[str, Any]
    budget_state: Mapping[str, Any]
    concurrency_state: Mapping[str, Any]
    settings: Mapping[str, Any]
    trees: Mapping[str, str] = field(default_factory=dict)
    #: Tar archives the checkpoint carries, for state too large or too structural to
    #: list file-by-file (a git repository, a database snapshot). Each declares the
    #: archive digest AND the per-entry content digests, so a restore is verified at
    #: content level and not merely by the archive bytes.
    archives: Mapping[str, Any] = field(default_factory=dict)
    version: str = MANIFEST_VERSION

    def to_json(self) -> dict[str, Any]:
        payload = {
            "archives": json.loads(canonical_json(dict(self.archives))),
            "baseline": self.baseline,
            "boundary": json.loads(canonical_json(dict(self.boundary))),
            "budget_state": json.loads(canonical_json(dict(self.budget_state))),
            "concurrency_state": json.loads(canonical_json(dict(self.concurrency_state))),
            "documents": json.loads(canonical_json(dict(self.documents))),
            "evaluator_closure": json.loads(canonical_json(dict(self.evaluator_closure))),
            "exact_t0_rule": EXACT_T0_RULE,
            "files": json.loads(canonical_json(dict(self.files))),
            "prompt_provenance": json.loads(canonical_json(dict(self.prompt_provenance))),
            "required_state_components": list(REQUIRED_STATE_COMPONENTS[self.baseline]),
            "settings": json.loads(canonical_json(dict(self.settings))),
            "state_components": dict(sorted(self.state_components.items())),
            "trees": dict(sorted(self.trees.items())),
            "version": self.version,
        }
        payload["manifest_digest"] = sha256_hex(canonical_json(payload).encode("utf-8"))
        return payload

    def digest(self) -> str:
        return self.to_json()["manifest_digest"]


def _require(payload: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in payload:
        raise CausalCheckpointError(f"manifest is missing required key {key!r}")
    value = payload[key]
    if not isinstance(value, kind):
        raise CausalCheckpointError(
            f"manifest key {key!r} must be {kind}, got {type(value).__name__}"
        )
    return value


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure structural validation. Raises on anything short of complete.

    Checks nothing about the filesystem — that is :func:`verify_checkpoint`'s job.
    Separated so a manifest can be validated in isolation, e.g. by a preflight that
    only has the bytes.
    """
    if not isinstance(payload, Mapping):
        raise CausalCheckpointError("manifest must be a JSON object")
    version = _require(payload, "version", str)
    if version != MANIFEST_VERSION:
        raise CausalCheckpointError(
            f"unsupported manifest version {version!r}; expected {MANIFEST_VERSION!r}"
        )
    baseline = _require(payload, "baseline", str)
    required = REQUIRED_STATE_COMPONENTS.get(baseline)
    if required is None:
        raise CausalCheckpointError(f"no required-state-component manifest for {baseline!r}")

    declared = _require(payload, "required_state_components", list)
    if list(declared) != list(required):
        raise CausalCheckpointError(
            f"manifest declares required components {declared} but {baseline} requires "
            f"{list(required)}"
        )

    components = _require(payload, "state_components", Mapping)
    unpinned = [
        name
        for name in required
        if not isinstance(components.get(name), str) or not components[name].strip()
    ]
    if unpinned:
        raise CausalCheckpointError(
            f"checkpoint is not exact -- unpinned {baseline} state components: {unpinned}. "
            f"{EXACT_T0_RULE}"
        )
    extra = sorted(set(components) - set(required))
    if extra:
        raise CausalCheckpointError(f"manifest pins unknown state components: {extra}")

    boundary = _require(payload, "boundary", Mapping)
    for key in ("completion_identity", "completion_seq"):
        if not str(boundary.get(key, "")).strip():
            raise CausalCheckpointError(f"boundary.{key} is required")

    files = _require(payload, "files", Mapping)
    for relative, entry in files.items():
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise CausalCheckpointError(f"file entry {relative!r} escapes the checkpoint")
        if not isinstance(entry, Mapping):
            raise CausalCheckpointError(f"file entry {relative!r} is not an object")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
            raise CausalCheckpointError(f"file entry {relative!r} has no sha256")
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            raise CausalCheckpointError(f"file entry {relative!r} has no byte count")

    archives = _require(payload, "archives", Mapping)
    for name, entry in archives.items():
        if not isinstance(entry, Mapping):
            raise CausalCheckpointError(f"archive {name!r} is not an object")
        for key in ("path", "extract_to", "sha256", "contents"):
            if key not in entry:
                raise CausalCheckpointError(f"archive {name!r} is missing {key!r}")
        if entry["path"] not in payload.get("files", {}):
            raise CausalCheckpointError(
                f"archive {name!r} references {entry['path']!r}, which the manifest does "
                "not content-address"
            )
        if payload["files"][entry["path"]]["sha256"] != entry["sha256"]:
            raise CausalCheckpointError(f"archive {name!r} digest disagrees with its file entry")
        if not isinstance(entry["contents"], Mapping) or not entry["contents"]:
            raise CausalCheckpointError(
                f"archive {name!r} declares no per-entry content digests, so a restore could "
                "not be verified at content level"
            )
        for relative in entry["contents"]:
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise CausalCheckpointError(
                    f"archive {name!r} entry {relative!r} escapes the target"
                )

    documents = _require(payload, "documents", Mapping)
    for name, entry in documents.items():
        if not isinstance(entry, Mapping):
            raise CausalCheckpointError(f"document {name!r} is not an object")
        for key in ("path", "payload", "sha256"):
            if key not in entry:
                raise CausalCheckpointError(f"document {name!r} is missing {key!r}")
        expected = sha256_hex(_document_bytes(entry["payload"]))
        if entry["sha256"] != expected:
            raise CausalCheckpointError(
                f"document {name!r} digest does not match its own payload"
            )

    closure = _require(payload, "evaluator_closure", Mapping)
    missing_closure = [
        part
        for part in EVALUATOR_CLOSURE_PARTS
        if not str(closure.get(part) or "").strip()
    ]
    if missing_closure:
        raise CausalCheckpointError(
            f"evaluator closure is incomplete -- {missing_closure} unpinned. A checkpoint "
            "whose scorer is not pinned cannot be shown to score the way the run did."
        )
    if not str(closure.get("digest") or "").strip():
        raise CausalCheckpointError("evaluator_closure.digest is required")
    if not str(closure.get("provenance_dir") or "").strip():
        raise CausalCheckpointError(
            "evaluator_closure.provenance_dir is required: the score-producing files must "
            "live in a sealed run-owned directory, so a later mutable working tree cannot "
            "invalidate the provenance"
        )

    for key in ("budget_state", "concurrency_state", "prompt_provenance", "settings"):
        _require(payload, key, Mapping)
    pending = payload["concurrency_state"].get("pending_sessions")
    if not isinstance(pending, list):
        raise CausalCheckpointError("concurrency_state.pending_sessions must be a list")

    recomputed = dict(payload)
    stated = recomputed.pop("manifest_digest", None)
    if not stated:
        raise CausalCheckpointError("manifest_digest is required")
    if stated != sha256_hex(canonical_json(recomputed).encode("utf-8")):
        raise CausalCheckpointError("manifest_digest does not match the manifest bytes")
    return dict(payload)


def _document_bytes(payload: Any) -> bytes:
    """Exactly the bytes a restore writes for a JSON document."""
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# Writing — atomic, complete-marker last
# ---------------------------------------------------------------------------


def write_checkpoint(
    checkpoint_dir: str | os.PathLike,
    manifest: CausalCheckpointManifest,
    *,
    payload_files: Mapping[str, str | os.PathLike],
) -> dict[str, Any]:
    """Write one checkpoint atomically. The COMPLETE marker is written last.

    ``payload_files`` maps each manifest ``files`` key to the source path to copy.
    Every copy is verified against the manifest digest before the marker is
    written, so a checkpoint directory that carries a marker is always internally
    consistent.
    """
    out = Path(checkpoint_dir)
    if out.exists() and any(out.iterdir()):
        raise CausalCheckpointError(f"refusing to write into a non-empty directory: {out}")
    payload = manifest.to_json()
    validate_manifest(payload)

    missing = sorted(set(payload["files"]) - set(payload_files))
    if missing:
        raise CausalCheckpointError(f"no source path supplied for manifest files: {missing}")
    extra = sorted(set(payload_files) - set(payload["files"]))
    if extra:
        raise CausalCheckpointError(f"payload files not declared in the manifest: {extra}")

    out.mkdir(parents=True, exist_ok=True)
    for relative, source in sorted(payload_files.items()):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        observed = digest_file(target)
        expected = payload["files"][relative]
        if observed != {"bytes": expected["bytes"], "sha256": expected["sha256"]}:
            raise CausalCheckpointError(
                f"{relative}: copied bytes do not match the manifest digest"
            )
    _write_atomic(out / MANIFEST_NAME, _document_bytes(payload))
    # Marker last, and only now: before this line the directory is deliberately
    # not a checkpoint.
    _write_atomic(out / COMPLETE_MARKER_NAME, (payload["manifest_digest"] + "\n").encode())
    return payload


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Reading and verifying
# ---------------------------------------------------------------------------


def read_checkpoint(checkpoint_dir: str | os.PathLike) -> dict[str, Any]:
    """Load a checkpoint manifest, refusing anything not atomically complete."""
    directory = Path(checkpoint_dir)
    manifest_path = directory / MANIFEST_NAME
    marker_path = directory / COMPLETE_MARKER_NAME
    if not manifest_path.is_file():
        raise CausalCheckpointError(f"no {MANIFEST_NAME} in {directory}")
    if not marker_path.is_file():
        raise CausalCheckpointError(
            f"{directory} has no {COMPLETE_MARKER_NAME} marker: the checkpoint was never "
            "completed and must not be restored"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    marker = marker_path.read_text(encoding="utf-8").strip()
    if marker != payload["manifest_digest"]:
        raise CausalCheckpointError(
            f"{COMPLETE_MARKER_NAME} holds {marker[:12]}… but the manifest digests to "
            f"{payload['manifest_digest'][:12]}…: the marker and manifest disagree"
        )
    return payload


def verify_checkpoint(checkpoint_dir: str | os.PathLike) -> dict[str, Any]:
    """Validate the manifest AND re-hash every file the checkpoint carries."""
    directory = Path(checkpoint_dir)
    payload = read_checkpoint(directory)
    drifted: list[str] = []
    for relative, entry in sorted(payload["files"].items()):
        path = directory / relative
        if not path.is_file():
            drifted.append(f"{relative}: missing")
            continue
        observed = digest_file(path)
        if observed["sha256"] != entry["sha256"] or observed["bytes"] != entry["bytes"]:
            drifted.append(
                f"{relative}: sha256 {observed['sha256'][:12]}… != {entry['sha256'][:12]}…"
            )
    if drifted:
        raise CausalCheckpointError(f"checkpoint bytes drifted: {drifted}")
    # The sealed evaluator provenance lives in the checkpoint too, so a later
    # mutable working tree cannot change what this checkpoint says it scored with.
    provenance = payload["evaluator_closure"]["provenance_dir"]
    if not any(name.startswith(f"{provenance}/") for name in payload["files"]):
        raise CausalCheckpointError(
            f"evaluator provenance directory {provenance!r} contributes no files to the "
            "checkpoint, so the scorer is not actually sealed here"
        )
    return {
        "baseline": payload["baseline"],
        "boundary": payload["boundary"],
        "evaluator_closure_digest": payload["evaluator_closure"]["digest"],
        "manifest_digest": payload["manifest_digest"],
        "n_archive_entries": sum(
            len(a["contents"]) for a in payload["archives"].values()
        ),
        "n_archives": len(payload["archives"]),
        "n_documents": len(payload["documents"]),
        "n_files": len(payload["files"]),
        "verified": True,
    }


# ---------------------------------------------------------------------------
# Restore plan — typed operations, never a trusted shell string
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestoreOp:
    """One restore step. ``sha256`` is what the step must produce."""

    kind: str
    target: str
    source: str | None = None
    payload: Any = None
    sha256: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "sha256": self.sha256,
            "source": self.source,
            "target": self.target,
        }


def restore_plan(payload: Mapping[str, Any]) -> list[RestoreOp]:
    """The typed plan that materialises this checkpoint into a fresh directory.

    Pure: builds the plan from the manifest alone and touches no filesystem. Every
    file and document op carries the digest the result must have, so a plan can be
    executed by anything and still be checked by :func:`verify_restored`.
    """
    validate_manifest(payload)
    ops: list[RestoreOp] = [RestoreOp(kind="assert_empty", target=".")]
    directories = sorted(
        {str(Path(relative).parent) for relative in payload["files"]}
        | {str(Path(entry["path"]).parent) for entry in payload["documents"].values()}
    )
    for directory in directories:
        if directory not in (".", ""):
            ops.append(RestoreOp(kind="mkdir", target=directory))
    for relative, entry in sorted(payload["files"].items()):
        ops.append(
            RestoreOp(kind="copy_file", source=relative, target=relative,
                      sha256=entry["sha256"])
        )
    for name in sorted(payload["archives"]):
        entry = payload["archives"][name]
        ops.append(
            RestoreOp(kind="extract_archive", source=entry["path"],
                      target=entry["extract_to"], sha256=entry["sha256"],
                      payload={"contents": dict(entry["contents"])})
        )
    for name in sorted(payload["documents"]):
        entry = payload["documents"][name]
        ops.append(
            RestoreOp(kind="write_json", target=entry["path"], payload=entry["payload"],
                      sha256=entry["sha256"])
        )
    return ops


def verify_restored(target_dir: str | os.PathLike, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash a materialised directory against the manifest. Fail closed."""
    validate_manifest(payload)
    target = Path(target_dir)
    problems: list[str] = []
    for relative, entry in sorted(payload["files"].items()):
        path = target / relative
        if not path.is_file():
            problems.append(f"{relative}: missing")
            continue
        observed = digest_file(path)
        if observed["sha256"] != entry["sha256"]:
            problems.append(f"{relative}: sha256 mismatch")
        elif observed["bytes"] != entry["bytes"]:
            problems.append(f"{relative}: byte count mismatch")
    for name in sorted(payload["documents"]):
        entry = payload["documents"][name]
        path = target / entry["path"]
        if not path.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        observed = sha256_hex(path.read_bytes())
        if observed != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 mismatch")
    # Archives are verified by CONTENT, not by archive bytes: what matters is that
    # the restored tree holds the same files, whatever the tar did.
    for name in sorted(payload["archives"]):
        entry = payload["archives"][name]
        root = target / entry["extract_to"]
        for relative, digest in sorted(entry["contents"].items()):
            path = root / relative
            if not path.is_file():
                problems.append(f"{entry['extract_to']}/{relative}: missing")
            elif sha256_hex(path.read_bytes()) != digest:
                problems.append(f"{entry['extract_to']}/{relative}: sha256 mismatch")
    if problems:
        raise CausalCheckpointError(f"restored state does not match the checkpoint: {problems}")
    return {
        "boundary": payload["boundary"],
        "manifest_digest": payload["manifest_digest"],
        "n_archive_entries": sum(len(a["contents"]) for a in payload["archives"].values()),
        "n_archives": len(payload["archives"]),
        "n_documents": len(payload["documents"]),
        "n_files": len(payload["files"]),
        "verified": True,
    }


