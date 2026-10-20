# Checkpoint and source-provenance utilities

This package contains the causal checkpoint manifest, integrity checks, and
canonical serialization used by the Shinka adapter. Both framework runners use
`source_snapshot.py` to capture the released runtime sources and resources with
content hashes, credential redaction, and persistent per-run snapshot references.
