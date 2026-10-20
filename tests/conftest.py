"""Tests run against this checkout, without model or embedding network calls."""
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for relative in (".", "shared/meta-agent-audit", "integrations/openevolve-takehome",
                 "integrations/shinka-takehome"):
    sys.path.insert(0, str(ROOT / relative))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("Regression tests must not make network calls")
    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)
