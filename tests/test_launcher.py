import itertools
import json
import subprocess
import sys

import pytest

from tests.helpers import ROOT


@pytest.mark.parametrize("method,framework,provider", itertools.product(
    ["baseline", "rsi-top"], ["openevolve", "shinka"], ["deepseek", "gemini"]))
def test_public_dry_run_is_credential_free_and_does_not_write(tmp_path, method, framework, provider):
    output = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), method, "--framework", framework,
         "--provider", provider, "--output", str(output), "--dry-run"],
        capture_output=True, text=True, check=True, cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    spec, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert spec["framework"] == framework and spec["provider"] == provider
    if method == "rsi-top":
        assert "--minimum-run-age-seconds" in spec["command"]
    assert not output.exists()
