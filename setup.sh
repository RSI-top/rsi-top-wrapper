#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v npm >/dev/null || { echo "Install Node.js >=22.19.0 (including npm)." >&2; exit 1; }
node -e 'const [a,b]=process.versions.node.split(".").map(Number); if(a<22 || (a===22 && b<19)){console.error("Node.js >=22.19.0 is required"); process.exit(1)}'
uv sync --frozen --python 3.11
npm ci --no-audit --no-fund
echo "Setup complete. See README.md for run commands."
