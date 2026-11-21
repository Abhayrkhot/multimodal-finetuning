#!/usr/bin/env bash
# Run a command in the CPU dev container, e.g.  scripts/dev.sh 'pytest -q'
# Build the image once with:  docker build -f docker/Dockerfile.dev -t mmftdev:py3.12 .
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$HOME/.cache/mmft-hf"
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -v "$HOME/.cache/mmft-hf:/hf" \
  -w /work mmftdev:py3.12 bash -lc "$*"
