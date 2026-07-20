#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_COMPUTE_CAP="80"

cargo build --release -p orchion-server --no-default-features --features cuda
