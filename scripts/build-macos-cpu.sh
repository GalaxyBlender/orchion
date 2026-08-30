#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export GGML_METAL=OFF
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
export CMAKE_OSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET}}"
cargo build --release -p orchion-server --no-default-features --features cpu
