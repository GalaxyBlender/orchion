#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cargo build --release -p orchion-server --no-default-features --features metal
