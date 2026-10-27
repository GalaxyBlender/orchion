#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-80}"

# Candle 0.9's cudarc supports CUDA 13.0 bindings, not newer 13.x version detection.
nvcc_version="$(nvcc --version)"
if [[ -z "${CUDARC_CUDA_VERSION:-}" && "$nvcc_version" == *"release 13."* ]]; then
  export CUDARC_CUDA_VERSION="13000"
fi

cargo build --release -p orchion-server --no-default-features --features cuda

# ort copies providers beside build artifacts, which can be outside target/ when
# Cargo's shared build-dir is configured. The server needs them beside its binary.
python3 - <<'PY'
import json
from pathlib import Path
import shutil
import subprocess

metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--no-deps"], text=True
))
build = Path(metadata["build_directory"]) / "release"
target = Path(metadata["target_directory"]) / "release"
for name in ("libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so"):
    source = build / name
    destination = target / name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
PY
