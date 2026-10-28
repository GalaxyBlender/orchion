# OCR memory regression tests

The opt-in tests in `libs/orchion-ocr/tests/runtime_memory.rs` exercise the public
`OcrEngine` interface with real models. They cover traditional OCR, standalone
layout detection, traditional OCR with layout, OCR-VL tasks, and OCR-VL with layout.
They are not mocked allocator tests or throughput benchmarks.

## Requirements

- Linux, local model assets, and representative document images.
- CUDA tests additionally require an NVIDIA GPU, `nvidia-smi`, and the CUDA build
  dependencies used by `scripts/build-linux-cuda.sh`.
- The complete matrix uses PP-OCRv5 Server, PP-OCRv6 Tiny, PP-DocLayoutV3 and
  PaddleOCR-VL-1.6 in the normal downloader cache layout. Individual cases only
  require their selected models.

The build example below targets an RTX 3080 with CUDA 13; select the matching
compute capability and CUDA bindings for other hardware/toolkit versions.

```sh
export ORCHION_OCR_TEST_MODELS="$PWD/data/models"
export ORCHION_OCR_TEST_IMAGES="/absolute/path/document.png"
export ORCHION_OCR_TEST_REPEATS=6

CUDA_COMPUTE_CAP=86 CUDARC_CUDA_VERSION=13000 \
  cargo test --release -p orchion-ocr --features ocr,ocr-vl,cuda \
  --test runtime_memory --no-run --message-format=json > target/ocr-memory-build.jsonl
```

With a shared Cargo build directory, provider libraries may not be next to the
test executable. Stage the executable and providers together before running:

```sh
python3 - <<'PY'
import json
from pathlib import Path
import shutil

rows = [json.loads(line) for line in Path("target/ocr-memory-build.jsonl").read_text().splitlines()]
executable = next(Path(row["executable"]) for row in rows
                  if row.get("reason") == "compiler-artifact"
                  and row["target"]["name"] == "runtime_memory"
                  and row.get("executable"))
destination = Path("target/ocr-memory-tests")
destination.mkdir(exist_ok=True)
shutil.copy2(executable, destination / "runtime_memory")
for name in ("libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so"):
    shutil.copy2(executable.parent.parent / name, destination / name)
PY

target/ocr-memory-tests/runtime_memory cuda_traditional_server \
  --exact --ignored --nocapture --test-threads=1
```

Run each CUDA test in its own process, sequentially, so an earlier model's CUDA
context or cached allocations cannot obscure the next model's measurements.
Other cases are `cuda_traditional_tiny`, `cuda_layout`,
`cuda_traditional_layout_server`, `cuda_traditional_layout_tiny`, `cuda_vl_tasks`,
`cuda_vl_layout`, `cpu_traditional_server`, `cpu_traditional_tiny`, `cpu_layout`,
and `cpu_vl_task` (four generated tokens per request).

## What is checked

- Repeated identical requests must succeed and retain the same text.
- One round warms every supplied image/task/format for ONNX. CUDA VL uses two
  rounds: spotting enlarges the image, and alternating the regular and enlarged
  shapes needs a second traversal before Candle's reusable allocator pool settles.
  Subsequent rounds must not grow post-request GPU memory by more than 128 MiB,
  or CPU RSS by more than
  256 MiB, relative to that warmed state. These allowances accommodate allocator
  granularity and runtime bookkeeping while catching the multi-GiB regression.
- CUDA requests must actually allocate GPU memory; registration failures are not
  an acceptable CPU substitute.
- Logs include per-request latency, process memory and memory after dropping the
  engine. A plateau is not proof that all idle allocations were returned.

For changing shapes, set `ORCHION_OCR_TEST_IMAGES` to a colon-separated sequence
of images, including narrow/wide text and different page sizes. Each sequence is
replayed in the same order. For VL use a small document crop first; the six task
types and three structured response formats each invoke real generation, capped
at 32 new tokens to keep this a bounded memory regression rather than a document
accuracy test. Increase `ORCHION_OCR_TEST_REPEATS` to 30 for longer confirmation.

The tests do not establish Metal/WebGPU behavior, accuracy on every model size,
or memory bounds for arbitrary page dimensions and generation lengths.
For layout without table-structure assets, use a prose page: a detected table
requires its configured structure model and is not a memory-allocation failure.
