# CUDA model unload regression tests

`scripts/test-model-unload.py` exercises the actual HTTP server. Each round
performs multiple inference requests, waits for the model to unload, checks
**per-process GPU memory on every adapter**, then makes the next request cold
load the model again. By default it exercises idle eviction; `--explicit`
uses `POST /api/models/unload` instead. It fails if there is no observable GPU
allocation, if post-unload usage exceeds the baseline allowance, or if successive
unloads accumulate GPU memory.

Run it **on the same Linux or Windows machine as an isolated CUDA-enabled
server**, using an unused local port. Start that server with a copy of the
configuration: enable only the model(s) under test, use local weights, disable
authentication, and set `idle_timeout = "12s"` for the enabled services. Set
`device = "cuda:0"` on ASR/TTS/OCR/OCR-VL and `gpu_layers = "all"` in each
LLM/embedding model's `runtime`. The model is still warm between requests, but idle cleanup
can run after the Tokio blocking workers from inference have exited. Wait for
the server to be ready and leave all models unloaded before starting the test.
Do not point these tests at a live workload.

On Linux the script uses `nvidia-smi` per-process accounting. On Windows it uses
WDDM GPU Process Memory counters for dedicated usage. It requires Python 3 and
curl. Use one fresh server process per case to keep runtime initialization for
different backends out of the baseline. For ASR supply WAV containing speech;
for OCR use a small image with readable text and configure a layout model on
the deployment. The OCR cases alternate between JSON and Markdown to exercise
plain recognition and layout. Repeat `--audio` or `--image` to exercise
different input sizes within a round.

```sh
# Replace $pid with the actual isolated server process ID.
python3 scripts/test-model-unload.py asr --url http://127.0.0.1:19090 --pid "$pid" \
  --model alibaba/qwen3-asr-1.7b --audio speech.wav --audio longer-speech.wav
python3 scripts/test-model-unload.py tts --url http://127.0.0.1:19090 --pid "$pid"
python3 scripts/test-model-unload.py ocr --url http://127.0.0.1:19090 --pid "$pid" --image document.png
python3 scripts/test-model-unload.py ocr-vl --url http://127.0.0.1:19090 --pid "$pid" --image document.png
python3 scripts/test-model-unload.py llm --url http://127.0.0.1:19090 --pid "$pid"
python3 scripts/test-model-unload.py embedding --url http://127.0.0.1:19090 --pid "$pid"
```

The default allowance is 256 MiB per adapter for persistent context/runtime
overhead, and no later unload may exceed the first by more than 128 MiB per
adapter. An unloaded model is not expected to force the entire CUDA runtime to
exit. Use `--rounds` and `--requests` to check longer runs.

For a focused Linux regression of Qwen3 ASR/TTS, the library test keeps a CUDA
primary context alive and drops the model on a new thread after three
inferences. Run each test separately with the appropriate weights:

```sh
export ORCHION_ASR_TEST_MODEL_DIR="$PWD/data/models/Qwen/Qwen3-ASR-0.6B"
export ORCHION_TTS_TEST_MODEL_DIR="$PWD/data/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
export CUDA_COMPUTE_CAP=86 # RTX 3080
cargo test --release --locked -p orchion-qwen3 --features asr,tts,cuda \
  --test cuda_unload asr_releases_gpu_memory_after_multiple_requests -- --exact --ignored --nocapture
cargo test --release --locked -p orchion-qwen3 --features asr,tts,cuda \
  --test cuda_unload tts_releases_gpu_memory_after_multiple_requests -- --exact --ignored --nocapture
```

### Temporary workspace while the model stays loaded

These Linux CUDA tests check a four-minute ASR input and a long Chinese TTS
request. They assert a peak CUDA memory-pool budget and verify that in-use bytes
return to the loaded baseline without unloading the model. Reserved bytes may
remain slightly above baseline because of allocator granularity. Run the tests
on an otherwise idle test GPU, with the model directories set as above:

```sh
cargo test --release --locked -p orchion-qwen3 --features asr,cuda \
  --test cuda_unload long_asr_bounds_workspace_and_releases_it_without_unloading_the_model \
  -- --exact --ignored --nocapture
cargo test --release --locked -p orchion-qwen3 --features tts,cuda \
  --test cuda_unload long_tts_bounds_workspace_and_releases_it_without_unloading_the_model \
  -- --exact --ignored --nocapture
```

For a Windows comparison, measure the isolated server PID's dedicated WDDM
memory after explicit model load, during inference, after the response, and
after unload. Sample through the whole request rather than using only a
post-request reading. Keep the GPU, model, input and generation options fixed
across binaries, and record whether another process loads a model on the same
adapter. Compare the outputs separately; these memory tests are not recognition
accuracy or speech quality benchmarks.
