# Vendored Qwen3-ASR: last-position logits

This is a before/after comparison of **the same vendored backend**, not
another [comparison with Crane](crane-performance.md). Both builds load
the original Qwen3-ASR 0.6B checkpoint. The change keeps all prompt
positions in the transformer and its KV cache, but normalizes and projects
only the last hidden state into the 151,936-token vocabulary. This avoids
materializing logits for the other prompt positions. Batch and streaming
inference both use this path. TTS is unchanged.

The [decoder-kernel follow-up](vendor-speech-performance.md) uses this
build as its before baseline.

**Result:** Keep the narrow optimization for lower peak memory on the
tested long CUDA batch and CPU streaming workloads. It did not produce
a general ASR speedup.

## CUDA ASR results

There were 5,256 MiB free before each measured process. Latency is the
median; brackets show the full observed range. Peaks are the largest
observed CUDA pool allocations.

| Workload | Before, ms [range] | After, ms [range] | Peak MiB, before → after |
| --- | ---: | ---: | ---: |
| Batch, 11 s | 227.3 [226.0–228.1] | 227.9 [226.1–228.7] | 1,619.1 → 1,619.1 |
| Batch, 66 s | 1,780.1 [1,767.4–1,783.8] | 1,769.4 [1,760.9–1,783.1] | 1,884.3 → 1,853.8 |
| Streaming, 11 s session | 489.6 [484.9–491.9] | 486.9 [480.0–489.2] | 1,586.0 → 1,586.0 |
| Streaming, first nonempty partial | 72.4 [71.7–74.2] | 72.5 [71.3–73.0] | — |
| Forced English, 11 s | 211.9 [211.2–212.8] | 211.4 [211.1–213.6] | 1,619.1 → 1,619.1 |
| Forced English, 66 s | 1,683.8 [1,682.9–1,685.0] | 1,701.3 [1,694.2–1,701.8] | 1,884.3 → 1,853.8 |

The 66 s batch saves **30.5 MiB** (1.6% of live peak, or 7.9% of
temporary pool workspace). Resident live allocation stayed at
1,497.5 MiB; the pool returned to 0 MiB live and reserved on model drop.
The change does not shrink the encoder's allocations or the KV cache,
so a request can still reach the same peak in an unaffected stage.
Host peak RSS for the auto-language batch process had a median of
2,182 MiB in both builds; streaming was 2,179 → 2,178 MiB. The
auto-language latency deltas are under 1% with overlapping ranges.
The forced-language long median was 1.0% higher in the single pair of
runs, so this is not a demonstrated CUDA speedup.

CUDA batch text, language and raw output matched exactly before and
after, including when English was forced. All streaming responses
(including feeds without a partial), their fields, and the final
transcript also matched.

The separate [CUDA workspace regression](model-unload-testing.md#temporary-workspace-while-the-model-stays-loaded)
on **240.5 s of synthetic input** (an eight-token limit) peaked at
**2,854 → 2,614 MiB**. In both builds, live allocation returned to the
loaded baseline and the reserved pool was 1,568 MiB after inference.
This stress test measures memory, not transcription accuracy or speed.

## CPU ASR results

| Workload | Before, ms [range] | After, ms [range] |
| --- | ---: | ---: |
| Batch, 11 s | 3,288.9 [3,242.4–3,322.5] | 3,180.4 [3,171.8–3,211.3] |
| Batch, 66 s | 29,906.2 [29,782.1–30,057.9] | 30,141.6 [30,029.9–30,492.7] |
| Streaming, 11 s session | 8,398.9 [8,287.7–8,403.8] | 8,132.1 [8,084.9–8,154.9] |
| Streaming, first nonempty partial | 1,082.2 [1,043.7–1,101.9] | 1,065.5 [1,057.0–1,073.2] |

Process peak RSS was **4,333 → 4,309 MiB** for batch (one process
handles both clips) and **3,988 → 3,860 MiB** for streaming. Short
batch and streaming session medians were about 3% lower; the long
batch median was 0.8% higher. With one fresh process per build per
workload, these are observations, not a general CPU speedup claim.
The batch transcripts and full streaming response sequences matched
exactly before and after.

## TTS regression on CUDA

No TTS implementation changed. All four first-repetition WAVs matched
byte for byte between the builds. Each request produced finite 24 kHz
audio, with no OOM. The timings below are smoke-check measurements, not
a TTS optimization comparison.

| Workload | Output | Median ms, before → after | CUDA pool peak MiB, before → after |
| --- | ---: | ---: | ---: |
| CustomVoice 0.6B, short | 6.00 s | 1,876.6 → 1,892.5 | 2,916.8 → 2,916.8 |
| CustomVoice 0.6B, long | 26.64 s | 8,500.3 → 8,531.3 | 2,918.0 → 2,918.0 |
| Base 0.6B, clone | 5.04 s | 1,837.1 → 1,839.1 | 2,911.8 → 2,911.8 |
| VoiceDesign 1.7B | 4.32 s | 1,540.0 → 1,543.9 | 4,737.2 → 4,737.2 |

The release unit tests and opt-in CUDA tests also passed, including
streaming sessions that are finished or abandoned, ASR/TTS unload and
reload, and the long ASR and TTS workspace budgets.

## Method

- Linux x86-64, AMD Ryzen AI 9 HX 470, RTX 3080 (10 GiB), NVIDIA
  driver 615.78.08, CUDA Toolkit 13.4, Candle 0.11.1, Rust 1.100.0,
  release build without the FlashAttention feature. Both builds use
  local dependencies and weights.
- Batch ASR uses the same 11 s English PCM as the Crane report, then
  the clip repeated six times (66 s). Streaming feeds the 11 s clip
  in 1 s calls, without real-time sleeps, to a session with the default
  2 s ASR chunk size. Time to first nonempty partial is compute wall
  time, not the time a microphone user would wait for the audio.
- One warm-up per process, then three timed requests per case. Timed
  requests exclude model loading and WAV writing. Automatic-language
  CUDA ASR alternates preserved before/after binaries (`ABBAAB`),
  in fresh processes per workload. CPU tests and forced-language CUDA
  tests use one process per build and workload. CPU runs use
  `RAYON_NUM_THREADS=8`, `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`.
- CUDA peak is the highest live allocation in the CUDA memory pool
  measured in the timed cases, not total process VRAM. Host RSS is
  process-wide, including model load, warm-up and inference, measured
  by Linux `wait4` (`ru_maxrss`). It cannot identify a single request
  or a specific buffer.
- TTS regression cases use the same weights, inputs and sampling
  settings in both binaries (`seed=42`, `top_k=50`, 512-frame limit).
  The saved WAVs are 24 kHz mono PCM16; this checks for changes, not
  audio quality.

These results cover only the listed weights and Linux CPU/CUDA paths.
They do not measure HTTP overhead, other accelerators, or transcription
accuracy on a corpus.

## Reproduce

Build the same benchmark harness once with the old vendor decoder and
once with the last-position change; keep a copy of each executable
before rebuilding. For the old build, use the `decoder.rs` and
`inference.rs` from Orchion `614f1d2` in a scratch tree, keeping the
benchmark harness and dependency lockfile fixed. Build without
downloading any dependencies once the local Cargo cache is populated:

```sh
export CUDA_COMPUTE_CAP=86 CUDARC_CUDA_VERSION=13000
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
cargo build --release --offline --locked -p orchion-crane \
  --features cuda,orchion-qwen3/cuda --example speech_bench
```

Use `jfk.pcm` and `reference.wav` from the
[speech benchmark setup](crane-performance.md#reproduce). The commands
below expect `exe` to point to either saved executable. Run again with
`device=cpu` for ASR CPU results. Record free GPU memory before each
CUDA run and capture each command's peak RSS separately (for example,
with `/usr/bin/time -f 'max_rss_kib=%M'` where available).

```sh
models="$PWD/data/models/Qwen"
asr="$models/Qwen3-ASR-0.6B"
"$exe" legacy "$device" asr "$asr" "$work/jfk.pcm"
"$exe" legacy "$device" asr-stream "$asr" "$work/jfk.pcm"
"$exe" legacy "$device" asr "$asr" "$work/jfk.pcm" english

mkdir -p "$output"
"$exe" legacy cuda:0 tts "$models/Qwen3-TTS-12Hz-0.6B-CustomVoice" "$output"
"$exe" legacy cuda:0 clone "$models/Qwen3-TTS-12Hz-0.6B-Base" \
  "$work/reference.wav" "$output"
"$exe" legacy cuda:0 design "$models/Qwen3-TTS-12Hz-1.7B-VoiceDesign" "$output"
```

Compare the `text=`, `language=`, `raw_output=` and `transcripts=` ASR
outputs between the two builds. Each TTS workload saves a WAV from the
first timed repetition for byte comparison.
