# Crane speech backend performance

This compares the production adapters in `libs/orchion-qwen3` (baseline,
Orchion `614f1d2`) and `libs/orchion-crane` (`crane-core` 26.9.0). Both
use Candle 0.11.1. The Crane adapter is not selected by the server.

The [vendored ASR follow-up](vendor-asr-prefill-performance.md) compares
the original and last-position-logits builds of the existing backend.
The [decoder-kernel follow-up](vendor-speech-performance.md) measures
subsequent ASR and TTS speedups against a fresh Crane reference run.

**Decision:** Crane is worth evaluating for batch ASR: it reduced latency
on CPU and CUDA and produced the same text on the two inputs tested.
Do not switch default TTS yet: the short CustomVoice and clone cases ran
faster on CUDA, but used more VRAM, and the long CUDA case failed.

## CUDA results

Linux x86-64, AMD Ryzen AI 9 HX 470, RTX 3080 (10 GiB), NVIDIA driver
615.78.08, CUDA Toolkit 13.4, Rust 1.100.0, `--release`, no FlashAttention
feature. The GPU also drove the desktop; it had 5,328 MiB free at the
start of the TTS run.

Times and RTF are medians of three requests; peak is the largest observed
CUDA pool high-water mark. Lower RTF is better.

| ASR 0.6B input | Baseline | Crane | RTF, baseline → Crane | Peak MiB, baseline → Crane |
| --- | ---: | ---: | ---: | ---: |
| 11 s | 237 ms | 130 ms | 0.0215 → 0.0118 | 1,619 → 1,870 |
| 66 s | 1.782 s | 0.681 s | 0.0270 → 0.0103 | 1,884 → 2,478 |

Crane cut batch ASR wall time by 45% and 62%, with peaks 251 MiB and
594 MiB higher. Both returned identical English transcripts on these
inputs (22 and 110 words). The long clip contains six repetitions, but
both transcripts contain five. This is not a recognition accuracy
evaluation and does not cover streaming ASR.

| TTS 0.6B workload | Baseline time / audio | Crane time / audio | RTF, baseline → Crane | Peak MiB, baseline → Crane |
| --- | ---: | ---: | ---: | ---: |
| CustomVoice, short | 1.880 s / 6.00 s | 1.228 s / 6.56 s | 0.313 → 0.187 | 2,917 → 3,543 |
| CustomVoice, long | 8.496 s / 26.64 s | CUDA out of memory | 0.319 → — | 2,918 → failed |
| Base, voice clone | 1.836 s / 5.04 s | 1.676 s / 7.68 s | 0.364 → 0.218 | 2,912 → 4,447 |

Crane short CustomVoice cut wall time by 35%, and time per second of
audio by 40%, at the cost of 626 MiB higher peak. Base clone also took
40% less time per second of audio, but was only 9% faster per request
and used 1,535 MiB more at peak. The two runtimes generated different
durations; the data does not establish equivalent synthesis quality.
This Crane version is not a replacement for the bounded CUDA TTS
decoder on a VRAM-constrained host. The Crane failure reproduced in a
second process with 5,329 MiB free before loading. The pool recorded
3,699 MiB before the error; failed allocations do not count toward that
peak, so it is not a measurement of the required VRAM.

The baseline performs high-rate waveform upsampling in 32-frame
windows; the published Crane batch decoder calls `chunked_decode` with
300 frames and 25 frames of left context. These are different memory
strategies. The timings do not profile which operation failed.

| CUDA load snapshot | Baseline time / used MiB | Crane time / used MiB |
| --- | ---: | ---: |
| ASR | 2.36 s / 1,498 | 2.65 s / 1,748 |
| CustomVoice | 28.29 s / 2,380 | 32.16 s / 2,658 |
| Base | 26.03 s / 2,413 | 2.83 s / 2,692 |

The load timings are one observation per process and filesystem
cache/scheduling was not controlled; the Base load timings in
particular should not be treated as a speedup. After dropping each
measured model the pool returned to 0 MiB used and reserved, including
the CustomVoice OOM recheck. This does not test server eviction or
reload.

## Lower VRAM without modifying Crane

### Batch ASR: `audio_config.conv_chunksize`

The published Crane crate reads this field from the `-hf` checkpoint's
`config.json`. It limits how many encoder windows are convolved together;
it does not split the user audio into separate transcription requests or
reset decoder context. The official 0.6B configuration sets it to 500.

With 2,588 MiB free before each run on the same RTX 3080, the 66-second
case behaved as follows. Times are medians of three completed requests.

| `conv_chunksize` | 11 s: time / peak MiB | 66 s: time / peak MiB |
| ---: | ---: | ---: |
| 500 (official) | 130 ms / 1,870 | CUDA out of memory |
| 32 | 129 ms / 1,870 | CUDA out of memory |
| 16 | 130 ms / 1,870 | 668 ms / 2,001 |
| 8 | 134 ms / 1,837 | 664 ms / 2,001 |

All successful transcripts matched the earlier run. In the earlier
test with more free VRAM, 500 reached 2,478 MiB on the 66-second clip.
At 16, the observed peak was 477 MiB lower without an observed
latency regression on this fixture. Resident allocation remained
1,748 MiB; this controls temporary convolution memory, not model
weights or all long-input allocations. Start with 16, and keep the
official cache untouched. To provision a separate local load directory:

```sh
model="$PWD/data/models/Qwen/Qwen3-ASR-0.6B-hf"
tuned="$(mktemp -d)"
for file in tokenizer_config.json tokenizer.json model.safetensors; do
  ln -s "$model/$file" "$tuned/$file"
done
jq '.audio_config.conv_chunksize = 16' "$model/config.json" > "$tuned/config.json"
bench crane cuda:0 asr "$tuned" "$work/jfk.pcm"
```

### TTS: bounded decoding via the public API

The opt-in `tts-stream` benchmark consumes
[`generate_speech_streaming`](https://docs.rs/crate/crane-core/26.9.0/source/src/models/qwen3_tts/model.rs)
and copies each PCM tensor to the host before requesting the next one.
The first chunk is 5 frames; subsequent chunks decode up to 25 new frames
plus 25 left-context frames instead of the batch decoder's 300 new
frames. The caller can still return a single WAV. This limits the
decode window, not the generation KV cache.

**End-to-end CUDA:** most of the RTX 3080's memory was already occupied
by other workloads. With 2,588 MiB free, both batch and streaming
loaded without the optional reference-audio encoder (Crane logged OOM
warnings while loading it), then failed with CUDA OOM on the short
warm-up. The earlier full-model batch run started with about 5.2 GiB
free. This follow-up does not show whether full-model streaming
completes the long prompt under those earlier conditions.

**Decoder only, CUDA:** to isolate the suspected bottleneck, the `codec`
benchmark loads the released native speech tokenizer, without the talker,
and feeds the same deterministic synthetic codec IDs to both schedules.
Its loaded CUDA pool use was 883 MiB. At 2,588 MiB free before process
start, with the same RTX 3080 and release settings:

| Identical input | Batch latency / peak | Streaming-window latency / peak |
| --- | ---: | ---: |
| 82 frames (6.56 s) | 143 ms / 1,767 MiB | 373 ms / 1,422 MiB |
| 333 frames (26.64 s) | CUDA out of memory | 1.353 s / 1,422 MiB |

The shorter decode windows saved 345 MiB (20% of total live peak, or
39% of temporary workspace) on 82 frames, but decoding alone took 2.6×
as long. The 333-frame case matches the original long TTS output
duration, not its content. These measurements exclude generation and
cannot be used as end-to-end TTS speed or memory numbers.

The 82-frame WAVs both had 157,440 samples at 24 kHz, but only
58,961 PCM16 samples matched exactly. The maximum absolute sample delta
was 35,547 and the RMS delta was 1,336 PCM16 units. So even for identical
codes, the 25-frame left context is **not waveform-equivalent** on this
fixture. The first 30 frames had a rounding-level delta (0.1 RMS); the
difference jumped when the first five history frames were dropped at
the next chunk. This lower-memory path should not replace the default
before testing audible seams and text fidelity with real speech. The
full streaming generator can also change the repetition-penalty history.

**End-to-end CPU:** streaming took a median 27.62 s for the short
prompt and produced 6.08 s of audio (RTF 4.54), versus 28.65 s and
6.08 s in the earlier batch run. This was a separate run, not a
controlled speedup comparison. The long prompt produced 29.04 s of
audio in each of three runs, with a median latency of 144.53 s
(RTF 4.98). A separate run
through the local Qwen3-ASR 0.6B model transcribed the saved short and
long clips back to their input text exactly; this is only a smoke
check, not a listening test.

If batch generation must stay unchanged, another public-API option is
`generate_codes` once, then `codes_to_pcm` on small overlapping windows,
discarding the overlap from each emitted chunk. This preserves the
batch-generated codec sequence but outputs PCM16 and still needs
waveform/seam verification; simply reusing a 25-frame overlap does not
resolve the difference above.

Base cloning does not expose a streaming iterator or a decode-window
option on `generate_voice_clone`. Shorter reference audio or splitting
target text into separate requests can reduce the amount decoded per
invocation, but can also change voice similarity or prosody. Neither is
a transparent substitute for bounded clone decoding.

## CPU results

Same Ryzen host and release binary, with the explicit `cpu` device and
F32 compute. Times and RTF are medians of three requests.

| ASR 0.6B input | Baseline | Crane | RTF, baseline → Crane |
| --- | ---: | ---: | ---: |
| 11 s | 4.946 s | 3.449 s | 0.450 → 0.314 |
| 66 s | 37.495 s | 19.105 s | 0.568 → 0.290 |

Crane cut batch ASR wall time by 30% and 49%. The transcripts matched
the CUDA run.

| TTS 0.6B workload | Baseline time / audio | Crane time / audio | RTF, baseline → Crane |
| --- | ---: | ---: | ---: |
| CustomVoice, short | 37.021 s / 7.20 s | 28.645 s / 6.08 s | 5.142 → 4.711 |
| CustomVoice, long | No completed sample | No completed sample | — |

CPU short TTS had 23% lower wall time, but also generated less audio;
the per-second speedup was 8%. The 240-second limit applied to the entire
process (load, warm-up, three short requests, then long requests). Both
processes reached that limit without completing their first long
request, so there is no CPU long-synthesis timing. CPU clone, 1.7B
models, Metal, Windows and VoiceDesign were not measured.

## Method

- Each backend/workload runs in a fresh process. One inference warms the
  model, then three sequential requests are timed per case. Load time is
  a single measurement per process, not a median; weights are local and
  the OS page cache is not flushed. HTTP and output encoding are not timed.
  The baseline uses its existing async adapter; Crane is called synchronously.
- ASR takes the same mono PCM16 input converted to `f32`; the long input
  repeats the 11-second clip six times. Both use automatic language
  detection and greedy decoding without a repetition penalty, with a
  512-token limit. The baseline loads the original Qwen checkpoint,
  while Crane uses the official
  [`Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/tree/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c)
  checkpoint. Repacking the original local tensor bytes using the
  official `-hf` header and tensor-name mapping reproduced the
  official SHA-256 (`d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35`),
  so the common tensor payloads are identical for this model.
- CustomVoice uses the same local 0.6B weights, Vivian, and short and long
  Chinese prompts. The Base clone case uses the same 0.6B weights and a
  5.28-second synthetic Ryan reference generated once by the baseline. The baseline
  opens the reference WAV per request; Crane takes predecoded PCM. Both
  run at temperature 0.7, top-p 0.8 and a 512-frame limit. The baseline
  uses seed 42 and top-k 50 to match the primary sampler constants in
  Crane; implementations and EOS behavior still differ.
- RTF is wall time divided by input audio duration for ASR, or output
  audio duration for TTS (lower is better). It helps when generated audio
  lengths differ; it does not measure pronunciation, similarity or how
  much of the prompt was spoken.
- CUDA memory is this process's default pool, not total device usage:
  `used_mib` is live allocation, `peak_mib` its high-water mark during
  the request (including model weights), and `reserved_mib` is backing
  memory held by the pool. The benchmark keeps the primary context
  alive when inspecting the pool after model drop.

## Reproduce

In the example below, set `models` to a directory with the original and
`-hf` Qwen3-ASR 0.6B checkpoints and the 0.6B CustomVoice and Base TTS
checkpoints. The baseline ASR directory must include `tokenizer.json`
(the benchmark used Orchion's existing ModelScope cache). Use an
otherwise idle GPU, and record how much VRAM is free before loading.
Set `CUDA_COMPUTE_CAP` and `CUDARC_CUDA_VERSION` for your machine.

```sh
models="$PWD/data/models/Qwen"
work="$(mktemp -d)"
curl -fLsS https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav \
  -o "$work/jfk.wav"
ffmpeg -nostdin -v error -i "$work/jfk.wav" -ac 1 -ar 16000 -f s16le "$work/jfk.pcm"

export CUDA_COMPUTE_CAP=86 CUDARC_CUDA_VERSION=13000
bench() {
  cargo run --quiet --release --locked -p orchion-crane \
    --features cuda,orchion-qwen3/cuda --example speech_bench -- "$@"
}

bench legacy cuda:0 reference "$models/Qwen3-TTS-12Hz-0.6B-CustomVoice" "$work/reference.wav"
bench legacy cuda:0 asr "$models/Qwen3-ASR-0.6B" "$work/jfk.pcm"
bench crane  cuda:0 asr "$models/Qwen3-ASR-0.6B-hf" "$work/jfk.pcm"
for backend in legacy crane; do
  bench "$backend" cuda:0 tts "$models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
  bench "$backend" cuda:0 clone "$models/Qwen3-TTS-12Hz-0.6B-Base" "$work/reference.wav"
done

custom="$models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
mkdir -p "$work"/{full-batch,full-stream,codec-batch,codec-stream}
bench crane cuda:0 tts "$custom" "$work/full-batch"
bench crane cuda:0 tts-stream "$custom" "$work/full-stream"
bench crane cuda:0 codec "$custom" batch "$work/codec-batch"
bench crane cuda:0 codec "$custom" stream "$work/codec-stream"
```

The optional output directory must exist. It receives a WAV from the
first successful repetition of each case; writing it is outside the
timed region. If a batch command runs out of memory, run the streaming
command separately. The codec and streaming cases use the already
released `crane-core` crate and the locally cached TTS model; with Cargo's
dependencies cached, add `--offline` to the `cargo run` invocation.

To reproduce CPU timings with the same build, pass `cpu` instead of
`cuda:0` and bound each process with `timeout 240`. The example also
supports CPU-only builds without the CUDA features.

The converted `jfk.pcm` fixture used in the measurements has SHA-256
`a29462b8ebd467318000e683b9117ade46230d3255ed2024e7db894abd9b38c9`.
