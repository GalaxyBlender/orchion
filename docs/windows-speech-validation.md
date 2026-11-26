# Windows CUDA speech validation

The current vendored ASR and TTS optimizations were built and tested
natively on Windows 11 (build 26300), with MSVC 19.52, Rust 1.100.0,
CUDA 13.4, NVIDIA driver 621.40, and an RTX 3080 running under WDDM.
The Release server build used `CUDA_COMPUTE_CAP=80`, cached
dependencies, and local checkpoints. Tests used an
isolated loopback server; the deployed server was not replaced.

Both the CUDA and non-CUDA Release builds of `orchion-qwen3` passed
26 tests. The opt-in real-model CUDA streaming lifecycle test also
passed. The Linux-only `cuda_unload` integration target does not run
on Windows; the HTTP regression below checks unload through WDDM
instead.

## HTTP regression

- ASR 0.6B recognized the JFK fixture with automatic detection and
  with English forced. The WebSocket returned five nonempty partials
  and a final transcript matching the batch result.
- TTS returned valid 24 kHz WAVs for 0.6B CustomVoice (6.00 s),
  0.6B Base voice cloning with reference audio (5.12 s), and 1.7B
  VoiceDesign (4.32 s). For each mode, repeating the request with
  `seed=42` and `top_k=50` returned a byte-identical WAV.
- The [model unload regression](model-unload-testing.md) sent two
  requests per round and explicitly unloaded the model, three times
  per model:

| Model | Loaded WDDM dedicated memory | After unload | Before load |
| --- | ---: | ---: | ---: |
| ASR 0.6B | 2,261–2,293 MiB | 235 MiB | 235 MiB |
| TTS 0.6B CustomVoice | 2,709 MiB | 235 MiB | 235 MiB |

WDDM values are for the server process on the test adapter, not peak
inference allocations. The 235 MiB baseline includes the initialized
CUDA runtime. No model memory accumulated across unloads.

## Standalone CUDA benchmark

The Windows offline cache lacked `crossterm_winapi 0.9.1`, so the
Crane example could not build. A temporary harness removed only the
Crane-specific workloads and dependencies from `speech_bench`. It
used the same production `orchion-qwen3` code, workload inputs, CUDA
pool counters, and locked dependency versions. No sources were
downloaded.

Each workload ran in a fresh process on the second RTX 3080, with
one warm-up and three timed runs. Latencies are medians, excluding
model load; profiling was disabled. RTF is elapsed time divided by
audio duration, so a TTS RTF above 1 is slower than real time.

| Workload | Audio | Latency | RTF | Live peak | Live after request | Reserved after request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASR automatic, short | 11.00 s | 482 ms | 0.044 | 1,619.1 MiB | 1,497.5 MiB | 2,016 MiB |
| ASR automatic, long | 66.00 s | 2,020 ms | 0.031 | 1,853.8 MiB | 1,497.5 MiB | 2,016 MiB |
| ASR forced English, short | 11.00 s | 439 ms | 0.040 | 1,619.1 MiB | 1,497.5 MiB | 1,984 MiB |
| ASR forced English, long | 66.00 s | 2,036 ms | 0.031 | 1,853.8 MiB | 1,497.5 MiB | 1,984 MiB |
| ASR streaming | 11.00 s | 1,106 ms | 0.101 | 1,586.0 MiB | 1,497.5 MiB | 2,016 MiB |
| TTS CustomVoice, short | 6.00 s | 6,714 ms | 1.119 | 2,917.1 MiB | 2,379.9 MiB | 2,464 MiB |
| TTS CustomVoice, long | 26.64 s | 29,220 ms | 1.097 | 2,918.2 MiB | 2,379.9 MiB | 2,464 MiB |
| TTS Base clone | 5.04 s | 6,443 ms | 1.278 | 2,912.1 MiB | 2,421.5 MiB | 2,528 MiB |
| TTS VoiceDesign | 4.32 s | 4,793 ms | 1.109 | 4,737.4 MiB | 4,308.0 MiB | 4,384 MiB |

The streaming benchmark emitted its first nonempty partial at a
median of 190 ms, feeding PCM without waiting for real-time capture.
Live allocation was stable across the three requests in each case;
Base clone retained an additional 7.8 MiB after warm-up versus
immediately after load. After model drop, live and reserved CUDA pool
bytes both returned to zero in every case. These counters exclude the
WDDM context overhead measured by the HTTP regression.

Batch transcripts and streaming partials and finals matched the
optimized Linux build exactly. All four fixed-seed TTS cases also
matched its full codec ID sequences and WAV SHA-256 hashes.

The [optimized Linux reference](vendor-tts-attention-performance.md#end-to-end-cuda-results)
took 6.30 s for long CustomVoice, versus 29.22 s on this Windows
host, with the same output and the same 2,918.2 MiB live CUDA peak.
In the separate synchronized Windows profile, CodePredictor took
18.52 s, Talker 6.46 s, and waveform decoding 3.01 s for this case.
The hosts and software stacks differ; these numbers are not a
Windows before/after comparison and do not identify the cause of the
gap. No Windows Crane latency comparison was measured.
