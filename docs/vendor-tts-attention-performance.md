# Vendored Qwen3-TTS: CUDA generation attention

Baseline: the [fused RoPE build](vendor-speech-performance.md). This
follow-up measures the Talker and CodePredictor generation paths
without changing the sampling algorithm, seed, `top_k`, or codec
decoder context.

## Where the time goes

Synchronized CUDA stage medians on the RTX 3080, after model warm-up:

| Workload | CodePredictor | Talker | Per-frame sampling | Waveform decode |
| --- | ---: | ---: | ---: | ---: |
| CustomVoice, 75 frames (6.00 s) | 844.3 ms | 322.2 ms | 77.8 ms | 197.8 ms |
| CustomVoice, 333 frames (26.64 s) | 3,749.9 ms | 1,703.9 ms | 346.1 ms | 832.1 ms |
| Base clone, 63 generated frames | 705.3 ms | 293.9 ms | 65.6 ms | 347.6 ms |
| VoiceDesign, 54 frames (4.32 s) | 618.2 ms | 406.2 ms | 56.4 ms | 133.1 ms |

Each generated frame contains one sampled semantic token and 15
acoustic tokens. CodePredictor runs 2-token prefill and then 14
single-token decoder steps per frame, reusing its request-local KV
buffers. Base cloning decodes 129 frames including the reference
prefix before trimming it out of the returned audio.

The synchronized stage values are diagnostic: they serialize GPU
work that can otherwise overlap with host work. Per-frame sampling
includes penalties, sampling itself, and token readback. It does not
include first-token sampling. The table does not include initial
prefill or miscellaneous generation work.

## Changes evaluated

- **Grouped single-token attention (not kept).** It removed expanded
  K/V copies, but all four fixed-seed WAVs changed. Short CustomVoice
  changed from 6.00 to 5.60 s, and long CustomVoice generated different
  audio even though its duration was unchanged. The long runtime fell
  from 6.64 to 5.89 s in one process per build; this is not a
  waveform-equivalent speedup.
- **Packed Q/K/V plus grouped attention (not kept).** VoiceDesign 1.7B
  loaded but failed its warm-up at 5,327 MiB initially free. Its CUDA
  live load peak rose from 4,375 to 4,863 MiB. In this implementation
  the loader retains the source weights until the model is assembled,
  so projections temporarily have both original and packed storage.
- **RoPE tables.** Store CUDA tables in the model's compute dtype so
  Q/K do not each recast the same values at every layer. Talker MRoPE
  also uses a resident table for up to 2,048 positions instead of
  rebuilding frequencies and trigonometric values in each layer;
  offsets beyond the cache use the original calculation. This does
  not alter the attention kernel or shrink model context.
- **Acoustic code writes.** CodePredictor writes integer codes directly
  to the already allocated result tensor instead of replacing the
  whole tensor after each code. This does not change code ordering,
  sampling, cache context, or waveform decode.

## End-to-end CUDA results

RTX 3080, local checkpoints, profiling disabled. Each build ran in
two fresh processes per workload, in after/before/before/after order,
with one warm-up and three timed requests per process. Times are
medians with the six-request range in brackets; model load is excluded.

| Workload | Output | Before, ms [range] | After, ms [range] | RTF before → after | Live pool peak, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CustomVoice, short | 6.00 s | 1,551 [1,471–1,668] | 1,383 [1,378–1,474] | 0.259 → 0.231 | 2,916.8 → 2,917.1 |
| CustomVoice, long | 26.64 s | 7,081 [6,677–8,181] | 6,298 [6,266–6,317] | 0.266 → 0.236 | 2,918.0 → 2,918.2 |
| Base clone | 5.04 s | 1,503 [1,485–2,073] | 1,411 [1,404–1,417] | 0.298 → 0.280 | 2,911.8 → 2,912.1 |
| VoiceDesign | 4.32 s | 1,252 [1,243–1,381] | 1,178 [1,170–1,180] | 0.290 → 0.273 | 4,737.2 → 4,737.4 |

Long CustomVoice latency fell 11.1% by the interleaved median. The
baseline was noisier; comparing its fastest measured request with
the slowest optimized request still shows a 5.4% reduction. The
[earlier Crane short CustomVoice run](vendor-speech-performance.md#interleaved-cuda-results)
took 1,242 ms at RTF 0.189 and peaked at 3,542.5 MiB, producing 6.56 s
of audio rather than 6.00 s. That is a historical reference, not a
same-round timing or waveform comparison.

The live peak and post-request model allocation increased by no more
than 0.3 MiB in these four workloads. Pool reservation was unchanged:
2,464 MiB for CustomVoice, 2,528 MiB for cloning, and 4,384 MiB for
VoiceDesign. Both live and reserved allocations returned to zero on
model drop. Peak process RSS rose by about 2–3 MiB.

## Output verification

The synchronized diagnostic runs compared every generated codec ID
against the baseline, including warm-ups. Short and long CustomVoice,
Base clone, and VoiceDesign matched exactly, as did the four exported
24 kHz PCM16 WAVs. Base cloning compared the generated codec IDs
before the original reference prefix was appended.

The vendored TTS library's 201 ordinary unit tests passed in an
isolated local copy. The offline test manifest omitted optional
backends and benchmark-only dependencies that were not used by `--lib`.
The CUDA RoPE cache test also passed, including the last cached row
and offsets past the cache. The workspace's 128 unit tests, one doc
test, six real-model CUDA checks, non-CUDA build, and strict Clippy
passed. CUDA and CPU ASR outputs (including streaming and forced
English on CUDA) matched the preceding build. The long TTS workspace
check returned to the loaded baseline with a 3,086 MiB live peak.

See [Windows CUDA speech validation](windows-speech-validation.md)
for the native Release server checks.

## Reproduce

Use the [offline build and local TTS fixtures](vendor-asr-prefill-performance.md#reproduce).
Run the `tts`, `clone`, and `design` workloads with profiling off for
end-to-end measurements. Set `ORCHION_TTS_PROFILE=1` for the diagnostic
run to print synchronized stage times and exact generated codec IDs.
It logs full codec ID sequences and is meant only for controlled
benchmarks, not ordinary inference. Compare WAVs from the same seed
and settings and compare RTF when audio duration differs across
backends.
