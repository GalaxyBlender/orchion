# Vendored speech performance: decoder kernels

This follow-up takes the [last-position logits build](vendor-asr-prefill-performance.md)
as the before baseline. It keeps the same production adapters and
model capabilities. Crane 26.9.0 is a reference measurement, not a
dependency of the production backend.

The [grouped decode follow-up](vendor-asr-decode-performance.md) profiles
the remaining ASR gap and tests a faster single-token CUDA path.

**Result:** On the RTX 3080, vendor ASR latency fell 11–21% in the
interleaved cases, and CustomVoice fell 22%. Vendor CUDA pool peaks
and the checked outputs did not change. Crane still leads on batch
ASR and short CustomVoice speed; the vendor uses less peak VRAM in
those cases and still completes the long TTS case.

## Changes kept

- **ASR:** use Candle's fused half-split RoPE on CUDA. The MRoPE
  frequency mapping and KV cache are unchanged. On CUDA, skip the
  all-zero mask for a single new decoder token. Prefill and non-CUDA
  decoding keep their masks.
- **TTS:** use the same fused RoPE primitive in the CUDA talker and
  code predictor. Sampling, codec generation order and waveform
  decoding are unchanged. CPU and Metal keep the existing rotation.

## Interleaved CUDA results

All these processes started with 5,328 MiB free. Latencies are medians
in milliseconds with measured ranges in brackets. Vendor peaks are
unchanged before → after; each build has six timed requests per case.
The separate Crane reference has three requests per successful case.

| Workload | Vendor before, ms [range] | Vendor after, ms [range] | Latency reduction | Crane, ms | Live peak MiB, vendor / Crane |
| --- | ---: | ---: | ---: | ---: | ---: |
| ASR batch, 11 s | 227.85 [227.3–229.0] | 185.80 [185.3–186.4] | 18.5% | 130.1 | 1,619.1 / 1,870.1 |
| ASR batch, 66 s | 1,784.25 [1,778.2–1,794.5] | 1,586.30 [1,568.5–1,604.8] | 11.1% | 683.9 | 1,853.8 / 2,478.4 |
| ASR streaming, 11 s | 490.30 [485.6–492.5] | 397.15 [393.3–399.3] | 19.0% | Not supported | 1,586.0 / — |
| ASR first streaming partial | 73.10 [72.2–73.4] | 57.85 [57.3–58.2] | 20.9% | Not supported | — |
| TTS CustomVoice, short | 1,895.75 [1,887.8–1,903.9] | 1,480.50 [1,471.9–1,483.4] | 21.9% | 1,242.3 | 2,916.8 / 3,542.5 |
| TTS CustomVoice, long | 8,595.55 [8,535.1–8,632.8] | 6,696.15 [6,623.0–6,739.7] | 22.1% | OOM | 2,918.0 / — |

Vendor short TTS generated 6.00 s of audio in both builds. Its
real-time factor fell from 0.316 to 0.247, versus 0.189 for Crane with
6.56 s of audio. Vendor long TTS continued to generate 26.64 s.
For the same 66 s ASR input, the optimized vendor is still 2.32×
Crane's latency, but its live CUDA pool peak is 625 MiB lower.
The CUDA batch ASR process peak host RSS was 2,182 → 2,181 MiB for
vendor, versus 1,936 MiB for Crane. Vendor CustomVoice host RSS was
2,165 → 2,163 MiB, measured across both prompts.
The vendor ASR and TTS pools returned to zero live and reserved bytes
after model drop.

## Other CUDA cases

These cases use one before process and one after process. Times are
medians of three requests. The live pool peaks were identical between
the vendor builds. Initial free VRAM was 5,256–5,328 MiB.

| Workload | Before / after | Latency reduction | Pool peak |
| --- | ---: | ---: | ---: |
| ASR forced English, 11 s | 235.9 / 173.1 ms | 26.6% | 1,619.1 MiB |
| ASR forced English, 66 s | 1,695.7 / 1,495.3 ms | 11.8% | 1,853.8 MiB |
| TTS Base clone, 5.04 s output | 1,848.8 / 1,485.2 ms | 19.7% | 2,911.8 MiB |
| TTS VoiceDesign 1.7B, 4.32 s output | 1,551.7 / 1,238.6 ms | 20.2% | 4,737.2 MiB |

## Functional comparison

- CUDA ASR batch results matched exactly with automatic language
  detection and with English forced. All streaming responses,
  including partials and the final transcript, matched.
- The four fixed-seed CUDA TTS WAVs matched byte for byte: CustomVoice
  short and long, Base clone, and VoiceDesign 1.7B. Their sample rates
  and durations did not change.
- The final build, restricted to CUDA, repeated those CUDA output
  checks and matched the CPU baseline on batch and streaming samples.
  It passed 64 unit tests, all six opt-in CUDA checks (including
  workspace budgets and unload/reload), and Clippy with warnings
  treated as errors.
- Crane produced the same automatic-language ASR transcripts on the
  test clips. At 5,328 MiB initial free VRAM, it completed the short
  CustomVoice case but OOMed on the long case and on Base clone
  warm-up. Those failures are not successful peak or latency samples.

## ASR experiments not kept

Batching the existing audio encoder attention windows separately avoided
computing scores that were previously masked. On the CUDA trial,
66-second batch latency and pool peak were essentially unchanged
(1,777.5 → 1,770.0 ms and 1,853.8 MiB in both builds). The forced-English
66-second result also gained five commas after “And so” while keeping the
same words. The experiment was dropped rather than adopting an output
change without a measured benefit.

Skipping the single-token mask on CPU also produced identical
transcripts but did not show a clear benefit. Its one-process medians
were 3.18 → 3.15 s (11 s batch), 30.14 → 30.94 s (66 s batch), and
8.13 → 8.07 s (11 s streaming). One long request took 80.55 s.
Process peak RSS was 4,309 → 4,354 MiB in batch and
3,860 → 4,048 MiB in streaming. With this run-to-run variation and
no paired CPU speedup, the final build keeps the non-CUDA path
unchanged. This restriction does not change the measured CUDA path.

## Remaining gap

The [TTS generation follow-up](vendor-tts-attention-performance.md)
profiles the Talker, CodePredictor, sampling, and waveform decoder
separately.

The longest batch ASR input has the largest latency gap. Profile the
convolution stem, audio transformer, text prefill and token decode
separately before choosing another optimization. In particular, a full
Crane batch is not a like-for-like memory budget: the vendor encoder
already caps GPU convolution batches to keep long requests in memory.

## Method

- Same Ryzen AI 9 HX 470 / RTX 3080 host, 0.6B weights, CUDA release
  build, and local fixtures as in the
  [last-position logits report](vendor-asr-prefill-performance.md#method).
  VoiceDesign uses the local 1.7B checkpoint. No upstream checkout,
  model download, or new dependency was required.
- Use the same benchmark harness with binaries from before and after
  the changes; only the vendor implementation differs. Measure after
  one warm-up, in fresh processes, with three timed requests per case.
  CUDA automatic-language ASR, streaming ASR and CustomVoice use
  two processes per build, in after/before/before/after order. Forced
  English, Base clone, VoiceDesign and Crane use one process per
  build and workload. The exploratory CPU timing compares against the
  after binary from the [previous follow-up](vendor-asr-prefill-performance.md#cpu-asr-results).
- GPU peaks are live CUDA pool high-water marks, not NVIDIA process
  memory. CPU peak RSS covers the whole process, including model
  loading. Report median request latency separately; load time and
  WAV writing are excluded.
- Use the fixed seed and compare the vendor WAV bytes; for ASR, compare
  automatic and forced-language batch results and all streaming
  responses. Comparisons against Crane are performance references
  only: the TTS waveforms and durations can differ across backends.
  Crane uses the original `-hf` packaging and configuration of the
  0.6B ASR weights, not the tuned convolution batch size. The results
  do not cover other ASR models, Metal, or Windows.

See [the benchmark commands](vendor-asr-prefill-performance.md#reproduce)
for vendor modes, and [the Crane report](crane-performance.md#reproduce)
for Crane modes. The before binary includes the last-position logits
change, but neither fused rotary kernel nor single-token mask removal.
Keep the benchmark harness, lockfile and checkpoint files the same
when rebuilding it in a scratch tree.
