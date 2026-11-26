# Vendored Qwen3-ASR: grouped CUDA token decoding

This follow-up starts from the [fused RoPE build](vendor-speech-performance.md).
It changes only single-token CUDA attention in the ASR text decoder.
Each group of query heads attends to its shared K/V head directly.
The old path materialized repeated K/V and a contiguous transposed K
on every token. The new path does neither. Prefill, cache lifetime,
the convolution batch limit, CPU, Metal, and TTS are unchanged.

**Result:** The 66 s CUDA batch request fell from **1,577 to 843 ms
(47%)** at the same pool peak. It is still 21% slower than Crane
with a convolution batch cap of 16, while using 147 MiB less peak
VRAM on that input.

## Where the time went

Synchronized phase medians on the RTX 3080 (three requests after warm-up):

| 66 s audio | Vendor before | Vendor after | Crane 26.9.0, stock |
| --- | ---: | ---: | ---: |
| Mel extraction and upload | 7.77 ms | 7.62 ms | Not timed separately |
| Audio encoder and projection | 64.0 ms | 65.4 ms | 63.6 ms |
| Text prefill | 62.46 ms | 63.47 ms | 64.4 ms |
| Token decoding (133 tokens) | 1,440.85 ms | 712.04 ms | 542.8 ms |
| Request | 1,581.1 ms | 854.3 ms | 680.4 ms |

Almost all of the gap came from token decoding, not audio encoding.
Profile timings are diagnostic, not the cross-run performance numbers.
Crane's published phase logging includes synchronization; vendor only
adds such synchronization when profiling is enabled.

## End-to-end CUDA results

Unprofiled release builds, same RTX 3080 and local checkpoints as the
[prior round](vendor-speech-performance.md#method), with 5,328 MiB
initial free VRAM. Vendor automatic batch and streaming numbers are
medians of six requests per case: two fresh processes per build in
after/before/before/after order, one warm-up and three timed requests
per process. Crane and forced-English results use three requests per
case. Load time is excluded. Peaks are live CUDA pool high-water marks,
not whole-process GPU usage.

| Workload | Vendor before | Vendor after | Crane stock | Crane tuned to 16 |
| --- | ---: | ---: | ---: | ---: |
| English batch, 11 s | 184.5 ms | 148.4 ms | 139.8 ms | 133.6 ms |
| English batch, 66 s | 1,577.2 ms | 842.5 ms | 683.6 ms | 698.4 ms |
| English streaming, 11 s | 393.5 ms | 339.5 ms | — | — |
| First nonempty streaming partial | 57.2 ms | 51.8 ms | — | — |
| Forced English, 11 s | 172.0 ms | 140.0 ms | — | — |
| Forced English, 66 s | 1,501.3 ms | 835.5 ms | — | — |

On the 66 s automatic case all before requests fell in 1,568.0–1,590.5 ms;
all after requests were in 838.9–845.0 ms. Vendor peak was unchanged:
1,619.1 MiB for 11 s and 1,853.8 MiB for 66 s, including forced English.
Streaming peaked at 1,586.0 MiB in both builds. Stock Crane peaked
at 1,870.1 and 2,478.4 MiB; tuned Crane at 1,870.1 and 2,000.9 MiB
respectively. Vendor batch peak host RSS was 2,182 → 2,184 MiB,
compared with 2,028 MiB stock and 2,029 MiB tuned Crane. The pool
returned to 0 MiB live and reserved after model drop.

On a separate Mandarin smoke test (29.04 s of previously synthesized
speech, then repeated to 174.24 s), vendor medians were 603.3 → 407.9 ms
and 8,481.6 → 3,326.5 ms. Peaks remained 1,829.3 and 2,180.8 MiB.
The input was the saved [CPU streaming long TTS clip](crane-performance.md#tts-bounded-decoding-via-the-public-api)
converted to 16 kHz mono PCM16. This is a secondary fixture, not a
multi-speaker accuracy benchmark.

## Functional verification

- Automatic English and Mandarin batch results matched exactly, including
  language and raw output. Both Crane configurations matched the automatic
  English baseline. All English streaming partials and final responses
  matched between vendor builds.
- Forced English matched on the 11 s clip. On the 66 s clip the new
  CUDA path adds five commas after “And so”; words and order are
  unchanged. The grouped GEMMs change floating-point evaluation order,
  so byte-identical transcripts are not guaranteed on other inputs.
- CPU batch and streaming outputs matched on a short fixture. This
  change does not claim a CPU or Metal speedup.
- The 240.5 s synthetic CUDA workspace regression still peaks at
  2,614 MiB and returns live allocation to the loaded baseline.
  Short and long CustomVoice WAVs matched the previous build byte
  for byte. All 64 release unit tests, six opt-in CUDA checks and
  strict Clippy passed.

## Alternatives

The alternative grouped path that still materialized transposed K was
slower (980.1 ms instead of 846.2 ms on a separate 66 s unprofiled
trial), and it did not restore the exact forced-English punctuation.
This build did not preallocate a larger cache. The
[buffered KV cache follow-up](vendor-asr-cache-performance.md) compares
in-place appends and packed Q/K/V against this baseline.

## Reproduce

Use the [existing speech benchmark setup](vendor-asr-prefill-performance.md#reproduce).
For profiling, add `ORCHION_ASR_PROFILE=1` to the invocation on either
backend. This enables the benchmark's INFO logger and synchronized
stage timings in the vendor code. Leave it unset for end-to-end
latency and peak measurements.

```sh
ORCHION_ASR_PROFILE=1 "$exe" legacy cuda:0 asr \
  "$models/Qwen3-ASR-0.6B" "$work/jfk.pcm"
ORCHION_ASR_PROFILE=1 "$exe" crane cuda:0 asr \
  "$models/Qwen3-ASR-0.6B-hf" "$work/jfk.pcm"
```

Build the before executable with the prior fused RoPE implementation but
without the grouped attention branch. Use identical local checkpoints,
release build flags and harness. For the tuned Crane comparison, use
[a separate model directory with `conv_chunksize=16`](crane-performance.md#batch-asr-audio_configconv_chunksize).
`asr-stream` feeds 1 s blocks without sleeping; first-partial times measure
computation, not audible latency from a microphone. `asr ... english`
exercises forced English. For both, compare the returned text and raw
output as well as end-to-end latency.
