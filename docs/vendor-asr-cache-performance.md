# Vendored Qwen3-ASR: buffered CUDA KV cache and packed Q/K/V

Baseline: the previous [grouped CUDA decode build](vendor-asr-decode-performance.md).
That build stopped expanding the K/V heads for attention, but still copied
the entire K/V history into a larger allocation on every token.

The [SwiGLU follow-up](vendor-asr-swiglu-performance.md) uses this build
as its baseline.

The CUDA grouped-query decoder now appends into a request-local cache that
grows in blocks of up to 128 positions, capped by the request's token
budget. It does not reserve decode capacity during prefill. The decoder
still attends to the full history; unused capacity is outside the views
passed to attention. CPU and Metal retain the original concatenation path.
The generation loop reuses the token tensor produced by argmax rather
than uploading the same ID again after checking for EOS.

## Packed Q/K/V projection

Single-token CUDA decoding projects Q, K and V with one GEMM instead
of three. Prefill still uses the original three GEMMs, taking views
of the packed weights. Packing replaces the old weights layer by
layer; the loaded model does not keep duplicate weights. It can
increase the CUDA pool's reserved pages at load even when live
inference allocations do not increase.

The intermediate build with the buffered cache and token reuse, but
without packed Q/K/V, took **760.4 ms** on the 66 s request versus
**844.3 ms** for its interleaved baseline. Both builds peaked at
1,853.8 MiB of live CUDA pool allocation.

## CUDA latency and memory

| Workload | Vendor before | Vendor after | Crane 26.9.0 | Crane, conv batch 16 |
| --- | ---: | ---: | ---: | ---: |
| English batch, 11 s | 149.8 ms | 138.1 ms | 130.0 ms | 129.2 ms |
| English batch, 66 s | 845.7 ms | 734.3 ms | 676.2 ms | 665.4 ms |
| Forced English batch, 11 s | 139.5 ms | 129.4 ms | — | — |
| Forced English batch, 66 s | 836.1 ms | 723.4 ms | — | — |
| Streaming, 11 s session | 341.2 ms | 320.9 ms | — | — |
| First nonempty streaming partial | 52.0 ms | 49.0 ms | — | — |

The 66 s automatic case had **13.2% lower latency** than the grouped-decode
baseline. Its six after requests were all between 731.1 and 736.4 ms,
versus 845.1–852.7 ms before. Median real-time factor fell from
0.0128 to 0.0111. It was still 8.6% slower than stock
Crane, and 10.4% slower than Crane with a convolution batch cap of 16.
In separate synchronized profiles of the 133-token decode, the median
decode phase went **712.7 → 625.2 → 593.6 ms** across the baseline,
buffered-cache-only build, and final build. Text prefill stayed around
62–64 ms.

| CUDA pool metric | Vendor before | Vendor after | Crane stock | Crane, conv batch 16 |
| --- | ---: | ---: | ---: | ---: |
| Batch live peak, 11 s | 1,619.1 MiB | 1,619.1 MiB | 1,870.1 MiB | 1,870.1 MiB |
| Batch live peak, 66 s | 1,853.8 MiB | 1,853.8 MiB | 2,478.4 MiB | 2,000.9 MiB |
| Streaming live peak | 1,586.0 MiB | 1,586.0 MiB | — | — |
| Live after request, model still loaded | 1,497.5 MiB | 1,497.5 MiB | 1,748.4 MiB | 1,748.4 MiB |
| Reserved after English batch request | 1,536–1,568 MiB | 1,760–1,792 MiB | 1,920 MiB | 1,920 MiB |

The 66 s case still used **147.1 MiB less live peak** than tuned Crane.
Vendor batch process peak host RSS was 2,184 → 2,183 MiB. Packing
raised the load-time live pool peak by 8 MiB (1,794.2 → 1,802.2).
The larger reserved pool was already present at model load and did
not grow across requests. It is not freed at each request boundary:
request-local live allocations return to the model baseline, and both
live and reserved pool allocations return to 0 MiB when the model is
dropped in the isolated benchmark.

## Output checks

Automatic and forced English batch results, all streaming partials
and final responses, and the short CPU smoke checks matched the
preceding vendor build exactly. This round adds no punctuation changes
to the previous forced-English result. Both Crane configurations
matched the automatic English result.

The same synthesized Mandarin fixture as the previous round also
matched exactly (29.04 s and 174.24 s repeated audio). Its median was
408.8 → 369.4 ms for the shorter case and 3,328.0 → 2,619.2 ms for the
longer case. The latter used 17.8 MiB more at peak (2,180.8 → 2,198.6
MiB), and live allocation still returned to the model baseline.

The 240.5 s synthetic CUDA workspace regression peaked at 2,638 MiB
(24 MiB above the previous build) and returned live allocation to the
loaded baseline. TTS was unchanged; the four fixed-seed WAVs (short
and long CustomVoice, Base clone, and VoiceDesign) matched byte for
byte. All 128 release unit tests, one doc test, six opt-in CUDA checks,
and release Clippy with `-D warnings` passed.

## Measurement method

The RTX 3080 runs started with approximately 5,328 MiB free. They use
the same local checkpoints and 11 s JFK PCM as the previous round,
repeated six times for the 66 s case. The vendor
automatic and streaming results use two fresh processes per build in
after/before/before/after order. Each process warms up once before
three timed requests. Forced English and each Crane configuration use
three timed requests per case. Request timings exclude model load.

Live peak is the CUDA pool high-water mark during inference. Post-request
live allocation checks whether the workspace was freed while the model
stayed loaded. Reserved pages are measured separately at load and after
inference; they need not be returned to the driver when live allocation
falls. Peak host RSS includes model load, warm-up and inference.

## Reproduce

Use the same [offline build and JFK fixture](vendor-asr-prefill-performance.md#reproduce).
Save the grouped-decode baseline binary before building this change.
Alternate fresh vendor processes on the same checkpoints and compare
`asr`, `asr ... english`, and `asr-stream` with profiling disabled.
The [Crane tuning comparison](crane-performance.md#batch-asr-audio_configconv_chunksize)
uses the same inputs and its local `-hf` checkpoint. Use
`ORCHION_ASR_PROFILE=1` only to measure the synchronized decode phase.
