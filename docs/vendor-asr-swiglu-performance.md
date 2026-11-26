# Vendored Qwen3-ASR: single-token CUDA SwiGLU

Baseline: the [buffered KV cache and packed Q/K/V build](vendor-asr-cache-performance.md).
This round ports two ideas from the published Crane 26.9.0 Qwen3
decoder: run the `gate` and `up` projections as one matrix multiply,
then compute `SiLU(gate) × up` in one CUDA kernel. Both changes are
limited to single-token CUDA decoding. Prefill uses the original two
projections through views of the packed weights. CPU and Metal keep
the original path.

The fused kernel operates on one BF16 row and explicitly rounds the
SiLU result to BF16 before multiplying, like the separate Candle
operations. Non-BF16 and other batch sizes use Candle's SwiGLU
operations. CUDA builds generate PTX with the existing `cudaforge`
toolchain rather than adding a runtime dependency on Crane.

## Isolating the two changes

With packed gate/up weights in both builds, switching from separate
activation and multiply to the fused kernel gave these medians over
six timed requests per build. Builds alternated in
fused/packed/packed/fused order in fresh processes.

| CUDA workload | Packed projection only | Packed projection + fused activation |
| --- | ---: | ---: |
| English batch, 11 s | 131.3 ms | 128.6 ms |
| English batch, 66 s | 703.0 ms | 692.9 ms |
| Streaming, 11 s | 308.2 ms | 303.1 ms |

The 66 s request in the initial baseline run had a 731.8 ms median.
In separate synchronized profiles on that input, the 133-token
decode went **596.8 → 566.4 → 555.8 ms** from baseline to packed
projection to fused activation, while text prefill stayed near 63 ms.
All three builds returned identical automatic and forced English text
and streaming responses. The two changes add no duplicate live weights:
their resident allocation was 1,497.5 MiB, including after every request.

## End-to-end CUDA comparison

RTX 3080, release builds, 5,328 MiB free before each batch process.
Automatic English and streaming results are medians of six requests
per build, with two fresh processes per build in
after/before/before/after order. Forced English, Mandarin and each
Crane build use three requests per case. Load and warm-up are excluded.

| Workload | Vendor before | Vendor after | Crane stock | Crane, convolution batch 16 |
| --- | ---: | ---: | ---: | ---: |
| English batch, 11 s | 137.7 ms | 130.0 ms | 129.7 ms | 129.4 ms |
| English batch, 66 s | 734.3 ms | 693.1 ms | 680.6 ms | 665.8 ms |
| Forced English batch, 11 s | 130.3 ms | 121.7 ms | — | — |
| Forced English batch, 66 s | 727.9 ms | 687.3 ms | — | — |
| Streaming, 11 s | 323.1 ms | 303.3 ms | — | — |
| First nonempty streaming partial | 49.4 ms | 46.1 ms | — | — |

The 66 s automatic case had **5.6% lower latency**. Its six runs moved
from 731.7–737.2 ms to 691.7–696.1 ms. It is now 1.8% slower than
stock Crane and 4.1% slower than tuned Crane. Both Crane configurations
matched the automatic English transcript.

| CUDA pool metric | Vendor before | Vendor after | Crane stock | Crane, convolution batch 16 |
| --- | ---: | ---: | ---: | ---: |
| Batch live peak, 11 s | 1,619.1 MiB | 1,619.1 MiB | 1,870.1 MiB | 1,870.1 MiB |
| Batch live peak, 66 s | 1,853.8 MiB | 1,853.8 MiB | 2,478.4 MiB | 2,000.9 MiB |
| Streaming live peak | 1,586.0 MiB | 1,586.0 MiB | — | — |
| Batch live after request | 1,497.5 MiB | 1,497.5 MiB | 1,748.4 MiB | 1,748.4 MiB |
| Automatic batch reserved after request | 1,760–1,792 MiB | 1,952 MiB | 1,920 MiB | 1,920 MiB |

The 66 s case used **147.1 MiB less live peak** than tuned Crane.
The extra reserved pages were already present at load, did not grow
across requests, and returned to zero on model drop in the isolated
benchmark. They do not represent live request allocations. Vendor
load-time live peak increased by 4 MiB (1,802.2 → 1,806.2 MiB), and
batch process peak host RSS was 2,182 → 2,184 MiB.

On the [same Mandarin smoke inputs](vendor-asr-cache-performance.md#output-checks),
29.04 s took 369.2 → 345.5 ms and 174.24 s took 2,624.5 → 2,510.0 ms.
Both outputs were identical. The longer case still peaked at
2,198.6 MiB; its reserved pool was 1,728 → 2,016 MiB at load and
after requests. All streaming partials and final outputs, forced
English, and the short CPU checks matched the previous vendor build.

## Regression checks

- The 240.5 s CUDA ASR workspace test still peaked at 2,638 MiB and
  returned live allocation to the loaded baseline. The pool retained
  2,016 MiB after the request. Repeated model load/unload passed.
- All four fixed-seed CUDA TTS WAVs matched the reference build byte
  for byte: short and long CustomVoice, Base cloning and VoiceDesign.
- 128 release unit tests, one doc test, six real-model CUDA checks,
  and two CUDA MLP/activation checks passed. The non-CUDA ASR/TTS
  build and release Clippy with `-D warnings` also passed.
- [Windows CUDA speech validation](windows-speech-validation.md)
  covers the native Release server, including batch and WebSocket ASR.

## Reproduce

Use the [existing offline build and fixtures](vendor-asr-prefill-performance.md#reproduce).
Save the previous round's binary as the baseline. To isolate the
projection change, build a second binary with
`TextMlp::activate_packed_decode` always calling Candle's
`swiglu`. Leave `ORCHION_ASR_PROFILE` unset for latency and memory
comparisons; enable it only for synchronized phase timings.

The two kernel-level checks can run on local CUDA hardware without
model weights:

```sh
export CUDA_COMPUTE_CAP=86 CUDARC_CUDA_VERSION=13000
for test in cuda_mlp_matches_separate_projections \
            fused_decode_preserves_bf16_intermediate_rounding; do
  cargo test --release --offline --locked -p qwen3-asr -p orchion-qwen3 \
    --features orchion-qwen3/asr,orchion-qwen3/cuda --lib -- --ignored "$test"
done
```
