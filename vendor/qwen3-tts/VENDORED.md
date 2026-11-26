# Vendored qwen3-tts

Library source and CUDA kernels from
[TrevorS/qwen3-tts-rs at `711ceee`](https://github.com/TrevorS/qwen3-tts-rs/tree/711ceee07cad92673f86de8997bdf54c30caa49f)
(MIT, as declared in the upstream README and manifest). That revision does not
include a standalone license file; [`LICENSE`](LICENSE) records the standard
MIT terms and upstream attribution. The upstream CLI, benchmarks and
non-runtime assets are omitted. This copy uses the workspace's Candle 0.11
series.

Orchion's patch bounds CUDA memory for waveform upsampling. Codec generation
and decoder attention still see the entire sequence. Upsampling processes
overlapping windows, discards samples affected by window-local padding, and
copies only the resulting PCM to the host. CPU and Metal still use the original
whole-waveform path. Remove the vendored dependency when the upstream runtime
provides equivalent bounded decoding.

The talker and code predictor also use Candle's fused half-split rotary
kernel on CUDA. The rotation, sampling, and waveform decoding are
otherwise unchanged.

The CUDA RoPE tables use the model's compute dtype. Talker MRoPE also
caches the first 2,048 positions, using the original calculation for
longer offsets. The code predictor writes acoustic token IDs into its
preallocated frame tensor without copying previous IDs. The
[performance report](../../docs/vendor-tts-attention-performance.md)
records the profile and output checks.
