# Vendored qwen3-asr

Source from the MIT-licensed [`qwen3-asr` 0.2.2 release](https://crates.io/crates/qwen3-asr/0.2.2)
([upstream commit `c5ef096`](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc)).
The upstream benchmark binary is omitted. The imported Rust source is formatted
by the workspace's `cargo fmt --all` and uses the workspace's Candle 0.11 series.

Orchion bounds the CUDA convolution batch in the offline audio encoder. It
preserves chunk order, the valid length of the final chunk, and the original
attention windows and decoder context. The convolution patch leaves the
incremental audio encoder unchanged.

The text decoder also projects only the last hidden position into the
vocabulary, an optimization inspired by the published MIT-licensed
`crane-core` 26.9.0 Qwen3 decoder. Batch and streaming requests still
process the entire input in the attention layers and KV cache.

The ASR text decoder uses Candle's fused half-split rotary kernel on
CUDA. It keeps the checkpoint's MRoPE frequency mapping. Single-token
decode on CUDA also omits the all-zero causal mask; prefill and the
non-CUDA paths retain their masks.

Single-token CUDA decode with grouped-query attention batches the
queries that share a KV head, avoiding the expanded key/value copies
and using the transposed key as a view. Prefill and masked attention
retain the existing implementation.

The grouped CUDA path grows the request-local KV cache in bounded
blocks after prefill, appending tokens in place. Attention only reads
the filled positions; no history is dropped. Non-CUDA cache appends
still concatenate.

CUDA single-token decode uses a packed Q/K/V projection. Prefill
uses individual views into the same weights without adding
duplicate resident tensors.

The CUDA decoder also packs the SwiGLU gate/up projection and fuses
BF16 single-token SiLU and multiply, following the published Crane
26.9.0 Qwen3 decoder optimizations. The custom kernel preserves
Candle's intermediate BF16 rounding and leaves prefill unchanged.
