# Vendored qwen3-asr

Source from the MIT-licensed [`qwen3-asr` 0.2.2 release](https://crates.io/crates/qwen3-asr/0.2.2)
([upstream commit `c5ef096`](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc)).
The upstream benchmark binary is omitted. The imported Rust source is formatted
by the workspace's `cargo fmt --all` and uses the workspace's Candle 0.11 series.

Orchion bounds the CUDA convolution batch in the offline audio encoder. It
preserves chunk order, the valid length of the final chunk, and the original
attention windows and decoder context. The incremental streaming path is
unchanged. Remove the vendored dependency when upstream provides equivalent
bounded convolution.
