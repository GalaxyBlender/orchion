use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Module, RmsNorm};
use std::collections::HashMap;

use crate::config::TextDecoderConfig;
use crate::linear::LinearW;

// ─── Weight helpers (dense / safetensors path) ────────────────────────────────

fn get_w(weights: &HashMap<String, Tensor>, name: &str) -> anyhow::Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
}

fn take_w(weights: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    weights
        .remove(name)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))
}

fn load_linear(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<LinearW> {
    Ok(LinearW::new(
        get_w(weights, &format!("{}.weight", prefix))?,
        None,
    ))
}

fn load_rms_norm(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        get_w(weights, &format!("{}.weight", prefix))?,
        eps,
    ))
}

// ─── MRoPE ───────────────────────────────────────────────────────────────────

pub(crate) fn compute_mrope_cos_sin(
    position_ids: &[Vec<i64>; 3],
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
    interleaved: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let seq_len = position_ids[0].len();

    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    let dim_map = if interleaved {
        build_interleaved_dim_map(mrope_section, half_dim)
    } else {
        build_contiguous_dim_map(mrope_section, half_dim)
    };

    let mut cos_vals = vec![0.0f32; seq_len * head_dim];
    let mut sin_vals = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        for j in 0..half_dim {
            let dim = dim_map[j];
            let pos = position_ids[dim][t] as f64;
            let angle = pos * inv_freq[j];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos_vals[t * head_dim + j] = c;
            sin_vals[t * head_dim + j] = s;
            cos_vals[t * head_dim + j + half_dim] = c;
            sin_vals[t * head_dim + j + half_dim] = s;
        }
    }

    let cos = Tensor::from_vec(cos_vals, (seq_len, head_dim), device)?;
    let sin = Tensor::from_vec(sin_vals, (seq_len, head_dim), device)?;

    Ok((cos, sin))
}

fn build_contiguous_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(total);
    for (dim, &size) in sections.iter().enumerate() {
        for _ in 0..size {
            if map.len() >= total {
                break;
            }
            map.push(dim);
        }
    }
    while map.len() < total {
        map.push(sections.len() - 1);
    }
    map
}

fn build_interleaved_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let n_dims = sections.len();
    let mut map = Vec::with_capacity(total);
    let mut counts = vec![0usize; n_dims];

    while map.len() < total {
        let prev_len = map.len();
        for dim in 0..n_dims {
            if map.len() >= total {
                break;
            }
            if counts[dim] < sections[dim] {
                map.push(dim);
                counts[dim] += 1;
            }
        }
        if map.len() == prev_len {
            break;
        }
    }
    map
}

/// Apply rotary embeddings using the tables prepared once for all decoder layers.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        return candle_nn::rotary_emb::rope(x, cos, sin).map_err(Into::into);
    }

    let x_rotated = rotate_half(x)?;
    (x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?).map_err(Into::into)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dims()[x.rank() - 1];
    let half = last_dim / 2;
    let x1 = x.narrow(x.rank() - 1, 0, half)?;
    let x2 = x.narrow(x.rank() - 1, half, half)?;
    let neg_x2 = (x2 * (-1.0f64))?;
    Tensor::cat(&[&neg_x2, &x1], x.rank() - 1).map_err(Into::into)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (bsz, num_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((bsz, num_kv, n_rep, seq, hd))?
        .reshape((bsz, num_kv * n_rep, seq, hd))
        .map_err(Into::into)
}

// ─── KV Cache ─────────────────────────────────────────────────────────────────

pub(crate) struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    pub(crate) fn new(num_layers: usize, buffered_limit: Option<usize>) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKvCache::new(buffered_limit))
                .collect(),
        }
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.layers[0].len
    }
}

struct LayerKvCache {
    data: Option<(Tensor, Tensor)>,
    len: usize,
    buffered_limit: Option<usize>,
}

impl LayerKvCache {
    fn new(buffered_limit: Option<usize>) -> Self {
        Self {
            data: None,
            len: 0,
            buffered_limit,
        }
    }

    fn append(&mut self, key: Tensor, value: Tensor) -> Result<(Tensor, Tensor)> {
        let len = self.len + key.dim(2)?;
        let (key, value) = match (&self.data, self.buffered_limit) {
            (None, _) => (key, value),
            (Some((k, v)), None) => (Tensor::cat(&[k, &key], 2)?, Tensor::cat(&[v, &value], 2)?),
            (Some((k, v)), Some(limit)) => {
                // Reserve only one block at a time, after prefill. The
                // grouped GEMMs can consume the resulting strided views.
                const GROW_TOKENS: usize = 128;
                let (k, v) = if len <= k.dim(2)? {
                    (k.clone(), v.clone())
                } else {
                    let capacity = len.saturating_add(GROW_TOKENS - 1).min(limit);
                    (self.grow(k, capacity)?, self.grow(v, capacity)?)
                };
                k.slice_set(&key, 2, self.len)?;
                v.slice_set(&value, 2, self.len)?;
                (k, v)
            }
        };
        let filled = (key.narrow(2, 0, len)?, value.narrow(2, 0, len)?);
        self.data = Some((key, value));
        self.len = len;
        Ok(filled)
    }

    fn grow(&self, old: &Tensor, capacity: usize) -> Result<Tensor> {
        let (batch, heads, _, head_dim) = old.dims4()?;
        let new = Tensor::zeros(
            (batch, heads, capacity, head_dim),
            old.dtype(),
            old.device(),
        )?;
        new.slice_set(&old.narrow(2, 0, self.len)?.contiguous()?, 2, 0)?;
        Ok(new)
    }
}

// ─── Text Attention (GQA + QK-norm + MRoPE) ───────────────────────────────────

struct TextAttention {
    q_proj: LinearW,
    k_proj: LinearW,
    v_proj: LinearW,
    qkv_proj: Option<LinearW>,
    o_proj: LinearW,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TextAttention {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let q = take_w(weights, &format!("{prefix}.q_proj.weight"))?;
        let k = take_w(weights, &format!("{prefix}.k_proj.weight"))?;
        let v = take_w(weights, &format!("{prefix}.v_proj.weight"))?;
        let (q, k, v, qkv_proj) = if q.device().is_cuda() {
            let (q_len, k_len, v_len) = (q.dim(0)?, k.dim(0)?, v.dim(0)?);
            let packed = Tensor::cat(&[q, k, v], 0)?;
            // Prefill uses the same three GEMMs, without retaining another
            // copy of the weights. The originals are dropped layer by layer.
            (
                packed.narrow(0, 0, q_len)?,
                packed.narrow(0, q_len, k_len)?,
                packed.narrow(0, q_len + k_len, v_len)?,
                Some(LinearW::new(packed, None)),
            )
        } else {
            (q, k, v, None)
        };
        Ok(Self {
            q_proj: LinearW::new(q, None),
            k_proj: LinearW::new(k, None),
            v_proj: LinearW::new(v, None),
            qkv_proj,
            o_proj: load_linear(weights, &format!("{}.o_proj", prefix))?,
            q_norm: load_rms_norm(weights, &format!("{}.q_norm", prefix), rms_norm_eps)?,
            k_norm: load_rms_norm(weights, &format!("{}.k_norm", prefix), rms_norm_eps)?,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut LayerKvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;
        let hd = self.head_dim;

        let (q, k, v) = match &self.qkv_proj {
            Some(proj) if seq_len == 1 && mask.is_none() => {
                let q_len = nqh * hd;
                let kv_len = nkvh * hd;
                let packed = proj.forward(x)?;
                (
                    packed.narrow(2, 0, q_len)?,
                    packed.narrow(2, q_len, kv_len)?,
                    packed.narrow(2, q_len + kv_len, kv_len)?,
                )
            }
            _ => (
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            ),
        };
        let q = q
            .reshape((bsz, seq_len, nqh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((bsz, seq_len, nkvh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((bsz, seq_len, nkvh, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        // QK normalization (applied per-head, on last dim)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // Append to KV cache
        let (k, v) = kv_cache.append(k, v)?;

        let n_rep = nqh / nkvh;
        let scale = 1.0 / (hd as f64).sqrt();
        if x.device().is_cuda() && seq_len == 1 && n_rep > 1 && mask.is_none() {
            // Treat the query heads sharing one KV head as rows of a single
            // matmul instead of materializing copies of the whole K/V history.
            let q = q.reshape((bsz, nkvh, n_rep, hd))?;
            let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let output = softmax_last_dim(&scores)?
                .matmul(&v)?
                .reshape((bsz, 1, nqh * hd))?;
            return self.o_proj.forward(&output).map_err(Into::into);
        }
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Attention
        let mut attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;

        if let Some(m) = mask {
            // Cast mask to attn's dtype (mask is F32, attn may be BF16).
            attn = attn.broadcast_add(&m.to_dtype(attn.dtype())?)?;
        }

        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz, seq_len, nqh * hd))?;
        self.o_proj.forward(&out).map_err(Into::into)
    }
}

// ─── SwiGLU MLP ───────────────────────────────────────────────────────────────

struct TextMlp {
    gate_proj: LinearW,
    up_proj: LinearW,
    gate_up_proj: Option<LinearW>,
    down_proj: LinearW,
}

impl TextMlp {
    fn load(weights: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let gate = take_w(weights, &format!("{prefix}.gate_proj.weight"))?;
        let up = take_w(weights, &format!("{prefix}.up_proj.weight"))?;
        let (gate, up, gate_up_proj) = if gate.device().is_cuda() {
            let (intermediate, hidden) = gate.dims2()?;
            let packed = Tensor::stack(&[gate, up], 0)?.reshape((2 * intermediate, hidden))?;
            // Preserve the prefill GEMMs, using views rather than keeping
            // another copy of the weights.
            (
                packed.narrow(0, 0, intermediate)?,
                packed.narrow(0, intermediate, intermediate)?,
                Some(LinearW::new(packed, None)),
            )
        } else {
            (gate, up, None)
        };
        Ok(Self {
            gate_proj: LinearW::new(gate, None),
            up_proj: LinearW::new(up, None),
            gate_up_proj,
            down_proj: load_linear(weights, &format!("{}.down_proj", prefix))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(proj) = &self.gate_up_proj {
            if x.dim(1)? == 1 {
                let activated = Self::activate_packed_decode(&proj.forward(x)?)?;
                return self.down_proj.forward(&activated).map_err(Into::into);
            }
        }
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?).map_err(Into::into)
    }

    fn activate_packed_decode(packed: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        if packed.dtype() == candle_core::DType::BF16 && packed.dim(0)? == 1 {
            return crate::swiglu::decode_bf16(packed);
        }
        candle_nn::ops::swiglu(packed)
    }
}

// ─── Text Decoder Layer ───────────────────────────────────────────────────────

struct TextDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: TextAttention,
    post_attention_layernorm: RmsNorm,
    mlp: TextMlp,
}

impl TextDecoderLayer {
    fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: load_rms_norm(
                weights,
                &format!("{}.input_layernorm", prefix),
                rms_norm_eps,
            )?,
            self_attn: TextAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_q_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                rms_norm_eps,
            )?,
            mlp: TextMlp::load(weights, &format!("{}.mlp", prefix))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut LayerKvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, cos, sin, kv_cache, mask)?;
        let x = (x + &h)?;

        // Pre-norm + MLP + residual
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        (&x + &h).map_err(Into::into)
    }
}

// ─── Causal mask ─────────────────────────────────────────────────────────────

pub(crate) fn create_causal_mask(
    seq_len: usize,
    past_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = past_len + seq_len;
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        for j in 0..total_len {
            if j > past_len + i {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, total_len), device).map_err(Into::into)
}

// ─── Text Decoder ────────────────────────────────────────────────────────────

pub(crate) struct TextDecoder {
    embed_tokens: Tensor,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearW,
}

impl TextDecoder {
    pub(crate) fn load(
        weights: &mut HashMap<String, Tensor>,
        prefix: &str,
        config: &TextDecoderConfig,
    ) -> Result<Self> {
        let embed_tokens = get_w(weights, &format!("{}.embed_tokens.weight", prefix))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = TextDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        let norm = load_rms_norm(weights, &format!("{}.norm", prefix), config.rms_norm_eps)?;
        // Tie lm_head weights to embed_tokens (weight tying).
        let lm_head = LinearW::new(embed_tokens.clone(), None);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Look up token embeddings. Returns the native dtype of the embedding table (BF16).
    pub(crate) fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens
            .index_select(input_ids, 0)
            .map_err(Into::into)
    }

    /// Update the KV cache for every input position and return next-token logits `[1, vocab]`.
    pub(crate) fn forward_last_logits(
        &self,
        hidden_states: &Tensor, // [1, seq, hidden]
        cos: &Tensor,           // [seq, head_dim], F32
        sin: &Tensor,           // [seq, head_dim], F32
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();

        // The fused CUDA kernel takes [seq, head_dim / 2]; MRoPE repeats
        // those values in the other half for the broadcast implementation.
        let compute_dtype = hidden.dtype();
        let prepare = |table: &Tensor| -> candle_core::Result<Tensor> {
            if hidden_states.device().is_cuda() {
                table
                    .narrow(1, 0, table.dim(1)? / 2)?
                    .to_dtype(compute_dtype)?
                    .contiguous()
            } else {
                table.to_dtype(compute_dtype)?.unsqueeze(0)?.unsqueeze(0)
            }
        };
        let cos = prepare(cos)?;
        let sin = prepare(sin)?;

        for (layer, cache) in self.layers.iter().zip(&mut kv_cache.layers) {
            hidden = layer.forward(&hidden, &cos, &sin, cache, mask)?;
        }

        // Earlier positions are needed for attention and the KV cache, not for
        // the final normalization or vocabulary projection.
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
        self.lm_head
            .forward(&self.norm.forward(&last)?)?
            .squeeze(1)
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn cuda_mlp_matches_separate_projections() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let gate = Tensor::new(&[[1f32, -2.], [-2., 1.], [0., 2.]], &device)?;
        let up = Tensor::new(&[[-1f32, 1.], [2., -1.], [1., 1.]], &device)?;
        let down = Tensor::new(&[[1f32, 2., -1.], [2., -1., 3.]], &device)?;
        let mut weights = HashMap::from([
            ("mlp.gate_proj.weight".into(), gate.clone()),
            ("mlp.up_proj.weight".into(), up.clone()),
            ("mlp.down_proj.weight".into(), down.clone()),
        ]);
        let mlp = TextMlp::load(&mut weights, "mlp")?;
        let input = Tensor::new(&[[[0.5f32, 1.], [-0.5, -2.]]], &device)?;
        for len in [1, 2] {
            let x = input.narrow(1, 0, len)?;
            let gate = LinearW::new(gate.clone(), None).forward(&x)?.silu()?;
            let up = LinearW::new(up.clone(), None).forward(&x)?;
            let expected = LinearW::new(down.clone(), None).forward(&(gate * up)?)?;
            let actual = mlp.forward(&x)?;
            assert_eq!(actual.dims(), expected.dims());
            assert!(
                (&actual - &expected)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?
                    < 1e-5
            );
        }
        Ok(())
    }

    #[test]
    fn buffered_cache_preserves_each_heads_history_across_growth() -> Result<()> {
        let mut cache = LayerKvCache::new(Some(140));
        for (start, end) in std::iter::once((0, 3)).chain((3..140).map(|i| (i, i + 1))) {
            let rows = |start: usize, end| {
                (start..end)
                    .chain(start + 1000..end + 1000)
                    .map(|n| n as f32)
                    .collect::<Vec<_>>()
            };
            let k = Tensor::from_vec(rows(start, end), (1, 2, end - start, 1), &Device::Cpu)?;
            let v = (&k + 2000.0)?;
            let (k, v) = cache.append(k, v)?;
            let expected = rows(0, end);
            assert_eq!(k.dims(), &[1, 2, end, 1]);
            assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, expected);
            assert_eq!(
                v.flatten_all()?.to_vec1::<f32>()?,
                expected.iter().map(|n| n + 2000.0).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn test_causal_mask_prefill() {
        let device = Device::Cpu;
        let mask = create_causal_mask(3, 0, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 3, 3]);
        let data: Vec<f32> = mask
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Row 0: [0, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        // Row 1: [0, 0, -inf]
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], f32::NEG_INFINITY);
        // Row 2: [0, 0, 0]
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    #[test]
    fn test_causal_mask_decode_step() {
        let device = Device::Cpu;
        // seq=1, past=5 → total_len=6; j > 5+0 never triggers → all zeros
        let mask = create_causal_mask(1, 5, &device).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 1, 6]);
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            data.iter().all(|&v| v == 0.0),
            "decode-step mask should be all zeros"
        );
    }

    #[test]
    fn test_build_contiguous_dim_map() {
        let sections = [2usize, 3, 3];
        let map = build_contiguous_dim_map(&sections, 8);
        assert_eq!(map, vec![0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_build_interleaved_dim_map() {
        let sections = [2usize, 2, 2];
        let map = build_interleaved_dim_map(&sections, 6);
        assert_eq!(map, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_compute_mrope_cos_sin_zero_positions() {
        let device = Device::Cpu;
        let seq_len = 4;
        let head_dim = 8;
        let mrope_section = [2usize, 3, 3];
        // All positions = 0 → angle = 0 → cos = 1, sin = 0 everywhere
        let position_ids: [Vec<i64>; 3] = [
            vec![0i64; seq_len],
            vec![0i64; seq_len],
            vec![0i64; seq_len],
        ];
        let (cos, sin) = compute_mrope_cos_sin(
            &position_ids,
            head_dim,
            10000.0,
            &mrope_section,
            false,
            &device,
        )
        .unwrap();
        assert_eq!(cos.dims(), &[seq_len, head_dim]);
        assert_eq!(sin.dims(), &[seq_len, head_dim]);
        let cos_data: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (&c, &s)) in cos_data.iter().zip(sin_data.iter()).enumerate() {
            assert!(
                (c - 1.0).abs() < 1e-6,
                "cos[{}] should be 1.0, got {}",
                i,
                c
            );
            assert!(s.abs() < 1e-6, "sin[{}] should be 0.0, got {}", i, s);
        }
    }

    #[test]
    fn test_compute_mrope_cos_sin_pythagorean() {
        // cos²+sin² must equal 1 at every (t, j)
        let device = Device::Cpu;
        let head_dim = 8;
        let mrope_section = [2usize, 2, 2];
        let position_ids: [Vec<i64>; 3] = [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let (cos, sin) = compute_mrope_cos_sin(
            &position_ids,
            head_dim,
            10000.0,
            &mrope_section,
            false,
            &device,
        )
        .unwrap();
        let cos_data: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin_data: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (&c, &s)) in cos_data.iter().zip(sin_data.iter()).enumerate() {
            let r = c * c + s * s;
            assert!(
                (r - 1.0).abs() < 1e-5,
                "cos²+sin² at [{}] = {}, want 1",
                i,
                r
            );
        }
    }

    #[test]
    fn test_rotate_half() {
        let device = Device::Cpu;
        // x = [[1, 2, 3, 4]] → rotate_half → [[-3, -4, 1, 2]]
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let rotated = rotate_half(&x).unwrap();
        let data: Vec<f32> = rotated.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![-3.0f32, -4.0, 1.0, 2.0]);
    }
}
