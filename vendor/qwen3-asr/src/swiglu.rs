use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, Layout, Result, Shape, Tensor};

include!(concat!(env!("OUT_DIR"), "/swiglu_ptx.rs"));

/// Fused activation for the contiguous BF16 gate/up output of one CUDA decode token.
pub(crate) fn decode_bf16(packed: &Tensor) -> Result<Tensor> {
    packed.apply_op1_no_bwd(&DecodeSwiGlu)
}

struct DecodeSwiGlu;

impl CustomOp1 for DecodeSwiGlu {
    fn name(&self) -> &'static str {
        "asr-decode-swiglu"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ASR fused SwiGLU is CUDA-only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let [1, 1, packed_width] = layout.dims() else {
            candle_core::bail!("ASR fused SwiGLU expects one decode row");
        };
        if *packed_width == 0 || packed_width % 2 != 0 {
            candle_core::bail!("ASR fused SwiGLU requires equally sized gate and up halves");
        }
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("ASR fused SwiGLU requires contiguous input");
        };
        let CudaStorageSlice::BF16(src) = &storage.slice else {
            candle_core::bail!("ASR fused SwiGLU requires BF16 input");
        };
        let width = *packed_width / 2;
        // Both the packed input and the CUDA indices must fit in u32.
        let width_arg = u32::try_from(*packed_width).map_err(candle_core::Error::wrap)? / 2;
        let src = src.slice(start..end);
        let dev = storage.device();
        let func = dev.get_or_load_custom_func("asr_swiglu_bf16", "asr_swiglu", SWIGLU)?;
        let config = LaunchConfig {
            grid_dim: (width_arg.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: The kernel initializes every element before the result can be read.
        let dst = unsafe { dev.alloc::<_>(width)? };
        let mut builder = func.builder();
        builder.arg(&src).arg(&dst).arg(&width_arg);
        // SAFETY: The pointers are BF16 slices with 2 * width inputs and width
        // outputs. The kernel bounds-checks threads and the launch uses Candle's stream.
        unsafe { builder.launch(config) }.w()?;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(dst),
                device: dev.clone(),
            },
            Shape::from_dims(&[1, 1, width]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_decode_preserves_bf16_intermediate_rounding() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let width = 4097;
        let row: Vec<_> = (0..width)
            .map(|i| (i as f32 - 2048.) / 31.)
            .chain((0..width).map(|i| (i as f32 % 129. - 64.) / 15.))
            .collect();
        let row = Tensor::from_vec(row, (1, 1, 2 * width), &device)?.to_dtype(DType::BF16)?;
        let packed = Tensor::cat(&[&row, &row], 1)?.narrow(1, 1, 1)?;
        let fused = decode_bf16(&packed)?;
        let expected = candle_nn::ops::swiglu(&packed)?;
        assert_eq!(
            fused.to_dtype(DType::F32)?.to_vec3::<f32>()?,
            expected.to_dtype(DType::F32)?.to_vec3::<f32>()?
        );
        Ok(())
    }
}
