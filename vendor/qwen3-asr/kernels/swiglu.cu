#include <cuda_bf16.h>

// One CUDA decode row: [gate | up] -> silu(gate) * up.
extern "C" __global__ void asr_swiglu_bf16(
    const __nv_bfloat16* packed, __nv_bfloat16* output, unsigned int width)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) {
        return;
    }
    const float g = __bfloat162float(packed[i]);
    // Keep the intermediate rounding of Candle's separate BF16 SiLU and multiply.
    const auto activated = __float2bfloat16_rn(g / (1.0f + expf(-g)));
    output[i] = __float2bfloat16_rn(
        __bfloat162float(activated) * __bfloat162float(packed[width + i]));
}
