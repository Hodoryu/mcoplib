
#include "sgl_kernel/tile.cuh"
#include "sgl_kernel/type.cuh"
#include "sgl_kernel/vec.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include <maca_fp8.h>
#include "../../../kernel/dispatch_utils.h"

struct SiluAndMulClampParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  float swiglu_limit;
};

template <typename DType, bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void  // maximize occupancy
    silu_mul_clamp_kernel(const SiluAndMulClampParams  params) {
  using namespace device;
  using DType2 = packed_t<DType>;
  constexpr auto kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize % 2 == 0 && kVecSize > 0);
  using Vec = AlignedVector<DType2, kVecSize / 2>;
  const auto bid = blockIdx.x;
  const auto tile = tile::Memory<Vec>::cta();
  const float limit = params.swiglu_limit;

  PDLWaitPrimary<kUsePDL>();
  const auto gate = tile.load(params.input, bid * 2 + 0);
  const auto up = tile.load(params.input, bid * 2 + 1);
  Vec out;

#pragma unroll
  for (uint32_t i = 0; i < kVecSize / 2; ++i) {
    const auto [g0_raw, g1_raw] = cast<fp32x2_t>(gate[i]);
    const float g0 = fminf(g0_raw, limit);
    const float g1 = fminf(g1_raw, limit);
    const float silu0 = g0 / (1.0f + expf(-g0));
    const float silu1 = g1 / (1.0f + expf(-g1));
    const auto [u0_raw, u1_raw] = cast<fp32x2_t>(up[i]);
    const float u0 = fmaxf(fminf(u0_raw, limit), -limit);
    const float u1 = fmaxf(fminf(u1_raw, limit), -limit);
    const float val0 = u0 * silu0;
    const float val1 = u1 * silu1;
    out[i] = cast<DType2>(fp32x2_t{val0, val1});
  }

  tile.store(params.output, out, bid);
  PDLTriggerSecondary<kUsePDL>();
}

void silu_and_mul_clamp_check(const torch::Tensor& input,
                        torch::Tensor& output, double swiglu_limit) {                  
    uint32_t kVecSize = 16 / input.element_size();
    const auto out_dim = input.size(-1);
    const auto num_tokens = input.numel() / out_dim;
    const auto H = output.size(-1);
    TORCH_CHECK(out_dim % kVecSize == 0, "out_dim must be divisible by vector size");
    const auto num_threads = H / kVecSize;
    TORCH_CHECK(num_threads <= 1024, "out_dim too large for single-block-per-row launch");

    const auto params = SiluAndMulClampParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .swiglu_limit = static_cast<float>(swiglu_limit),
    };
     if (num_tokens == 0) {
      return;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const dim3 grid(static_cast<unsigned int>(num_tokens));
    const dim3 block(num_threads);
    if (input.dtype() == at::ScalarType::BFloat16) {
      silu_mul_clamp_kernel<maca_bfloat16, false>
              <<<grid, block, 0, stream>>>(params);
    } else if(input.dtype() == at::ScalarType::Half)
    {
      silu_mul_clamp_kernel<half, false>
              <<<grid, block, 0, stream>>>(params);
    } else 
    {
      TORCH_CHECK(0, "not support this datatype");
    }
}
