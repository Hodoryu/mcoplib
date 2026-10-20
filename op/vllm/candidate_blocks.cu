#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int THREADS = 256;
constexpr int SCORE_THREADS = 512;
constexpr int MAX_SHARED_BITMAP_BYTES = 65536;

__global__ void block_scores_kernel(const float* logits, const int* starts,
                                    const int* ends, float* scores, int64_t sr,
                                    int64_t sc, int64_t ss, int64_t se,
                                    int width, int nblocks, int block_size,
                                    int row_repeat) {
  int row = blockIdx.y;
  int start = starts ? starts[(row / row_repeat) * ss] : 0;
  int end = ends[(row / row_repeat) * se];
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nblocks;
       b += gridDim.x * blockDim.x) {
    float best = -INFINITY;
    int64_t base = int64_t(start) + int64_t(b) * block_size;
    for (int j = 0; j < block_size; ++j) {
      int64_t col = base + j;
      if (col >= end || col >= width) continue;
      float x = logits[int64_t(row) * sr + col * sc];
      if (isnan(x) || x > best) best = x;
    }
    if (end > start && b == (end - start - 1) / block_size) best = INFINITY;
    scores[int64_t(row) * nblocks + b] = best;
  }
}

// The production decode layout has eight FP32 logits per candidate block.
// Aligned 16-byte loads replace eight scalar loads per score.
__global__ void block_scores_vec4_kernel(const float* logits, const int* ends,
                                         float* scores, int64_t se, int width,
                                         int nblocks, int row_repeat) {
  int row = blockIdx.y;
  int end = ends[int64_t(row / row_repeat) * se];
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nblocks;
       b += gridDim.x * blockDim.x) {
    int col = b * 8;
    float best = -INFINITY;
    if (col + 8 <= end) {
      const float4* p = reinterpret_cast<const float4*>(logits + int64_t(row) * width + col);
      float4 x = p[0], y = p[1];
      float v[8] = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (isnan(v[j]) || v[j] > best) best = v[j];
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (col + j >= end) break;
        float x = logits[int64_t(row) * width + col + j];
        if (isnan(x) || x > best) best = x;
      }
    }
    if (end > 0 && b == (end - 1) / 8) best = INFINITY;
    scores[int64_t(row) * nblocks + b] = best;
  }
}

__global__ void block_scores_aligned_vec4_kernel(
    const float* logits, const int* ends, float* scores, int64_t se,
    int width, int nblocks, int block_size, int row_repeat) {
  int row = blockIdx.y;
  int end = ends[int64_t(row / row_repeat) * se];
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nblocks;
       b += gridDim.x * blockDim.x) {
    int col = b * block_size;
    float best = -INFINITY;
    for (int j = 0; j < block_size; j += 4) {
      int x = col + j;
      if (x + 4 <= end && x + 4 <= width) {
        float4 packed = *reinterpret_cast<const float4*>(
            logits + int64_t(row) * width + x);
        float values[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
          if (isnan(values[lane]) || values[lane] > best) best = values[lane];
      } else {
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
          int tail = x + lane;
          if (tail >= end || tail >= width) continue;
          float value = logits[int64_t(row) * width + tail];
          if (isnan(value) || value > best) best = value;
        }
      }
    }
    if (end > 0 && b == (end - 1) / block_size) best = INFINITY;
    scores[int64_t(row) * nblocks + b] = best;
  }
}

__global__ void store_candidates_kernel(const float* values, const int64_t* indices,
                                        int* out, int64_t osr, int64_t osc,
                                        int k, int out_k) {
  int row = blockIdx.y;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < out_k;
       col += gridDim.x * blockDim.x) {
    int value = -1;
    if (col < k && values[int64_t(row) * k + col] > -INFINITY)
      value = static_cast<int>(indices[int64_t(row) * k + col]);
    out[int64_t(row) * osr + int64_t(col) * osc] = value;
  }
}

__global__ void store_all_candidates_kernel(const float* scores, int* out,
                                             int64_t osr, int64_t osc,
                                             int nblocks, int out_k) {
  int row = blockIdx.y;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < out_k;
       col += gridDim.x * blockDim.x) {
    int value = -1;
    if (col < nblocks && scores[int64_t(row) * nblocks + col] > -INFINITY)
      value = col;
    out[int64_t(row) * osr + int64_t(col) * osc] = value;
  }
}

constexpr int TOPK_THREADS = 1024;
constexpr int RADIX_BITS = 8;
constexpr int RADIX_BINS = 1 << RADIX_BITS;
constexpr int RADIX_GROUPS = 8;

__device__ __forceinline__ uint32_t ordered_float_key(float x) {
  if (isnan(x)) return 0xffffffffu;
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Select an unordered runtime-sized top-k set with four byte-radix passes and
// write int32 indices directly. Full sorting is unnecessary for candidate
// masking. The
// per-group histograms reduce shared-atomic contention while staying within
// the C600U shared-memory budget (256 * 8 * 4 = 8 KiB).
__global__ void select_topk_unordered_radix_kernel(const float* scores, int* out,
                                                   int64_t osr, int64_t osc,
                                                   int nblocks, int topk) {
  __shared__ uint32_t histogram[RADIX_BINS * RADIX_GROUPS];
  __shared__ uint32_t prefix;
  __shared__ int remaining;
  __shared__ uint32_t threshold;
  __shared__ int output_count;
  __shared__ int equal_count;
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int group = tid & (RADIX_GROUPS - 1);

  if (tid == 0) {
    prefix = 0;
    remaining = topk;
  }
  __syncthreads();

#pragma unroll
  for (int pass = 0; pass < 4; ++pass) {
    int shift = 24 - pass * RADIX_BITS;
    for (int i = tid; i < RADIX_BINS * RADIX_GROUPS; i += blockDim.x)
      histogram[i] = 0;
    __syncthreads();

    uint32_t current_prefix = prefix;
    for (int i = tid; i < nblocks; i += blockDim.x) {
      uint32_t key = ordered_float_key(scores[int64_t(row) * nblocks + i]);
      if (pass == 0 || (key >> (shift + RADIX_BITS)) == current_prefix) {
        int digit = (key >> shift) & (RADIX_BINS - 1);
        atomicAdd(histogram + digit * RADIX_GROUPS + group, 1u);
      }
    }
    __syncthreads();

    // Collapse the privatized histograms before the short serial bin search.
    for (int bin = tid; bin < RADIX_BINS; bin += blockDim.x) {
      uint32_t count = 0;
#pragma unroll
      for (int g = 0; g < RADIX_GROUPS; ++g)
        count += histogram[bin * RADIX_GROUPS + g];
      histogram[bin * RADIX_GROUPS] = count;
    }
    __syncthreads();

    if (tid == 0) {
      int rank = remaining;
      int selected = 0;
      for (int bin = RADIX_BINS - 1; bin >= 0; --bin) {
        int count = histogram[bin * RADIX_GROUPS];
        if (rank > count) {
          rank -= count;
        } else {
          selected = bin;
          break;
        }
      }
      prefix = pass == 0 ? uint32_t(selected)
                         : (prefix << RADIX_BITS) | uint32_t(selected);
      remaining = rank;
      if (pass == 3) threshold = prefix;
    }
    __syncthreads();
  }

  if (tid == 0) {
    output_count = 0;
    equal_count = 0;
  }
  __syncthreads();

  uint32_t kth_key = threshold;
  int take_equal = remaining;
  for (int i = tid; i < nblocks; i += blockDim.x) {
    float value = scores[int64_t(row) * nblocks + i];
    uint32_t key = ordered_float_key(value);
    bool selected = key > kth_key;
    if (key == kth_key) selected = atomicAdd(&equal_count, 1) < take_equal;
    if (selected) {
      int slot = atomicAdd(&output_count, 1);
      // Match store_candidates_kernel: selected NaNs and -inf become padding.
      out[int64_t(row) * osr + int64_t(slot) * osc] =
          value > -INFINITY ? i : -1;
    }
  }
}

__global__ void candidate_bitmap_scatter_kernel(
    const int* candidates, const int* starts, uint32_t* bitmap, int64_t cr,
    int64_t cc, int64_t ss, int width, int nblocks, int block_size, int k,
    int nwords, int row_repeat) {
  int row = blockIdx.y;
  int start = starts ? starts[(row / row_repeat) * ss] : 0;
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < k;
       col += gridDim.x * blockDim.x) {
    int b = candidates[int64_t(row) * cr + int64_t(col) * cc];
    if (b < 0) continue;
    if (int64_t(start) + int64_t(b) * block_size >= width) b = nblocks;
    atomicOr(bitmap + int64_t(row) * nwords + (b >> 5), 1u << (b & 31));
  }
}

// One CTA per row: clear a compact bitmap in shared memory, set candidate
// bits, then publish it. This merges clear + scatter without a global barrier.
__global__ void candidate_bitmap_kernel(const int* candidates, const int* starts,
                                         uint32_t* bitmap, int64_t cr, int64_t cc,
                                         int64_t ss, int width, int nblocks,
                                         int block_size, int k, int nwords,
                                         int row_repeat) {
  extern __shared__ uint32_t local[];
  int row = blockIdx.x;
  int start = starts ? starts[(row / row_repeat) * ss] : 0;
  for (int i = threadIdx.x; i < nwords; i += blockDim.x) local[i] = 0;
  __syncthreads();
  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    int b = candidates[int64_t(row) * cr + int64_t(col) * cc];
    if (b < 0) continue;
    if (int64_t(start) + int64_t(b) * block_size >= width) b = nblocks;
    atomicOr(local + (b >> 5), 1u << (b & 31));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nwords; i += blockDim.x)
    bitmap[int64_t(row) * nwords + i] = local[i];
}

__device__ __forceinline__ bool bitmap_test(const uint32_t* bits, int b) {
  return ((bits[b >> 5] >> (b & 31)) & 1u) != 0;
}

__global__ void mask_candidates_bitmap_kernel(
    float* logits, const int* starts, const int* ends, const uint32_t* bitmap,
    int64_t sr, int64_t sc, int64_t ss, int64_t se, int width, int nblocks,
    int nwords, int block_size, int row_repeat) {
  int row = blockIdx.y;
  int start = starts ? starts[(row / row_repeat) * ss] : 0;
  int end = ends[(row / row_repeat) * se];
  const uint32_t* bits = bitmap + int64_t(row) * nwords;
  bool edge = bitmap_test(bits, nblocks);
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width;
       col += gridDim.x * blockDim.x) {
    bool keep = false;
    if (col >= start && col < end) {
      keep = bitmap_test(bits, (col - start) / block_size);
      keep |= col == width - 1 && edge;
    }
    if (!keep) logits[int64_t(row) * sr + int64_t(col) * sc] = -INFINITY;
  }
}

template <int GROUP, bool HAS_TAIL>
__global__ void mask_candidates_bitmap_general_vec4_kernel(
    float* logits, const int* starts, const int* ends, const uint32_t* bitmap,
    int64_t ss, int64_t se, int width, int nblocks, int nwords,
    int block_size, int row_repeat) {
  int row = blockIdx.y;
  int start = starts ? starts[(row / row_repeat) * ss] : 0;
  int end = ends[(row / row_repeat) * se];
  const uint32_t* bits = bitmap + int64_t(row) * nwords;
  bool edge = bitmap_test(bits, nblocks);
  int vectors = HAS_TAIL ? (width + 3) / 4 : width / 4;
  int base = blockIdx.x * blockDim.x * GROUP + threadIdx.x;
#pragma unroll
  for (int group = 0; group < GROUP; ++group) {
    int v = base + group * blockDim.x;
    if (v >= vectors) continue;
    int col = v * 4;
    bool keep[4];
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      int x = col + lane;
      bool in_width = !HAS_TAIL || x < width;
      keep[lane] = !in_width;
      if (in_width && x >= start && x < end) {
        keep[lane] = bitmap_test(bits, (x - start) / block_size);
        keep[lane] |= x == width - 1 && edge;
      }
    }
    if (keep[0] && keep[1] && keep[2] && keep[3]) continue;
    float* p = logits + int64_t(row) * width + col;
    if ((!HAS_TAIL || col + 4 <= width) &&
        !keep[0] && !keep[1] && !keep[2] && !keep[3]) {
      *reinterpret_cast<float4*>(p) =
          make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    } else {
#pragma unroll
      for (int lane = 0; lane < 4; ++lane)
        if ((!HAS_TAIL || col + lane < width) && !keep[lane]) p[lane] = -INFINITY;
    }
  }
}

template <int GROUP>
__global__ void mask_candidates_bitmap_vec4_kernel(float* logits, const int* ends,
                                                   const uint32_t* bitmap,
                                                   int64_t se, int width,
                                                   int nblocks, int nwords,
                                                   int row_repeat) {
  int row = blockIdx.y;
  int end = ends[int64_t(row / row_repeat) * se];
  const uint32_t* bits = bitmap + int64_t(row) * nwords;
  bool edge = bitmap_test(bits, nblocks);
  int vectors = width / 4;
  int base = blockIdx.x * blockDim.x * GROUP + threadIdx.x;
#pragma unroll
  for (int j = 0; j < GROUP; ++j) {
    int v = base + j * blockDim.x;
    if (v >= vectors) continue;
    int col = v * 4;
    int b = v / 2;
    bool selected = bitmap_test(bits, b);
    if (selected && col + 4 <= end) continue;
    float* p = logits + int64_t(row) * width + col;
    if (!selected || col >= end) {
      if (edge && col == width - 4 && end > width - 1) {
        p[0] = p[1] = p[2] = -INFINITY;
      } else {
        *reinterpret_cast<float4*>(p) = make_float4(-INFINITY, -INFINITY,
                                                      -INFINITY, -INFINITY);
      }
    } else {
      for (int j = 0; j < 4; ++j)
        if (col + j >= end) p[j] = -INFINITY;
    }
  }
}

// Each CTA owns a logits tile and builds its own bitmap. Repeated bitmap work
// removes the global bitmap allocation and the second launch.
template <int TILE_VECTORS>
__global__ void mask_candidates_fused_vec4_kernel(
    float* logits, const int* ends, const int* candidates, int64_t se,
    int64_t cr, int64_t cc, int width, int nblocks, int nwords, int k,
    int row_repeat) {
  extern __shared__ uint32_t bits[];
  int row = blockIdx.y;
  for (int i = threadIdx.x; i < nwords; i += blockDim.x) bits[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int b = candidates[int64_t(row) * cr + int64_t(i) * cc];
    if (b < 0) continue;
    if (int64_t(b) * 8 >= width) b = nblocks;
    atomicOr(bits + (b >> 5), 1u << (b & 31));
  }
  __syncthreads();
  int end = ends[int64_t(row / row_repeat) * se];
  bool edge = bitmap_test(bits, nblocks);
  int begin = blockIdx.x * TILE_VECTORS;
  int limit = min(begin + TILE_VECTORS, width / 4);
  for (int v = begin + threadIdx.x; v < limit; v += blockDim.x) {
    int col = v * 4;
    int b = v / 2;
    bool selected = bitmap_test(bits, b);
    if (selected && col + 4 <= end) continue;
    float* p = logits + int64_t(row) * width + col;
    if (!selected || col >= end) {
      if (edge && col == width - 4 && end > width - 1) {
        p[0] = p[1] = p[2] = -INFINITY;
      } else {
        *reinterpret_cast<float4*>(p) = make_float4(-INFINITY, -INFINITY,
                                                    -INFINITY, -INFINITY);
      }
    } else {
      for (int j = 0; j < 4; ++j)
        if (col + j >= end) p[j] = -INFINITY;
    }
  }
}

template <bool RESTORE>
__global__ void candidate_values_kernel(float* logits, const int* ends,
                                        const int* candidates, float* saved,
                                        int64_t se, int64_t cr, int64_t cc,
                                        int width, int nblocks, int k,
                                        int row_repeat) {
  int row = blockIdx.y;
  int end = ends[int64_t(row / row_repeat) * se];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k;
       i += gridDim.x * blockDim.x) {
    int b = candidates[int64_t(row) * cr + int64_t(i) * cc];
    if (b < 0) continue;
    float* p = logits + int64_t(row) * width;
    float* t = saved + (int64_t(row) * k + i) * 8;
    if (b >= nblocks) {
      if (end > width - 1) {
        if (RESTORE) p[width - 1] = t[0];
        else t[0] = p[width - 1];
      }
      continue;
    }
    int col = b * 8;
    if (RESTORE) {
      if (col + 8 <= end) {
        reinterpret_cast<float4*>(p + col)[0] = reinterpret_cast<float4*>(t)[0];
        reinterpret_cast<float4*>(p + col)[1] = reinterpret_cast<float4*>(t)[1];
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (col + j < end) p[col + j] = t[j];
      }
    } else {
      reinterpret_cast<float4*>(t)[0] = reinterpret_cast<float4*>(p + col)[0];
      reinterpret_cast<float4*>(t)[1] = reinterpret_cast<float4*>(p + col)[1];
    }
  }
}

template <int TILE_VECTORS>
__global__ void fill_neg_inf_vec4_kernel(float* logits, int width) {
  int row = blockIdx.y;
  int vectors = width / 4;
  float4 negative = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
  int begin = blockIdx.x * TILE_VECTORS;
  int limit = min(begin + TILE_VECTORS, vectors);
  for (int v = begin + threadIdx.x; v < limit; v += blockDim.x)
    reinterpret_cast<float4*>(logits + int64_t(row) * width)[v] = negative;
}

void check_common(const torch::Tensor& logits, const c10::optional<torch::Tensor>& starts,
                  const torch::Tensor& ends, int64_t block_size, int64_t row_repeat) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.scalar_type() == at::kFloat,
              "logits must be a CUDA float32 matrix");
  TORCH_CHECK(ends.is_cuda() && ends.scalar_type() == at::kInt && ends.dim() == 1,
              "ends must be CUDA int32 vector");
  TORCH_CHECK(ends.device() == logits.device(),
              "logits and ends must be on the same device");
  TORCH_CHECK(block_size > 0 && row_repeat > 0,
              "block_size and row_repeat must be positive");
  TORCH_CHECK(ends.numel() >= (logits.size(0) + row_repeat - 1) / row_repeat,
              "ends is too short");
  if (starts) TORCH_CHECK(starts->is_cuda() && starts->device() == logits.device() &&
                          starts->scalar_type() == at::kInt && starts->dim() == 1 &&
                          starts->numel() >=
                              (logits.size(0) + row_repeat - 1) / row_repeat,
                          "starts must be a sufficiently long CUDA int32 vector on "
                          "the logits device");
}

void launch_check() {
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "candidate CUDA launch failed: ", cudaGetErrorString(err));
}

void launch_block_scores(const torch::Tensor& logits,
                         const c10::optional<torch::Tensor>& starts,
                         const torch::Tensor& ends, torch::Tensor& scores,
                         int block_size, int row_repeat, cudaStream_t stream) {
  int rows = logits.size(0), width = logits.size(1), nblocks = scores.size(1);
  if (!starts && block_size == 8 && width % 8 == 0 && logits.stride(1) == 1 &&
      logits.stride(0) == width) {
    block_scores_vec4_kernel<<<dim3((nblocks + SCORE_THREADS - 1) / SCORE_THREADS, rows),
                               SCORE_THREADS, 0, stream>>>(
        logits.data_ptr<float>(), ends.data_ptr<int>(), scores.data_ptr<float>(),
        ends.stride(0), width, nblocks, row_repeat);
  } else if (!starts && block_size % 4 == 0 && width % 4 == 0 &&
             logits.stride(1) == 1 && logits.stride(0) == width) {
    block_scores_aligned_vec4_kernel
        <<<dim3((nblocks + SCORE_THREADS - 1) / SCORE_THREADS, rows),
            SCORE_THREADS, 0, stream>>>(
            logits.data_ptr<float>(), ends.data_ptr<int>(), scores.data_ptr<float>(),
            ends.stride(0), width, nblocks, block_size, row_repeat);
  } else {
    block_scores_kernel<<<dim3((nblocks + THREADS - 1) / THREADS, rows),
                          THREADS, 0, stream>>>(
        logits.data_ptr<float>(), starts ? starts->data_ptr<int>() : nullptr,
        ends.data_ptr<int>(), scores.data_ptr<float>(), logits.stride(0),
        logits.stride(1), starts ? starts->stride(0) : 0, ends.stride(0),
        width, nblocks, block_size, row_repeat);
  }
  launch_check();
}
}  // namespace

void select_candidate_blocks(torch::Tensor logits, c10::optional<torch::Tensor> starts,
                             torch::Tensor ends, int64_t topk_blocks,
                             int64_t block_size, torch::Tensor out,
                             int64_t row_repeat) {
  check_common(logits, starts, ends, block_size, row_repeat);
  const at::cuda::CUDAGuard device_guard(logits.device());
  int rows = logits.size(0), width = logits.size(1);
  TORCH_CHECK(topk_blocks >= 0 && out.is_cuda() && out.scalar_type() == at::kInt &&
              out.device() == logits.device() && out.dim() == 2 &&
              out.size(0) == rows && out.size(1) == topk_blocks,
              "out must be int32 [rows, topk_blocks]");
  if (!rows || !topk_blocks) return;
  if (!width) { out.fill_(-1); return; }
  int nblocks = (width + block_size - 1) / block_size;
  auto scores = torch::empty({rows, nblocks}, logits.options());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  launch_block_scores(logits, starts, ends, scores, block_size, row_repeat, stream);

  if (topk_blocks >= nblocks) {
    store_all_candidates_kernel
        <<<dim3((topk_blocks + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
            scores.data_ptr<float>(), out.data_ptr<int>(), out.stride(0),
            out.stride(1), nblocks, topk_blocks);
    launch_check();
    return;
  }

  // One CTA per row wins while each row is moderate, or when enough rows expose
  // parallelism for larger score vectors. The threshold pass writes unordered
  // int32 indices directly and is independent of starts/layout/block size.
  bool use_unordered_radix =
      nblocks <= 65536 ||
      (nblocks <= 131072 && rows >= 16) ||
      (nblocks <= 262144 && rows >= 32);
  if (use_unordered_radix) {
    select_topk_unordered_radix_kernel<<<rows, TOPK_THREADS, 0, stream>>>(
        scores.data_ptr<float>(), out.data_ptr<int>(), out.stride(0),
        out.stride(1), nblocks, topk_blocks);
    launch_check();
    return;
  }

  auto top = scores.topk(std::min<int64_t>(topk_blocks, nblocks), -1, true, true);
  auto values = std::get<0>(top), indices = std::get<1>(top);
  int k = values.size(1);
  store_candidates_kernel<<<dim3((topk_blocks + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
      values.data_ptr<float>(), indices.data_ptr<int64_t>(), out.data_ptr<int>(),
      out.stride(0), out.stride(1), k, topk_blocks);
  launch_check();
}

static void apply_candidate_mask_fill_scatter(torch::Tensor logits,
                                              torch::Tensor ends,
                                              torch::Tensor candidates,
                                              int64_t row_repeat);

void apply_candidate_mask(torch::Tensor logits, c10::optional<torch::Tensor> starts,
                          torch::Tensor ends, torch::Tensor candidates,
                          int64_t block_size, int64_t row_repeat) {
  check_common(logits, starts, ends, block_size, row_repeat);
  const at::cuda::CUDAGuard device_guard(logits.device());
  int rows = logits.size(0), width = logits.size(1);
  TORCH_CHECK(candidates.is_cuda() && candidates.scalar_type() == at::kInt &&
              candidates.device() == logits.device() && candidates.dim() == 2 &&
              candidates.size(0) == rows,
              "candidates must be int32 [rows, k]");
  if (!rows || !width) return;
  if (!starts && block_size == 8 && candidates.size(1) == 2048 &&
      logits.stride(1) == 1 && logits.stride(0) == width &&
      ((width == 524288 && rows >= 4 && rows <= 8) ||
       (width == 1048576 && rows >= 2 && rows <= 8))) {
    apply_candidate_mask_fill_scatter(logits, ends, candidates, row_repeat);
    return;
  }
  int nblocks = (width + block_size - 1) / block_size;
  int k = candidates.size(1);
  int nwords = (nblocks + 1 + 31) / 32;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  bool contiguous = logits.stride(1) == 1 && logits.stride(0) == width;
  bool specialized_bitmap =
      !starts && block_size == 8 && width % 4 == 0 && contiguous;
  bool bitmap_fits_in_shared =
      nwords * sizeof(uint32_t) <= MAX_SHARED_BITMAP_BYTES;
  if (bitmap_fits_in_shared && specialized_bitmap && width <= 1048576 &&
      (rows == 1 || (rows == 2 && width <= 524288))) {
    constexpr int TILE_VECTORS = 4096;
    int vectors = width / 4;
    mask_candidates_fused_vec4_kernel<TILE_VECTORS>
        <<<dim3((vectors + TILE_VECTORS - 1) / TILE_VECTORS, rows), THREADS,
            nwords * sizeof(uint32_t), stream>>>(
            logits.data_ptr<float>(), ends.data_ptr<int>(), candidates.data_ptr<int>(),
            ends.stride(0), candidates.stride(0), candidates.stride(1),
            width, nblocks, nwords, k, row_repeat);
    launch_check();
    return;
  }

  torch::Tensor bitmap;
  if (bitmap_fits_in_shared) {
    bitmap = torch::empty({rows, nwords}, logits.options().dtype(at::kInt));
    candidate_bitmap_kernel<<<rows, THREADS, nwords * sizeof(uint32_t), stream>>>(
        candidates.data_ptr<int>(), starts ? starts->data_ptr<int>() : nullptr,
        reinterpret_cast<uint32_t*>(bitmap.data_ptr<int>()), candidates.stride(0),
        candidates.stride(1), starts ? starts->stride(0) : 0, width, nblocks,
        block_size, k, nwords, row_repeat);
    launch_check();
  } else {
    bitmap = torch::zeros({rows, nwords}, logits.options().dtype(at::kInt));
    if (k) {
      candidate_bitmap_scatter_kernel
          <<<dim3((k + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
              candidates.data_ptr<int>(), starts ? starts->data_ptr<int>() : nullptr,
              reinterpret_cast<uint32_t*>(bitmap.data_ptr<int>()),
              candidates.stride(0), candidates.stride(1),
              starts ? starts->stride(0) : 0, width, nblocks, block_size, k,
              nwords, row_repeat);
      launch_check();
    }
  }

  constexpr int MASK_GROUP = 2;
  const uint32_t* bitmap_data =
      reinterpret_cast<const uint32_t*>(bitmap.data_ptr<int>());
  if (specialized_bitmap) {
    int vectors = width / 4;
    mask_candidates_bitmap_vec4_kernel<MASK_GROUP>
        <<<dim3((vectors + THREADS * MASK_GROUP - 1) /
                     (THREADS * MASK_GROUP),
                 rows),
            THREADS, 0, stream>>>(logits.data_ptr<float>(), ends.data_ptr<int>(),
                                  bitmap_data, ends.stride(0), width, nblocks,
                                  nwords, row_repeat);
  } else if (contiguous) {
    int vectors = (width + 3) / 4;
    auto grid = dim3((vectors + THREADS * MASK_GROUP - 1) /
                         (THREADS * MASK_GROUP),
                     rows);
    if (width % 4 == 0) {
      mask_candidates_bitmap_general_vec4_kernel<MASK_GROUP, false>
          <<<grid, THREADS, 0, stream>>>(
              logits.data_ptr<float>(), starts ? starts->data_ptr<int>() : nullptr,
              ends.data_ptr<int>(), bitmap_data,
              starts ? starts->stride(0) : 0, ends.stride(0), width, nblocks,
              nwords, block_size, row_repeat);
    } else {
      mask_candidates_bitmap_general_vec4_kernel<MASK_GROUP, true>
          <<<grid, THREADS, 0, stream>>>(
              logits.data_ptr<float>(), starts ? starts->data_ptr<int>() : nullptr,
              ends.data_ptr<int>(), bitmap_data,
              starts ? starts->stride(0) : 0, ends.stride(0), width, nblocks,
              nwords, block_size, row_repeat);
    }
  } else {
    mask_candidates_bitmap_kernel
        <<<dim3((width + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
            logits.data_ptr<float>(), starts ? starts->data_ptr<int>() : nullptr,
            ends.data_ptr<int>(), bitmap_data, logits.stride(0),
            logits.stride(1), starts ? starts->stride(0) : 0, ends.stride(0),
            width, nblocks, nwords, block_size, row_repeat);
  }
  launch_check();
}

static void apply_candidate_mask_fill_scatter(torch::Tensor logits,
                                              torch::Tensor ends,
                                              torch::Tensor candidates,
                                              int64_t row_repeat) {
  int rows = logits.size(0), width = logits.size(1);
  int k = candidates.size(1), nblocks = width / 8;
  auto saved = torch::empty({rows, k, 8}, logits.options());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (k) {
    candidate_values_kernel<false>
        <<<dim3((k + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
            logits.data_ptr<float>(), ends.data_ptr<int>(), candidates.data_ptr<int>(),
            saved.data_ptr<float>(), ends.stride(0), candidates.stride(0),
            candidates.stride(1), width, nblocks, k, row_repeat);
    launch_check();
  }
  int vectors = width / 4;
  constexpr int FILL_TILE_VECTORS = 4096;
  fill_neg_inf_vec4_kernel<FILL_TILE_VECTORS>
      <<<dim3((vectors + FILL_TILE_VECTORS - 1) / FILL_TILE_VECTORS, rows),
          THREADS, 0, stream>>>(logits.data_ptr<float>(), width);
  launch_check();
  if (k) {
    candidate_values_kernel<true>
        <<<dim3((k + THREADS - 1) / THREADS, rows), THREADS, 0, stream>>>(
            logits.data_ptr<float>(), ends.data_ptr<int>(), candidates.data_ptr<int>(),
            saved.data_ptr<float>(), ends.stride(0), candidates.stride(0),
            candidates.stride(1), width, nblocks, k, row_repeat);
    launch_check();
  }
}
