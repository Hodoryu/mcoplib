#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kShortRowThreads = 256;
constexpr int kFewRowsThreads = 1024;
constexpr int kMidRowsThreads = 512;
constexpr int kLargeRowsThreads = 256;
constexpr int kSmallRegionsThreads = 1024;
constexpr int kDecodeTopK = 512;
constexpr int kRadixKeyCacheMaxRegions = 3072;
constexpr int kDecodeFastPathMinRegions = 2048;
constexpr int kOrderedCompactionMaxRegions = 12288;
constexpr int kRadixBuckets = 256;
constexpr int kRadixPasses = 4;
#ifdef USE_MACA
constexpr int kWaveSize = 64;
constexpr uint64_t kFullWaveMask = 0xffffffffffffffffULL;
#else
constexpr int kWaveSize = 32;
constexpr uint32_t kFullWaveMask = 0xffffffffu;
#endif

static_assert((kDecodeTopK & (kDecodeTopK - 1)) == 0,
              "bitonic sort size must be a power of two");
static_assert(kOrderedCompactionMaxRegions >= kDecodeFastPathMinRegions,
              "ordered compaction range must not be empty");
static_assert(kSmallRegionsThreads == 1024,
              "small-rank path requires one thread per first 1024 regions");
static_assert(kShortRowThreads % kWaveSize == 0 &&
                  kSmallRegionsThreads % kWaveSize == 0 &&
                  kMidRowsThreads % kWaveSize == 0 &&
                  kFewRowsThreads % kWaveSize == 0 &&
                  kLargeRowsThreads % kWaveSize == 0,
              "all block sizes must contain complete waves");

// Inclusive prefix for a boolean flag. Shared memory contains one count per
// wave, so the scan works for both CUDA warps and MetaX waves.
template <int BLOCK_THREADS>
__device__ __forceinline__ int block_flag_prefix(bool flag, int* wave_counts) {
  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWaveSize - 1);
  const int wave = tid / kWaveSize;

  const uint64_t ballot =
      static_cast<uint64_t>(__ballot_sync(kFullWaveMask, flag));
  if (lane == 0) wave_counts[wave] = __popcll(ballot);
  __syncthreads();

  const uint64_t lower_mask =
      lane == 0 ? 0 : ((uint64_t{1} << lane) - 1);
  int prefix = __popcll(ballot & lower_mask) + static_cast<int>(flag);
#pragma unroll
  for (int preceding_wave = 0; preceding_wave < kWaves; ++preceding_wave) {
    if (preceding_wave < wave) prefix += wave_counts[preceding_wave];
  }
  return prefix;
}

// Exclusive prefix for an arbitrary non-negative count. Each wave scans in
// registers; only one total per wave crosses shared memory. On C600U this
// replaces a 1024-element shared-memory Blelloch scan and its ~20 barriers
// with two block barriers.
template <int BLOCK_THREADS>
__device__ __forceinline__ int block_count_exclusive_prefix(
    int count, int* wave_offsets, int* block_total) {
  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & (kWaveSize - 1);
  const int wave = tid / kWaveSize;

  int inclusive = count;
#pragma unroll
  for (int offset = 1; offset < kWaveSize; offset <<= 1) {
    const int preceding =
        __shfl_up_sync(kFullWaveMask, inclusive, offset);
    if (lane >= offset) inclusive += preceding;
  }
  if (lane == kWaveSize - 1) wave_offsets[wave] = inclusive;
  __syncthreads();

  if (wave == 0) {
    const int wave_count = lane < kWaves ? wave_offsets[lane] : 0;
    int wave_inclusive = wave_count;
#pragma unroll
    for (int offset = 1; offset < kWaveSize; offset <<= 1) {
      const int preceding =
          __shfl_up_sync(kFullWaveMask, wave_inclusive, offset);
      if (lane >= offset) wave_inclusive += preceding;
    }
    if (lane < kWaves) wave_offsets[lane] = wave_inclusive - wave_count;
    if (lane == kWaves - 1) *block_total = wave_inclusive;
  }
  __syncthreads();
  return wave_offsets[wave] + inclusive - count;
}

// Numeric comparison deliberately makes -0.0f and +0.0f share a key, which
// matches the Triton implementation used by the original operator.
__device__ __forceinline__ uint32_t ordered_float_key(float score) {
  const uint32_t bits = __float_as_uint(score);
  return score < 0.0f ? ~bits : (bits | 0x80000000u);
}

template <int BLOCK_THREADS>
__global__ void region_topk_ids_fill_all_kernel(
    const int32_t* __restrict__ lengths, int32_t* __restrict__ out,
    int32_t seq_regions, int64_t stride_out_row, int32_t topk) {
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  int visible = lengths[row];
  visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);
  int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;
  for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
    row_out[slot] = slot < visible ? slot : -1;
  }
}

// Direct stable rank is faster than four atomic radix histograms for the
// production topk=512 boundary range. The separate kernel keeps its key tile
// out of the wide-row path's shared-memory footprint.
template <int BLOCK_THREADS>
__global__ void region_topk_ids_small_rank_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ out, int32_t seq_regions,
    int64_t stride_logits_row, int64_t stride_out_row, int32_t topk) {
  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  __shared__ uint32_t keys[BLOCK_THREADS + 1];
  __shared__ int selected_wave_counts[kWaves];
  __shared__ int extra_rank;

  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  const float* row_logits =
      logits + static_cast<int64_t>(row) * stride_logits_row;
  int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;
  int visible = lengths[row];
  visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);

  if (topk >= visible) {
    for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
      row_out[slot] = slot < visible ? slot : -1;
    }
    return;
  }

  for (int region = tid; region < visible; region += BLOCK_THREADS) {
    keys[region] = ordered_float_key(row_logits[region]);
  }
  __syncthreads();

  const uint32_t key = tid < visible ? keys[tid] : 0;
  bool selected = false;
  if (tid < visible) {
    int rank = 0;
    for (int other_id = 0; other_id < visible; ++other_id) {
      const uint32_t other_key = keys[other_id];
      rank += other_key > key || (other_key == key && other_id < tid);
    }
    selected = rank < topk;
  }

  bool extra_selected = false;
  if (visible > BLOCK_THREADS) {
    const uint32_t extra_key = keys[BLOCK_THREADS];
    const int extra_precedes =
        key > extra_key || (key == extra_key && tid < BLOCK_THREADS);
    block_count_exclusive_prefix<BLOCK_THREADS>(
        extra_precedes, selected_wave_counts, &extra_rank);
    if (tid == 0) extra_selected = extra_rank < topk;
  }

  const int position =
      block_flag_prefix<BLOCK_THREADS>(selected, selected_wave_counts) - 1;
  if (selected) row_out[position] = tid;
  if (tid == 0 && extra_selected) {
    int selected_before_extra = 0;
#pragma unroll
    for (int wave = 0; wave < kWaves; ++wave) {
      selected_before_extra += selected_wave_counts[wave];
    }
    row_out[selected_before_extra] = BLOCK_THREADS;
  }
}

template <int BLOCK_THREADS, bool CACHE_CANDIDATES, bool CACHE_RADIX_KEYS>
__global__ void region_topk_ids_kernel(
    const float* __restrict__ logits, const int32_t* __restrict__ lengths,
    int32_t* __restrict__ out, int32_t seq_regions,
    int64_t stride_logits_row, int64_t stride_out_row, int32_t topk) {
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  const float* row_logits =
      logits + static_cast<int64_t>(row) * stride_logits_row;
  int32_t* row_out = out + static_cast<int64_t>(row) * stride_out_row;

  constexpr int kWaves = BLOCK_THREADS / kWaveSize;
  __shared__ uint32_t histogram[kRadixBuckets];
  __shared__ int selected_wave_counts[kWaves];
  __shared__ int tied_wave_counts[kWaves];
  __shared__ int candidate_ids[kDecodeTopK];
  __shared__ int candidate_total;
  __shared__ int candidate_overflow;
  __shared__ uint32_t threshold;
  __shared__ int remaining;
  __shared__ int threshold_ties;
  __shared__ int emit_cursor;
  __shared__ int ties_seen;

  int visible = lengths[row];
  visible = visible < 0 ? 0 : (visible > seq_regions ? seq_regions : visible);

  // Write selected ids and -1 padding in one pass. This avoids clearing all
  // topk slots, synchronizing, and then overwriting the visible prefix.
  if (topk >= visible) {
    for (int slot = tid; slot < topk; slot += BLOCK_THREADS) {
      row_out[slot] = slot < visible ? slot : -1;
    }
    return;
  }

  // If only one id is excluded, find the stable minimum directly. On equal
  // scores the largest id is discarded, matching stable descending top-k.
  if (visible == topk + 1) {
    const int lane = tid & (kWaveSize - 1);
    const int wave = tid / kWaveSize;
    uint32_t bottom_key = UINT32_MAX;
    int bottom_id = -1;
    for (int region = tid; region < visible; region += BLOCK_THREADS) {
      const uint32_t key = ordered_float_key(row_logits[region]);
      if (key < bottom_key || (key == bottom_key && region > bottom_id)) {
        bottom_key = key;
        bottom_id = region;
      }
    }
#pragma unroll
    for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
      const uint32_t other_key =
          __shfl_down_sync(kFullWaveMask, bottom_key, offset);
      const int other_id =
          __shfl_down_sync(kFullWaveMask, bottom_id, offset);
      if (lane + offset < kWaveSize &&
          (other_key < bottom_key ||
           (other_key == bottom_key && other_id > bottom_id))) {
        bottom_key = other_key;
        bottom_id = other_id;
      }
    }
    if (lane == 0) {
      histogram[wave] = bottom_key;
      candidate_ids[wave] = bottom_id;
    }
    __syncthreads();

    if (wave == 0) {
      bottom_key = lane < kWaves ? histogram[lane] : UINT32_MAX;
      bottom_id = lane < kWaves ? candidate_ids[lane] : -1;
#pragma unroll
      for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
        const uint32_t other_key =
            __shfl_down_sync(kFullWaveMask, bottom_key, offset);
        const int other_id =
            __shfl_down_sync(kFullWaveMask, bottom_id, offset);
        if (lane + offset < kWaveSize &&
            (other_key < bottom_key ||
             (other_key == bottom_key && other_id > bottom_id))) {
          bottom_key = other_key;
          bottom_id = other_id;
        }
      }
      if (lane == 0) emit_cursor = bottom_id;
    }
    __syncthreads();

    const int excluded_id = emit_cursor;
    for (int region = tid; region < visible; region += BLOCK_THREADS) {
      if (region != excluded_id) {
        row_out[region - static_cast<int>(region > excluded_id)] = region;
      }
    }
    return;
  }

  // Exact rank is faster than four radix passes for a single short tile.
  if (BLOCK_THREADS == kShortRowThreads && visible <= kShortRowThreads) {
    const uint32_t key =
        tid < visible ? ordered_float_key(row_logits[tid]) : 0;
    histogram[tid] = key;
    __syncthreads();

    bool selected = false;
    if (tid < visible) {
      int rank = 0;
      for (int other_id = 0; other_id < visible; ++other_id) {
        const uint32_t other_key = histogram[other_id];
        rank += other_key > key || (other_key == key && other_id < tid);
      }
      selected = rank < topk;
    }
    const int position =
        block_flag_prefix<BLOCK_THREADS>(selected, selected_wave_counts) - 1;
    if (selected) row_out[position] = tid;
    return;
  }

  if (tid == 0) {
    threshold = 0;
    remaining = topk;
    threshold_ties = 0;
  }
  __syncthreads();

  // The observed 2K decode shapes have only two or three values per thread.
  // Cache transformed keys in registers so four radix passes do not reload
  // the same logits from global memory.
  constexpr int kCachedKeysPerThread =
      (kRadixKeyCacheMaxRegions + BLOCK_THREADS - 1) / BLOCK_THREADS;
  uint32_t cached_radix_keys[kCachedKeysPerThread];
  if (CACHE_RADIX_KEYS) {
#pragma unroll
    for (int item = 0; item < kCachedKeysPerThread; ++item) {
      const int region = tid + item * BLOCK_THREADS;
      cached_radix_keys[item] =
          region < visible ? ordered_float_key(row_logits[region]) : 0;
    }
  }

#pragma unroll
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    for (int bucket = tid; bucket < kRadixBuckets;
         bucket += BLOCK_THREADS) {
      histogram[bucket] = 0;
    }
    __syncthreads();

    const int shift = 24 - pass * 8;
    const int high = shift + 8;
    if (CACHE_RADIX_KEYS) {
#pragma unroll
      for (int item = 0; item < kCachedKeysPerThread; ++item) {
        const int region = tid + item * BLOCK_THREADS;
        if (region < visible) {
          const uint32_t key = cached_radix_keys[item];
          const bool matches =
              pass == 0 || ((key >> high) == (threshold >> high));
          if (matches) {
            const int bucket = static_cast<int>((key >> shift) & 0xffu);
            atomicAdd(&histogram[bucket], 1u);
          }
        }
      }
    } else {
      for (int region = tid; region < visible; region += BLOCK_THREADS) {
        const uint32_t key = ordered_float_key(row_logits[region]);
        const bool matches =
            pass == 0 || ((key >> high) == (threshold >> high));
        if (matches) {
          const int bucket = static_cast<int>((key >> shift) & 0xffu);
          atomicAdd(&histogram[bucket], 1u);
        }
      }
    }
    __syncthreads();

    // Scan buckets in descending key order. Exactly one thread owns the
    // interval containing `remaining`, avoiding a serial 256-bin scan on
    // every radix pass.
    const int descending_bucket = kRadixBuckets - 1 - tid;
    const int bucket_count =
        tid < kRadixBuckets
            ? static_cast<int>(histogram[descending_bucket])
            : 0;
    const int target_rank = remaining;
    // Every wave must snapshot the old rank before the winning thread
    // publishes the next pass's rank into shared memory.
    __syncthreads();
    const int count_above = block_count_exclusive_prefix<BLOCK_THREADS>(
        bucket_count, selected_wave_counts, &candidate_total);
    if (tid < kRadixBuckets && count_above < target_rank &&
        count_above + bucket_count >= target_rank) {
      remaining = target_rank - count_above;
      threshold |= static_cast<uint32_t>(descending_bucket) << shift;
      if (pass == kRadixPasses - 1) threshold_ties = bucket_count;
    }
    __syncthreads();
  }

  // When every value tied at the cutoff is selected, exactly topk values
  // satisfy key >= threshold. Medium-width rows are compacted tile-by-tile in
  // logical-id order. Wider rows use one block scan followed by a fixed 512-id
  // sort. A partially selected cutoff tie uses the stable fallback below.
  if (BLOCK_THREADS > kShortRowThreads && topk == kDecodeTopK &&
      visible >= kDecodeFastPathMinRegions &&
      threshold_ties == remaining) {
    if (visible <= kOrderedCompactionMaxRegions) {
      if (tid == 0) emit_cursor = 0;
      __syncthreads();
      for (int base = 0; base < visible; base += BLOCK_THREADS) {
        // Every thread snapshots the cursor before block_flag_prefix's
        // barrier. Thread 0 can then advance the shared cursor without racing
        // the global writes, which use this private snapshot.
        const int output_base = emit_cursor;
        const int region = base + tid;
        const bool selected =
            region < visible &&
            ordered_float_key(row_logits[region]) >= threshold;
        const int selected_prefix = block_flag_prefix<BLOCK_THREADS>(
            selected, selected_wave_counts);
        if (selected) {
          row_out[output_base + selected_prefix - 1] = region;
        }
        if (tid == 0) {
          int tile_selected = 0;
#pragma unroll
          for (int wave = 0; wave < kWaves; ++wave) {
            tile_selected += selected_wave_counts[wave];
          }
          emit_cursor = output_base + tile_selected;
        }
        __syncthreads();
      }
      return;
    }

    constexpr int kCandidateCacheCapacity = 8;
    int cached_candidate_ids[kCandidateCacheCapacity];
    int local_candidates = 0;
    bool local_overflow = false;
    for (int region = tid; region < visible; region += BLOCK_THREADS) {
      const uint32_t key = ordered_float_key(row_logits[region]);
      if (key >= threshold) {
        if (CACHE_CANDIDATES) {
          if (local_candidates < kCandidateCacheCapacity) {
            cached_candidate_ids[local_candidates] = region;
          } else {
            local_overflow = true;
          }
        }
        ++local_candidates;
      }
    }
    if (CACHE_CANDIDATES) {
      if (tid == 0) candidate_overflow = 0;
      __syncthreads();
      if (local_overflow) atomicExch(&candidate_overflow, 1);
    }
    const int candidate_offset = block_count_exclusive_prefix<BLOCK_THREADS>(
        local_candidates, selected_wave_counts, &candidate_total);

    if (candidate_total == topk) {
      if (CACHE_CANDIDATES && candidate_overflow == 0) {
        for (int index = 0; index < local_candidates; ++index) {
          candidate_ids[candidate_offset + index] =
              cached_candidate_ids[index];
        }
      } else {
        int position = candidate_offset;
        for (int region = tid; region < visible; region += BLOCK_THREADS) {
          const uint32_t key = ordered_float_key(row_logits[region]);
          if (key >= threshold) candidate_ids[position++] = region;
        }
      }
      __syncthreads();

      // All candidate ids are unique. A fixed-size bitonic network gives
      // deterministic ascending order without another global-memory pass.
#pragma unroll
      for (int size = 2; size <= kDecodeTopK; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
          for (int index = tid; index < kDecodeTopK;
               index += BLOCK_THREADS) {
            const int partner = index ^ stride;
            if (partner > index) {
              const int lhs = candidate_ids[index];
              const int rhs = candidate_ids[partner];
              const bool ascending = (index & size) == 0;
              if ((ascending && lhs > rhs) ||
                  (!ascending && lhs < rhs)) {
                candidate_ids[index] = rhs;
                candidate_ids[partner] = lhs;
              }
            }
          }
          __syncthreads();
        }
      }
      for (int slot = tid; slot < kDecodeTopK; slot += BLOCK_THREADS) {
        row_out[slot] = candidate_ids[slot];
      }
      return;
    }
  }

  if (tid == 0) {
    emit_cursor = 0;
    ties_seen = 0;
  }
  __syncthreads();

  for (int base = 0; base < visible; base += BLOCK_THREADS) {
    const int region = base + tid;
    const bool live = region < visible;
    uint32_t key = 0;
    if (live) key = ordered_float_key(row_logits[region]);
    const bool tied = live && key == threshold;
    const bool greater = live && key > threshold;

    const int tied_prefix =
        block_flag_prefix<BLOCK_THREADS>(tied, tied_wave_counts);
    const bool selected =
        greater || (tied && ties_seen + tied_prefix <= remaining);
    const int selected_prefix =
        block_flag_prefix<BLOCK_THREADS>(selected, selected_wave_counts);

    if (selected) {
      const int position = emit_cursor + selected_prefix - 1;
      if (position < topk) row_out[position] = region;
    }
    __syncthreads();
    if (tid == 0) {
      int tile_selected = 0;
      int tile_tied = 0;
#pragma unroll
      for (int wave = 0; wave < kWaves; ++wave) {
        tile_selected += selected_wave_counts[wave];
        tile_tied += tied_wave_counts[wave];
      }
      emit_cursor += tile_selected;
      ties_seen += tile_tied;
    }
    __syncthreads();
  }
}

template <int BLOCK_THREADS>
void launch_region_topk_ids_kernel(
    const float* logits, const int32_t* lengths, int32_t* out, int32_t rows,
    int32_t seq_regions, int64_t stride_logits_row, int64_t stride_out_row,
    int32_t topk, bool cache_candidates, bool cache_radix_keys,
    cudaStream_t stream) {
  if (cache_candidates) {
    // Candidate caching is used only for >=32K regions; radix-key caching is
    // used only for <=3K, so the two modes are mutually exclusive.
    region_topk_ids_kernel<BLOCK_THREADS, true, false>
        <<<static_cast<unsigned int>(rows), BLOCK_THREADS, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
  } else if (cache_radix_keys) {
    region_topk_ids_kernel<BLOCK_THREADS, false, true>
        <<<static_cast<unsigned int>(rows), BLOCK_THREADS, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
  } else {
    region_topk_ids_kernel<BLOCK_THREADS, false, false>
        <<<static_cast<unsigned int>(rows), BLOCK_THREADS, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
  }
}

cudaError_t launch_region_topk_ids(
    const float* logits, const int32_t* lengths, int32_t* out, int32_t rows,
    int32_t seq_regions, int64_t stride_logits_row, int64_t stride_out_row,
    int32_t topk, cudaStream_t stream) {
  if (rows == 0 || topk == 0) return cudaSuccess;

  if (topk >= seq_regions) {
    region_topk_ids_fill_all_kernel<kShortRowThreads>
        <<<static_cast<unsigned int>(rows), kShortRowThreads, 0, stream>>>(
            lengths, out, seq_regions, stride_out_row, topk);
  } else if (topk == kDecodeTopK && rows <= 32 &&
             seq_regions > kDecodeTopK + 1 && seq_regions <= 1025) {
    region_topk_ids_small_rank_kernel<kSmallRegionsThreads>
        <<<static_cast<unsigned int>(rows), kSmallRegionsThreads, 0, stream>>>(
            logits, lengths, out, seq_regions, stride_logits_row,
            stride_out_row, topk);
  } else if (topk == kDecodeTopK && rows <= 32 &&
             seq_regions <= 1024 && seq_regions > kShortRowThreads) {
    launch_region_topk_ids_kernel<kSmallRegionsThreads>(
        logits, lengths, out, rows, seq_regions, stride_logits_row,
        stride_out_row, topk, false, false, stream);
  } else if (rows <= 32 && seq_regions > kShortRowThreads) {
    launch_region_topk_ids_kernel<kFewRowsThreads>(
        logits, lengths, out, rows, seq_regions, stride_logits_row,
        stride_out_row, topk, seq_regions >= 32768,
        seq_regions <= kRadixKeyCacheMaxRegions, stream);
  } else if (rows <= 64 && seq_regions > kShortRowThreads) {
    launch_region_topk_ids_kernel<kMidRowsThreads>(
        logits, lengths, out, rows, seq_regions, stride_logits_row,
        stride_out_row, topk, seq_regions >= 32768, false, stream);
  } else {
    launch_region_topk_ids_kernel<kLargeRowsThreads>(
        logits, lengths, out, rows, seq_regions, stride_logits_row,
        stride_out_row, topk, false, false, stream);
  }
  return cudaGetLastError();
}

}  // namespace

torch::Tensor region_topk_ids(const torch::Tensor& logits,
                              const torch::Tensor& lengths, int64_t topk) {
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor");
  TORCH_CHECK(logits.device() == lengths.device(),
              "logits and lengths must be on the same device");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must have dtype float32");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt32,
              "lengths must have dtype int32");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
  TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(logits.stride(1) == 1, "logits stride(1) must be 1");
  TORCH_CHECK(logits.stride(0) >= logits.size(1),
              "logits row stride must cover seq_regions");
  TORCH_CHECK(lengths.numel() == logits.size(0),
              "lengths size must match logits rows");
  TORCH_CHECK(topk >= 0, "topk must be non-negative");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<int32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max() &&
                  topk <= std::numeric_limits<int32_t>::max(),
              "rows, seq_regions, and topk must fit in int32");

  at::cuda::OptionalCUDAGuard const device_guard(logits.device());
  auto out = torch::empty({logits.size(0), topk},
                          logits.options().dtype(torch::kInt32));
  const cudaError_t status = launch_region_topk_ids(
      logits.data_ptr<float>(), lengths.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(), static_cast<int32_t>(logits.size(0)),
      static_cast<int32_t>(logits.size(1)), logits.stride(0), out.stride(0),
      static_cast<int32_t>(topk), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess,
              "region_topk_ids launch failed: ", cudaGetErrorString(status));
  return out;
}
