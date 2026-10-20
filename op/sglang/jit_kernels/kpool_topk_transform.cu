// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic AOT staged-radix version with the two read-before-clear barriers
// present in the supplied JIT kernel.  It is kept outside mcoplib so this
// suspected intermittent GLM-5 correctness fix can be tested in isolation.
//
// The operator selects K pool groups per score row, expands every selected
// group to pool_size consecutive token indices, and fuses an optional page
// table lookup or ragged offset.  K remains a compile-time template constant;
// the ATen entry point dispatches the six supported specializations.

#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int kDefaultThreads = 1024;
constexpr int kMediumThreads = 512;
constexpr int kLongStagingSize = 4096;
constexpr std::size_t kLongDynamicSmem =
    2 * kLongStagingSize * sizeof(int32_t);  // 32 KiB
// Both GLM-5 3-Ki workloads have score width <= 1024, so no threshold
// bucket can contain more than 1024 candidates.  A dedicated instance avoids
// reserving the long-row 4096-entry double buffer for these short rows.
constexpr int kShortStagingSize = 1024;

// Packed staged entries hold a 16-bit row-relative index plus the top two
// bytes of the ordered FP32 key.  Packing is only valid while every row
// index fits 16 bits, i.e. score_width <= 65536.
constexpr int64_t kPackMaxScoreWidth = 65536;
constexpr std::size_t kShortDynamicSmem =
    2 * kShortStagingSize * sizeof(int32_t);  // 8 KiB
constexpr int kRadix = 256;
// With the warp-shuffle suffix scan, per-pass barrier cost no longer
// dominates short rows, so the 512-thread medium path is retired: widths in
// (1024, 2048] join the default 1024-thread scalar dispatch (measured -5.4%
// on the S=2048 chunk-1 shape against the same-day control legs).
constexpr int64_t kMediumMaxScoreWidth = 1024;
// The GLM-5 prefill mid-band splits at the measured load-pattern crossover.
// Long rows are scalar-load bandwidth bound -- chunk7 (S=14336, scalar) scans
// the same ~54 MB as chunk8 (S=16128, vec8) yet runs ~30% slower -- so vec8
// wins from S=8192 up (chunks 4-7 measured -3.3..-20.2%).  Below that the
// vec8 body under-parallelizes the short rows (S=4096 +20.6%, S=6144 +4.6%:
// only length/8 of 1024 threads stay active and each concentrates 8
// histogram atomics), so 2048 < width < 8192 keeps the scalar dispatch.
constexpr int64_t kVec8MinScoreWidth = 8192;
// Widths at or above this keep the iteration-tuned 8-KiB staging footprint on
// constrained parts; the mid-band vec8 rows keep the 32-KiB buffer their
// scalar dispatch used, isolating the load-width change.
constexpr int64_t kVec8ShortStagingMinScoreWidth = 16126;
constexpr int32_t kVec8Stride = 2;  // Two float4 loads per thread and iteration.

struct KpoolTopKParams {
  const float* __restrict__ score;
  const int32_t* __restrict__ lengths;
  const int32_t* __restrict__ row_starts;
  int32_t* __restrict__ output;
  const int32_t* __restrict__ page_table;
  const int32_t* __restrict__ page_table_row_index;
  const int32_t* __restrict__ topk_indices_offset;
  const int32_t* __restrict__ seq_lens;
  int64_t score_stride;
  int64_t output_stride;
  int64_t page_table_stride;
  int32_t pool_size;
  int32_t token_topk;
  int32_t out_cols;
};

// float → fp16(压缩) → uint16(读比特) → 保序变换(修正符号) → 取高字节(分桶key)，四步把一个 float 变成 radix sort 能直接用的 0~255 桶号，且桶号的大小顺序和原始 float 一致。
__device__ __forceinline__ uint8_t ordered_fp16_high_byte(float value) {  //强制内联展开，调用时强制展开
  const __half half_value = __float2half_rn(value); //round to nearest, 1+5+10
  const uint16_t bits = __half_as_ushort(half_value);
  const uint16_t key =
      (bits & 0x8000u) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u); // uint16_t 0-65535
  return static_cast<uint8_t>(key >> 8); // uint8_t 0-255
}

// vectorize half2
__device__ __forceinline__ void ordered_fp16_high_bytes(
    float first, float second, uint8_t& first_key, uint8_t& second_key) {
  const __half2 half_values = __floats2half2_rn(first, second);
  const uint16_t first_bits = __half_as_ushort(__low2half(half_values));
  const uint16_t second_bits = __half_as_ushort(__high2half(half_values));
  const uint16_t first_ordered =
      (first_bits & 0x8000u) ? static_cast<uint16_t>(~first_bits)
                             : static_cast<uint16_t>(first_bits | 0x8000u);
  const uint16_t second_ordered =
      (second_bits & 0x8000u) ? static_cast<uint16_t>(~second_bits)
                              : static_cast<uint16_t>(second_bits | 0x8000u);
  first_key = static_cast<uint8_t>(first_ordered >> 8);
  second_key = static_cast<uint8_t>(second_ordered >> 8);
}

__device__ __forceinline__ uint32_t ordered_fp32_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

struct Vec8Layout {
  const float4* values;
  int32_t prefix;
  int32_t vec4_length;
  int32_t tail;
  int32_t vec8_length;
};

__device__ __forceinline__ Vec8Layout make_vec8_layout(
    const float* row_input, int32_t length) {
  const uintptr_t row_address = reinterpret_cast<uintptr_t>(row_input);
  const int32_t prefix_unclamped = static_cast<int32_t>(
      (alignof(float4) - (row_address & (alignof(float4) - 1))) / sizeof(float));
  const int32_t prefix =
      (row_address & (alignof(float4) - 1)) == 0
      ? 0
      : (length < prefix_unclamped ? length : prefix_unclamped);
  const int32_t vec4_length = (length - prefix) / 4;
  return {
      reinterpret_cast<const float4*>(row_input + prefix),
      prefix,
      vec4_length,
      prefix + vec4_length * 4,
      vec4_length / kVec8Stride,
  };
}

__device__ __forceinline__ int32_t transform_token(
    int32_t raw_token,
    const int32_t* __restrict__ page_table_row,
    const int32_t* __restrict__ topk_indices_offset,
    int32_t offset) {
  if (page_table_row != nullptr) {
    return page_table_row[raw_token];
  }
  if (topk_indices_offset != nullptr) {
    return raw_token + offset;
  }
  return raw_token;
}

template <int K, int Threads, bool UseVec8, int StagingSize, bool PackStaged>
__device__ void radix_topk_exact_with_staged_fast_path(
    const float* __restrict__ input,
    int32_t* __restrict__ selected_indices,
    int32_t row_start,
    int32_t length) {
  static_assert(Threads >= kRadix + 1, "radix histogram requires at least 257 threads");
  static_assert(!UseVec8 || Threads == kDefaultThreads, "vec8 is tuned for 1024 threads");
  static_assert(StagingSize >= K, "staging buffer must hold at least K candidates");

  // Packed staged entry layout (PackStaged only): row-relative index in
  // bits 0-15, ordered-FP32-key byte 0 in bits 16-23, byte 1 in bits
  // 24-31.  Refinement rounds 0/1 then read their threshold bytes from
  // shared memory instead of reloading the score from global memory.
  constexpr uint32_t kPackIndexMask = 0xFFFFu;

  // Two staged buffers preserve the fast JIT path for normal score
  // distributions. If a threshold bucket exceeds this kernel instance's
  // StagingSize, an exact full-row rescan fallback is used instead of silently
  // truncating candidates as the attachment does.
  alignas(128) __shared__ uint32_t histogram_buffer[2][kRadix + 128];
  alignas(128) __shared__ int32_t output_counter;
  alignas(128) __shared__ int32_t threshold_bin_id;
  alignas(128) __shared__ int32_t num_input[2];
  alignas(128) __shared__ int32_t last_remain;

  // State used only by the exact overflow fallback.
  alignas(128) __shared__ uint32_t prefix_mask;
  alignas(128) __shared__ uint32_t prefix_value;
  alignas(128) __shared__ int32_t candidate_count;
  alignas(128) __shared__ int32_t next_remain;
  alignas(128) __shared__ int32_t tie_remaining;

  extern __shared__ int32_t staged_indices[];

  const int32_t tid = threadIdx.x;
  const float* row_input = input + row_start;
  auto& histogram = histogram_buffer[0];
  // PackStaged only: histogram_buffer[1] collects the byte-1 histogram
  // during staging.  When the whole staged pool shares one byte-0 bin, it is
  // exactly the histogram round 0's restage pass would have built, which
  // lets the refine loop skip that pass (see below).
  auto& histogram_byte1 = histogram_buffer[1];
  int32_t remain = K;

  // Single-warp suffix sum: 256 bins / 32 lanes = 8 registers per lane, an
  // in-lane sequential pass, and a five-step shuffle scan replace the
  // eight-round all-thread Hillis-Steele sweep and its per-round barriers.
  // The result layout matches the original: hist[i] holds the count of
  // bins >= i.  The trailing barrier publishes the scan to all threads,
  // exactly like the last round of the sweep did.
  const auto run_suffix_sum_on = [&](uint32_t* hist) {
    if (tid < 32) {
      uint32_t values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = hist[tid * 8 + j];
      }
#pragma unroll
      for (int j = 6; j >= 0; --j) {
        values[j] += values[j + 1];
      }
      const uint32_t lane_total = values[0];
      uint32_t lane_suffix = lane_total;
#pragma unroll
      for (int distance = 1; distance < 32; distance <<= 1) {
        const uint32_t neighbor =
            __shfl_down_sync(0xffffffffu, lane_suffix, distance);
        if (tid + distance < 32) {
          lane_suffix += neighbor;
        }
      }
      const uint32_t higher_lanes = lane_suffix - lane_total;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        hist[tid * 8 + j] = values[j] + higher_lanes;
      }
    }
    __syncthreads();
  };
  const auto run_suffix_sum = [&] { run_suffix_sum_on(&histogram_buffer[0][0]); };

  // Stage 1: coarse 8-bit histogram over the valid score region.
  if (tid < kRadix + 1) {
    histogram[tid] = 0;
  }
  __syncthreads();

  if constexpr (UseVec8) {
    const Vec8Layout layout = make_vec8_layout(row_input, length);
    for (int32_t index = tid; index < layout.prefix; index += Threads) {
      ::atomicAdd(&histogram[ordered_fp16_high_byte(row_input[index])], 1u);
    }
    for (int32_t vec_index = tid; vec_index < layout.vec8_length;
         vec_index += Threads) {
      const int32_t base = vec_index * kVec8Stride;
      const float4 first = layout.values[base];
      const float4 second = layout.values[base + 1];
      uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
      ordered_fp16_high_bytes(first.x, first.y, key0, key1);
      ordered_fp16_high_bytes(first.z, first.w, key2, key3);
      ordered_fp16_high_bytes(second.x, second.y, key4, key5);
      ordered_fp16_high_bytes(second.z, second.w, key6, key7);
      ::atomicAdd(&histogram[key0], 1u);
      ::atomicAdd(&histogram[key1], 1u);
      ::atomicAdd(&histogram[key2], 1u);
      ::atomicAdd(&histogram[key3], 1u);
      ::atomicAdd(&histogram[key4], 1u);
      ::atomicAdd(&histogram[key5], 1u);
      ::atomicAdd(&histogram[key6], 1u);
      ::atomicAdd(&histogram[key7], 1u);
    }
    for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
         vec_index < layout.vec4_length;
         vec_index += Threads) {
      const float4 values = layout.values[vec_index];
      uint8_t key0, key1, key2, key3;
      ordered_fp16_high_bytes(values.x, values.y, key0, key1);
      ordered_fp16_high_bytes(values.z, values.w, key2, key3);
      ::atomicAdd(&histogram[key0], 1u);
      ::atomicAdd(&histogram[key1], 1u);
      ::atomicAdd(&histogram[key2], 1u);
      ::atomicAdd(&histogram[key3], 1u);
    }
    for (int32_t index = layout.tail + tid; index < length; index += Threads) {
      ::atomicAdd(&histogram[ordered_fp16_high_byte(row_input[index])], 1u);
    }
  } else {
    for (int32_t index = tid; index < length; index += Threads) {
      ::atomicAdd(&histogram[ordered_fp16_high_byte(row_input[index])], 1u);
    }
  }
  __syncthreads();

  run_suffix_sum();
  if (tid < kRadix && histogram[tid] > static_cast<uint32_t>(remain) &&
      histogram[tid + 1] <= static_cast<uint32_t>(remain)) {
    threshold_bin_id = tid;
    num_input[0] = 0;
    output_counter = 0;
  }
  __syncthreads();

  const int32_t coarse_threshold = threshold_bin_id;
  remain -= static_cast<int32_t>(histogram[coarse_threshold + 1]);

  if (remain == 0) {
    const auto append_if_above_threshold = [&](int32_t index, uint8_t coarse_bin) {
      if (coarse_bin > coarse_threshold) {
        const int32_t position = ::atomicAdd(&output_counter, 1);
        selected_indices[position] = index;
      }
    };

    if constexpr (UseVec8) {
      const Vec8Layout layout = make_vec8_layout(row_input, length);
      for (int32_t index = tid; index < layout.prefix; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
      for (int32_t vec_index = tid; vec_index < layout.vec8_length;
           vec_index += Threads) {
        const int32_t base = vec_index * kVec8Stride;
        const float4 first = layout.values[base];
        const float4 second = layout.values[base + 1];
        uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
        ordered_fp16_high_bytes(first.x, first.y, key0, key1);
        ordered_fp16_high_bytes(first.z, first.w, key2, key3);
        ordered_fp16_high_bytes(second.x, second.y, key4, key5);
        ordered_fp16_high_bytes(second.z, second.w, key6, key7);
        const int32_t index = layout.prefix + base * 4;
        append_if_above_threshold(index, key0);
        append_if_above_threshold(index + 1, key1);
        append_if_above_threshold(index + 2, key2);
        append_if_above_threshold(index + 3, key3);
        append_if_above_threshold(index + 4, key4);
        append_if_above_threshold(index + 5, key5);
        append_if_above_threshold(index + 6, key6);
        append_if_above_threshold(index + 7, key7);
      }
      for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
           vec_index < layout.vec4_length;
           vec_index += Threads) {
        const float4 values = layout.values[vec_index];
        uint8_t key0, key1, key2, key3;
        ordered_fp16_high_bytes(values.x, values.y, key0, key1);
        ordered_fp16_high_bytes(values.z, values.w, key2, key3);
        const int32_t index = layout.prefix + vec_index * 4;
        append_if_above_threshold(index, key0);
        append_if_above_threshold(index + 1, key1);
        append_if_above_threshold(index + 2, key2);
        append_if_above_threshold(index + 3, key3);
      }
      for (int32_t index = layout.tail + tid; index < length; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
    } else {
      for (int32_t index = tid; index < length; index += Threads) {
        append_if_above_threshold(index, ordered_fp16_high_byte(row_input[index]));
      }
    }
    __syncthreads();
    return;
  }

  // Match the JIT kernel's barrier before reusing histogram storage.  Every
  // thread must consume histogram[coarse_threshold + 1] above before any
  // thread clears that element for the first FP32 refinement pass.
  __syncthreads();

  // Collect the threshold bucket and build the first FP32-byte histogram.
  if (tid < kRadix + 1) {
    histogram[tid] = 0;
    if constexpr (PackStaged) {
      histogram_byte1[tid] = 0;
    }
  }
  __syncthreads();

  const auto append_or_stage_candidate = [&](int32_t index, float value, uint8_t coarse_bin) {
    if (coarse_bin > coarse_threshold) {
      const int32_t position = ::atomicAdd(&output_counter, 1);
      selected_indices[position] = index;
    } else if (coarse_bin == coarse_threshold) {
      const int32_t position = ::atomicAdd(&num_input[0], 1);
      if (C10_LIKELY(position < StagingSize)) {
        const uint32_t key = ordered_fp32_key(value);
        staged_indices[position] = PackStaged
            ? static_cast<int32_t>((static_cast<uint32_t>(index) & kPackIndexMask)
                                   | (((key >> 24) & 0xFFu) << 16)
                                   | (((key >> 16) & 0xFFu) << 24))
            : index;
        const int32_t sub_bin = static_cast<int32_t>((key >> 24) & 0xFFu);
        ::atomicAdd(&histogram[sub_bin], 1u);
        if constexpr (PackStaged) {
          ::atomicAdd(&histogram_byte1[(key >> 16) & 0xFFu], 1u);
        }
      }
    }
  };

  if constexpr (UseVec8) {
    const Vec8Layout layout = make_vec8_layout(row_input, length);
    for (int32_t index = tid; index < layout.prefix; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
    for (int32_t vec_index = tid; vec_index < layout.vec8_length;
         vec_index += Threads) {
      const int32_t base = vec_index * kVec8Stride;
      const float4 first = layout.values[base];
      const float4 second = layout.values[base + 1];
      uint8_t key0, key1, key2, key3, key4, key5, key6, key7;
      ordered_fp16_high_bytes(first.x, first.y, key0, key1);
      ordered_fp16_high_bytes(first.z, first.w, key2, key3);
      ordered_fp16_high_bytes(second.x, second.y, key4, key5);
      ordered_fp16_high_bytes(second.z, second.w, key6, key7);
      const int32_t index = layout.prefix + base * 4;
      append_or_stage_candidate(index, first.x, key0);
      append_or_stage_candidate(index + 1, first.y, key1);
      append_or_stage_candidate(index + 2, first.z, key2);
      append_or_stage_candidate(index + 3, first.w, key3);
      append_or_stage_candidate(index + 4, second.x, key4);
      append_or_stage_candidate(index + 5, second.y, key5);
      append_or_stage_candidate(index + 6, second.z, key6);
      append_or_stage_candidate(index + 7, second.w, key7);
    }
    for (int32_t vec_index = layout.vec8_length * kVec8Stride + tid;
         vec_index < layout.vec4_length;
         vec_index += Threads) {
      const float4 values = layout.values[vec_index];
      uint8_t key0, key1, key2, key3;
      ordered_fp16_high_bytes(values.x, values.y, key0, key1);
      ordered_fp16_high_bytes(values.z, values.w, key2, key3);
      const int32_t index = layout.prefix + vec_index * 4;
      append_or_stage_candidate(index, values.x, key0);
      append_or_stage_candidate(index + 1, values.y, key1);
      append_or_stage_candidate(index + 2, values.z, key2);
      append_or_stage_candidate(index + 3, values.w, key3);
    }
    for (int32_t index = layout.tail + tid; index < length; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
  } else {
    for (int32_t index = tid; index < length; index += Threads) {
      const float value = row_input[index];
      append_or_stage_candidate(index, value, ordered_fp16_high_byte(value));
    }
  }
  __syncthreads();

  // Exact fallback for adversarial/dense score distributions.  Every refine
  // round rescans the original row while narrowing an FP32 prefix.  This path
  // is slower but cannot drop candidates when the staged buffers overflow.
  if (num_input[0] > StagingSize) {
    if (tid == 0) {
      prefix_mask = 0;
      prefix_value = 0;
    }
    __syncthreads();

#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
      if (tid < kRadix + 1) {
        histogram[tid] = 0;
      }
      if (tid == 0) {
        candidate_count = 0;
      }
      __syncthreads();

      const int shift = 24 - round * 8;
      for (int32_t index = tid; index < length; index += Threads) {
        const float value = input[row_start + index];
        if (ordered_fp16_high_byte(value) != coarse_threshold) {
          continue;
        }
        const uint32_t key = ordered_fp32_key(value);
        if (round > 0 && (key & prefix_mask) != prefix_value) {
          continue;
        }
        ::atomicAdd(&candidate_count, 1);
        ::atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
      }
      __syncthreads();

      // The whole current prefix is required; no further radix refinement is
      // necessary and every matching index can be emitted directly.
      if (candidate_count == remain) {
        for (int32_t index = tid; index < length; index += Threads) {
          const float value = input[row_start + index];
          if (ordered_fp16_high_byte(value) != coarse_threshold) {
            continue;
          }
          const uint32_t key = ordered_fp32_key(value);
          if (round == 0 || (key & prefix_mask) == prefix_value) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = index;
          }
        }
        __syncthreads();
        return;
      }

      run_suffix_sum();
      if (tid < kRadix && histogram[tid] > static_cast<uint32_t>(remain) &&
          histogram[tid + 1] <= static_cast<uint32_t>(remain)) {
        threshold_bin_id = tid;
        next_remain = remain - static_cast<int32_t>(histogram[tid + 1]);
        tie_remaining = next_remain;
      }
      __syncthreads();

      const int32_t refine_threshold = threshold_bin_id;
      for (int32_t index = tid; index < length; index += Threads) {
        const float value = input[row_start + index];
        if (ordered_fp16_high_byte(value) != coarse_threshold) {
          continue;
        }
        const uint32_t key = ordered_fp32_key(value);
        if (round > 0 && (key & prefix_mask) != prefix_value) {
          continue;
        }
        const int32_t bin = static_cast<int32_t>((key >> shift) & 0xFFu);
        if (bin > refine_threshold) {
          const int32_t position = ::atomicAdd(&output_counter, 1);
          selected_indices[position] = index;
        } else if (round == 3 && bin == refine_threshold) {
          const int32_t slot = ::atomicAdd(&tie_remaining, -1);
          if (slot > 0) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = index;
          }
        }
      }
      __syncthreads();

      remain = next_remain;
      if (remain == 0 || round == 3) {
        return;
      }

      if (tid == 0) {
        prefix_mask |= 0xFFu << shift;
        prefix_value |= static_cast<uint32_t>(refine_threshold) << shift;
      }
      __syncthreads();
    }
    return;
  }

  // Normal staged path: refine only the threshold candidates in shared
  // memory, matching the low-overhead behavior of the supplied JIT kernel.
  // Round 0 is usually redundant: when the whole staged pool shares one
  // byte-0 bin it emits nothing, drops nothing, and restages the identical
  // pool, and the staging-time byte-1 histogram already holds what its
  // element pass would have built.
  bool skipped_round0 = false;
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    const int buffer_index = (round & 1) ^ (skipped_round0 ? 1 : 0);
    const int32_t current_count = num_input[buffer_index];

    if (current_count == remain) {
      const int32_t output_base = output_counter;
      for (int32_t index = tid; index < current_count; index += Threads) {
        const uint32_t entry =
            static_cast<uint32_t>(staged_indices[buffer_index * StagingSize + index]);
        selected_indices[output_base + index] =
            PackStaged ? static_cast<int32_t>(entry & kPackIndexMask)
                       : static_cast<int32_t>(entry);
      }
      __syncthreads();
      return;
    }

    uint32_t* const scan_src =
        (PackStaged && round == 1 && skipped_round0) ? &histogram_byte1[0]
                                                     : &histogram[0];
    run_suffix_sum_on(scan_src);
    if (tid < kRadix && scan_src[tid] > static_cast<uint32_t>(remain) &&
        scan_src[tid + 1] <= static_cast<uint32_t>(remain)) {
      threshold_bin_id = tid;
      num_input[buffer_index ^ 1] = 0;
      last_remain = remain - static_cast<int32_t>(scan_src[tid + 1]);
    }
    __syncthreads();

    const int32_t refine_threshold = threshold_bin_id;
    const int32_t emitted = static_cast<int32_t>(scan_src[refine_threshold + 1]);
    remain -= emitted;
    const int shift = 24 - round * 8;

    if (remain == 0) {
      for (int32_t index = tid; index < current_count; index += Threads) {
        const uint32_t entry =
            static_cast<uint32_t>(staged_indices[buffer_index * StagingSize + index]);
        const int32_t score_index =
            PackStaged ? static_cast<int32_t>(entry & kPackIndexMask)
                       : static_cast<int32_t>(entry);
        const int32_t bin =
            (PackStaged && round <= 1)
                ? static_cast<int32_t>((entry >> (16 + 8 * round)) & 0xFFu)
                : static_cast<int32_t>(
                      (ordered_fp32_key(input[row_start + score_index]) >> shift) & 0xFFu);
        if (bin > refine_threshold) {
          const int32_t position = ::atomicAdd(&output_counter, 1);
          selected_indices[position] = score_index;
        }
      }
      __syncthreads();
      return;
    }

    if constexpr (PackStaged) {
      if (round == 0 && emitted == 0 &&
          scan_src[refine_threshold] == static_cast<uint32_t>(current_count)) {
        // The entire pool sits in the threshold bin and nothing is above it,
        // so the restage pass is a no-op on the pool and the staging-time
        // byte-1 histogram already matches it exactly.  Keep the pool in the
        // current buffer and let round 1 scan histogram_byte1 instead.
        skipped_round0 = true;
        continue;
      }
    }

    // Do not clear the suffix sums until all threads have consumed
    // histogram[refine_threshold + 1] and made the same remain decision.
    // Without this JIT barrier, fast threads can race ahead and make slower
    // threads observe zero, creating divergent refinement state.
    __syncthreads();

    if (tid < kRadix + 1) {
      histogram[tid] = 0;
    }
    __syncthreads();
    for (int32_t index = tid; index < current_count; index += Threads) {
      const uint32_t entry =
          static_cast<uint32_t>(staged_indices[buffer_index * StagingSize + index]);
      const int32_t score_index =
          PackStaged ? static_cast<int32_t>(entry & kPackIndexMask)
                     : static_cast<int32_t>(entry);
      const int32_t bin =
          (PackStaged && round <= 1)
              ? static_cast<int32_t>((entry >> (16 + 8 * round)) & 0xFFu)
              : static_cast<int32_t>(
                    (ordered_fp32_key(input[row_start + score_index]) >> shift) & 0xFFu);
      if (bin > refine_threshold) {
        const int32_t position = ::atomicAdd(&output_counter, 1);
        selected_indices[position] = score_index;
      } else if (bin == refine_threshold) {
        if (round == 3) {
          const int32_t slot = ::atomicAdd(&last_remain, -1);
          if (slot > 0) {
            const int32_t position = ::atomicAdd(&output_counter, 1);
            selected_indices[position] = score_index;
          }
        } else {
          const int32_t position = ::atomicAdd(&num_input[buffer_index ^ 1], 1);
          if (C10_LIKELY(position < StagingSize)) {
            staged_indices[(buffer_index ^ 1) * StagingSize + position] =
                static_cast<int32_t>(entry);
            const int32_t sub_bin =
                (PackStaged && round == 0)
                    ? static_cast<int32_t>((entry >> 24) & 0xFFu)
                    : static_cast<int32_t>(
                          (ordered_fp32_key(input[row_start + score_index]) >> (shift - 8)) & 0xFFu);
            ::atomicAdd(&histogram[sub_bin], 1u);
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int K, int Threads, bool UseVec8, int StagingSize, bool PackStaged>
__global__ __launch_bounds__(Threads) void kpool_topk_transform_kernel(
    KpoolTopKParams params) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int32_t tid = threadIdx.x;
  const int32_t length = params.lengths[row];
  const int32_t row_start = params.row_starts == nullptr ? 0 : params.row_starts[row];
  const float* score_row = params.score + row * params.score_stride;
  int32_t* output_row = params.output + row * params.output_stride;

  const int64_t page_table_row =
      params.page_table_row_index == nullptr ? row : static_cast<int64_t>(params.page_table_row_index[row]);
  const int32_t* page_table_entry =
      params.page_table == nullptr ? nullptr : params.page_table + page_table_row * params.page_table_stride;
  const int32_t offset = params.topk_indices_offset == nullptr ? 0 : params.topk_indices_offset[row];
  const bool append_tail = params.seq_lens != nullptr;
  const int32_t full_pool_token_len = length * params.pool_size;
  const int32_t history_len =
      full_pool_token_len < params.token_topk ? full_pool_token_len : params.token_topk;
  const int32_t tail_count = append_tail ? params.seq_lens[row] % params.pool_size : 0;

  // If every valid group is required, selection is unnecessary.
  if (length <= K) {
    // With no Top-K selection the history is the identity sequence.  Its tail
    // continues that same sequence, so both regions collapse into one bounds
    // check and no per-element division/modulo is required.
    const int32_t valid_columns = history_len + tail_count;
    if (page_table_entry == nullptr) {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        output_row[column] = column < valid_columns ? column + offset : -1;
      }
    } else {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        output_row[column] =
            column < valid_columns ? page_table_entry[column] : -1;
      }
    }
    return;
  }

  // Selecting K groups from K + 1 candidates only requires excluding the
  // single minimum-score group. Avoid the full multi-pass radix selection.
  if (length == K + 1) {
    alignas(8) __shared__ unsigned long long minimum_key_index;

    if (tid == 0) {
      minimum_key_index = ~0ULL;
    }
    __syncthreads();

    for (int32_t index = tid; index < length; index += Threads) {
      const uint32_t key = ordered_fp32_key(score_row[row_start + index]);
      const unsigned long long candidate =
          (static_cast<unsigned long long>(key) << 32) | static_cast<uint32_t>(index);
      ::atomicMin(&minimum_key_index, candidate);
    }
    __syncthreads();

    const int32_t excluded_group =
        static_cast<int32_t>(minimum_key_index & 0xFFFFFFFFULL);
    for (int32_t column = tid; column < params.out_cols; column += Threads) {
      if (column < history_len) {
        const int32_t group_rank = column / params.pool_size;
        const int32_t slot = column % params.pool_size;
        const int32_t group = group_rank + (group_rank >= excluded_group);
        const int32_t raw_token = group * params.pool_size + slot;
        output_row[column] = transform_token(
            raw_token, page_table_entry, params.topk_indices_offset, offset);
      } else if (append_tail && column < history_len + tail_count) {
        const int32_t raw_token = full_pool_token_len + column - history_len;
        output_row[column] = transform_token(
            raw_token, page_table_entry, params.topk_indices_offset, offset);
      } else {
        output_row[column] = -1;
      }
    }
    return;
  }

  __shared__ int32_t selected_groups[K];
  radix_topk_exact_with_staged_fast_path<K, Threads, UseVec8, StagingSize, PackStaged>(
      score_row, selected_groups, row_start, length);

  // GLM-5 fixes pool_size at four.  Specialize the output expansion so the
  // compiler can replace integer division/modulo with shifts and masks, and
  // hoist the uniform page-table/offset choice out of the element loop.
  if (params.pool_size == 4) {
    if (page_table_entry == nullptr) {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        if (column < history_len) {
          const int32_t group_rank = column >> 2;
          const int32_t slot = column & 3;
          const int32_t raw_token = (selected_groups[group_rank] << 2) + slot;
          output_row[column] = raw_token + offset;
        } else if (append_tail && column < history_len + tail_count) {
          output_row[column] =
              full_pool_token_len + column - history_len + offset;
        } else {
          output_row[column] = -1;
        }
      }
    } else {
      for (int32_t column = tid; column < params.out_cols; column += Threads) {
        if (column < history_len) {
          const int32_t group_rank = column >> 2;
          const int32_t slot = column & 3;
          const int32_t raw_token = (selected_groups[group_rank] << 2) + slot;
          output_row[column] = page_table_entry[raw_token];
        } else if (append_tail && column < history_len + tail_count) {
          const int32_t raw_token =
              full_pool_token_len + column - history_len;
          output_row[column] = page_table_entry[raw_token];
        } else {
          output_row[column] = -1;
        }
      }
    }
    return;
  }

  for (int32_t column = tid; column < params.out_cols; column += Threads) {
    if (column < history_len) {
      const int32_t group_rank = column / params.pool_size;
      const int32_t slot = column % params.pool_size;
      const int32_t raw_token = selected_groups[group_rank] * params.pool_size + slot;
      output_row[column] = transform_token(
          raw_token, page_table_entry, params.topk_indices_offset, offset);
    } else if (append_tail && column < history_len + tail_count) {
      const int32_t raw_token = full_pool_token_len + column - history_len;
      output_row[column] = transform_token(
          raw_token, page_table_entry, params.topk_indices_offset, offset);
    } else {
      output_row[column] = -1;
    }
  }
}

template <auto* Kernel, std::size_t MaxDynamicSmem>
void set_kernel_smem_once() {
  [[maybe_unused]] static const cudaError_t result = [] {
    return ::cudaFuncSetAttribute(
        Kernel, ::cudaFuncAttributeMaxDynamicSharedMemorySize, MaxDynamicSmem);
  }();
  TORCH_CHECK(
      result == cudaSuccess,
      "kpool_topk_transform cudaFuncSetAttribute failed: ",
      ::cudaGetErrorString(result));
}

template <
    int K,
    int Threads,
    bool UseVec8,
    int StagingSize,
    bool PackStaged,
    std::size_t DynamicSmem>
void launch_kpool_topk_transform_impl(
    const KpoolTopKParams& params, int64_t batch_size, cudaStream_t stream) {
  set_kernel_smem_once<
      kpool_topk_transform_kernel<K, Threads, UseVec8, StagingSize, PackStaged>,
      DynamicSmem>();
  const dim3 grid{static_cast<uint32_t>(batch_size)};
  const dim3 block{Threads};
  kpool_topk_transform_kernel<K, Threads, UseVec8, StagingSize, PackStaged>
      <<<grid, block, DynamicSmem, stream>>>(params);
}

template <int K, bool PackStaged>
void launch_kpool_topk_transform(
    const KpoolTopKParams& params,
    int64_t batch_size,
    bool use_short_staging,
    bool use_medium_threads,
    bool use_vec8,
    bool constrained_shared_memory,
    int64_t score_width,
    cudaStream_t stream) {
  if (use_vec8) {
    if (constrained_shared_memory && batch_size >= 32 &&
        score_width >= kVec8ShortStagingMinScoreWidth) {
      launch_kpool_topk_transform_impl<
          K, kDefaultThreads, true, kShortStagingSize, PackStaged, kShortDynamicSmem>(
          params, batch_size, stream);
    } else {
      launch_kpool_topk_transform_impl<
          K, kDefaultThreads, true, kLongStagingSize, PackStaged, kLongDynamicSmem>(
          params, batch_size, stream);
    }
  } else if (use_short_staging) {
    if (constrained_shared_memory && batch_size < 32) {
      launch_kpool_topk_transform_impl<
          K, kDefaultThreads, false, kShortStagingSize, PackStaged, kShortDynamicSmem>(
          params, batch_size, stream);
    } else {
      launch_kpool_topk_transform_impl<
          K, kMediumThreads, false, kShortStagingSize, PackStaged, kShortDynamicSmem>(
          params, batch_size, stream);
    }
  } else if (use_medium_threads) {
    launch_kpool_topk_transform_impl<
        K, kMediumThreads, false, kLongStagingSize, PackStaged, kLongDynamicSmem>(
        params, batch_size, stream);
  } else {
    launch_kpool_topk_transform_impl<
        K, kDefaultThreads, false, kLongStagingSize, PackStaged, kLongDynamicSmem>(
        params, batch_size, stream);
  }
}

void check_cuda_tensor(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as score");
}

void check_int32_vector(
    const at::Tensor& tensor, const char* name, int64_t batch_size, const at::Device& device) {
  check_cuda_tensor(tensor, name, device);
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must have dtype int32");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous(), name, " must be a contiguous 1D tensor");
  TORCH_CHECK(tensor.size(0) == batch_size, name, " must have batch_size elements");
}

const int32_t* optional_int32_data(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? tensor->data_ptr<int32_t>() : nullptr;
}

}  // namespace

void kpool_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& output,
    int64_t pool_size,
    std::optional<at::Tensor> page_table_opt,
    std::optional<at::Tensor> topk_indices_offset_opt,
    std::optional<at::Tensor> row_starts_opt,
    std::optional<at::Tensor> seq_lens_opt,
    std::optional<at::Tensor> page_table_row_index_opt) {
  TORCH_CHECK(score.is_cuda(), "score must be a CUDA tensor");
  const at::Device device = score.device();
  TORCH_CHECK(score.scalar_type() == at::ScalarType::Float, "score must have dtype float32");
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1, "score must be 2D with stride(1) == 1");

  const int64_t batch_size = score.size(0);
  check_int32_vector(lengths, "lengths", batch_size, device);

  check_cuda_tensor(output, "output", device);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Int, "output must have dtype int32");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(), "output must be a contiguous 2D tensor");
  TORCH_CHECK(output.size(0) == batch_size, "output batch dimension must match score");

  TORCH_CHECK(pool_size > 1, "pool_size must be greater than 1");
  TORCH_CHECK(pool_size <= std::numeric_limits<int32_t>::max(), "pool_size exceeds int32 range");
  TORCH_CHECK(
      !(page_table_opt.has_value() && topk_indices_offset_opt.has_value()),
      "page_table and topk_indices_offset are mutually exclusive");
  TORCH_CHECK(
      !page_table_row_index_opt.has_value() || page_table_opt.has_value(),
      "page_table_row_index requires page_table");

  const int64_t tail_cols = seq_lens_opt.has_value() ? pool_size - 1 : 0;
  TORCH_CHECK(output.size(1) > tail_cols, "output has too few columns");
  const int64_t token_topk = output.size(1) - tail_cols;
  TORCH_CHECK(token_topk % pool_size == 0, "token_topk must be divisible by pool_size");
  const int64_t group_topk = token_topk / pool_size;
  TORCH_CHECK(
      group_topk == 128 || group_topk == 160 || group_topk == 192 ||
          group_topk == 224 || group_topk == 256 || group_topk == 512,
      "unsupported group_topk: ",
      group_topk);
  TORCH_CHECK(
      output.size(1) <= std::numeric_limits<int32_t>::max(),
      "output column count exceeds int32 range");

  if (topk_indices_offset_opt.has_value()) {
    check_int32_vector(*topk_indices_offset_opt, "topk_indices_offset", batch_size, device);
  }
  if (row_starts_opt.has_value()) {
    check_int32_vector(*row_starts_opt, "row_starts", batch_size, device);
  }
  if (seq_lens_opt.has_value()) {
    check_int32_vector(*seq_lens_opt, "seq_lens", batch_size, device);
  }
  if (page_table_row_index_opt.has_value()) {
    check_int32_vector(*page_table_row_index_opt, "page_table_row_index", batch_size, device);
  }

  const int32_t* page_table_ptr = nullptr;
  int64_t page_table_stride = 0;
  if (page_table_opt.has_value()) {
    const at::Tensor& page_table = *page_table_opt;
    check_cuda_tensor(page_table, "page_table", device);
    TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int, "page_table must have dtype int32");
    TORCH_CHECK(
        page_table.dim() == 2 && page_table.stride(1) == 1,
        "page_table must be 2D with stride(1) == 1");
    if (!page_table_row_index_opt.has_value()) {
      TORCH_CHECK(page_table.size(0) == batch_size, "page_table batch dimension must match score");
    } else {
      TORCH_CHECK(page_table.size(0) > 0, "page_table must contain at least one row");
    }
    page_table_ptr = page_table.data_ptr<int32_t>();
    page_table_stride = page_table.stride(0);
  }

  if (batch_size == 0) {
    return;
  }

  const KpoolTopKParams params{
      .score = score.data_ptr<float>(),
      .lengths = lengths.data_ptr<int32_t>(),
      .row_starts = optional_int32_data(row_starts_opt),
      .output = output.data_ptr<int32_t>(),
      .page_table = page_table_ptr,
      .page_table_row_index = optional_int32_data(page_table_row_index_opt),
      .topk_indices_offset = optional_int32_data(topk_indices_offset_opt),
      .seq_lens = optional_int32_data(seq_lens_opt),
      .score_stride = score.stride(0),
      .output_stride = output.stride(0),
      .page_table_stride = page_table_stride,
      .pool_size = static_cast<int32_t>(pool_size),
      .token_topk = static_cast<int32_t>(token_topk),
      .out_cols = static_cast<int32_t>(output.size(1)),
  };

  // Short rows use an exactly-sized 1024-candidate double buffer (8 KiB),
  // while longer rows retain the 4096-candidate buffer and exact overflow
  // fallback. Keep the implementations separate so the short kernel does not
  // reserve long-row shared memory.
  const int64_t score_width = score.size(1);
  const bool use_short_staging = score_width <= kShortStagingSize;
  const bool use_medium_threads =
      score.size(1) > group_topk + 1 && score.size(1) <= kMediumMaxScoreWidth;
  const bool use_vec8 = score.size(1) >= kVec8MinScoreWidth;
  // Select the launch policy from the resource that constrains this kernel,
  // rather than a product name.  GPUs exposing at most 64 KiB shared memory
  // per multiprocessor benefit from the smaller staging footprint and the
  // wide short-row block; higher-capacity devices retain the established
  // C600U policy.  This also lets future SKUs inherit the appropriate path.
  static const bool constrained_shared_memory = [] {
    int device_id = 0;
    cudaDeviceProp properties{};
    const cudaError_t get_device_status = cudaGetDevice(&device_id);
    const cudaError_t get_properties_status =
        get_device_status == cudaSuccess
        ? cudaGetDeviceProperties(&properties, device_id)
        : get_device_status;
    TORCH_CHECK(
        get_properties_status == cudaSuccess,
        "kpool_topk_transform failed to query GPU properties: ",
        cudaGetErrorString(get_properties_status));
    return properties.sharedMemPerMultiprocessor <= 64 * 1024;
  }();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const auto launch_kpool = [&](auto group_topk_constant) {
    constexpr int K = decltype(group_topk_constant)::value;
    if (score_width <= kPackMaxScoreWidth) {
      launch_kpool_topk_transform<K, true>(
          params, batch_size, use_short_staging, use_medium_threads,
          use_vec8, constrained_shared_memory, score_width, stream);
    } else {
      launch_kpool_topk_transform<K, false>(
          params, batch_size, use_short_staging, use_medium_threads,
          use_vec8, constrained_shared_memory, score_width, stream);
    }
  };
  switch (group_topk) {
    case 128:
      launch_kpool(std::integral_constant<int, 128>{});
      break;
    case 160:
      launch_kpool(std::integral_constant<int, 160>{});
      break;
    case 192:
      launch_kpool(std::integral_constant<int, 192>{});
      break;
    case 224:
      launch_kpool(std::integral_constant<int, 224>{});
      break;
    case 256:
      launch_kpool(std::integral_constant<int, 256>{});
      break;
    case 512:
      launch_kpool(std::integral_constant<int, 512>{});
      break;
    default:
      TORCH_CHECK(false, "unsupported group_topk: ", group_topk);
  }

  const cudaError_t result = ::cudaGetLastError();
  TORCH_CHECK(
      result == cudaSuccess,
      "kpool_topk_transform kernel launch failed: ",
      ::cudaGetErrorString(result));
}
