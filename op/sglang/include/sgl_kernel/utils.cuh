/// \file utils.cuh
/// \brief Core CUDA/device utilities: type aliases, PDL helpers,
///        typed pointer access, kernel launch wrapper, and error checking.
///
/// This header is included (directly or transitively) by nearly every
/// JIT kernel. It provides:
/// - Scalar/packed type aliases (`fp16_t`, `bf16_t`, `fp8_e4m3_t`, ...).
/// - `SGL_DEVICE` macro (forced-inline device function qualifier).
/// - `kWarpThreads` constant (32).
/// - PDL (Programmatic Dependent Launch) helpers for Hopper (sm_90+).
/// - Typed `load_as` / `store_as` for void-pointer access.
/// - `pointer::offset` for safe void-pointer arithmetic.
/// - `host::LaunchKernel` - kernel launcher with optional PDL.
/// - `host::RuntimeDeviceCheck` - CUDA error checking.

#pragma once

// #include <sgl_kernel/utils.h>

// #include <dlpack/dlpack.h>
// #include <tvm/ffi/extra/c_env_api.h>

#include <cstddef>
#include <type_traits>
#ifndef USE_ROCM
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#else
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef  
#define  
#endif
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
inline constexpr auto cudaSuccess = hipSuccess;
#define cudaStreamPerThread hipStreamPerThread
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaLaunchKernel hipLaunchKernel
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif

#ifndef USE_ROCM
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8_e5m2_t = __nv_fp8_e5m2;

using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x2_e5m2_t = __nv_fp8x2_e5m2;

using fp32x4_t = float4;
#else
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __hip_bfloat16;
using fp8_e4m3_t = uint8_t;
using fp8_e5m2_t = uint8_t;
using fp32x2_t = float2;
using fp16x2_t = half2;
using bf16x2_t = __hip_bfloat162;
using fp8x2_e4m3_t = uint16_t;
using fp8x2_e5m2_t = uint16_t;
using fp32x4_t = float4;
#endif

/*
 * LDG Support
 */
#ifndef USE_ROCM
#define SGLANG_LDG(arg) __ldg(arg)
#else
#define SGLANG_LDG(arg) *(arg)
#endif

// DLPack device type for the current platform
// #ifndef USE_ROCM
// inline constexpr auto kDLGPU = kDLCUDA;
// #else
// inline constexpr auto kDLGPU = kDLROCM;
// #endif

namespace device {

/// \brief Macro: forced-inline device function qualifier.
#define SGL_DEVICE __forceinline__ __device__

// Architecture detection: SGL_CUDA_ARCH is injected by load_jit() and is
// available in both host and device compilation passes, whereas __CUDA_ARCH__
// is only defined by nvcc during the device pass.
// #if !defined(USE_ROCM)
// #if !defined(SGL_CUDA_ARCH)
// #error "SGL_CUDA_ARCH is not defined. JIT compilation must inject -DSGL_CUDA_ARCH via load_jit()."
// #endif
// #if defined(__CUDA_ARCH__)
// static_assert(
//     __CUDA_ARCH__ == SGL_CUDA_ARCH, "SGL_CUDA_ARCH mismatch: injected arch flag does not match device target");
// #endif
// #define SGL_ARCH_HOPPER_OR_GREATER (SGL_CUDA_ARCH >= 900)
// #define SGL_ARCH_BLACKWELL_OR_GREATER ((SGL_CUDA_ARCH >= 1000) && (CUDA_VERSION >= 12090))
// #else  // USE_ROCM
#define SGL_ARCH_HOPPER_OR_GREATER 0
#define SGL_ARCH_BLACKWELL_OR_GREATER 0
// #endif

// Maximum vector size in bytes supported by current architecture.
// Pre-Blackwell / AMD: 128-bit (16 bytes)
// Blackwell or greater: 256-bit (32 bytes)
inline constexpr std::size_t kMaxVecBytes = SGL_ARCH_BLACKWELL_OR_GREATER ? 32 : 16;

/// \brief Number of threads per warp (always 32 on NVIDIA/AMD GPUs).
inline constexpr auto kWarpThreads = 32u;
/// \brief Full warp active mask (all 32 lanes).
#ifndef USE_ROCM
inline constexpr auto kFullMask = 0xffffffffu;
#else
inline constexpr auto kFullMask = 0xffffffffffffffffULL;
#endif

/**
 * \brief PDL (Programmatic Dependent Launch): wait for the primary kernel.
 *
 * On Hopper (sm_90+), inserts a `griddepcontrol.wait` instruction to
 * synchronize with a preceding kernel in the same stream. On older
 * architectures or ROCm this is a no-op.
 */
// template <bool kUsePDL>
// SGL_DEVICE void PDLWaitPrimary() {
// #if SGL_ARCH_HOPPER_OR_GREATER
//   if constexpr (kUsePDL) {
//     asm volatile("griddepcontrol.wait;" ::: "memory");
//   }
// #endif
// }

// /**
//  * \brief PDL: trigger dependent (secondary) kernel launch.
//  *
//  * On Hopper (sm_90+), inserts a `griddepcontrol.launch_dependents`
//  * instruction. On older architectures or ROCm this is a no-op.
//  */
// template <bool kUsePDL>
// SGL_DEVICE void PDLTriggerSecondary() {
// #if SGL_ARCH_HOPPER_OR_GREATER
//   if constexpr (kUsePDL) {
//     asm volatile("griddepcontrol.launch_dependents;" :::);
//   }
// #endif
// }

#ifndef SGL_ARCH_HOPPER_OR_GREATER
#define SGL_ARCH_HOPPER_OR_GREATER 0
#endif

template <bool kUsePDL>
SGL_DEVICE void PDLWaitPrimary() {
#if defined(_CUDA_ARCH) && (CUDA_ARCH_ >= 900) && SGL_ARCH_HOPPER_OR_GREATER
if constexpr (kUsePDL)
{ asm volatile("griddepcontrol.wait;" ::: "memory"); }
#else
(void)kUsePDL;
#endif
}

template <bool kUsePDL>
SGL_DEVICE void PDLTriggerSecondary() {
#if defined(_CUDA_ARCH) && (CUDA_ARCH_ >= 900) && SGL_ARCH_HOPPER_OR_GREATER
if constexpr (kUsePDL)
{ asm volatile("griddepcontrol.launch_dependents;" :::); }
#else
(void)kUsePDL;
#endif
} 

// template <std::integral T, std::integral U>
// SGL_DEVICE constexpr auto div_ceil(T a, U b) {
//   return (a + b - 1) / b;
// }

/**
 * \brief Load data with the specified type and offset from a void pointer.
 * \tparam T The type to load.
 * \param ptr The base pointer.
 * \param offset The offset in number of elements of type T.
 */
template <typename T>
SGL_DEVICE T load_as(const void* ptr, int64_t offset = 0) {
  return static_cast<const T*>(ptr)[offset];
}

/**
 * \brief Store data with the specified type and offset to a void pointer.
 * \tparam T The type to store.
 * \param ptr The base pointer.
 * \param val The value to store.
 * \param offset The offset in number of elements of type T.
 * \note we use type_identity_t to force the caller to explicitly specify
 * the template parameter `T`, which can avoid accidentally using the wrong type.
 */
// template <typename T>
// SGL_DEVICE void store_as(void* ptr, std::type_identity_t<T> val, int64_t offset = 0) {
//   static_cast<T*>(ptr)[offset] = val;
// }

/// \brief Safe void-pointer arithmetic (byte-level by default).
namespace pointer {

// we only allow void * pointer arithmetic for safety

// template <typename T = char, std::integral... U>
// SGL_DEVICE auto offset(void* ptr, U... offset) -> void* {
//   return static_cast<T*>(ptr) + (... + offset);
// }

// template <typename T = char, std::integral... U>
// SGL_DEVICE auto offset(const void* ptr, U... offset) -> const void* {
//   return static_cast<const T*>(ptr) + (... + offset);
// }

}  // namespace pointer

}  // namespace device


