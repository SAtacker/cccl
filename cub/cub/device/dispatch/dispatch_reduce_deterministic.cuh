/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeterministicDeviceReduce provides device-wide, parallel operations for
 *       computing a reduction across a sequence of data items residing within
 *       device-accessible memory. Current reduction operator supported is cub::Sum
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <iterator>

#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/transform_output_iterator.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH
#include <cuda/std/functional>
_CCCL_SUPPRESS_DEPRECATED_POP

#include <stdio.h>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace rfa_detail
{

template <typename ReductionOpT, typename InitT, typename InputIteratorT>
using AccumT = cub::detail::accumulator_t<ReductionOpT, InitT, cub::detail::value_t<InputIteratorT>>;

template <typename OutputIteratorT, typename InputIteratorT>
using InitT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::value_t<InputIteratorT>>;

template <typename FloatType = float, typename std::enable_if<std::is_floating_point<FloatType>::value>::type* = nullptr>
struct deterministic_sum_t
{
  using DeterministicAcc = detail::rfa_detail::ReproducibleFloatingAccumulator<FloatType>;

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc acc, FloatType f)
  {
    acc += f;
    return acc;
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc acc, float4 f)
  {
    acc += f;
    return acc;
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc acc, double4 f)
  {
    acc += f;
    return acc;
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(FloatType f, DeterministicAcc acc)
  {
    return this->operator()(acc, f);
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(float4 f, DeterministicAcc acc)
  {
    return this->operator()(acc, f);
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(double4 f, DeterministicAcc acc)
  {
    return this->operator()(acc, f);
  }

  _CCCL_HOST _CCCL_DEVICE DeterministicAcc operator()(DeterministicAcc lhs, DeterministicAcc rhs)
  {
    DeterministicAcc rtn = lhs;
    rtn += rhs;
    return rtn;
  }
};

} // namespace rfa_detail

/**
 * @brief Deterministically Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block in deterministic fashion
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] even_share
 *   Even-share descriptor for mapping an equal number of tiles onto each
 *   thread block
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::DeterministicReducePolicy::BLOCK_THREADS)) void DeterministicDeviceReduceKernel(
  InputIteratorT d_in,
  AccumT* d_out,
  OffsetT num_items,

  ReductionOpT reduction_op,
  TransformOpT transform_op,
  const int reduce_grid_size)
{
  using BlockReduceT =
    BlockReduce<AccumT,
                ChainedPolicyT::ActivePolicy::DeterministicReducePolicy::BLOCK_THREADS,
                ChainedPolicyT::ActivePolicy::DeterministicReducePolicy::BLOCK_ALGORITHM>;
  // Shared memory storage
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  using FloatType                 = typename AccumT::ftype;
  constexpr int BinLength         = AccumT::MAXINDEX + AccumT::MAXFOLD;
  constexpr auto ITEMS_PER_THREAD = ChainedPolicyT::DeterministicReducePolicy::ITEMS_PER_THREAD;
  constexpr auto BLOCK_THREADS    = ChainedPolicyT::DeterministicReducePolicy::BLOCK_THREADS;
  constexpr auto TILE_SIZE        = BLOCK_THREADS * ITEMS_PER_THREAD;
  const int GRID_DIM              = reduce_grid_size;
  const int tid                   = BLOCK_THREADS * blockIdx.x + threadIdx.x;

  FloatType* shared_bins = detail::rfa_detail::get_shared_bin_array<FloatType, BinLength>();

#pragma unroll
  for (int index = threadIdx.x; index < BinLength; index += ChainedPolicyT::DeterministicReducePolicy::BLOCK_THREADS)
  {
    shared_bins[index] = detail::rfa_detail::RFA_bins<FloatType>::initialize_bins(index);
  }

  CTA_SYNC();

  AccumT thread_aggregate{};
  int count = 0;

#pragma unroll
  for (int i = tid; i < num_items; i += ITEMS_PER_THREAD * GRID_DIM * BLOCK_THREADS)
  {
    FloatType items[ITEMS_PER_THREAD] = {};
    for (int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      const int idx = i + j * GRID_DIM * BLOCK_THREADS;
      if (idx < num_items)
      {
        items[j] = transform_op(d_in[idx]);
      }
    }
    FloatType abs_max = fabs(items[0]);

#pragma unroll
    for (auto j = 1; j < ITEMS_PER_THREAD; j++)
    {
      abs_max = fmax(fabs(items[j]), abs_max);
    }

    thread_aggregate.set_max_val(abs_max);
#pragma unroll
    for (auto j = 0; j < ITEMS_PER_THREAD; j++)
    {
      thread_aggregate.unsafe_add(items[j]);
      count++;
      if (count >= thread_aggregate.endurance())
      {
        thread_aggregate.renorm();
        count = 0;
      }
    }
  }

  AccumT block_aggregate = BlockReduceT(temp_storage).Reduce(thread_aggregate, [](AccumT lhs, AccumT rhs) -> AccumT {
    AccumT rtn = lhs;
    rtn += rhs;
    return rtn;
  });

  // Output result
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy_single(d_out + blockIdx.x, block_aggregate);
  }
}

/**
 * @brief Deterministically Reduce a single tile kernel entry point (single-block). Can be used
 *        to aggregate privatized thread block reductions from a previous
 *        multi-block reduction pass.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::__identity>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::SingleTilePolicy::BLOCK_THREADS), 1) void DeterministicDeviceReduceSingleTileKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
  ReductionOpT reduction_op,
  InitT init,
  TransformOpT transform_op)
{
  using BlockReduceT =
    BlockReduce<AccumT,
                ChainedPolicyT::SingleTilePolicy::BLOCK_THREADS,
                ChainedPolicyT::SingleTilePolicy::BLOCK_ALGORITHM>;
  // Shared memory storage
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = init;
    }

    return;
  }

  using FloatType         = typename AccumT::ftype;
  constexpr int BinLength = AccumT::MAXINDEX + AccumT::MAXFOLD;

  FloatType* shared_bins = detail::rfa_detail::get_shared_bin_array<FloatType, BinLength>();

#pragma unroll
  for (int index = threadIdx.x; index < BinLength;
       index += ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS)
  {
    shared_bins[index] = detail::rfa_detail::RFA_bins<FloatType>::initialize_bins(index);
  }
  CTA_SYNC();

  constexpr auto BLOCK_THREADS = ChainedPolicyT::ActivePolicy::SingleTilePolicy::BLOCK_THREADS;

  AccumT thread_aggregate{};

  // Consume block aggregates of previous kernel
#pragma unroll
  for (int i = threadIdx.x; i < num_items; i += BLOCK_THREADS)
  {
    thread_aggregate += transform_op(d_in[i]);
  }

  AccumT block_aggregate = BlockReduceT(temp_storage).Reduce(thread_aggregate, reduction_op, num_items);
  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
  }
}

/******************************************************************************
 * Policy
 ******************************************************************************/

/**
 * @tparam AccumT
 *   Accumulator data type
 *
 * OffsetT
 *   Signed integer type for global offsets
 *
 * ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 */
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct DeviceDeterministicReducePolicy
{
  //---------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //---------------------------------------------------------------------------

  /// SM30
  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 2;

    // ReducePolicy (GTX670: 154.0 @ 48M 4B items)
    using DeterministicReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_DEFAULT>;

    // SingleTilePolicy
    using SingleTilePolicy = DeterministicReducePolicy;
  };

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B
    // items)
    using DeterministicReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = DeterministicReducePolicy;
  };

  /// SM60
  struct Policy600 : ChainedPolicy<600, Policy600, Policy350>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using DeterministicReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = DeterministicReducePolicy;
  };

  using MaxPolicy = Policy600;
};

/******************************************************************************
 * Single-problem dispatch
 *****************************************************************************/
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction in deterministic fashion
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam InitT
 *   Initial value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename SelectedPolicy = DeviceDeterministicReducePolicy<
            rfa_detail::AccumT<cub::Sum, rfa_detail::InitT<OutputIteratorT, InputIteratorT>, InputIteratorT>,
            OffsetT,
            cub::Sum>,
          typename InitT        = rfa_detail::InitT<OutputIteratorT, InputIteratorT>,
          typename AccumT       = rfa_detail::AccumT<cub::Sum, InitT, InputIteratorT>,
          typename TransformOpT = ::cuda::std::__identity>
struct DeterministicDispatchReduce : SelectedPolicy
{
  using deterministic_add_t = rfa_detail::deterministic_sum_t<AccumT>;
  using ReductionOpT        = deterministic_add_t;

  using deterministic_accum_t = typename deterministic_add_t::DeterministicAcc;

  using AcumFloatTransformT = detail::rfa_detail::rfa_float_transform_t<AccumT>;

  using OutputIteratorTransformT = thrust::transform_output_iterator<AcumFloatTransformT, OutputIteratorT>;
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorTransformT d_out;

  /// Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  TransformOpT transform_op;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DeterministicDispatchReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorTransformT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    int ptx_version,
    TransformOpT transform_op = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
      , transform_op(transform_op)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DeterministicDispatchReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorTransformT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitT init,
    cudaStream_t stream,
    bool debug_synchronous,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_items(num_items)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }
  //---------------------------------------------------------------------------
  // Small-problem (single tile) invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke a single block block to reduce in-core deterministically
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeSingleTile(SingleTileKernelT single_tile_kernel)
  {
    cudaError error = cudaSuccess;
    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        break;
      }

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
              (long long) stream,
              ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif

      // Invoke single_reduce_sweep_kernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
        .doit(single_tile_kernel, d_in, d_out, num_items, reduction_op, init, transform_op);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Normal problem size invocation (two-pass)
  //---------------------------------------------------------------------------

  /**
   * @brief Invoke two-passes to reduce deteerministically
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam ReduceKernelT
   *   Function type of cub::DeterministicDeviceReduceKernel
   *
   * @tparam SingleTileKernelT
   *   Function type of cub::DeterministicDeviceReduceSingleTileKernel
   *
   * @param[in] reduce_kernel
   *   Kernel function pointer to parameterization of cub::DeterministicDeviceReduceKernel
   *
   * @param[in] single_tile_kernel
   *   Kernel function pointer to parameterization of
   *   cub::DeterministicDeviceReduceSingleTileKernel
   */
  template <typename ActivePolicyT, typename ReduceKernelT, typename SingleTileKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(ReduceKernelT reduce_kernel, SingleTileKernelT single_tile_kernel)
  {
    cudaError error = cudaSuccess;
    do
    {
      const auto tile_size = ActivePolicyT::DeterministicReducePolicy::BLOCK_THREADS
                           * ActivePolicyT::DeterministicReducePolicy::ITEMS_PER_THREAD;
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      int sm_count;
      error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      KernelConfig reduce_config;
      error = CubDebug(reduce_config.Init<typename ActivePolicyT::DeterministicReducePolicy>(reduce_kernel));
      if (cudaSuccess != error)
      {
        break;
      }

      const int reduce_device_occupancy = reduce_config.sm_occupancy * sm_count;
      const int max_blocks              = reduce_device_occupancy * CUB_SUBSCRIPTION_FACTOR(0);
      const int resulting_grid_size     = (num_items + tile_size - 1) / tile_size;

      // Get grid size for device_reduce_sweep_kernel
      const int reduce_grid_size = resulting_grid_size > max_blocks ? max_blocks : resulting_grid_size;

      // Temporary storage allocation requirements
      void* allocations[1]       = {};
      size_t allocation_sizes[1] = {
        reduce_grid_size * sizeof(deterministic_accum_t) // bytes needed for privatized block
                                                         // reductions
      };

      // Alias the temporary allocations from the single storage blob (or
      // compute the necessary size of the blob)
      error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        return cudaSuccess;
      }

      // Alias the allocation for the privatized per-block reductions
      deterministic_accum_t* d_block_reductions = (deterministic_accum_t*) allocations[0];

      // Log device_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeterministicDeviceReduceKernel<<<%d, %d, 0, %lld>>>(), %d items "
              "per thread, %d SM occupancy\n",
              reduce_grid_size,
              ActivePolicyT::DeterministicReducePolicy::BLOCK_THREADS,
              (long long) stream,
              ActivePolicyT::DeterministicReducePolicy::ITEMS_PER_THREAD,
              reduce_config.sm_occupancy);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        reduce_grid_size, ActivePolicyT::DeterministicReducePolicy::BLOCK_THREADS, 0, stream)
        .doit(reduce_kernel,
              d_in,
              d_block_reductions,
              static_cast<int>(num_items),
              reduction_op,
              transform_op,
              reduce_grid_size);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

// Log single_reduce_sweep_kernel configuration
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DeterministicDeviceReduceSingleTileKernel<<<1, %d, 0, %lld>>>(), "
              "%d items per thread\n",
              ActivePolicyT::SingleTilePolicy::BLOCK_THREADS,
              (long long) stream,
              ActivePolicyT::SingleTilePolicy::ITEMS_PER_THREAD);
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Invoke DeterministicDeviceReduceSingleTileKernel
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        1, ActivePolicyT::SingleTilePolicy::BLOCK_THREADS, 0, stream)
        .doit(single_tile_kernel,
              d_block_reductions,
              d_out,
              reduce_grid_size, // triple_chevron is not type safe, make sure to use int
              reduction_op,
              init,
              ::cuda::std::__identity{});

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors
      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /// Invocation Deterministic
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using SingleTilePolicyT = typename ActivePolicyT::SingleTilePolicy;
    using MaxPolicyT        = typename DeterministicDispatchReduce::MaxPolicy;

    // Force kernel code-generation in all compiler passes
    if (num_items <= (SingleTilePolicyT::BLOCK_THREADS * SingleTilePolicyT::ITEMS_PER_THREAD))
    {
      return InvokeSingleTile<ActivePolicyT>(
        DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          InputIteratorT,
          OutputIteratorTransformT,
          OffsetT,
          ReductionOpT,
          InitT,
          deterministic_accum_t,
          TransformOpT>);
    }
    else
    {
      return InvokePasses<ActivePolicyT>(
        DeterministicDeviceReduceKernel<typename DeterministicDispatchReduce::MaxPolicy,
                                        InputIteratorT,
                                        OffsetT,
                                        ReductionOpT,
                                        deterministic_accum_t,
                                        TransformOpT>,
        DeterministicDeviceReduceSingleTileKernel<
          MaxPolicyT,
          deterministic_accum_t*,
          OutputIteratorTransformT,
          int, // Always used with int
               // offsets
          ReductionOpT,
          InitT,
          deterministic_accum_t>);
    }
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchHelper(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorTransformT d_out,
    OffsetT num_items,
    deterministic_add_t reduction_op,
    InitT init,
    cudaStream_t stream,
    TransformOpT transform_op = {})
  {
    using MaxPolicyT = typename DeterministicDispatchReduce::MaxPolicy;

    cudaError error = cudaSuccess;
    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DeterministicDispatchReduce dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_items,
        reduction_op,
        init,
        stream,
        ptx_version,
        transform_op);

      // Dispatch to chained policy
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DispatchHelper(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorTransformT d_out,
    OffsetT num_items,
    deterministic_add_t reduction_op,
    InitT init,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return DispatchHelper(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, init, stream);
  }

  /**
   * @brief Internal dispatch routine for computing a device-wide deterministic reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    rfa_detail::InitT<OutputIteratorT, InputIteratorT> init = {},
    cudaStream_t stream                                     = {},
    TransformOpT transform_op                               = {})
  {
    OutputIteratorTransformT d_out_transformed = thrust::make_transform_output_iterator(d_out, AcumFloatTransformT{});

    return DispatchHelper(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out_transformed,
      num_items,
      deterministic_add_t{},
      init,
      stream,
      transform_op);
  }
};
} // namespace detail
CUB_NAMESPACE_END