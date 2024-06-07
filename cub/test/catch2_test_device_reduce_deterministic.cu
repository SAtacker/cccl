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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "cub/block/block_load.cuh"
#include "cub/block/block_reduce.cuh"
#include "cub/thread/thread_load.cuh"
#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/detail/rfa.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include <catch2/catch.hpp>

// %PARAM% TEST_LAUNCH lid 0:1:2

// // today
// float levels[42];
// runtime_populate(levels);

// // shreyes

// template <class Limit>
// constexpr float next_limit()
// {
//   return compile_time_compute_next(Limit);
// }

// template <float Limit>
// int limit(float x, int bin)
// {
//   if (Limit < x)
//   {
//     return bin;
//   }
//   else
//   {
//     return limit<next_limit<Limit>>(x, bin + 1);
//   }
// }

using float_type_list = c2h::type_list<float, double>;

template <int NOMINAL_BLOCK_THREADS_4B, int NOMINAL_ITEMS_PER_THREAD_4B>
struct AgentReducePolicy
{
  /// Number of items per vectorized load
  static constexpr int VECTOR_LOAD_LENGTH = 4;

  /// Cooperative block-wide reduction algorithm to use
  static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING;

  /// Cache load modifier for reading input elements
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::CacheLoadModifier::LOAD_DEFAULT;
  constexpr static int ITEMS_PER_THREAD                 = NOMINAL_ITEMS_PER_THREAD_4B;
  constexpr static int BLOCK_THREADS                    = NOMINAL_BLOCK_THREADS_4B;
};

template <int ItemsPerThread, int BlockSize>
struct hub_t
{
  struct Policy : cub::ChainedPolicy<300, Policy, Policy>
  {
    constexpr static int ITEMS_PER_THREAD = ItemsPerThread;

    using ReducePolicy = AgentReducePolicy<BlockSize, ItemsPerThread>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;

    // SegmentedReducePolicy
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy;
};

template <typename type>
void reduce_1()
{
  const int num_items = 42;
  thrust::device_vector<type> input(num_items, 1.0f);
  thrust::device_vector<type> output_p1(1);
  thrust::device_vector<type> output_p2(1);

  const type* d_input = thrust::raw_pointer_cast(input.data());

  std::size_t temp_storage_bytes{};

  using deterministic_dispatch_t_p1 =
    cub::detail::DeterministicDispatchReduce<decltype(d_input), decltype(output_p1.begin()), int, hub_t<4, 256>>;

  using deterministic_dispatch_t_p2 =
    cub::detail::DeterministicDispatchReduce<decltype(d_input), decltype(output_p1.begin()), int, hub_t<4, 128>>;

  deterministic_dispatch_t_p1::Dispatch(nullptr, temp_storage_bytes, d_input, output_p1.begin(), num_items);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);

  deterministic_dispatch_t_p1::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output_p1.begin(), num_items);

  type const res_p1 = output_p1[0];

  deterministic_dispatch_t_p2::Dispatch(nullptr, temp_storage_bytes, d_input, output_p2.begin(), num_items);

  deterministic_dispatch_t_p2::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output_p2.begin(), num_items);

  type const res_p2 = output_p1[0];

  REQUIRE(res_p1 == res_p2);
}

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::DeterministicSum, deterministic_sum);
// DECLARE_LAUNCH_WRAPPER(, 000reduce_2);

CUB_TEST("Deterministic Device reduce works with float and double", "[reduce][deterministic]", float_type_list)
{
  using type          = typename c2h::get<0, TestType>;
  const int num_items = 42;
  thrust::device_vector<type> input(num_items, 1.0f);
  thrust::device_vector<type> output(1);

  const type* d_input = thrust::raw_pointer_cast(input.data());

  deterministic_sum(d_input, output.begin(), num_items);

  type const res = output[0];

  REQUIRE(res == num_items);
}

CUB_TEST("Deterministic Device reduce works with float and double and is deterministic",
         "[reduce][deterministic]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;
  reduce_1<type>();
}
