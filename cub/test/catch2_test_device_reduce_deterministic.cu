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

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/detail/rfa.cuh>

#include <thrust/device_vector.h>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"
#include <catch2/catch.hpp>

// %PARAM% TEST_LAUNCH lid 0:1:2

template <int tpb = 256, int itpt = 20, int ipvl = 2>
struct custom_values
{
  const int threads_per_block  = tpb;
  const int items_per_thread   = itpt;
  const int items_per_vec_load = ipvl;
};

template <custom_values& c1, custom_values& c2, custom_values& c3>
struct DeterministicDeviceReducePolicyHubTemplate
{
  template <typename AccumT, typename OffsetT, typename ReductionOpT>
  struct policy
  {
    //---------------------------------------------------------------------------
    // Architecture-specific tuning policies
    //---------------------------------------------------------------------------

    /// SM30
    struct Policy300 : cub::ChainedPolicy<300, Policy300, Policy300>
    {
      static constexpr int threads_per_block  = c1.threads_per_block;
      static constexpr int items_per_thread   = c1.items_per_thread;
      static constexpr int items_per_vec_load = c1.items_per_vec_load;

      // ReducePolicy (GTX670: 154.0 @ 48M 4B items)
      using ReducePolicy =
        cub::AgentReducePolicy<threads_per_block,
                               items_per_thread,
                               AccumT,
                               items_per_vec_load,
                               cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                               cub::LOAD_DEFAULT>;

      // SingleTilePolicy
      using SingleTilePolicy = ReducePolicy;

      // SegmentedReducePolicy
      using SegmentedReducePolicy = ReducePolicy;
    };

    /// SM35
    struct Policy350 : cub::ChainedPolicy<350, Policy350, Policy300>
    {
      static constexpr int threads_per_block  = c2.threads_per_block;
      static constexpr int items_per_thread   = c2.items_per_thread;
      static constexpr int items_per_vec_load = c2.items_per_vec_load;

      // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B
      // items)
      using ReducePolicy =
        cub::AgentReducePolicy<threads_per_block,
                               items_per_thread,
                               AccumT,
                               items_per_vec_load,
                               cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                               cub::LOAD_LDG>;

      // SingleTilePolicy
      using SingleTilePolicy = ReducePolicy;

      // SegmentedReducePolicy
      using SegmentedReducePolicy = ReducePolicy;
    };

    /// SM60
    struct Policy600 : cub::ChainedPolicy<600, Policy600, Policy350>
    {
      static constexpr int threads_per_block  = c3.threads_per_block;
      static constexpr int items_per_thread   = c3.items_per_thread;
      static constexpr int items_per_vec_load = c3.items_per_vec_load;

      // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
      using ReducePolicy =
        cub::AgentReducePolicy<threads_per_block,
                               items_per_thread,
                               AccumT,
                               items_per_vec_load,
                               cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                               cub::LOAD_LDG>;

      // SingleTilePolicy
      using SingleTilePolicy = ReducePolicy;

      // SegmentedReducePolicy
      using SegmentedReducePolicy = ReducePolicy;
    };

    using MaxPolicy = Policy300;
  };
};

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::DeterministicSum, deterministic_sum);

using float_type_list = c2h::type_list<float, double>;

CUB_TEST("Deterministic Device reduce works with float and double", "[reduce][deterministic]", float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  SECTION("base test")
  {
    const int num_items = 42;
    thrust::device_vector<type> input(num_items, 1.0f);
    thrust::device_vector<type> output(1);

    const type* d_input = thrust::raw_pointer_cast(input.data());

    deterministic_sum(d_input, output.begin(), num_items);

    type const res = output[0];

    REQUIRE(res == num_items);
  }

  SECTION("custom policy test")
  {
    const int num_items = 42;
    thrust::device_vector<type> input(num_items, 1.0f);
    thrust::device_vector<type> output(1);

    const type* d_input = thrust::raw_pointer_cast(input.data());

    std::size_t temp_storage_bytes{};

    custom_values c1, c2, c3;
    cub::DeviceReduce::DeterministicSum<decltype(d_input),
                                        decltype(output.begin()),
                                        int,
                                        DeterministicDeviceReducePolicyHubTemplate<c1, c2, c3>>(
      nullptr, temp_storage_bytes, d_input, output.begin(), num_items);
    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    cub::DeviceReduce::DeterministicSum<decltype(d_input),
                                        decltype(output.begin()),
                                        int,
                                        DeterministicDeviceReducePolicyHubTemplate<c1, c2, c3>>(
      thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output.begin(), num_items);

    type const res = output[0];

    REQUIRE(res == num_items);
  }

  SECTION("custom policy test with custom values")
  {
    const int num_items = 42;
    thrust::device_vector<type> input(num_items, 1.0f);
    thrust::device_vector<type> output(1);

    const type* d_input = thrust::raw_pointer_cast(input.data());

    std::size_t temp_storage_bytes{};

    custom_values<128, 128, 128> c1, c2, c3;
    cub::DeviceReduce::DeterministicSum<decltype(d_input),
                                        decltype(output.begin()),
                                        int,
                                        DeterministicDeviceReducePolicyHubTemplate<c1, c2, c3>>(
      nullptr, temp_storage_bytes, d_input, output.begin(), num_items);
    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    cub::DeviceReduce::DeterministicSum<decltype(d_input),
                                        decltype(output.begin()),
                                        int,
                                        DeterministicDeviceReducePolicyHubTemplate<c1, c2, c3>>(
      thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, d_input, output.begin(), num_items);

    type const res = output[0];

    REQUIRE(res == num_items);
  }
}
