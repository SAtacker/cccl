/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * @file
 * Thread utilities for sequential reduction over statically-sized array types
 */

#pragma once

#include <cub/config.cuh>

#include <type_traits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/rfa.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

CUB_NAMESPACE_BEGIN

/// Internal namespace (to prevent ADL mishaps between static functions when mixing different CUB installations)
namespace internal
{

/**
 * @brief Sequential reduction over statically-sized array types
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT
ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix, Int2Type<LENGTH> /*length*/)
{
  AccumT retval = prefix;
  if constexpr ((std::is_invocable_v<ReductionOp, detail::ReproducibleFloatingAccumulator<float>, float4>
                 || std::is_invocable_v<ReductionOp, detail::ReproducibleFloatingAccumulator<double>, double4>)
                && (std::is_convertible_v<T, float> || std::is_convertible_v<T, double>)
                && (std::is_same_v<AccumT, detail::ReproducibleFloatingAccumulator<float>>
                    || std::is_same_v<AccumT, detail::ReproducibleFloatingAccumulator<double>>) )
  {
    constexpr int float4_inp_len = LENGTH / 4;

    // printf("jere float4_inp_len %d LENGTH %d\n", float4_inp_len, LENGTH);
    auto* float4_input = reinterpret_cast<std::conditional_t<std::is_same_v<T, float>, float4, double4>*>(input);
    // cuda::std::array<float4, float4_inp_len> float4_input;
//     std::conditional_t<std::is_same_v<T, float>, float4, double4> float4_input[float4_inp_len] = {0};
// #pragma unroll
//     for (int i = 0; i < LENGTH; ++i)
//     {
//       auto j = i / 4;

//       float4_input[j].x = input[i];

//       if (i + 1 < LENGTH)
//       {
//         float4_input[j].y = input[i + 1];
//       }
//       else
//       {
//         float4_input[j].y = 0.0f;
//       }
//       if (i + 2 < LENGTH)
//       {
//         float4_input[j].z = input[i + 2];
//       }
//       else
//       {
//         float4_input[j].z = 0.0f;
//       }
//       if (i + 3 < LENGTH)
//       {
//         float4_input[j].w = input[i + 3];
//       }
//       else
//       {
//         float4_input[j].w = 0.0f;
//       }
//     }
#pragma unroll
    for (int i = 0; i < float4_inp_len; ++i)
    {
      retval = reduction_op(retval, float4_input[i]);
    }

    return retval;
  }
  else
  {
    // printf("here\n");
#pragma unroll
    for (int i = 0; i < LENGTH; ++i)
    {
      retval = reduction_op(retval, input[i]);
    }

    return retval;
  }
}

/**
 * @brief Perform a sequential reduction over @p LENGTH elements of the @p input array,
 *        seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T* input, ReductionOp reduction_op, PrefixT prefix)
{
  return ThreadReduce(input, reduction_op, prefix, Int2Type<LENGTH>());
}

/**
 * @brief Perform a sequential reduction over @p LENGTH elements of the @p input array.
 *        The aggregate is returned.
 *
 * @tparam LENGTH
 *   LengthT of input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 */
template <int LENGTH, typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T* input, ReductionOp reduction_op)
{
  T prefix = input[0];
  return ThreadReduce<LENGTH - 1>(input + 1, reduction_op, prefix);
}

/**
 * @brief Perform a sequential reduction over the statically-sized @p input array,
 *        seeded with the specified @p prefix. The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 *
 * @param[in] prefix
 *   Prefix to seed reduction with
 */
template <int LENGTH,
          typename T,
          typename ReductionOp,
          typename PrefixT,
          typename AccumT = detail::accumulator_t<ReductionOp, PrefixT, T>>
_CCCL_DEVICE _CCCL_FORCEINLINE AccumT ThreadReduce(T (&input)[LENGTH], ReductionOp reduction_op, PrefixT prefix)
{
  return ThreadReduce(input, reduction_op, prefix, Int2Type<LENGTH>());
}

/**
 * @brief Serial reduction with the specified operator
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 */
template <int LENGTH, typename T, typename ReductionOp>
_CCCL_DEVICE _CCCL_FORCEINLINE T ThreadReduce(T (&input)[LENGTH], ReductionOp reduction_op)
{
  return ThreadReduce<LENGTH>((T*) input, reduction_op);
}

} // namespace internal
CUB_NAMESPACE_END
