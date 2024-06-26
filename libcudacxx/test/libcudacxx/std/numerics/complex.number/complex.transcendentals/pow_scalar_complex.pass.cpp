//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   pow(const T& x, const complex<T>& y);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const T& a, const cuda::std::complex<T>& b, cuda::std::complex<T> x)
{
  cuda::std::complex<T> c = pow(a, b);
  is_about(real(c), real(x));
  assert(cuda::std::abs(imag(c)) < T(1.e-6));
}

template <class T>
__host__ __device__ void test()
{
  test(T(2), cuda::std::complex<T>(2), cuda::std::complex<T>(4));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    for (unsigned j = 0; j < N; ++j)
    {
      cuda::std::complex<T> r = pow(real(testcases[i]), testcases[j]);
      cuda::std::complex<T> z = exp(testcases[j] * log(cuda::std::complex<T>(real(testcases[i]))));
      if (cuda::std::isnan(real(r)))
      {
        assert(cuda::std::isnan(real(z)));
      }
      else
      {
        assert(real(r) == real(z));
      }
      if (cuda::std::isnan(imag(r)))
      {
        assert(cuda::std::isnan(imag(z)));
      }
      else
      {
        assert(imag(r) == imag(z));
      }
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif
  test_edges<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_edges<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_edges<__nv_bfloat16>();
#endif

  return 0;
}
