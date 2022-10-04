/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  simple-example-of-vectors.cpp
 *
 *  Description:
 *    Example of vector operations in SYCL.
 *
 **************************************************************************/

#define SYCL_SIMPLE_SWIZZLES

#include <CL/sycl.hpp>

using namespace cl::sycl;

class vector_example;

/* The purpose of this sample code is to demonstrate
 * the usage of vectors inside SYCL kernels. */
int main() {
  int ret = 0;
  const int size = 64;
  cl::sycl::float4 dataA[size];
  cl::sycl::float3 dataB[size];

  /* Here, the vectors are initialised. float4 is a short name for
   * cl::sycl::vec<float, 4> - the other short names are similar. */
  for (int i = 0; i < size; i++) {
    dataA[i] = float4(2.0f, 1.0f, 1.0f, static_cast<float>(i));
    dataB[i] = float3(0.0f, 0.0f, 0.0f);
  }

  {
    /* In previous samples, we've mostly seen scalar types, though it is
     * perfectly possible to pass vectors to buffers. */
    buffer<cl::sycl::float4, 1> bufA(dataA, range<1>(size));
    buffer<cl::sycl::float3, 1> bufB(dataB, range<1>(size));

    queue myQueue;

    myQueue.submit([&](handler& cgh) {
      auto ptrA = bufA.get_access<access::mode::read_write>(cgh);
      auto ptrB = bufB.get_access<access::mode::read_write>(cgh);

      /* This kernel demonstrates the following:
       *   You can access the individual elements of a vector by using the
       *   functions x(), y(), z(), w() and so on, as described in the spec
       *
       *   "Swizzles" can be used by calling a function equivalent to the
       *   swizzle you need, for example, xxw(), or any combination of the
       *   elements. The swizzle need not be the same size as the original
       *   vector
       *
       *   Vectors can also be scaled easily using operator overloads */
      cgh.parallel_for<vector_example>(range<3>(4, 4, 4), [=](item<3> item) {
        auto in = ptrA[item.get_linear_id()];
        auto w = in.w();
        vec<float,3> swizzle = in.xyz();
        auto scaled = swizzle * w;
        ptrB[item.get_linear_id()] = scaled;
      });
    });
  }

  for (int i = 0; i < size; i++) {
    if (static_cast<float>(dataB[i].y()) != i) {
      ret = -1;
    }
  }

  return ret;
}
