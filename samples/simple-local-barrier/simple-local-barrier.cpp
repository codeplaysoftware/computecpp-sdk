/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  simple-local-barrier.cpp
 *
 *  Description:
 *    Sample code demonstrating local barriers in device code.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>

using namespace cl::sycl;

/* This sample demonstrates the usage of a local_barrier inside
 * a command group. The program creates an array with 64 integers
 * and swaps each pair. */
int main() {
  int ret = -1;
  const int size = 64;
  cl::sycl::cl_int data[size];
  std::iota(std::begin(data), std::end(data), 0);

  {
    default_selector selector;

    queue myQueue(selector);

    buffer<cl::sycl::cl_int, 1> buf(data, range<1>(size));

    myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      accessor<cl::sycl::cl_int, 1, access::mode::read_write,
               access::target::local>
          tile(range<1>(2), cgh);

      cgh.parallel_for<class example_kernel>(
          nd_range<1>(range<1>(size), range<1>(2)), [=](nd_item<1> item) {
            size_t idx = item.get_global_linear_id();
            int pos = idx & 1;
            int opp = pos ^ 1;

            tile[pos] = ptr[item.get_global_linear_id()];

            /* Memory accesses made to local memory before this function call
             * will be resolved immediately after. The same is true for
             * global memory (if access::fence_space::global_space is used),
             * but coherence will only apply *within a work group*! */
            item.barrier(access::fence_space::local_space);

            ptr[idx] = tile[opp];
          });
    });
  }

  /* Basic error checking - should always pass! */
  if (data[3] == 2.0f) {
    ret = 0;
  }

  return ret;
}
