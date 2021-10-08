/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  vector-addition-examples.cpp
 *
 *  Description:
 *    Shows different code generation for vector addition kernels
 **************************************************************************/

#include <iostream>

#include <CL/sycl.hpp>

using namespace cl;

/* Base vector add function. */
void vecAdd(const float* a, const float* b, float* c, size_t id) {
  c[id] = a[id] + b[id];
}

/* Masked variant where the store is hidden behind a runtime branch. */
void vecAddMasked(const float* a, const float* b, float* c, size_t id) {
  float v = a[id] + b[id];
  if (v < 0.0f) {
    c[id] = v;
  }
}

/* Variant where the variable value is predicated on a branch. */
void vecAddPredicated(const float* a, const float* b, float* c, size_t id) {
  float v = a[id] + b[id];
  if (v < 0.0f) {
    v = 0.0f;
  }
  c[id] = v;
}

class VecAddKernel;
class VecAddKernelMasked;
class VecAddKernelPredicated;

void zeroBuffer(sycl::buffer<float, 1> b) {
  constexpr auto dwrite = sycl::access::mode::discard_write;
  auto h = b.get_access<dwrite>();
  for (auto i = 0u; i < b.get_range()[0]; i++) {
    h[i] = 0.f;
  }
}

void sumBuffer(sycl::buffer<float, 1> b) {
  constexpr auto read = sycl::access::mode::read;
  auto h = b.get_access<read>();
  auto sum = 0.0f;
  for (auto i = 0u; i < b.get_range()[0]; i++) {
    sum += h[i];
  }
  std::cout << "computation result: " << sum << std::endl;
}

/* This sample shows three different vector addition functions. It
 * is possible to inspect the assembly generated by these samples
 * using the ComputeSuite tooling to compare the different approaches.
 * The general flow is that the output buffer is zeroed, the calculation
 * scheduled, then the sum printed for each of the functions. */
int main(int argc, char* argv[]) {
  constexpr auto read = sycl::access::mode::read;
  constexpr auto write = sycl::access::mode::write;
  constexpr auto dwrite = sycl::access::mode::discard_write;
  constexpr const size_t N = 100000;
  const sycl::range<1> VecSize{N};

  sycl::buffer<float> bufA{VecSize};
  sycl::buffer<float> bufB{VecSize};
  sycl::buffer<float> bufC{VecSize};

  {
    auto h_a = bufA.get_access<dwrite>();
    auto h_b = bufB.get_access<dwrite>();
    for (auto i = 0u; i < N; i++) {
      h_a[i] = sin(i);
      h_b[i] = cos(i);
    }
  }

  sycl::queue myQueue;

  {
    zeroBuffer(bufC);
    auto cg = [&](sycl::handler& h) {
      auto a = bufA.get_access<read>(h);
      auto b = bufB.get_access<read>(h);
      auto c = bufC.get_access<write>(h);

      h.parallel_for<VecAddKernel>(
          VecSize, [=](sycl::id<1> i) { vecAdd(&a[0], &b[0], &c[0], i[0]); });
    };
    myQueue.submit(cg);
    sumBuffer(bufC);
  }
  {
    zeroBuffer(bufC);
    auto cg = [&](sycl::handler& h) {
      auto a = bufA.get_access<read>(h);
      auto b = bufB.get_access<read>(h);
      auto c = bufC.get_access<write>(h);

      h.parallel_for<VecAddKernelMasked>(VecSize, [=](sycl::id<1> i) {
        vecAddMasked(&a[0], &b[0], &c[0], i[0]);
      });
    };
    myQueue.submit(cg);
    sumBuffer(bufC);
  }
  {
    zeroBuffer(bufC);
    auto cg = [&](sycl::handler& h) {
      auto a = bufA.get_access<read>(h);
      auto b = bufB.get_access<read>(h);
      auto c = bufC.get_access<write>(h);

      h.parallel_for<VecAddKernelPredicated>(VecSize, [=](sycl::id<1> i) {
        vecAddPredicated(&a[0], &b[0], &c[0], i[0]);
      });
    };
    myQueue.submit(cg);
    sumBuffer(bufC);
  }

  return 0;
}
