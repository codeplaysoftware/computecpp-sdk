/***************************************************************************
 *
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  reinterpret.cpp
 *
 *  Description:
 *    Sample code showing the reinterpret buffer feature of SYCL 1.2.1
 *
 **************************************************************************/

#include <CL/sycl.hpp>

int main() {
  cl::sycl::range<1> r(128);
  cl::sycl::buffer<float, 1> buf_float(r);
  cl::sycl::queue q;

  {
    auto acc = buf_float.get_access<cl::sycl::access::mode::write>();
    for (auto i = 0u; i < r.size(); i++) {
      acc[i] = i + 1;
    }
  }

  /* buf_int is a new SYCL buffer, with the same total size as buf_float,
   * but will provide uint32_t elements instead. However, the device memory
   * is the *same*.
   * This is actually true of the host memory, too, though there isn't any
   * host memory explicitly used here. In effect the reinterpret is a way
   * to obtain a differently-shaped handle over the top of SYCL's internal
   * data structures. */
  auto buf_int = buf_float.reinterpret<uint32_t>(r);
  q.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf_int.get_access<cl::sycl::access::mode::read_write>(cgh);
    /* This kernel will multiply IEEE-754 32-bit floats by two, by manipulating
     * the exponent directly */
    cgh.parallel_for<class mult>(r, [=](cl::sycl::item<2> i) {
      constexpr auto mask = 0x7F800000u;
      constexpr auto mantissa_shift = 23u;
      auto& elem = acc[i];
      auto exponent = (elem & mask) >> mantissa_shift;
      exponent++;
      elem &= ~mask;
      elem |= (exponent << mantissa_shift);
    });
  });

  auto ret = 0;
  {
    auto acc = buf_float.get_access<cl::sycl::access::mode::read>();
    for (auto i = 0u; i < r.size(); i++) {
      ret += (acc[i] != (2 * i + 2));
    }
  }

  return ret;
}
