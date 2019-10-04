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
 *  simple-vector-add.cpp
 *
 *  Description:
 *    Example of a vector addition in SYCL.
 *
 **************************************************************************/

/* This example is a very small one designed to show how compact SYCL code
 * can be. That said, it includes no error checking and is rather terse. */
#include <CL/sycl.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  using namespace cl::sycl;

  constexpr access::mode sycl_read = access::mode::read;
  constexpr access::mode sycl_write = access::mode::write;

  queue deviceQueue;
  range<1> numOfItems{N};
  buffer<T, 1> bufferA(VA.data(), numOfItems);
  buffer<T, 1> bufferB(VB.data(), numOfItems);
  buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    };
    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, kern);
  });
}

int main() {
  const size_t sample_size = 4;
  using namespace cl::sycl;

  auto arrA = std::array<int, sample_size>();
  auto arrB = std::array<int, sample_size>();
  auto arrC = std::array<int, sample_size>();

  using std::begin;
  using std::end;
  std::iota(begin(arrA), end(arrA), 0);
  std::iota(begin(arrB), end(arrB), 0);

  simple_vadd(arrA, arrB, arrC);
  auto sumOfAandB = std::array<float, sample_size>();
  std::transform(begin(arrA), end(arrA), begin(arrB), begin(sumOfAandB),
                 std::plus<int>());
  auto result = std::equal(begin(arrC), end(arrC), begin(sumOfAandB));
  if (!result) {
    std::cout << "The result of simple_vadd(arrA, arrB, arrC) is incorrect! \n";
    return 1;
  }

  auto arrD = std::array<float, sample_size>();
  auto arrE = std::array<float, sample_size>();
  auto arrF = std::array<float, sample_size>();
  std::iota(begin(arrD), end(arrD), 0.0f);
  std::iota(begin(arrE), end(arrE), 0.0f);

  simple_vadd(arrD, arrE, arrF);
  auto sumOfDandE = std::array<float, sample_size>();
  std::transform(begin(arrD), end(arrD), begin(arrE), begin(sumOfDandE),
                 std::plus<float>());
  result = std::equal(begin(arrF), end(arrF), begin(sumOfDandE));
  if (!result) {
    std::cout << "The result of simple_vadd(arrD, arrE, arrF) is incorrect! \n";
    return 1;
  }

  return 0;
}
