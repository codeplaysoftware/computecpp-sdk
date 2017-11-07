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
 *  using-function-objects.cpp
 *
 *  Description:
 *    Sample code that demonstrates how to use function objects as kernel
 *    functors in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <random>

using namespace cl::sycl;

/* Here we define a function object, i.e. an object that has a function call
 * operator. Functors can be used to define kernels in SYCL instead of lambdas.
 * The only requirements on the functor objects you define is that they must be
 * standard layout (see C++ specification for details or use the C++11 type
 * trait is_standard_layout<class> to check). */
class my_functor {
  using rw_acc_t =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

 public:
  /* Here we construct the functor object with the accessor that we intend to
   * pass to the kernel. Just as with lambdas the accessor must be available
   * inside the functor, either by being constructed internally or by being
   * passed in on construction. */
  my_functor(rw_acc_t ptr) : m_ptr(ptr) {
    /* Generate a random number in [1, 100]. This is executed on host. */
    std::random_device hwRand;
    std::uniform_int_distribution<> r(1, 100);
    m_randumNum = r(hwRand);
  }

  /* We define a function call operator with a prototype that matches the
   * requirements of the parallel_for API, i.e. with a single item parameter,
   * that defines the kernel. In this case the kernel simply assigns the random
   * number to the accessor. */
  void operator()(item<1> item) { m_ptr[item.get()] = m_randumNum; }

  /* A member function to retrieve the random number. Functors are still
   * standard C++ classes and can be used as normal on the host. */
  int get_random() { return m_randumNum; }

 private:
  /* We define an accessor that will be initialized on construction of the
   * functor object and made available in the kernel on the device. */
  rw_acc_t m_ptr;

  int m_randumNum;
};

int main() {
  const int size = 64;
  int data[size] = {0};

  /* We define a variable for storing the random number that is generated. */
  int randomNumber;

  try {
    queue myQueue;

    buffer<int, 1> buf(data, range<1>(size));

    myQueue.submit([&](handler& cgh) {

      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      /* We create an instance of the functor object, passing it the accessor
       * created just before. */
      my_functor functor(ptr);

      /* We retrieve the random number that is generated when constructing the
       * functor, as this is just a standard C++ object. */
      randomNumber = functor.get_random();

      /* Here we call the parallel_for() API with an instance of the functor
       * instead of a lambda as seen in other samples. */
      cgh.parallel_for(range<1>(size), functor);
    });
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  /* We check the result is correct. */
  auto success =
      std::all_of(std::begin(data), std::end(data),
                  [randomNumber](int i) { return i == randomNumber; });
  if (success) {
    std::cout << "Data has the random number " << randomNumber << "."
              << std::endl;
    return 0;
  } else {
    std::cout << "Data does not have the random number " << randomNumber << "."
              << std::endl;
    return 1;
  }
}
