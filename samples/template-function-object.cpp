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
 *  template-function-object.cpp
 *
 *  Description:
 *    Sample code that demonstrates how to template function objects in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

using namespace cl::sycl;
using namespace std::placeholders;

/* In SYCL, C++ classes can be used as kernels - not just lambdas. One
 * requirement is that the class must be standard layout, which is used
 * to provide certain guarantees to the runtime. C++ Reference has a good
 * description of standard layout at:
 * http://en.cppreference.com/w/cpp/language/data_members#Standard_layout
 *
 * This template class defines a kernel function that performs a vector add.
 * It contains three accessors: two inputs that are access::mode::read and an
 * output that is access::mode::write. A templated class allows you to have a
 * generic kernel that can be instantiated over different types. */
template <typename dataT>
class vector_add_kernel {
 public:
  using read_accessor =
      accessor<dataT, 1, access::mode::read, access::target::global_buffer>;
  using write_accessor =
      accessor<dataT, 1, access::mode::write, access::target::global_buffer>;
  vector_add_kernel(read_accessor ptrA, read_accessor ptrB, write_accessor ptrC)
      : m_ptrA(ptrA), m_ptrB(ptrB), m_ptrC(ptrC) {}

  /* We define this object's function call operator to match the parallel_for
   * call which takes a range rather than an nd_range. operator()() takes an
   * item rather than an nd_item. */
  void operator()(item<1> item) {
    /* This is a standard vector add, as seen in other samples. */
    m_ptrC[item] = m_ptrA[item] + m_ptrB[item];
  }

 private:
  read_accessor m_ptrA;
  read_accessor m_ptrB;
  write_accessor m_ptrC;
};

/* This function will be bound to later on in the kernel submission code. */
template <typename dataT>
void vector_add(buffer<dataT, 1>* a, buffer<dataT, 1>* b, buffer<dataT, 1>* c,
                int count, handler& cgh) {
  auto a_dev = a->template get_access<access::mode::read>(cgh);
  auto b_dev = b->template get_access<access::mode::read>(cgh);
  auto c_dev = c->template get_access<access::mode::write>(cgh);

  cgh.parallel_for(range<1>(count),
                   vector_add_kernel<dataT>(a_dev, b_dev, c_dev));
}

/* Using the code above, two vector additions are performed: one on ints,
 * the other on floats. */
int main() {
  const unsigned count = 1024;
  bool pass = false;

  queue myQueue;

  try {
    std::vector<float> a(count), b(count), c(count);
    std::fill(a.begin(), a.end(), 0.f);
    std::fill(b.begin(), b.end(), 1.f);

    {
      buffer<float, 1> bufA(a.data(), range<1>(count));
      buffer<float, 1> bufB(b.data(), range<1>(count));
      buffer<float, 1> bufC(c.data(), range<1>(count));

      /* We create a function object for the command_group by using std::bind
       * with the vector_add function template and the accessor and scalar
       * parameters. This submission instantiates the template with float. */
      myQueue.submit(
          std::bind(vector_add<float>, &bufA, &bufB, &bufC, count, _1));
    }

    /* Check the results after the data is copied back to the host. */
    pass = std::all_of(c.begin(), c.end(), [](float i) { return i == 1.0f; });

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  try {
    std::vector<int> a(count), b(count), c(count);
    std::fill(a.begin(), a.end(), 0.f);
    std::fill(b.begin(), b.end(), 1.f);

    {
      buffer<int, 1> bufA(a.data(), range<1>(count));
      buffer<int, 1> bufB(b.data(), range<1>(count));
      buffer<int, 1> bufC(c.data(), range<1>(count));

      /* This is the same as above, except we now instantiate for int. */
      myQueue.submit(
          std::bind(vector_add<int>, &bufA, &bufB, &bufC, count, _1));
    }

    pass =
        pass && std::all_of(c.begin(), c.end(), [](int i) { return i == 1; });

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 1;
  }

  if (pass) {
    std::cout << "The results are as expected." << std::endl;
    return 0;
  } else {
    std::cout << "The results are not as expected." << std::endl;
    return 1;
  }
}
