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
 *  placeholder-accessors.cpp
 *
 *  Description:
 *    Sample code that illustrates how to use placeholder accessors.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

namespace {
class Worker;
}

class Doubler {
  /* These are placeholder accessors a class members so they are visible to both
   * perform_doubling and operator(). As placeholder accessors, they do not have
   * a buffer associated with them yet. It will be attached during the command
   * group */
  accessor<const uint8_t, 1, access::mode::read, access::target::global_buffer,
           access::placeholder::true_t>
      in_accessor;
  accessor<uint8_t, 1, access::mode::discard_write,
           access::target::global_buffer, access::placeholder::true_t>
      out_accessor;

 public:
  void perform_doubling(const uint8_t* input, uint8_t* output, size_t items) {
    try {
      queue myQueue;

      /* Create buffers from the provided pointers */
      auto in_buffer = buffer<const uint8_t, 1>(input, items);
      auto out_buffer = buffer<uint8_t, 1>(output, items);

      myQueue.submit([&](handler& cgh) {
        /* Associate buffers with the accessors */
        cgh.require(in_buffer, in_accessor);
        cgh.require(out_buffer, out_accessor);

        /* The accessors can now be used to access the buffers created above */
        cgh.parallel_for<Worker>(range<1>(items), *this);
      });
    } catch (const exception& e) {
      std::cerr << "SYCL exception caught: " << e.what() << "\n";
      throw;
    }
  }

  void operator()(item<1> item) { out_accessor[item] = in_accessor[item] * 2; }
};

/* Input data, and array length */
constexpr size_t num_items = 10;
const std::array<uint8_t, num_items> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

int main() {
  Doubler doubler{};

  std::array<uint8_t, num_items> output{};
  doubler.perform_doubling(values.data(), output.data(), num_items);

  /* We check that the result is correct. */
  if (!std::equal(
          output.begin(), output.end(), values.begin(),
          [](const auto out, const auto in) { return out == in * 2; })) {
    std::cerr << "Resulting output is wrong!\n";
    return 1;
  }
  return 0;
}
