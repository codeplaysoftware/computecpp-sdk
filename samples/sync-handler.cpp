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
 *  sync-handler.cpp
 *
 *  Description:
 *    Sample code that demonstrates the use of a synchronous error handler to
 *    demonstrate error handling.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <iostream>

using namespace cl::sycl;

namespace {
class Worker;

/* A custom device selector */
class PickySelector : public device_selector {
  int operator()(const device&) const final {
    /* Here logic would exist to find a specific device to do our computation
     * with. As this is just an example, it's hardcoded to find nothing. In a
     * real program, it would inspect the device information and return priority
     * numbers depending on that device's suitablity. */
    return -1;
  }
};
}  // namespace

int main() {
  /* Our output array */
  constexpr size_t number_of_sevens = 7;
  std::array<size_t, number_of_sevens> sevens;

  try {
    /* Use our custom device selector to find a specific device. In this example
     * we require a very specific piece of hardware. If that isn't present, the
     * code should just run on the host. */
    PickySelector selector;

    /* Attempt to create a queue for this using this selector. If the selector
     * fails to find a device, this will throw an exception. */
    queue myQueue(selector);

    /* If the selector couldn't find a suitable device, the following code
     * will not run. */
    buffer<size_t, 1> buf(sevens.data(), range<1>(number_of_sevens));
    myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<Worker>(range<1>(number_of_sevens),
                               [=](item<1> item) { ptr[item] = 7; });
    });
  } catch (const exception& e) {
    /* Report the exception to the user. */
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    std::cerr << "Running on host...\n";

    /* Can't find any suitable devices, so just run the code on the host
     * system normally. */
    sevens.fill(7);
  }

  /* Check the array has been populated with sevens correctly. */
  if (std::any_of(sevens.begin(), sevens.end(),
                  [](const auto x) { return x != 7; })) {
    std::cerr << "A seven was not set!\n";
    return 1;
  }

  return 0;
}
