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
 *  parallel_for.cpp
 *
 *  Description:
 *    Samples using the parallel_for API in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

/* We define the number of work items to enqueue. */
const int nElems = 64u;

int main() {
  /* We define and initialize data to be copied to the device. */
  int data[nElems] = {0};

  try {
    default_selector selector;
    queue myQueue(selector, [](exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (std::exception& e) {
          std::cout << e.what();
        }
      }
    });

    buffer<int, 1> buf(data, range<1>(nElems));

    myQueue.submit([&](handler& cgh) {

      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      /* We create an nd_range to describe the work space that the kernel is
       * to be executed across. Here we create a linear (one dimensional)
       * nd_range, which creates a work item per element of the vector. The
       * first parameter of the nd_range is the range of global work items
       * and the second is the range of local work items (i.e. the work group
       * range). */
      auto myRange = nd_range<1>(range<1>(nElems), range<1>(nElems / 4));

      /* We construct the lambda outside of the parallel_for function call,
       * though it can be inline inside the function call too. For this
       * parallel_for API the lambda is required to take a single parameter;
       * an item<N> of the same dimensionality as the nd_range - in this
       * case one. Other kernel dispatches might have different parameters -
       * for example, the single_task takes no arguments. */
      auto myKernel = ([=](nd_item<1> item) {
        /* Items have various methods to extract ids and ranges. The
         * specification has full details of these. Here we use the
         * item::get_global() to retrieve the global id as an id<1>.
         * This particular kernel will set the ith element to the value
         * of i. */
        ptr[item.get_global()] = item.get_global()[0];
      });

      /* We call the parallel_for() API with two parameters; the nd_range
       * we constructed above and the lambda that we constructed. Because
       * the kernel is a lambda we *must* specify a template parameter to
       * use as a name. */
      cgh.parallel_for<class assign_elements>(myRange, myKernel);
    });

  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return 2;
  }

  /* Check the result is correct. */
  int result = 0;
  for (int i = 0; i < nElems; i++) {
    if (data[i] != i) {
      std::cout << "The results are incorrect (element " << i << " is "
                << data[i] << "!\n";
      result = 1;
    }
  }
  if (result != 1) {
    std::cout << "The results are correct." << std::endl;
  }
  return result;
}
