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
 *  async_handler.cpp
 *
 *  Description:
 *    Sample showing the use of the hierarchical API in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

using namespace cl::sycl;

/* Helper function to compute a globalID from a group and item in a
 * hierarchical parallel_for_work_item context */
static inline cl::sycl::id<1> get_global_id(cl::sycl::group<1>& group,
                                            cl::sycl::item<1>& item) {
  cl::sycl::range<1> globalR = group.get_global_range();
  cl::sycl::range<1> localR = item.get_range();
  cl::sycl::id<1> lID = item.get();
  return cl::sycl::id<1>(group.get(0) * localR.get(0) + lID.get(0));
}

class PrivateMemory;

/* This sample showcases the syntax of the private_memory interface. */
int main() {
  int ret = 0;
  const int nItems = 64;
  const int nLocals = 16;
  int data[nItems] = {0};

  /* Any data on the device will be copied back to the host
   * after the block ends. */
  {
    default_selector selector;

    queue myQueue(selector);

    /* We need to create a buffer in order to access data
     * from the SYCL devices. */
    buffer<int, 1> buf(data, range<1>(nItems));

    /* This command group enqueues a kernel on myQueue
     * that adds the thread id to each element of the
     * data array. */
    myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);
      /* We create a linear (one dimensional) group range, which
       * creates a thread per element of the vector. */
      auto groupRange = range<1>(nItems / nLocals);
      /* We create a linear (one dimensional) local range which defines the
       * workgroup size. */
      auto localRange = range<1>(nLocals);
      /* Kernel functions executed by a parallel_for that takes an
       * nd_range receive a single parameter of type item. */
      auto hierarchicalKernel = [=](group<1> groupID) {
        /* Unlike variables allocated in a parallel_for_work_group scope,
         * privateObj is allocated per workitem and lives in thread-private
         * memory. */
        private_memory<int> privateObj(groupID);

        parallel_for_work_item(groupID, [&](item<1> itemID) {
          /* Assign the thread global id into private memory. */
          privateObj(itemID) = get_global_id(groupID, itemID).get(0);
        });

        parallel_for_work_item(groupID, [&](item<1> itemID) {
          /* Retrieve the global id stored in the previous
           * parallel_for_work_item call and store it in global memory. */
          ptr[get_global_id(groupID, itemID).get(0)] = privateObj(itemID);
        });
      };
      cgh.parallel_for_work_group<class PrivateMemory>(groupRange, localRange,
                                                       hierarchicalKernel);
    });
  }

  for (int i = 0; i < nItems; i++) {
    if (data[i] != i) {
      ret = 1;
    }
  }

  return ret;
}
