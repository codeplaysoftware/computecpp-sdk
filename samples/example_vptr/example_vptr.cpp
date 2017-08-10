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
 *  example_sycl_application.cpp
 *
 *  Description:
 *    Sample code that walks through the basics of executing matrix addition
 *    using the virtual pointer in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

#include "pointer_alias.hpp"
#include "virtual_ptr.hpp"

using namespace cl::sycl;

const size_t N = 100;
const size_t M = 150;

/* This sample creates three device-only arrays, then initialises them
 * on the device. After that, it adds two of them together, storing the
 * result in the third buffer. It then verifies the result of the kernel
 * on the host by using a host accessor to gain access to the data. */
int main() {
  /* Destroying SYCL objects like queues and buffers blocks until all work
   * associated with those objects is completed. */
  {
    queue myQueue;
    cl::sycl::codeplay::PointerMapper pMap;

    /* Create device-only 2D buffers of floats for the matrices. */
    /*
    buffer<float, 2> a(range<2>{N, M});
    buffer<float, 2> b(range<2>{N, M});
    buffer<float, 2> c(range<2>{N, M});
    */
    float* a = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));
    float* b = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));
    float* c = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));

    /* This kernel enqueue will initialise buffer a. The accessor "A" has
     * write access. */
    myQueue.submit([&](handler& cgh) {
      auto A = pMap.get_access<access::mode::write>(a, cgh);
      cgh.parallel_for<class init_a>(range<1>(N*M), [=](id<1> index) {
          A[index] = index[0] * 2;
      });
    });

    /* This kernel enqueue will likewise initialise buffer b. The only
     * accessor it specifies is a write accessor to b, so the runtime
     * can use this information to recognise that these kernels are
     * actually independent of each other. Therefore, they can be enqueued
     * to the device with no dependencies between each other. */
    myQueue.submit([&](handler& cgh) {
      auto B = pMap.get_access<access::mode::write>(b, cgh);
      cgh.parallel_for<class init_b>(range<1>{N*M}, [=](id<1> index) {
          B[index] = index[0] * 2014;
      });
    });

    /* This kernel will actually perform the computation C = A * B. Since
     * A and B are only read from, we specify read accessors for those two
     * buffers, which the SYCL runtime recognises as a dependency on the
     * previous kernels. If the data were initialised on a different device,
     * or on the host, the SYCL runtime would ensure that the data were
     * copied between contexts etc. properly. */
    myQueue.submit([&](handler& cgh) {
      auto A = pMap.get_access<access::mode::write>(a, cgh);
      auto B = pMap.get_access<access::mode::write>(b, cgh);
      auto C = pMap.get_access<access::mode::write>(c, cgh);
      cgh.parallel_for<class matrix_add>(
          range<1>{N*M}, [=](id<1> index) { 
              C[index] = A[index] + B[index]; });
    });

    /* A host accessor will copy data from the device and, under most
     * circumstances, allocate space for it for the user (it will not
     * allocate space when the map_allocator is used and an initial host
     * pointer is provided, as this instructs the runtime to map the data
     * into the host's memory). Since this code is attempting to access
     * buffer c, which had write access in the third kernel, the device is
     * assumed to have the most recent copy. Therefore, the runtime will
     * wait for the device to finish executing the third kernel before
     * copying data from the device to the host. Because it is read only,
     * were we to use buffer c on the device again, no copy would be issued
     * (and in fact, the operator[]() exposed here does not return an lvalue,
     * and cannot be assigned to).*/
    auto cAcc = pMap.get_access<access::mode::read, access::target::host_buffer>(c);
    auto C = cl::sycl::codeplay::get_host_ptr_as<float>(cAcc);
    std::cout << "Result:" << std::endl;
    for (size_t i = 0; i < N*M; i++) {
      //for (size_t j = 0; j < M; j++) {
        std::cout << C[i] << std::endl;
        if (C[i] != (i) * (2 + 2014)) {
          std::cout << "Wrong value " << C[i] << " for element " << i << std::endl;
          return -1;
        }
      //}
    }
    /* End scope of myQueue, this waits for any remaining operations on the
     * queue to complete. */
  }

  std::cout << "Good computation!" << std::endl;
  return 0;
}
