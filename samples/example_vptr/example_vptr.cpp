/***************************************************************************
 *
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  example_vptr.cpp
 *
 *  Description:
 *    Sample code that demonstrates the use of the virtual pointer intrface in
 *SYCL on matrix addition.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

#include "virtual_ptr.hpp"

using namespace cl::sycl;

const size_t N = 100;
const size_t M = 150;

/* This sample allocates three device-only matrices, using the virtual pointer
 * and SYCLmalloc. It initalises the first two in parallel on the device. After
 * that, it adds them together, storing the result in the third matrix. It then
 * verifies the result on the host by using:
 *  - pointer arithmetic on the virtual pointer to index the matrix
 *  - host accessor to gain access to the data. */
int main() {
  {
    queue myQueue;
    cl::sycl::codeplay::PointerMapper pMap;

    /* Allocate the matrices using SYCLmalloc. a, b and c are virtual pointers,
     * pointing to device buffers.
     */
    float* a = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));
    float* b = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));
    float* c = static_cast<float*>(SYCLmalloc(N * M * sizeof(float), pMap));

    /* This kernel will initialise the buffer pointed to by a. The accessor "A"
     * has write access. We retrieve it directly from the PointerMapper, using
     * the virtual pointer.*/
    myQueue.submit([&](handler& cgh) {
      auto A = pMap.get_access<access::mode::write,
                               access::target::global_buffer, float>(a, cgh);
      cgh.parallel_for<class init_a>(
          range<1>(N * M), [=](id<1> index) { A[index] = index[0] * 2; });
    });

    /* Similarly, this kernel will initialise the buffer pointed to by b. */
    myQueue.submit([&](handler& cgh) {
      auto B = pMap.get_access<access::mode::write,
                               access::target::global_buffer, float>(b, cgh);
      cgh.parallel_for<class init_b>(
          range<1>{N * M}, [=](id<1> index) { B[index] = index[0] * 2014; });
    });

    /* This kernel will perform the computation C = A + B. */
    myQueue.submit([&](handler& cgh) {
      auto A = pMap.get_access<access::mode::read,
                               access::target::global_buffer, float>(a, cgh);
      auto B = pMap.get_access<access::mode::read,
                               access::target::global_buffer, float>(b, cgh);
      auto C = pMap.get_access<access::mode::write,
                               access::target::global_buffer, float>(c, cgh);
      cgh.parallel_for<class matrix_add>(range<1>{N * M}, [=](id<1> index) {
        C[index] = A[index] + B[index];
      });
    });

    /* On the host, the result stored in the buffer of virtual pointer "c" are
     * checked. The matrix is accessed row by row, using pointer arithmetics on
     * the virtual pointer. */
    float* c_row = c;
    for (size_t i = 0; i < N; i++) {
      /* Get the number of elements by which the row is offset. */
      auto row_offset = pMap.get_offset(c_row) / sizeof(float);

      /* Create a host accessor to access the data on the host. */
      auto C = pMap.get_access<access::mode::read, access::target::host_buffer,
                               float>(c_row);
      for (size_t j = 0; j < M; j++) {
        if (C[row_offset + j] != (i * M + j) * (2 + 2014)) {
          std::cout << "Wrong value " << C[row_offset + j] << " for element "
                    << i * M + j << std::endl;
          return -1;
        }
        c_row++;
      }
    }
    /* End scope of myQueue, this waits for any remaining operations on the
     * queue to complete. */
  }

  std::cout << "Good computation!" << std::endl;
  return 0;
}
