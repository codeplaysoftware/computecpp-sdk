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
 *  Sample code that demonstrates the use of the virtual pointer interface in
 *  SYCL on matrix addition.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>
#include <vptr/virtual_ptr.hpp>

class init_a;
class init_b;
class matrix_add;
/* This sample allocates three device-only matrices, using the virtual pointer
 * and SYCLmalloc. It initalises the first two in parallel on the device. After
 * that, it adds them together, storing the result in the third matrix. It then
 * verifies the result on the host by using:
 *  - pointer arithmetic on the virtual pointer to index the matrix
 *  - host accessor to gain access to the data. */
int main() {
  using namespace cl::sycl;
  
  constexpr size_t row = 100;
  constexpr size_t col = 150;
  constexpr size_t totalEntries = row * col;
  constexpr size_t totalBytes = row * col * sizeof(float);
  constexpr float multiplier1 = 2.0f;
  constexpr float multiplier2 = 2014.0f;
  {
    queue myQueue;
    cl::sycl::codeplay::PointerMapper pMap;

    // Allocate the matrices using SYCLmalloc. a, b and c are virtual pointers,
    // pointing to device buffers.
    float* vptrA = static_cast<float*>(SYCLmalloc(totalBytes, pMap));
    float* vptrB = static_cast<float*>(SYCLmalloc(totalBytes, pMap));
    float* vptrC = static_cast<float*>(SYCLmalloc(totalBytes, pMap));

    // This kernel will initialise the buffer pointed to by a. The accessor "A"
    // has write access. We retrieve it directly from the PointerMapper, using
    // the virtual pointer.
    myQueue.submit([&](handler& cgh) {
      constexpr size_t dim = 1;

      auto accA = pMap.get_access<access::mode::discard_write,
                                  access::target::global_buffer, float>(vptrA, cgh);
      auto functor = [=](item<dim> index) {
                       accA[index] = index[0] * multiplier1;
                     };

      cgh.parallel_for<init_a>(range<dim>{totalEntries}, functor);
    });

    //Similarly, this kernel will initialise the buffer pointed to by b.
    myQueue.submit([&](handler& cgh) {
      constexpr size_t dim = 1;      

      auto accB = pMap.get_access<access::mode::discard_write,
                                  access::target::global_buffer, float>(vptrB, cgh);
      auto functor = [=](item<dim> index) {
                       accB[index] = index[0] * multiplier2;
                     };

      cgh.parallel_for<init_b>(range<dim>{totalEntries}, functor);
      });

    //This kernel will perform the computation c = a + b.
    myQueue.submit([&](handler& cgh) {
      auto accA = pMap.get_access<access::mode::read,
                                  access::target::global_buffer, float>(vptrA, cgh);
      auto accB = pMap.get_access<access::mode::read,
                                  access::target::global_buffer, float>(vptrB, cgh);
      auto accC = pMap.get_access<access::mode::discard_write,
                                  access::target::global_buffer, float>(vptrC, cgh);

      constexpr size_t dim = 1;
      auto functor = [=](item<dim> index) {
                       accC[index] = accA[index] + accB[index];
                     };
      
      cgh.parallel_for<matrix_add>(range<dim>{totalEntries}, functor);
    });

    // On the host, the result stored in the buffer of virtual pointer "c" are
    // checked. The matrix is accessed row by row, using pointer arithmetics on
    // the virtual pointer.
    auto c_row = vptrC;
    for (size_t i = 0; i < row; ++i) {
      /* Get the number of elements by which the row is offset. */
      auto row_offset = pMap.get_element_offset<float>(c_row);

      /* Create a host accessor to access the data on the host. */
      auto accC = pMap.get_access<access::mode::read,
                                  access::target::host_buffer, float>(c_row);
      for (size_t j = 0; j < col; ++j) {
        if (accC[row_offset + j] != (i * col + j) * (multiplier1 + multiplier2)) {
          std::cout << "Wrong value " << accC[row_offset + j] << " for element "
                    << i * col + j << '\n';
          break;
        }
        ++c_row;
      }
    }
    // End scope of myQueue, this waits for any remaining operations on the
    // queue to complete.
  }
  return 0;
}
