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
 *  example.cpp
 *
 *  Description:
 *   Generated user code that can wrap cuda kernel and execute it.
 *
 * Authors:
 *
 *    Mehdi Goli     Codeplay Software Ltd.
 *    Ruyman Reyes   Codeplay Software Ltd.
 *
 **************************************************************************/
#include <math.h>
#include <iostream>

// Header added by the source to source tool
#include "compatibility_definitions.hpp"

// Generated class:: Kernel dispatch.
// This signature must be variadic
template <typename... Args>
struct ___CudaConverterFunctor___add
    : public cl::sycl::codeplay::Generic_Kernel_Functor<
          ___CudaConverterFunctor___add<Args...>> {
  using parent = cl::sycl::codeplay::Generic_Kernel_Functor<
      ___CudaConverterFunctor___add<Args...>>;
  using parent::__syncthreads;
  using parent::blockDim;
  using parent::blockIdx;
  using parent::gridDim;
  using parent::threadIdx;

  ___CudaConverterFunctor___add(Args... args) : parent(args...) {}
  // if shared memory is used this code will be added
  template <typename T>
  T* SharedMemory() {
    return parent::template get_local_mem<T>();
  }
  // kernel executor
  template <typename... params_t>
  void __execute__(params_t... params) {
    add(params...);
  }
  __global__ void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = (blockDim.x * gridDim.x);
    for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
  }
};

int main(int argc, char* argv[]) {
  // Size of vectors
  int n = 1 << 20;
  // Host input vectors
  float* h_a;
  float* h_b;

  // Device input vectors
  float* d_a;
  float* d_b;

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(float);

  // Allocate memory for each vector on host
  h_a = (float*)malloc(bytes);
  h_b = (float*)malloc(bytes);

  // Added by the conversion tool
  cl::sycl::queue deviceQueue((cl::sycl::gpu_selector()));

  // Allocate memory for each vector on GPU
  // Original: cudaMalloc(&d_a, bytes);
  d_a = static_cast<float*>(cl::sycl::codeplay::SYCLmalloc(
      bytes, cl::sycl::codeplay::get_global_pointer_mapper()));
  // Original: cudaMalloc(&d_b, bytes);
  d_b = static_cast<float*>(cl::sycl::codeplay::SYCLmalloc(
      bytes, cl::sycl::codeplay::get_global_pointer_mapper()));

  int i;
  // Initialize vectors on host
  // initialize x and y arrays on the host
  for (int i = 0; i < n; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // Copy host vectors to device
  // Original: cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice );
  cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::HostToDevice>(deviceQueue, h_a, d_a, bytes,
                                              true);

  // Original: cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice );
  cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::HostToDevice>(deviceQueue, h_b, d_b, bytes,
                                              true);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // Execute the kernel
  // Original: vecAdd<<<numBlocks, blockSize>>>(n, d_a, d_b);
  deviceQueue.submit(cl::sycl::codeplay::CudaCommandGroup<
                     ___CudaConverterFunctor___add<int, float*, float*>>(
      numBlocks, blockSize, n, d_a, d_b));

  // Copy array back to host
  // Original: cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
  cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::DeviceToHost>(deviceQueue, d_b, h_b, bytes,
                                              true);

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  float maxError = 0.0f;
  for (int i = 0; i < n; i++) maxError = fmax(maxError, fabs(h_b[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Release device memory
  // Original: cudaFree(d_a);
  SYCLfree(d_a, cl::sycl::codeplay::get_global_pointer_mapper());
  // Original: cudaFree(d_b);
  SYCLfree(d_b, cl::sycl::codeplay::get_global_pointer_mapper());

  // Release host memory
  free(h_a);
  free(h_b);

  return 0;
}
