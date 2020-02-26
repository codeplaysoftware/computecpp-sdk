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

// Header added by the source to source tool
#include "compatibility_definitions.hpp"

// Generated class:: Kernel dispatch.
// This signature must be variadic
template <typename... Args>
struct ___CudaConverterFunctor___vecAdd
    : public cl::sycl::codeplay::Generic_Kernel_Functor<
          ___CudaConverterFunctor___vecAdd<Args...>> {
  using parent = cl::sycl::codeplay::Generic_Kernel_Functor<
      ___CudaConverterFunctor___vecAdd<Args...>>;
  using parent::__syncthreads;
  using parent::blockDim;
  using parent::blockIdx;
  using parent::gridDim;
  using parent::threadIdx;

  ___CudaConverterFunctor___vecAdd(Args... args) : parent(args...) {}
  // if shared memory is used this code will be added
  template <typename T>
  T* SharedMemory() {
    return parent::template get_local_mem<T>();
  }
  // kernel executor
  template <typename... params_t>
  void __execute__(params_t... params) {
    vecAdd(params...);
  }
  // The original cuda kernel
  __global__ void vecAdd(double* a, double* b, double* c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    double* smem = SharedMemory<double>();
    if (id < n) {
      smem[threadIdx.x] = a[id] + b[id];
      c[id] = smem[threadIdx.x];
    }
  }
};

int main(int argc, char* argv[]) {
  // Size of vectors
  int n = 1024;

  // Host input vectors
  double* h_a;
  double* h_b;
  // Host output vector
  double* h_c;

  // Device input vectors
  double* d_a;
  double* d_b;
  // Device output vector
  double* d_c;

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(double);

  // Allocate memory for each vector on host
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);

  // Added by the conversion tool
  cl::sycl::queue deviceQueue((cl::sycl::gpu_selector()));

  // Allocate memory for each vector on GPU
  // Original: cudaMalloc(&d_a, bytes);
  d_a = static_cast<double*>(cl::sycl::codeplay::SYCLmalloc(
      bytes, cl::sycl::codeplay::get_global_pointer_mapper()));
  // Original: cudaMalloc(&d_b, bytes);
  d_b = static_cast<double*>(cl::sycl::codeplay::SYCLmalloc(
      bytes, cl::sycl::codeplay::get_global_pointer_mapper()));
  // Original: cudaMalloc(&d_c, bytes);
  d_c = static_cast<double*>(cl::sycl::codeplay::SYCLmalloc(
      bytes, cl::sycl::codeplay::get_global_pointer_mapper()));

  int i;
  // Initialize vectors on host
  for (i = 0; i < n; i++) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
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

  dim3 blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = dim3(256, 1, 1);

  // Number of thread blocks in grid
  gridSize = dim3((int)ceil((float)n / blockSize.x), 1, 1);
  // Shared memory size in byte for SYCL
  // Original :  shared memory size in byte
  int sharedmem = blockSize.x * sizeof(int);
  using data_type = cl::sycl::codeplay::acc_t<uint8_t>;
  // Execute the kernel
  // Original: vecAdd<<<gridSize, blockSize, sharedmem>>>(d_a, d_b, d_c, n);
  deviceQueue.submit(
      cl::sycl::codeplay::CudaCommandGroup<
          ___CudaConverterFunctor___vecAdd<double*, double*, double*, int>>(
          gridSize, blockSize, sharedmem, d_a, d_b, d_c, n));

  // Copy array back to host
  // Original: cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );
  cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::DeviceToHost>(deviceQueue, d_c, h_c, bytes,
                                              true);

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  double sum = 0;
  for (i = 0; i < n; i++) sum += h_c[i];
  printf("final result: %f\n", sum / n);

  // Release device memory
  // Original: cudaFree(d_a);
  SYCLfree(d_a, cl::sycl::codeplay::get_global_pointer_mapper());
  // Original: cudaFree(d_b);
  SYCLfree(d_b, cl::sycl::codeplay::get_global_pointer_mapper());
  // Original: cudaFree(d_c);
  SYCLfree(d_c, cl::sycl::codeplay::get_global_pointer_mapper());

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
