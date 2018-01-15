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
 *  opencl-c-interop.cpp
 *
 *  Description:
 *    Sample code that shows the interoperability between OpenCL and SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cl::sycl;

/* The source of the OpenCL C kernel is stored in C strings as usual.
 * It is convenient to use C++11 raw string literals to store OpenCL C
 * source inside your code, as everything between the delimiters (here
 * "EOK") is considered part of the string. Therefore there is no need
 * to escape double quotes, backslashes and so on. */
const char* kernel_src_pow_x = R"EOK(
  __kernel void call_pow(__global float *input, __global float *output, int element_num) {
  int globalID = get_global_id(0);
  if(globalID < element_num)
    output[globalID] = pow(input[globalID],(input[globalID]/(globalID+1)));
}
)EOK";

int main() {
  /* This is the maximum absolute precision error between host and device
   * libraries. */
  float maximum_precision_error = 0.0f;

  const int nElems = 64;
  float input[nElems], call_pow[nElems], std_math_pow[nElems],
      err_host_device[nElems];
  for (int i = 0; i < nElems; i++) {
    input[i] = static_cast<float>(i);
    call_pow[i] = 0.0f;
    std_math_pow[i] = 0.0f;
    err_host_device[i] = 0.0f;
  }

  {
    gpu_selector gpuselector;

    queue gpu_queue(gpuselector, [](cl::sycl::exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (cl::sycl::exception e) {
          std::cout << e.what() << std::endl;
        }
      }
    });

    /* Retrieve the underlying cl_context of the context associated with the
     * queue. */
    cl_context clContext = gpu_queue.get_context().get();

    /* Retrieve the underlying cl_device_id of the device asscociated with the
     * queue. */
    cl_device_id clDeviceId = gpu_queue.get_device().get();

    /* Retrieve the underlying cl_command_queue of the queue. */
    cl_command_queue clCommandQueue = gpu_queue.get();

    /* Create variable to store OpenCL errors. */
    ::cl_int err = 0;

    /* Store the kernel source in a string. */
    string_class kernelSourceString = string_class(kernel_src_pow_x);

    /* Determine kernel source data and size. */
    auto kerneklSourceData = kernelSourceString.data();
    size_t kernelSourceSize = kernelSourceString.size();

    /* Create a cl_program object from a source string. */
    cl_program clProgram = clCreateProgramWithSource(
        clContext, 1, &kerneklSourceData, &kernelSourceSize, &err);

    /* Output an error if the create program fails. */
    if (err != CL_SUCCESS) {
      std::cout << "Failed to create program from source." << std::endl;
    }

    /* Build the cl_program object. */
    err = clBuildProgram(clProgram, 1, &clDeviceId, nullptr, nullptr, nullptr);

    /* Output an error if the build fails. */
    if (err != CL_SUCCESS) {
      std::cout << "Failed to build program." << std::endl;
    }

    /* Create a cl_kernel object from a kernel name string. */
    cl_kernel clKernel = clCreateKernel(clProgram, "call_pow", &err);

    /* Output an error if the build fails. */
    if (err != CL_SUCCESS) {
      std::cout << "Failed to create the kernel." << std::endl;
    }

    /* Create a SYCl kernel using the inter-op constructor. */
    kernel pow_kernel(clKernel);
    auto inputOpenCL = clCreateBuffer(clContext, CL_MEM_READ_ONLY,
                                      nElems * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to create input cl_mem object." << std::endl;
    }

    auto outputOpenCL = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY,
                                       nElems * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to create output cl_mem object." << std::endl;
    }

    err = clEnqueueWriteBuffer(clCommandQueue, inputOpenCL, CL_TRUE, 0,
                               nElems * sizeof(float), input, 0, nullptr,
                               nullptr);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to transfer data to device";
    }

    gpu_queue.submit([&](handler& cgh) {
      /* Normally, SYCL sets kernel arguments for the user. However, when
       * using the interoperability features, it is unable to do this and
       * the user must set the arguments manually. */
      cgh.set_arg(0, inputOpenCL);
      cgh.set_arg(1, outputOpenCL);
      cgh.set_arg(2, nElems);

      cgh.parallel_for(range<1>(nElems), pow_kernel);
    });
    gpu_queue.wait_and_throw();

    err = clEnqueueReadBuffer(clCommandQueue, outputOpenCL, CL_TRUE, 0,
                              nElems * sizeof(float), err_host_device, 0,
                              nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to transfer data from device";
    }

    buffer<float, 1> input_buffer(input, range<1>(nElems));
    buffer<float, 1> call_pow_buffer(call_pow, range<1>(nElems));

    /* This submission performs the same calculation but in SYCL code. */
    gpu_queue.submit([&](handler& cgh) {
      auto in = input_buffer.get_access<access::mode::read>(cgh);
      auto out = call_pow_buffer.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class pow_comp>(range<1>(nElems), [=](item<1> item) {
        size_t idx = item[0];
        out[idx] = cl::sycl::pow(in[idx], (in[idx] / (idx + 1)));
      });
    });

    clReleaseDevice(clDeviceId);
    clReleaseCommandQueue(clCommandQueue);
    clReleaseContext(clContext);
  }

  /* Finally, this loop performs a host-side comparison. */
  for (int i = 0; i < nElems; i++) {
    std_math_pow[i] = std::pow(input[i], input[i] / (i + 1));
    maximum_precision_error =
        std::max(maximum_precision_error,
                 std::max(std::fabs(err_host_device[i] - std_math_pow[i]),
                          std::fabs(call_pow[i] - std_math_pow[i])));
  }
  std::cout << "Maximum Absolute Error " << std::fabs(maximum_precision_error)
            << std::endl;

  return 0;
}
