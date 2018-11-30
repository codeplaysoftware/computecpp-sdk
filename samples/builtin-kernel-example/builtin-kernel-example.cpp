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
 *  builtin_kernel_example.cpp
 *
 *  Description:
 *    Example of using an OpenCL builtin kernel with SYCL via the codeplay
 *    extension
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <array>
#include <iostream>

int main() {

  queue testQueue(TEST_SELECTOR, TEST_ASYNC_HANDLER);
  context testContext = testQueue.get_context();

  device dev = testContext.get_devices()[0];

  /*
   * Using the device properties, we can query which built in kernels
   * are supported
   */
  auto builtinKernels = dev.get_info<info::device::built_in_kernels>();

  if (builtinKernels.size() == 0) {
     std::cout << "[EXIT] No built-in kernels available for testing " 
               << std::endl;
      return 0;
  }

  /*
   * The only builtin kernel supported by this example is the
   * ComputeAorta copy buffer, defined as:
   *    copy_buffer(__global * in, __global * out)
   * And copies input into output.
   */
  const std::string kAortaTestKernelName{"copy_buffer"};

  auto kernelNamePos =
    std::find(std::begin(builtinKernels), std::end(builtinKernels),
        kAortaTestKernelName);
  if (kernelNamePos == std::end(builtinKernels)) {
      std::cout << "[EXIT] Only ComputeAorta test built-in kernel is supported "
                << " on this example " << std::endl;
      return 0;
  }

  const float goldenValue = 1234.0f;
  float input = goldenValue;
  float output = 0;

  {
    buffer<float, 1> buf(&input, range<1>(1));
    buffer<float, 1> bufOut(range<1>(1));
    bufOut.set_final_data(&output);

    program syclProgram(testContext);

    /*
     * The "create_from_built_in_kernel" method from the 
     * ComputeCpp program class uses the clCreateProgramWithBultinKernels
     * to load the OpenCL builtin kernels into the SYCL program object.
     */
    syclProgram.create_from_built_in_kernel(kAortaTestKernelName);

    kernel kernelC(syclProgram.get_kernel(kAortaTestKernelName));

    testQueue.submit([&](handler& cgh) {
      auto accIn = buf.get_access<access::mode::read>(cgh);
      auto accOut = bufOut.get_access<access::mode::write>(cgh);

      /* Using the OpenCL interoperability to set the kernel 
       * arguments to match the accessors to the builtin kernel
       * arguments.
       */
      cgh.set_arg(0, accIn);
      cgh.set_arg(1, accOut);

      auto myRange = range<1>{1};
      /* The kernel can be dispatched to the device using the 
       * parallel_for dispatch function.
       * In this case, since the range is 1, the single_task could
       * also have been used.
       */
      cgh.parallel_for(myRange, kernelC);
    });
  }

  int retVal = 0;
  if (input != output) {
    std::cout << " The result of the builtin kernel is not expected!" 
              << std::endl;
    retVal = 1;
  }

  return retVal;
}
