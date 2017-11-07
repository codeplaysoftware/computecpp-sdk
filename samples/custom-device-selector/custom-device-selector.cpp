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
 *  custom-device-selector.cpp
 *
 *  Description:
 *    Sample code that shows how to write a custom device selector in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

using namespace cl::sycl;

/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
class custom_selector : public device_selector {
 public:
  custom_selector() : device_selector() {}

  /* The selection is performed via the () operator in the base
   * selector class.This method will be called once per device in each
   * platform. Note that all platforms are evaluated whenever there is
   * a device selection. */
  int operator()(const device& device) const override {
    /* We only give a valid score to devices that support SPIR. */
    if (device.has_extension(cl::sycl::string_class("cl_khr_spir"))) {
      if (device.get_info<info::device::device_type>() ==
          info::device_type::cpu) {
        return 50;
      }
      if (device.get_info<info::device::device_type>() ==
          info::device_type::gpu) {
        return 100;
      }
    }
    /* Devices with a negative score will never be chosen. */
    return -1;
  }
};

int main() {
  const int dataSize = 64;
  int ret = -1;
  float data[dataSize] = {0.f};

  range<1> dataRange(dataSize);
  buffer<float, 1> buf(data, dataRange);

  /* We create an object of custom_selector type and use it
   * like any other selector. */
  custom_selector selector;
  queue myQueue(selector);

  myQueue.submit([&](handler& cgh) {
    auto ptr = buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class example_kernel>(dataRange, [=](item<1> item) {
      size_t idx = item.get_linear_id();
      ptr[item.get_linear_id()] = static_cast<float>(idx);
    });
  });

  /* A host accessor can be used to force an update from the device to the
   * host, allowing the data to be checked. */
  accessor<float, 1, access::mode::read_write, access::target::host_buffer>
      hostPtr(buf);

  if (hostPtr[10] == 10.0f) {
    ret = 0;
  }

  return ret;
}
