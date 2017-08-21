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
 *   accessor.cc
 *
 *  Description:
 *   Tests for the get_access() method in the pointer mapper
 *
 **************************************************************************/

#include "gtest/gtest.h"

#include <CL/sycl.hpp>
#include <iostream>

#include "pointer_alias.hpp"
#include "virtual_ptr.hpp"

using sycl_acc_target = cl::sycl::access::target;
const sycl_acc_target sycl_acc_host = sycl_acc_target::host_buffer;

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace cl::sycl::codeplay;

TEST(accessor, basic_test) {
  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);
    void *myPtr = SYCLmalloc(100 * sizeof(float), pMap);
    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(PointerMapper::is_nullptr(myPtr));

    ASSERT_TRUE(PointerMapper::is_nullptr(nullptr));

    ASSERT_EQ(pMap.count(), 1u);

    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler &h) {
      auto accB = pMap.get_access<sycl_acc_rw>(myPtr, h);
      h.single_task<class foo1>([=]() { accB[0] = 1.0f; });
    });

    // Only way of reading the value is using a host accessor
    {
      auto hostAcc = pMap.get_access<sycl_acc_rw, sycl_acc_host>(myPtr);
      ASSERT_EQ(hostAcc[0], 1.0f);
    }
    SYCLfree(myPtr, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}

TEST(accessor, two_buffers) {
  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);
    void *ptrA = SYCLmalloc(100 * sizeof(int), pMap);
    ASSERT_NE(ptrA, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(ptrA));
    ASSERT_EQ(pMap.count(), 1u);

    void *ptrB = SYCLmalloc(10 * sizeof(int), pMap);

    ASSERT_NE(ptrB, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(ptrA));
    ASSERT_EQ(pMap.count(), 2u);

    try {
      cl::sycl::queue q([&](cl::sycl::exception_list e) {
        std::cout << "Error " << std::endl;
      });

      q.submit([&](cl::sycl::handler &h) {
        auto accB1 = pMap.get_access<sycl_acc_rw>(ptrA, h);
        auto accB2 = pMap.get_access<sycl_acc_rw>(ptrB, h);
        h.single_task<class foo2>([=]() {
          cl::sycl::codeplay::get_device_ptr_as<int>(accB1)[0] = 1;
          cl::sycl::codeplay::get_device_ptr_as<int>(accB2)[0] = 2;
        });
      });

      q.wait_and_throw();
      // Only way of reading the value is using a host accessor
      {
        auto hostAccA = pMap.get_access<sycl_acc_rw, sycl_acc_host>(ptrA);
        ASSERT_EQ(cl::sycl::codeplay::get_host_ptr_as<int>(hostAccA)[0], 1);
      }
      {
        auto hostAccB = pMap.get_access<sycl_acc_rw, sycl_acc_host>(ptrB);
        ASSERT_EQ(cl::sycl::codeplay::get_host_ptr_as<int>(hostAccB)[0], 2);
      }
    } catch (std::out_of_range e) {
      FAIL();
    }

    ASSERT_EQ(pMap.count(), 2u);
    SYCLfree(ptrA, pMap);
    ASSERT_EQ(pMap.count(), 1u);
    SYCLfree(ptrB, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}

TEST(accessor, allocator) {
  // an allocator type
  using alloc_t = cl::sycl::detail::aligned_mem::aligned_allocator<uint8_t>;

  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);

    // add a pointer with the base allocator type
    void *ptrA = SYCLmalloc(100 * sizeof(int), pMap);

    // get the buffer with the base allocator type
    auto bufA = pMap.get_buffer(ptrA);

    ASSERT_EQ(pMap.count(), 1u);

    // add a pointer with the allocator type
    void *ptrB = SYCLmalloc<alloc_t>(100 * sizeof(int), pMap);

    // get the buffer with the allocator type
    cl::sycl::buffer<uint8_t, 1, alloc_t> bufB = pMap.get_buffer<buffer_data_type_t, alloc_t>(ptrB);

    ASSERT_EQ(pMap.count(), 2u);
  }
}
