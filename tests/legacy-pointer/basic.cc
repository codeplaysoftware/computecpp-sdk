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
 *   basic.cc
 *
 *  Description:
 *   Basic tests of the pointer mapper utility header
 *
 **************************************************************************/

#include "gtest/gtest.h"

#include <CL/sycl.hpp>
#include <iostream>

#include "legacy-pointer/legacy_pointer.hpp"

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace codeplay;

using buffer_id = legacy::PointerMapper::buffer_id;
using buffer_t = legacy::PointerMapper::buffer_t;

TEST(pointer_mapper, basic_test) {
  {
    ASSERT_EQ(legacy::getPointerMapper().count(), 0u);
    void* myPtr = legacy::malloc(100 * sizeof(float));
    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(legacy::PointerMapper::is_nullptr(myPtr));

    void* totallyInvalidPtr = reinterpret_cast<void*>(0x0000F0F0F1);

    ASSERT_TRUE(legacy::PointerMapper::is_nullptr(totallyInvalidPtr));

    ASSERT_EQ(legacy::getPointerMapper().count(), 1u);

    // Obtain the buffer id
    buffer_id bId = legacy::getPointerMapper().get_buffer_id(myPtr);
    // Obtain the buffer
    // Note that the scope of this buffer ends when the buffer
    // is freed
    //
    buffer_t b = legacy::getPointerMapper().get_buffer(bId);

    cl::sycl::queue q;
    q.submit([&b](cl::sycl::handler& h) {
      auto accB = b.get_access<sycl_acc_rw>(h);
      h.single_task<class foo1>([=]() { accB[0] = 1.0f; });
    });

    // Only way of reading the value is using a host accessor
    {
      auto hostAcc = b.get_access<sycl_acc_rw>();
      ASSERT_EQ(hostAcc[0], 1.0f);
    }
    legacy::free(myPtr);
    ASSERT_EQ(legacy::getPointerMapper().count(), 0u);
  }
}

TEST(pointer_mapper, two_buffers) {
  {
    ASSERT_EQ(legacy::getPointerMapper().count(), 0u);
    void* ptrA = legacy::malloc(100 * sizeof(float));
    ASSERT_NE(ptrA, nullptr);
    ASSERT_FALSE(legacy::PointerMapper::is_nullptr(ptrA));
    ASSERT_EQ(legacy::getPointerMapper().count(), 1u);

    void* ptrB = legacy::malloc(10 * sizeof(int));
    ASSERT_NE(ptrB, nullptr);
    ASSERT_FALSE(legacy::PointerMapper::is_nullptr(ptrA));
    ASSERT_EQ(legacy::getPointerMapper().count(), 2u);

    // Obtain the buffer id
    buffer_id bId1 = legacy::getPointerMapper().get_buffer_id(ptrA);
    // Obtain the buffer id
    buffer_id bId2 = legacy::getPointerMapper().get_buffer_id(ptrB);

    // Obtain the buffer
    // Note that the scope of this buffer ends when the buffer
    // is freed
    try {
      buffer_t b1 = legacy::getPointerMapper().get_buffer(bId1);
      buffer_t b2 = legacy::getPointerMapper().get_buffer(bId2);

      cl::sycl::queue q;
      q.submit([&b1, &b2](cl::sycl::handler& h) {
        auto accB1 = b1.get_access<sycl_acc_rw>(h);
        auto accB2 = b2.get_access<sycl_acc_rw>(h);
        h.single_task<class foo2>([=]() {
          accB1[0] = 1.0f;
          accB2[0] = 2.0f;
        });
      });

      // Only way of reading the value is using a host accessor
      {
        auto hostAccA = b1.get_access<sycl_acc_rw>();
        ASSERT_EQ(hostAccA[0], 1.0f);
      }
      {
        auto hostAccB = b2.get_access<sycl_acc_rw>();
        ASSERT_EQ(hostAccB[0], 2.0f);
      }
    } catch (std::out_of_range e) {
      FAIL();
    }

    ASSERT_EQ(legacy::getPointerMapper().count(), 2u);
    legacy::free(ptrA);
    ASSERT_EQ(legacy::getPointerMapper().count(), 1u);
    legacy::free(ptrB);
    ASSERT_EQ(legacy::getPointerMapper().count(), 0u);
  }
}
