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
 *  offset.cc
 *
 *  Description:
 *   Test using offsets with the  pointer utility class
 *
 **************************************************************************/

#include "gtest/gtest.h"

#include <CL/sycl.hpp>
#include <iostream>

#include "pointer_alias.hpp"
#include "virtual_ptr.hpp"

using sycl_acc_target = cl::sycl::access::target;
const sycl_acc_target sycl_acc_host = sycl_acc_target::host_buffer;
const sycl_acc_target sycl_acc_buffer = sycl_acc_target::global_buffer;

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace cl::sycl::codeplay;

using buffer_t = PointerMapper::buffer_t;

struct kernel {
  using acc_type = cl::sycl::accessor<buffer_data_type_t, 1,
                                      sycl_acc_rw, sycl_acc_buffer>;
  acc_type accB_;
  int i_;
  int j_;
  int SIZE_;
  int offset_;

  kernel(acc_type accB, int i, int j, int SIZE, int offset)
      : accB_(accB), i_(i), j_(j), SIZE_(SIZE), offset_(offset){};

  void operator()() {
    auto float_off = offset_ / sizeof(float);
    float *ptr = cl::sycl::codeplay::get_device_ptr_as<float>(accB_);
    ptr[float_off] = i_ * SIZE_ + j_;
  };
};

TEST(offset, basic_test) {
  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);
    float *myPtr = static_cast<float *>(SYCLmalloc(100 * sizeof(float), pMap));

    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(PointerMapper::is_nullptr(myPtr));

    ASSERT_EQ(pMap.count(), 1u);

    myPtr += 3;

    auto b = pMap.get_buffer(myPtr);

    size_t offset = pMap.get_offset(myPtr);
    ASSERT_EQ(offset, 3 * sizeof(float));

    cl::sycl::queue q;
    q.submit([&b, offset](cl::sycl::handler &h) {
      auto accB = b.get_access<sycl_acc_rw>(h);
      h.single_task<class foo1>([=]() {
        cl::sycl::codeplay::get_device_ptr_as<float>(accB)[offset] = 1.0f;
      });
    });

    // Only way of reading the value is using a host accessor
    {
      auto hostAcc = b.get_access<sycl_acc_rw, sycl_acc_host>();
      ASSERT_EQ(cl::sycl::codeplay::get_host_ptr_as<float>(hostAcc)[offset],
                1.0f);
    }

    SYCLfree(myPtr, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}

TEST(offset, 2d_indexing) {
  PointerMapper pMap;
  {
    const unsigned SIZE = 8;
    ASSERT_EQ(pMap.count(), 0u);
    float *myPtr =
        static_cast<float *>(SYCLmalloc(SIZE * SIZE * sizeof(float), pMap));

    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(PointerMapper::is_nullptr(myPtr));

    ASSERT_EQ(pMap.count(), 1u);

    cl::sycl::queue q;

    float *actPos = myPtr;
    for (unsigned i = 0; i < SIZE; i++) {
      for (unsigned j = 0; j < SIZE; j++) {
        // Obtain the buffer
        // Note that the scope of this buffer ends when the buffer
        // is freed
        //
        auto b = pMap.get_buffer(actPos);

        size_t offset = pMap.get_offset(actPos);
        ASSERT_EQ(offset, (i * SIZE + j) * sizeof(float));

        q.submit([b, offset, i, j, SIZE](cl::sycl::handler &h) mutable {
          auto accB = b.get_access<sycl_acc_rw>(h);
          h.single_task(kernel(accB, i, j, SIZE, offset));
        });

        // We move to the next ptr
        actPos++;
      }  // for int j
    }    // for int i

    // Only way of reading the value is using a host accessor
    {
      auto b = pMap.get_buffer(myPtr);
      ASSERT_EQ(b.get_size(), SIZE * SIZE * sizeof(float));
      auto hostAcc = b.get_access<sycl_acc_rw, sycl_acc_host>();
      float *fPtr = cl::sycl::codeplay::get_host_ptr_as<float>(hostAcc);
      for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
          ASSERT_EQ(fPtr[i * SIZE + j], i * SIZE + j);
        }
      }
    }
    SYCLfree(myPtr, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}
