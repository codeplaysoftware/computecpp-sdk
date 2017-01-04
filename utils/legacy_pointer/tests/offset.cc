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
 *   Test using offsets with the legacy pointer utility class
 *
 **************************************************************************/

#include "gtest/gtest.h"

#include <CL/sycl.hpp>
#include <iostream>

#include "legacy_pointer.hpp"

using sycl_acc_target = cl::sycl::access::target;
const  sycl_acc_target sycl_acc_host = sycl_acc_target::host_buffer;
const  sycl_acc_target sycl_acc_buffer = sycl_acc_target::global_buffer;

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace codeplay;

using buffer_id = legacy::PointerMapper::buffer_id;
using buffer_t = legacy::PointerMapper::buffer_t;


struct kernel {
  using acc_type = 
            cl::sycl::accessor<legacy::PointerMapper::buffer_data_type,
                                1, sycl_acc_rw, sycl_acc_buffer>;
  acc_type accB_;
  int i_;
  int j_;
  int SIZE_;
  int offset_;

  kernel(acc_type accB, int i, int j, int SIZE, int offset)
    : accB_(accB), i_(i), j_(j), SIZE_(SIZE), offset_(offset) { };

  void operator()() {
    auto float_off = offset_ / sizeof(float);
    float * ptr = reinterpret_cast<float *>(&*accB_.get_pointer());
    ptr[float_off] = i_ * SIZE_ + j_;
  };
};

TEST(offset, basic_test) {
  {
    ASSERT_EQ(legacy::getPointerMapper().count(), 0);
    float * myPtr = static_cast<float *>(
                        legacy::malloc(100 * sizeof(float)));

    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(legacy::PointerMapper::is_nullptr(myPtr));

    ASSERT_EQ(legacy::getPointerMapper().count(), 1);

    myPtr += 3;

    // Obtain the buffer id
    buffer_id bId = legacy::getPointerMapper().get_buffer_id(myPtr);
    ASSERT_NE(bId, 0); // Only one buffer

    // Obtain the buffer
    // Note that the scope of this buffer ends when the buffer
    // is freed
    //
    buffer_t b = legacy::getPointerMapper().get_buffer(bId);

    off_t offset = legacy::getPointerMapper().get_offset(myPtr);
    ASSERT_EQ(offset, 3*sizeof(float));
    
    cl::sycl::queue q;
    q.submit([&b,offset](cl::sycl::handler& h) {
        auto accB = b.get_access<sycl_acc_rw>(h);
        h.single_task<class foo1>([=]() {
              accB[offset] = 1.0f;
            });
        });

    // Only way of reading the value is using a host accessor
    {
      auto hostAcc = b.get_access<sycl_acc_rw, sycl_acc_host>();
      ASSERT_EQ(hostAcc[offset], 1.0f);
    }
    legacy::free(myPtr);
    ASSERT_EQ(legacy::getPointerMapper().count(), 0);
  }
}


TEST(offset, 2d_indexing) {
  {
    const unsigned SIZE = 8;
    ASSERT_EQ(legacy::getPointerMapper().count(), 0);
    float * myPtr = static_cast<float *>(
                        legacy::malloc(SIZE * SIZE * sizeof(float)));

    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(legacy::PointerMapper::is_nullptr(myPtr));

    ASSERT_EQ(legacy::getPointerMapper().count(), 1);

    // Obtain the buffer id
    buffer_id bId = legacy::getPointerMapper().get_buffer_id(myPtr);
    ASSERT_NE(bId, 0); // Only one buffer
    
    cl::sycl::queue q;

    float * actPos = myPtr;
    for (int i = 0; i < SIZE; i++) {   
      for (int j = 0; j < SIZE; j++) {
        // Obtain the buffer id
        buffer_id bId = 
          legacy::getPointerMapper().get_buffer_id(actPos);
        ASSERT_NE(bId, 0); // Only one buffer

        // Obtain the buffer
        // Note that the scope of this buffer ends when the buffer
        // is freed
        //
        buffer_t b = legacy::getPointerMapper().get_buffer(bId);

        off_t offset = legacy::getPointerMapper().get_offset(actPos);
        ASSERT_EQ(offset, (i * SIZE + j) * sizeof(float));
        
        q.submit([b, offset, i, j, SIZE](cl::sycl::handler& h) mutable {
            auto accB = b.get_access<sycl_acc_rw>(h);
            h.single_task(kernel(accB, i, j, SIZE, offset));
        });
        q.wait_and_throw();
        // We move to the next ptr
        actPos++;
      }  // for int j
    }  // for int i

    // Only way of reading the value is using a host accessor
    {
      buffer_t b = legacy::getPointerMapper().get_buffer(bId);
      ASSERT_EQ(b.get_size(), SIZE*SIZE*sizeof(float));
      auto hostAcc = b.get_access<sycl_acc_rw, sycl_acc_host>();
      float * fPtr = reinterpret_cast<float *>(&*hostAcc.get_pointer());
      for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
          ASSERT_EQ(fPtr[i * SIZE + j], i * SIZE + j);
        }
      }
    }
    legacy::free(myPtr);
    ASSERT_EQ(legacy::getPointerMapper().count(), 0);
  }
}
