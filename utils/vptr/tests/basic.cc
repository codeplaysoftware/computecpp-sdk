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
#include <random>

#include "pointer_alias.hpp"
#include "virtual_ptr.hpp"

using sycl_acc_target = cl::sycl::access::target;
const sycl_acc_target sycl_acc_host = sycl_acc_target::host_buffer;

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace cl::sycl::codeplay;

TEST(pointer_mapper, basic_test) {
  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);
    void *myPtr = SYCLmalloc(100 * sizeof(float), pMap);
    ASSERT_NE(myPtr, nullptr);

    ASSERT_FALSE(PointerMapper::is_nullptr(myPtr));

    ASSERT_TRUE(PointerMapper::is_nullptr(nullptr));

    ASSERT_EQ(pMap.count(), 1u);

    auto b = pMap.get_buffer(myPtr);

    cl::sycl::queue q;
    q.submit([&b](cl::sycl::handler &h) {
      auto accB = b.get_access<sycl_acc_rw>(h);
      h.single_task<class foo1>([=]() {
        cl::sycl::codeplay::get_device_ptr_as<float>(accB)[0] = 1.0f;
      });
    });

    // Only way of reading the value is using a host accessor
    {
      auto hostAcc = b.get_access<sycl_acc_rw, sycl_acc_host>();
      ASSERT_EQ(cl::sycl::codeplay::get_host_ptr_as<float>(hostAcc)[0], 1.0f);
    }
    SYCLfree(myPtr, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}

TEST(pointer_mapper, two_buffers) {
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

    // Obtain the buffer
    // Note that the scope of this buffer ends when the pointer is freed
    try {
      auto b2 = pMap.get_buffer(ptrB);
      auto b1 = pMap.get_buffer(ptrA);

#ifdef COMPUTECPP_INTERFACE
      ASSERT_NE(b2.get_impl().get(), b1.get_impl().get());
#endif  // COMPUTECPP_INTERFACE

      cl::sycl::queue q([&](cl::sycl::exception_list e) {
        std::cout << "Error " << std::endl;
      });

      q.submit([&b1, &b2](cl::sycl::handler &h) {
        auto accB1 = b1.get_access<sycl_acc_rw>(h);
        auto accB2 = b2.get_access<sycl_acc_rw>(h);
        h.single_task<class foo2>([=]() {
          cl::sycl::codeplay::get_device_ptr_as<int>(accB1)[0] = 1;
          cl::sycl::codeplay::get_device_ptr_as<int>(accB2)[0] = 2;
        });
      });

      q.wait_and_throw();
      // Only way of reading the value is using a host accessor
      {
        auto hostAccA = b1.get_access<sycl_acc_rw, sycl_acc_host>();
        ASSERT_EQ(cl::sycl::codeplay::get_host_ptr_as<int>(hostAccA)[0], 1);
      }
      {
        auto hostAccB = b2.get_access<sycl_acc_rw, sycl_acc_host>();
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

TEST(pointer_mapper, reuse_ptr) {
  PointerMapper pMap;
  {
    ASSERT_EQ(pMap.count(), 0u);

    // First we insert a large buffer
    void *initial = SYCLmalloc(100 * sizeof(int), pMap);
    ASSERT_NE(initial, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(initial));
    ASSERT_EQ(pMap.count(), 1u);

    // Now we insert a small one, that will be reused
    void *reused = SYCLmalloc(10 * sizeof(int), pMap);
    ASSERT_NE(reused, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(reused));
    ASSERT_EQ(pMap.count(), 2u);

    // Another large buffer
    void *end = SYCLmalloc(100 * sizeof(int), pMap);
    ASSERT_NE(end, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(end));
    ASSERT_EQ(pMap.count(), 3u);

    // We free the intermediate one
    SYCLfree(reused, pMap);
    ASSERT_EQ(pMap.count(), 2u);

    void *shouldBeTheSame = SYCLmalloc(10 * sizeof(int), pMap);
    ASSERT_NE(shouldBeTheSame, nullptr);
    ASSERT_FALSE(PointerMapper::is_nullptr(shouldBeTheSame));
    ASSERT_EQ(pMap.count(), 3u);
    ASSERT_EQ(shouldBeTheSame, reused);
  }
}

TEST(pointer_mapper, multiple_alloc_free) {
  PointerMapper pMap;
  size_t numAllocations = (1 << 9);
  size_t maxAllocSize = 1251;

  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_int_distribution<int> uniform_dist(1, maxAllocSize);

  {
    int *ptrToFree = nullptr;

    for (auto i = 0; i < numAllocations / 2; i++) {
      int *current =
          static_cast<int *>(SYCLmalloc(uniform_dist(e1) * sizeof(int), pMap));
      // We choose a random pointer to free from the entire range
      if (uniform_dist(e1) % 2) {
        ptrToFree = current;
      }
    }
    ASSERT_EQ(pMap.count(), (numAllocations / 2));
    SYCLfree(ptrToFree, pMap);
    ASSERT_EQ(pMap.count(), (numAllocations / 2) - 1);
    int *ptrInBetween = static_cast<int *>(SYCLmalloc(50 * sizeof(int), pMap));
    for (auto i = 0; i < numAllocations / 2; i++) {
      int *current =
          static_cast<int *>(SYCLmalloc(uniform_dist(e1) * sizeof(int), pMap));
      // We choose a random pointer to free from the entire range
      if (uniform_dist(e1) % 2) {
        ptrToFree = current;
      }
    }
    ASSERT_EQ(pMap.count(), numAllocations);
    SYCLfree(ptrInBetween, pMap);
    ASSERT_EQ(pMap.count(), (numAllocations - 1));
  }

  pMap.clear();
  ASSERT_EQ(pMap.count(), 0u);
}

TEST(pointer_mapper, default_access) {
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
      auto accDev = pMap.get_access(myPtr, h);
      h.single_task<class foo3>([=]() {
        accDev[0] = 1.0f;
      });
    });

    {
      auto hostAcc = pMap.get_access<sycl_acc_rw, sycl_acc_host>(myPtr);
      ASSERT_EQ(hostAcc[0], 1.0f);
    }

    SYCLfree(myPtr, pMap);
    ASSERT_EQ(pMap.count(), 0u);
  }
}
