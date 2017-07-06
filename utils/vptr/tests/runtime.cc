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
 *   runtime.cc
 *
 *  Description:
 *   Tests for the pointer_mapper registered in the runtime
 *
 **************************************************************************/

#include "gtest/gtest.h"

#include <CL/sycl.hpp>
#include <iostream>

#include "virtual_ptr.hpp"
#include "pointer_alias.hpp"

using sycl_acc_target = cl::sycl::access::target;
const sycl_acc_target sycl_acc_host = sycl_acc_target::host_buffer;

using sycl_acc_mode = cl::sycl::access::mode;
const sycl_acc_mode sycl_acc_rw = sycl_acc_mode::read_write;

using namespace codeplay;

TEST(runtime, basic_test) {
  // Ensure that the pointer mapper in the runtime is null
  auto runtime_pmppr_null = cl::sycl::detail::get_pointer_mapper();
  ASSERT_TRUE(PointerMapper::is_nullptr(runtime_pmppr_null));

  // Create a non-null pointer mapper
  PointerMapper pMap;

  ASSERT_EQ(pMap.count(), 0u);

  void *myPtr = SYCLmalloc(100 * sizeof(float), pMap);
  ASSERT_NE(myPtr, nullptr);

  ASSERT_FALSE(PointerMapper::is_nullptr(myPtr));

  ASSERT_TRUE(PointerMapper::is_nullptr(nullptr));

  ASSERT_EQ(pMap.count(), 1u);

  // Regiester it in the runtime
  cl::sycl::detail::register_pointer_mapper(&pMap);

  // Ensure that the pointer mapper in the runtime now isn't null
  auto runtime_pmppr = cl::sycl::detail::get_pointer_mapper();
  ASSERT_FALSE(PointerMapper::is_nullptr(runtime_pmppr));
}
