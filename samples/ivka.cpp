/***************************************************************************
 *
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  ivka.cpp
 *
 *  Description:
 *    Sample showing the different kinds of things that are valid and not
 *    valid when used as kernel args.
 *
 **************************************************************************/

#include <ivka/ivka.hpp>

#include <CL/sycl.hpp>

struct Foo {
  int foo;
};

struct Bar : Foo {
  int bar;
};

using namespace cl::sycl;
using mode = cl::sycl::access::mode;
using target = cl::sycl::access::target;

static_assert(is_valid_kernel_arg<
    accessor<int, 1, mode::read, target::global_buffer>>::value, "");
static_assert(is_valid_kernel_arg<double>::value, "");
static_assert(is_valid_kernel_arg<Foo>::value, "");
static_assert(!is_valid_kernel_arg<Bar>::value, "");
static_assert(!is_valid_kernel_arg<cl::sycl::queue>::value, "");

int main() { return 0; }
