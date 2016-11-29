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
 *  hello_world.cpp
 *
 *  Description:
 *    Sample code that illustrates a simple hello world in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

using namespace cl::sycl;

/* This sample executes a single-threaded kernel on the default device
 * (as determined by the SYCL implementation) whose only function is to
 * output the canonical hello world string. */
int main() {
  default_selector selector;
  queue myQueue(selector);

  myQueue.submit([&](handler& cgh) {
    /* The stream object allows output to be generated from the kernel. It
     * takes three parameters in its constructor: the total size of the
     * memory to be allocated to it, the maximum size of any one statement
     * in the stream and the command group handler (to bind it to a
     * particular kernel execution, among other things). */
    stream os(1024, 80, cgh);

    cgh.single_task<class hello_world>([=]() {
      /* We use the stream operator on the stream object we created above to
       * print to stdout from the device. */
      os << "Hello, World!" << endl;

    });
  });

  return 0;
}
