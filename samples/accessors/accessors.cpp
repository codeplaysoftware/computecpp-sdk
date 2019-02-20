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
 *  accessor.cpp
 *
 *  Description:
 *    Sample code that illustrates how to make data available on a device
 *    using accessors in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

class multiply;

int main() {
  /* We define the data to be passed to the device. */
  int data = 5;

  /* The scope we create here defines the lifetime of the buffer object, in SYCL
   * the lifetime of the buffer object dictates synchronization using RAII. */
  try {
    /* We can also create a queue that uses the default selector in
     * the queue's default constructor. */
    queue myQueue;

    /* We define a buffer in order to maintain data across the host and one or
     * more devices. We construct this buffer with the address of the data
     * defined above and a range specifying a single element. */
    buffer<int, 1> buf(&data, range<1>(1));

    myQueue.submit([&](handler& cgh) {
      /* We define accessors for requiring access to a buffer on the host or on
       * a device. Accessors are are like pointers to data we can use in
       * kernels to access the data. When constructing the accessor you must
       * specify the access target and mode. SYCL also provides the
       * get_access() as a buffer member function, which only requires an
       * access mode - in this case access::mode::read_write.
       * (make_access<>() has a second template argument which defaults
       * to access::mode::global) */
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      cgh.single_task<multiply>([=]() {
        /* We use the subscript operator of the accessor constructed above to
         * read the value, multiply it by itself and then write it back to the
         * accessor again. */
        ptr[0] = ptr[0] * ptr[0];
      });
    });

    /* queue::wait() will block until kernel execution finishes,
     * successfully or otherwise. */
    myQueue.wait();

  } catch (exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }

  /* We check that the result is correct. */
  if (data == 25) {
    std::cout << "Hurray! 5 * 5 is " << data << '\n';
    return 0;
  } else {
    std::cout << "Oops! Something went wrong... 5 * 5 is not " << data << "!\n";
    return 1;
  }
}
