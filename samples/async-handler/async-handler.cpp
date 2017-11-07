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
 *  async-handler.cpp
 *
 *  Description:
 *    Sample code that demonstrates the use of an asynchronous handler for
 *    exceptions in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace cl::sycl;

/*  Async Handler Example
 *
 *  This sample code illustrates how to use an async handler.
 *
 *  Command groups with invalid local range are issued to queues
 *  and/or context that have an asynchronous handler associated with it.
 *
 *  The expected outcome of the sample code is to trigger the
 *  asynchronous handler a number of times. */
int main() {
  bool error = false;
  unsigned nTimesCall = 0;

  /* An async handler functor can be used to handle SYCL asynchronous
   * errors (i.e., errors that happen while the command group is
   * executing on the device and cannot trigger a normal exception).
   * The async_handler is called when the cl::sycl::queue::wait_and_throw()
   * method is invoked. Note that exceptions are ignored if no wait and
   * throw is issued.
   *
   * This async handler takes the list of asynchronous exceptions
   * that have been captured by the runtime while executing the kernels.
   * a cl::sycl::exception_list object is a collection of std::exception_ptrs.
   * std::exception_ptrs can only be rethrown, at which point the exception
   * can be caught as usual and handled through the regular try/catch pair.
   *
   * In this example we re-throw the first exception and capture it to
   * set a boolean variable. It would be possible to catch every single
   * exception individually from the list by reordering the code such that
   * the for loop encapsulates the try/catch. */
  auto asyncHandler = [&error, &nTimesCall](cl::sycl::exception_list eL) {
    ++nTimesCall;
    for (auto& e : eL) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception& e) {
        /* We set to true since we have caught a SYCL exception.
         * More complex checking of error classes can be done here,
         * like checking for specific SYCL sub-classes or
         * recovering a potential underlying OpenCL error. */
        error = true;
        std::cout << " I have caught an exception! " << std::endl;
        std::cout << e.what() << std::endl;
      }
    }
  };

  /* This command group causes an asynchronous SYCL error, since
   * the global size cannot be evenly divided by the local size
   * (in fact, the local size is also orders of magnitude larger).
   * Either of these is considered an error and will be passed to
   * the async_handler. */
  auto cgh_error = [&](handler& cgh) {
    auto myRange = nd_range<2>(range<2>(6, 2), range<2>(20000, 20000));

    cgh.parallel_for<class kernel0>(myRange, [=](nd_item<2> itemID) {});
  };

  default_selector mySelector;

  /* Construct a queue with an async_handler. */
  {
    queue myQueue(mySelector, asyncHandler);
    myQueue.submit(cgh_error);

    /* The asynchronous handler is called at this point.
     * Exceptions are migrated to the caller thread. */
    myQueue.wait_and_throw();
  }

  /* Construct a context with an async_handler. */
  {
    context myContext(asyncHandler);
    queue myQueue(myContext, mySelector);
    myQueue.submit(cgh_error);

    /* The asynchronous handler is called at this point.
     * Exceptions are migrated to the caller thread. */
    myQueue.wait_and_throw();
  }

  /* Construct a context without an async_handler. */
  {
    context myContext(mySelector, false);
    queue myQueue(myContext, mySelector);
    myQueue.submit(cgh_error);

    /* Even though the method says "throw", since no asynchronous handler was
     * specified when constructing the queue, no exceptions will be returned
     * to the user and they are considered lost. */
    myQueue.wait_and_throw();
  }

  std::cout << " The asynchronous handler has been called " << nTimesCall
            << " times " << std::endl;

  /* It is expected that by this point the handler should have been called
   * twice, even though there were 3 erroneous command groups issued to
   * the runtime. */
  return !((nTimesCall == 2) && error);
}
