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
 *  smart-pointer.cpp
 *
 *  Description:
 *    Sample code that shows how SYCL can use custom allocators.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

// Custom stack allocator
#include "stack_allocator.hpp"

using namespace cl::sycl;

int main() {
  const unsigned int nElems = 12;
  std::shared_ptr<int> p(new int[nElems]);
  bool correct = true;
  for (unsigned int i = 0; i < nElems; i++) {
    p.get()[i] = 0;
  }

  queue myQueue;

  {
    /* Buffers can take a shared_ptr as a parameter, and they
     * will share ownership of the pointer.
     * Data will be copied back only if the user still keeps
     * a reference of the data. */
    {
      buffer<int, 1> buf(p, range<1>(nElems));

      myQueue.submit([&](handler& cgh) {
        auto myRange = nd_range<2>(range<2>(6, 2), range<2>(2, 1));

        auto ptr = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class kernel0>(myRange, [=](nd_item<2> itemID) {
          ptr[itemID.get_global_linear_id()] =
              (int) (itemID.get_global_linear_id());
        });
      });
      {
        /* The runtime will make data available via hA. It might use mapped
         * memory, temporary objects or internal allocations. It is up to
         * the implementation. The original pointer is not updated. */
        auto hA = buf.get_access<access::mode::read>();

        int sum = 0;
        for (unsigned int i = 0; i < nElems; i++) {
          sum += hA[i];
        }

        if (sum != 66) {
          correct = false;
        }
      }
    }

    /* Data now available in the original pointer, because the buffer
     * has been destroyed. */
    int sum = 0;
    for (unsigned int i = 0; i < nElems; i++) {
      sum += p.get()[i];
    }

    if (sum != 66) {
      correct = false;
    }
  }

  {
    /* Custom allocators can be used - here we use a stack_allocator from
     * https://github.com/charles-salvia/charles/blob/master/stack_allocator.hpp
     */
    {
      buffer<int, 1, stack_allocator<int, nElems>> buf{range<1>{nElems}};
      /* buffer::set_final_data() tells the runtime that the data should be
       * copied to p when the buffer is destroyed. */
      buf.set_final_data(p);

      myQueue.submit([&](handler& cgh) {
        auto myRange = nd_range<2>(range<2>(6, 2), range<2>(2, 1));

        auto ptr = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class kernel1>(myRange, [=](nd_item<2> itemID) {
          ptr[itemID.get_global_linear_id()] = itemID.get_global_linear_id();
        });
      });
    }

    /* The buffers have now been destroyed, and the data copied in to p. */
    int sum = 0;
    for (unsigned int i = 0; i < nElems; i++) {
      sum += p.get()[i];
    }

    if (sum != 66) {
      correct = false;
    }
  }

  {
    {
      /* The property "use_host_ptr" tells the runtime that the user
       * pointer passed to the constructor should be used to store all
       * data, rather than new internal allocations. When using this,
       * all host accessors update the user-given host memory. This can
       * improve performance, though you should always profile to see
       * if it actually makes a difference. */
      buffer<int, 1> buf(p, range<1>(nElems),
                         {property::buffer::use_host_ptr()});

      myQueue.submit([&](handler& cgh) {
        auto myRange = nd_range<2>(range<2>(6, 2), range<2>(2, 1));

        auto ptr = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class kernel2>(myRange, [=](nd_item<2> itemID) {
          ptr[itemID.get_global_linear_id()] =
              (int) (itemID.get_global_linear_id());
        });
      });

      {
        /* Host accessors will actually block on creation, so in this case we
         * know kernel2 has finished by the time hA is available. */
        auto hA = buf.get_access<access::mode::read>();

        int sum = 0;
        for (unsigned int i = 0; i < nElems; i++) {
          sum += p.get()[i];
        }

        if (sum != 66) {
          correct = false;
        }
      }
    }
    /* Normally data is copied to the host when buffers are destroyed, which
     * would happen here, but since a map_allocator and host_accessor were
     * used in concert no data copy happens (i.e., it has already happened. */
  }

  return correct ? 0 : 1;
}
