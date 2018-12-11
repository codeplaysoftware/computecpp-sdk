/***************************************************************************
 *
 *  Copyright Codeplay Software Limited
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
 *  example_vptr.cpp
 *
 *  Description:
 *  Sample code that demonstrates the use of the use_onchip_memory extension
 *  to SYCL provided by ComputeCpp.
 *
 **************************************************************************/

#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>
#include <iostream>

namespace sycl = cl::sycl;
namespace access = sycl::access;

// Sample kernel
// -------------
// Since kernels are no longer allowed to be declared inline, I dislike
// declaring a random name in the global namespace that can potentially be
// repurposed.
//
// This kernel performs the same operation as std::iota, but also scales
// the result by two.
//
class scaled_iota {
 public:
  using accessor = sycl::accessor<int, 1, access::mode::discard_write,
                                  access::target::global_buffer>;

  explicit scaled_iota(accessor deviceAccess) noexcept
      : m_deviceAccess{std::move(deviceAccess)} {}

  void operator()(sycl::nd_item<2> itemId) const noexcept {
    const auto linearId = itemId.get_global_linear_id();
    m_deviceAccess[linearId] = linearId * 2;
  }

 private:
  accessor m_deviceAccess;
};

namespace codeplay = sycl::codeplay;

template <class Policy>
void use_with_policy(Policy policy, sycl::queue& queue) {
  auto hostData = sycl::vector_class<int>(1024);
  {
    auto taskContext = queue.get_context();

    // Notice that the Codeplay property takes a policy argument: this is used
    // to indicate whether the property is advantageous or genuinely necessary.
    //
    auto deviceData = sycl::buffer<int, 1>{
        hostData.data(),                  //
        sycl::range<1>(hostData.size()),  //
        sycl::property_list{
            sycl::property::buffer::context_bound(taskContext),
            codeplay::property::buffer::use_onchip_memory(policy)  // <--------
        }  //                                            ^---------------------
    };

    deviceData.set_final_data(hostData.data());

    queue.submit([&](sycl::handler & cgh) {
      constexpr auto dimension_size = 2;
      auto r = sycl::nd_range<dimension_size>{
          sycl::range<dimension_size>{
              hostData.size() / dimension_size,  //
              dimension_size                     //
          },
          sycl::range<dimension_size>{dimension_size, 1}  //
      };
      cgh.parallel_for(
          r,
          scaled_iota(deviceData.get_access<access::mode::discard_write>(cgh)));
    });

    queue.wait_and_throw();
  }
}

// Codeplay policy extensions have two different enabling mechanisms: the first
// is to indicate that a policy is preferred. Using this policy means that if
// the system supports the feature, then the feature will be enabled. If the
// feature is not present on the system, then it will not be enabled.
//
// Puns aside, this is the preferred default.
//
void how_to_use_with_prefer() {
  auto queue = sycl::queue{};
  ::use_with_policy(codeplay::property::prefer, queue);
}

// Alternatively, if you can guarantee that your system will support this
// policy, or if it is expected any system using your software must support the
// policy, then you can use the require tag to indicate that the feature is
// required by your software.
//
// In the event that the property isn't supported, an exception will be thrown.
//
void how_to_use_with_require() {
  try {
    auto queue = sycl::queue{};
    ::use_with_policy(codeplay::property::require, queue);
  } catch (const sycl::exception& e) {
    std::cerr << "An error occurred: " << e.what()
              << "\n"
                 "\n"
                 "This particular error has occurred because you are requiring "
                 "the policy use_onchip_memory be available, and your hardware "
                 "doesn't support the use_onchip_memory, so the SYCL ecosystem "
                 "will raise an error.\n";
  }
}

int main() {
  // Using the on-chip memory policy with the require tag.
  how_to_use_with_require();

  // Using the on-chip memory policy with the prefer tag.
  how_to_use_with_prefer();
}
