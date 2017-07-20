#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(cl::sycl::nd_item<1> item, int n, cl::sycl::global_ptr<float> x,
         cl::sycl::global_ptr<float> y) {
  int index = item.get_global_linear_id();
  int stride = item.get_global_range(0);
  for (int i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

using namespace cl::sycl::access;

int main(void) {
  int N = 1 << 20;  // 1M elements

  // encapsulate data in SYCL buffers
  cl::sycl::buffer<float> x(N);
  cl::sycl::buffer<float> y(N);

  // initialize x and y arrays on the host
  {
    auto px = x.get_access<mode::write, target::host_buffer>();
    auto py = y.get_access<mode::write, target::host_buffer>();
    for (int i = 0; i < N; i++) {
      px[i] = 1.0f;
      py[i] = 2.0f;
    }
  }

  { // create a scope to define the lifetime of the SYCL objects
    // create a SYCL queue for a GPU
    cl::sycl::queue gpu_queue((cl::sycl::gpu_selector()));

    // submit this work to the SYCL queue
    gpu_queue.submit([&](cl::sycl::handler &cgh) {
      // request access to the data on the OpenCL GPU
      auto aX = x.get_access<mode::read>(cgh);
      auto aY = y.get_access<mode::read_write>(cgh);

      // Run kernel on 1M elements on the OpenCL GPU
      cgh.parallel_for<class add_functor>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(N), cl::sycl::range<1>(256)),
          [=](cl::sycl::nd_item<1> it) { add(it, N, aX, aY); });
    });
  }

  // Check for errors (all values should be 3.0f)
  auto py = y.get_access<mode::read, target::host_buffer>();
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) maxError = fmax(maxError, fabs(py[i] - 3.0f));

  std::cout << "Max error: " << maxError << std::endl;
  // Free memory: destructors do this automatically

  return 0;
}
