#include <CL/sycl.hpp>

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
void add(cl::sycl::nd_item<1> item, int n,
	     cl::sycl::global_ptr<float> x, cl::sycl::global_ptr<float> y)
{
	int index = item.get_local(0);
	int stride = item.get_local_range(0);
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	int N = 1 << 20; // 1M elements
	// encapsulate data in SYCL buffers
	cl::sycl::buffer<float> x(N);
	cl::sycl::buffer<float> y(N);

	// initialize x and y arrays on the host
	{
		auto px = x.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
		auto py = y.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
		for (int i = 0; i < N; i++) {
			px[i] = 1.0f;
			py[i] = 2.0f;
		}
	}

	{ // create a scope to define the lifetime of the SYCL objects
		cl::sycl::gpu_selector selectgpu;
		cl::sycl::device gpu_device(selectgpu);
		cl::sycl::queue gpu_queue(gpu_device);
		gpu_queue.submit([&](cl::sycl::handler &cgh) {
			auto aX = x.get_access<cl::sycl::access::mode::read>(cgh);
			auto aY = y.get_access<cl::sycl::access::mode::read_write>(cgh);
			// Run kernel on 1M elements on the OpenCL GPU
			cgh.parallel_for<class add_functor>(
				cl::sycl::nd_range<1>(cl::sycl::range<1>(256), 
				                      cl::sycl::range<1>(256)),
				[=](cl::sycl::nd_item<1> it) {
					add(it, N, aX, aY);
				});
		});
	}

	{ // Check for errors (all values should be 3.0f)
		auto py = y.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
		float maxError = 0.0f;
		for (int i = 0; i < N; i++)
			maxError = fmax(maxError, fabs(py[i] - 3.0f));
		std::cout << "Max error: " << maxError << std::endl;
	}

	// Free memory: destructors do this automatically
	return 0;
}


