## ComputeCpp SDK - Code Samples

### SYCL Basics

Sample File                     | Description 
------------------------------- | ---------------------------------------------------------------------------------------------------------
accessors.cpp                   | Making data available on a device using accessors in SYCL.
parallel-for.cpp                | Using the parallel_for API in SYCL.
simple-vector-add.cpp           | A "Hello World" example of a simple vector addition in SYCL.
example-sycl-application.cpp    | Another "Hello World" example that walks through the basics** of executing a matrix add in SYCL.
simple-example-of-vectors.cpp   | Example of vector operations in SYCL - also demonstrates **swizzles.
sync-handler.cpp                | Synchronous error handling in SYCL.
async-handler.cpp               | Asynchronous error handling in SYCL.
simple-local-barrier.cpp        | Basic use of local barriers in device code.
using-function-objects.cpp      | Using function objects as kernels in SYCL.
template-function-object.cpp    | Using template function objects as kernels in SYCL.
reduction.cpp                   | Example of a reduction operation in SYCL. A good of memory management in SYCL.
matrix-multiply.cpp             | Tiled matrix multiplication - comparing SYCL and OpenMP implementations.
simple-private-memory.cpp       | Utilizing private memory on a device using the hierarchical API in SYCL.
images.cpp                      | Basic use of SYCL image and sampler objects.
custom-device-selector.cpp      | Writing a custom device selector in SYCL.
ivka.cpp                        | Sample showing the different kinds of things that are valid and not valid when used as kernel arguments.

---

### More Advanced SYCL

Sample File                     | Description 
------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------
monte-carlo-pi.cpp              | Monte-Carlo Pi approximation algorithm in SYCL. Shows how to query the maximum number of work-items in a work-group to check if a kernel can be executed with the initially desired work-group size.
scan.cpp                        | Example implementation of a parallel inclusive scan with a given associative binary operation in SYCL.
gaussian-blur.cpp               | This sample implements a Gaussian Blur filter, blurring a JPG or PNG image from the command line. The original image file is not modified. More advanced use of SYCL image and sampler objects.
reinterpret.cpp                 | Demonstration of the buffer reinterpret feature of SYCL 1.2.1. Buffers of one type can be transformed into buffers of another type, similar to reinterpret_cast feature in standard C++.
placeholder-accessors.cpp       | Example use of SYCL placeholder accessors.
builtin-kernel-example.cpp      | Using an OpenCL built-in kernel with SYCL via the codeplay extension.
use-onchip-memory.cpp           | Demonstrating the use_onchip_memory extension to SYCL provided by ComputeCpp.
smart-pointer.cpp               | Custom Allocators in SYCL. Dependency: *stack_allocator.hpp*
vptr.cpp                        | Using the Virtual Pointer interface in SYCL on matrix addition kernel. Dependency: */include/vptr/virtual_ptr.hpp*
opencl-c-interop.cpp            | OpenCL/SYCL interopability example.
