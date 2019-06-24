## ComputeCpp SDK - Code Samples

| Sample File | Sample Description |
| ------------- | ------------- |
| `accessors.cpp`  | Sample code that illustrates how to make data **available on a device** using **accessors** in SYCL.  |
| `parallel-for.cpp`  | Sample code showing how to use the **parallel_for** API in SYCL.  |
| `simple-vector-add.cpp`  | A "Hello World" example of a **simple vector addition** in SYCL.  |
| `example-sycl-application.cpp`  | Another "Hello World" sample code that **walks through the basics** of executing a matrix add in SYCL.  |
| `simple-example-of-vectors.cpp`  | Example of **vector** operations in SYCL - also demonstrates **swizzles**.  |
| `sync-handler.cpp`  | Sample code that demonstrates the use of a **synchronous error handler** to demonstrate error handling.  |
| `async-handler.cpp`  | Sample code that demonstrates the use of an **asynchronous handler** for exceptions in SYCL.  |
| `simple-local-barrier.cpp`  | Sample code demonstrating a basic use of **local barriers** in device code.  |
| `custom-device-selector.cpp`  | Sample code that shows how to write a **custom device selector** in SYCL.  |
| `ivka.cpp`  | Sample showing the different kinds of things that are **valid** and **not valid** when used as **kernel arguments**.  |
| `reduction.cpp`  | Example of a **reduction operation** in SYCL. It is a good demonstration of **memory management** in SYCL. |
| `matrix-multiply.cpp`  | Example of **tiled matrix multiplication** in SYCL. It compares similarities/differences to an alternative OpenMP implementation.  |
| `monte-carlo-pi.cpp`  | Example of Monte-Carlo Pi approximation algorithm in SYCL. It also demonstrates how to **query** the **maximum** number of **work-items** in a work-group to check if a kernel can be executed with the initially desired work-group size.  |
| `simple-private-memory.cpp`  | Sample showing how to utilize private memory on a device using the hierarchical API in SYCL.  |
| `images.cpp`  |  Sample code that demonstrates a **basic** use of SYCL **image** and **sampler** objects.  |
| `gaussian-blur.cpp`  | This code implements a **Gaussian Blur filter**, blurring a JPG or PNG image from the command line. The original image file is not modified. Demonstrates a **more advanced** use of SYCL **image** and **sampler** objects. |
| `reinterpret.cpp`  | Sample code showing the **reinterpret buffer** feature of SYCL 1.2.1. Buffers of **one type** can be **transformed** into buffers of **another type**, similar to C++'s reinterpret_cast feature. |
| `scan.cpp`  | Example of a **parallel inclusive scan** with a given associative **binary operation** in SYCL.  |
| `placeholder-accessors.cpp`  | Sample code that illustrates how to use placeholder accessors.  |
| `using-function-objects.cpp`  | Sample code that demonstrates how to use **function objects as kernels** in SYCL.  |
| `template-function-object.cpp`  |  Sample code that demonstrates how to **template function objects as kernels** in SYCL.  |
| `builtin-kernel-example.cpp`  | Example of using an **OpenCL built-in kernel** with SYCL via the codeplay extension.  |
| `use-onchip-memory.cpp`  | Sample code that demonstrates the use of the **use_onchip_memory extension** to SYCL provided by ComputeCpp.  |
| `smart-pointer.cpp`  | Sample code that shows how SYCL can use custom allocators. <br> Dependency: `stack_allocator.hpp` |
| `vptr.cpp`  | Sample code that demonstrates the use of the **virtual pointer interface** in SYCL on matrix addition. <br> Dependency: `/include/vptr/virtual_ptr.hpp` |
| `opencl-c-interop.cpp`  | Sample code that shows the **interoperability** between **OpenCL and SYCL**.  |
