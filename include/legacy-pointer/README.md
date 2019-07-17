# Legacy Pointer Interface Utility

This folder contains a legacy pointer interface utility header, that
facilitates integrating SYCL applications into codebases that rely on
malloc/free allocations on the device.

The utility header offers an interface to simluate the traditional C
behaviour of malloc a pointer on the device and free it later.

The malloc returns a non-dereferenciable pointer that can be used later
in SYCL code to retrieve the buffer it refers to, plus an offset into
it.

## Usage
---
Include the `legacy_pointer.hpp` file in your program, and replace your
device malloc/free operations with `codeplay::legacy::malloc/free`

To retrieve the buffer from the pointer, just use the function
`codeplay::legacy::PointerMapper::get_buffer_id` to obtain the
buffer id from the PointerMapper class, and then the method
`codeplay::legacy::PointerMapper::get_buffer` to obtain the SYCL
buffer.

## Building tests
---
```bash
mkdir build && cd build # in SDK root
cmake .. -DComputeCpp_DIR=/path/to/computecpp -DCOMPUTECPP_SDK_BUILD_TESTS=ON
make legacy-basic legacy-offset
```
