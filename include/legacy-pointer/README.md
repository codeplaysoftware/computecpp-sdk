This folder contains a legacy pointer interface utility header, that
facilitates integrating SYCL applications into codebases that rely on
malloc/free allocations on the device.

The utility header offers an interface to simluate the traditional C
behaviour of malloc a pointer on the device and free it later.

The malloc returns a non-dereferenciable pointer that can be used later
in SYCL code to retrieve the buffer it refers to, plus an offset into
it.

Usage
=====

Include the *legacy\_pointer.hpp* file in your program, and replace your
device malloc/free operations with **codeplay::legacy::malloc/free**

To retrieve the buffer from the pointer, just use the function
**codeplay::legacy::PointerMapper::get\_buffer\_id** to obtain the
buffer id from the PointerMapper class, and then the method
**codeplay::legacy::PointerMapper::get\_buffer** to obtain the SYCL
buffer.

Building tests
==============

1.  mkdir build \# in SDK root

2.  cd build

3.  cmake ../
    -DCOMPUTECPP\_PACKAGE\_ROOT\_DIR=/path/to/computecpp/package \\
    -DCOMPUTECPP\_SDK\_BUILD\_TESTS=1

4.  make legacy-basic legacy-offset
