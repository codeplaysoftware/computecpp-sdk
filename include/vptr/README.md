# Software Managed Virtual Pointer

This folder contains the implementation of a software-managed virtual
pointer that facilitates a raw pointer interface for SYCL buffers. This
virtual address space is non-dereferenceable on the host, but accessors
can be obtained in the SYCL command groups.

This differs from the `legacy_pointer` interface in that a contiguous
virtual address space of `sizeof(size_t)` bits is created. When
allocating a buffer using `SYCLmalloc`, a virtual address in the range
of 1 to 2 <sup>`sizeof(size_t)`</sup> is returned. This is, in
practice, a number that identifies a certain SYCL buffer. The pointer
can be offset, so that positions inside a given buffer can be identified
and passed around in higher-level functions. This enables pointer
arithmetic on the virtual addresses. From any virtual pointer address,
the methods `get_buffer` and `get_offset` can always be used to retrieve
the SYCL buffer associated with it, together with the offset from the
base address of the given buffer.
```cpp
// Create the Pointer Mapper structure
PointerMapper pMap;
// Create a SYCL buffer of 10 floats
// This pointer is a number that identifies the buffer
// in the pointer mapper structure
float * a = static_cast<float *>(SYCLmalloc(10 * sizeof(float), pMap));
// Create a SYCL buffer of 25 integers
int * b = static_cast<int *>(SYCLmalloc(25 * sizeof(int), pMap));
// Create a pointer to the 5th element
// This simply adds 5 * sizeof(float *) to the base address.
float * c = a + 5;
// Retrieve the buffer
assert(pMap.get_buffer(a) == pMap.get_buffer(c))
// Substracting the value of the offset from the base address of the
// buffer recovers the offset into it
assert(pMap.get_offset(c) == 5 * sizeof(float))

// Invalid usage: no-dereference on the host
// float myVal = *c;
// Valid access on host: Use host-accessor
{
  auto syclAcc = pMap.get_buffer(a).get_access<access::mode::read>();
  float myVal = syclAcc[0];
}

// Free the pointers
SYCLfree(a, pMap);
SYCLfree(b, pMap);
```
Developers looking for a simple replacement of malloc/free functions
should use the Legacy Pointer interface. In situations where developers
need complete addressing of the entire device memory space, developers
should use this software-managed virtual pointer.

Note that multiple PointerMapper objects can be instantiated
simultaneously.

## Usage
---
Include the `virtual_ptr` header file in your program. Replace your
device `malloc` and `free` operations with `codeplay::SYCLmalloc`
and `codeplay::SYCLfree`. These functions are not thread-safe, even
though the underlying SYCL buffer objects are thread-safe.

To retrieve the SYCL buffer from the virtual pointer, use the
`codeplay::PointerMapper::get_buffer` function. The offset into the
SYCL buffer on the device side can be retrieved using the
`codeplay::PointerMapper::get_offset` function. A pointer of size
zero can be malloc’ed and free’d but will throw an exception if
accessed.

See the tests for basic usage examples. Note that the pointer cannot be
dereferenced on the host, but host accessors can be constructed once the
buffer is retrieved.

## Experimental ComputeCpp Integration
---

ComputeCpp (since version 0.2.1) supports integration with the PointerMapper.
This is an experimental feature at the moment, and not intended for
general usage. To enable testing of the experimental interface support,
pass `COMPUTECPP_INTERFACE=ON` to the CMake configuration line.

## Building tests
---
```bash
mkdir build && cd build # in SDK root
cmake .. -DComputeCpp_DIR=/path/to/computecpp -DCOMPUTECPP_SDK_BUILD_TESTS=ON
make
```
