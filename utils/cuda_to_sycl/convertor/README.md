# The SYCL Convertor

An `*.cu` file is a combination of Host CUDA API and device CUDA kernel.

The SYCL convertor executes a SYCL `*.cpp` file which is converted from 
a CUDA `*.cu` file, by using the followign rules.

1. For the device kernel, we will introduce a SYCL functor that will encapsulate 
all the CUDA code, enabling direct re-use of CUDA kernels without major 
modifications whenever using simple kernel cases.

1. For the host code, the CUDA API is replaced with equivalent SYCL/C++ code. 

## CUDA device kernel

For example:
 
```cpp
__global__ void vecAdd(double *a, double *b, double *c, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  double *smem = SharedMemory<double>();
  if (id < n) {
    smem[threadIdx.x] = a[id] + b[id];
    c[id] = smem[threadIdx.x];
  }
}
```
should be refactored as: 

```cpp
// Generated class:: Kernel dispatch.
// This signature must be variadic
template <typename... Args>
struct ___CudaConverterFunctor___vecAdd
    : public cl::sycl::codeplay::Generic_Kernel_Functor<
          ___CudaConverterFunctor___vecAdd<Args...>> {
  using parent = cl::sycl::codeplay::Generic_Kernel_Functor<
      ___CudaConverterFunctor___vecAdd<Args...>>;
  using parent::__syncthreads;
  using parent::blockDim;
  using parent::blockIdx;
  using parent::gridDim;
  using parent::threadIdx;

  ___CudaConverterFunctor___vecAdd(Args... args) : parent(args...) {}
  // if shared memory is used this code will be added
  template <typename T>
  T* SharedMemory() {
    return parent::template get_local_mem<T>();
  }
  // kernel executor
  template <typename... params_t>
  void __execute__(params_t... params) {
    vecAdd(params...);
  }
  // The original cuda kernel
  __global__ void vecAdd(double* a, double* b, double* c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    double* smem = SharedMemory<double>();
    if (id < n) {
      smem[threadIdx.x] = a[id] + b[id];
      c[id] = smem[threadIdx.x];
    }
  }
};
```

## CUDA host API conversion

* Memory creation:
```cpp
   double *d_a;
    cudaMalloc(&d_a, bytes);  
```
should be refactored as:
```cpp
    // Device input vectors
    double *d_a;
    d_a = static_cast<double*> (cl::sycl::codeplay::SYCLmalloc(sizeof(bytes), 
                                    cl::sycl::codeplay::get_global_pointer_mapper()));
```
* Explicit memory operation: 

`cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);`

should be refactored as

 `cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::HostToDevice>(deviceQueue, h_a, d_a, bytes,
                                              true);`


`cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );`
 
 should be refactored as:

 `   cl::sycl::codeplay::cuda_copy_conversion<
      cl::sycl::codeplay::Kind::DeviceToHost>(deviceQueue, d_c, h_c, bytes,
                                              true);`

* CUDA chevron 

For example: 

`vecAdd<<<gridSize, blockSize, sharedmem>>>(d_a, d_b, d_c, n);`

should be refactored as:

`deviceQueue.submit(
      cl::sycl::codeplay::CudaCommandGroup<
          ___CudaConverterFunctor___vecAdd<double*, double*, double*, int>>(
          gridSize, blockSize, sharedmem, d_a, d_b, d_c, n));`


# Convertor backend:

The SYCL kernel functor is inherited the SYCL generic functor.
The command group captures the SYCL functor kernel as a variadic nested template type, with the types requited for cuda device kernel.
It reconstructs the SYCL kernel functor type with the extra types required to execute the converted SYCL kernel functor and instantiate it
the SYCL dispatcher by passing the new type constructed for SYCL kernel functor.
The kernel dispatcher then instantiates the new constructed SYCL kernel functor inside the functor operator when the device kernel is called. 
At this time the nd_item is provided and each thead can construct their threadIdx, blockIdx. Also they can call the __synch_threds() to access the barrier and executes any cuda kernels embeded in the SYCL kernel functor.

# Limitations

* Not all input CUDA code can be automatically converted into SYCL code, since there is not always a one-to-one mapping between the two, or the mapping is not obvious. In particular, CUDA kernels heavily optimized for a specific CUDA architecture would need to be re-written manually to achieve comparable performance on the target architecture, even if they can be converted directly.

* This document also focuses on use cases where only one CUDA stream is present (the default stream). Multiple stream code is simpler to translate, since the SYCL queue can be mapped directly to a CUDA stream. However this code is difficult to read, so has been left out of the code examples and descriptions.

* Vendor or device specific extensions on the destination SYCL platform are not covered by this document. Any CUDA capability that cannot be translated into standard SYCL code is not supported.

* Local memory access in CUDA must be through using a utility class.
for example: 

```cpp
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
// specialize it based on type (specifically double) to avoid unaligned memory
// access compile errors
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};
```


# Building and executing the examples:

* build:

 COMPUTECPP_DIR=/path/to/computecpp/ make

 * execute
  
  ./add

  ./add_stride

