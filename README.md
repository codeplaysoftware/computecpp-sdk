# ComputeCpp SDK Readme

![ComputeCpp™]

## Introduction

[![Build Status]](https://travis-ci.org/codeplaysoftware/computecpp-sdk)

This is the README for the ComputeCpp SDK, a collection of sample code,
utilities and tools for Codeplay’s ComputeCpp, an implementation of the
SYCL programming standard. You can find more information at:

* The Codeplay developer website http://developer.codeplay.com
* The SYCL community website http://sycl.tech

## Contents

* CMakeLists.txt
  * The entry point for this project’s CMake configuration. Adds `samples/`
    subdirectory and optionally adds tests.
* LICENSE.txt
  * The license this package is available under: Apache 2.0
* README.md
  * This readme file
* cmake/
  * Contains a CMake module for integrating ComputeCpp with existing
    projects. See later in this document for a description of the
    CMake module provided. There are also toolchains for generic
    GCC-like setups as well as the poky toolchain.
* include/
  * add this directory to your include search path to be able to use
    SDK code in your own projects (for example, the virtual pointer utility).
* samples/
  * A collection of sample SYCL code, tested on ComputeCpp, provided
    as a learning resource and starting point for new SYCL software.
    The samples are built with CMake.
* demos/
  * A collection of graphical applications that use SYCL to accelerate
    calculations.
* tests/
  * Tests for the utilities in the SDK.
* tools/
  * Useful tools for working with SYCL code and ComputeCpp. Includes
    a simple driver script for compiling SYCL code with ComputeCpp,
    an example Makefile and other utilities.
* util/
  * Example utility code that we expect will be useful for other
    projects as a starting point.

## Supported Platforms

The master branch of computecpp-sdk is regularly tested with the "Supported" 
hardware listed on [the ComputeCpp Supported Platforms page]. 

## Pre-requisites

* The samples should work with any SYCL 1.2.1 implementation, though
  have only been tested with ComputeCpp.
* OpenCL 1.2-capable hardware and drivers with SPIR 1.2/SPIR-V support
* C++11-compliant compiler and libstdc++ on GNU/Linux (GCC 4.9+, Clang 3.6+)
* Microsoft Visual C++ 2015/2017 on Windows
* CMake 3.4.3 and newer

## Getting Started

On the Codeplay website there is a [step-by-step guide] to building and running 
the ComputeCpp SDK samples. There is also a [guide to SYCL] that serves as an
introduction to SYCL development. Additionally, there is an
[Integration guide] should you wish to add ComputeCpp to existing projects.

## Setup

CMake files are provided as a build system for this software. CMake
version 3.4.3 is required at minimum, though later versions of CMake
should continue to be compatible.

At minimum, one CMake variable is required to get the sample code
building - `ComputeCpp_DIR`. This variable should point to the root
directory of the ComputeCpp install (i.e. the directory with the the
folders bin, include, lib and so on). You can also specify
`COMPUTECPP_SDK_BUILD_TESTS` to add the tests/ subdirectory to the
build, which will build Gtest-based programs testing the legacy pointer
and virtual pointer classes. Some samples have optional OpenMP support.
You can enable it by setting `COMPUTECPP_SDK_USE_OPENMP` to ON in
CMake.

You can additionally specify `CMAKE_BUILD_TYPE` and
`CMAKE_INSTALL_PREFIX` to choose a Debug or Release build and the
location you’d like to be used when the "install" target is built. The
install target currently will copy all the sample binaries to the
directory of your choosing.

Lastly, the SDK will build targeting `spir64` IR by default. This will
work on most devices, but won’t work on NVIDIA (for example). To that
end, you can specify `-DCOMPUTECPP_BITCODE=target`, which can be any of
`spir[64]`, `spirv[64]` or `ptx64`.

If you would like to crosscompile the SDK targeting some other platform,
there are toolchain files available in the cmake/toolchains directory.
They require certain variables pointing to the root of the toolchain you
are using to be set in the environment. They cannot be specificed in the
CMake cache. The toolchains will identify which variables have not been
set when used.

## Troubleshooting

The sample code should compile without error on our supported platforms.
If you run into trouble, or think you have found a bug, we have a support
forum available through the [ComputeCpp website].

## Maintainers

This SDK is maintained by [Codeplay Software Ltd.]
If you have any problems, please contact sycl@codeplay.com.

## Acknowledgements

This repository contains code written by Sean Barrett (`stb_image` code
in the Gaussian blur sample) and Charles Salvia (the stack allocator
used in the smart pointer sample). Please see the files for their
respective licences.

## Contributions

This SDK is licensed under the Apache 2.0 license. Patches are very
welcome! If you have an idea for a new sample, different build system
integration or even a fix for something that is broken, please get in
contact.

[ComputeCpp™]: https://www.codeplay.com/public/uploaded/public/computecpp_logo.png
[Build Status]: https://travis-ci.org/codeplaysoftware/computecpp-sdk.svg?branch=master
[step-by-step guide]: https://developer.codeplay.com/computecppce/latest/getting-started-guide
[Integration guide]: https://developer.codeplay.com/computecppce/latest/integration-guide
[FAQ page]: https://developer.codeplay.com/computecppce/latest/faq
[Codeplay Software Ltd.]: https://www.codeplay.com
[ComputeCpp website]: https://developer.codeplay.com
[guide to SYCL]: https://developer.codeplay.com/products/computecpp/ce/guides/sycl-guide
[the ComputeCpp Supported Platforms page]: https://developer.codeplay.com/products/computecpp/ce/guides/platform-support