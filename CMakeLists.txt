cmake_minimum_required(VERSION 3.4.3)

if (COMPUTECPP_SDK_BUILD_DEMOS)
  # Policy required to enable MSVC_RUNTIME_LIBRARY property
  # This needs to be set before the first project for the policy
  # to have an effect. This is only needed when building the demos
  cmake_policy(SET CMP0091 NEW)
endif()

project(ComputeCpp-SDK)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(COMPUTECPP_SDK_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(COMPUTECPP_SDK_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)
option(COMPUTECPP_SDK_BUILD_DEMOS "Build the SDK demos" OFF)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

add_subdirectory(samples)
if(COMPUTECPP_SDK_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if (COMPUTECPP_SDK_BUILD_DEMOS)
  add_subdirectory(demos)
endif()
