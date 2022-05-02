cmake_minimum_required(VERSION 3.10.2)
project(ComputeCpp-SDK)

option(COMPUTECPP_SDK_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(COMPUTECPP_SDK_USE_SUBGROUPS "Enable subgroup support in samples" OFF)
option(COMPUTECPP_SDK_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)
option(COMPUTECPP_SDK_BUILD_DEMOS "Build the SDK demos" OFF)

if(COMPUTECPP_SDK_BUILD_DEMOS AND
    CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
  # Policy required to enable MSVC_RUNTIME_LIBRARY property
  # This is only needed when building the demos.
  cmake_policy(SET CMP0091 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND COMPUTECPP_USER_FLAGS "-no-serial-memop")
find_package(ComputeCpp REQUIRED)

enable_testing()

add_subdirectory(samples)

if(COMPUTECPP_SDK_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(COMPUTECPP_SDK_BUILD_DEMOS)
  add_subdirectory(demos)
endif()
