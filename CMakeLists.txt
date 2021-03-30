cmake_minimum_required(VERSION 3.4.3)

project(ComputeCpp-SDK)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(COMPUTECPP_SDK_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(COMPUTECPP_SDK_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)
option(COMPUTECPP_SDK_INCLUDE_AFTER "Force integration header to be included after main" OFF)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

add_subdirectory(samples)
if(COMPUTECPP_SDK_BUILD_TESTS)
  add_subdirectory(tests)
endif()
