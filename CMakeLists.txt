cmake_minimum_required(VERSION 3.2.2)

project(ComputeCpp-SDK)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(COMPUTECPP_SDK_USE_OPENMP "Enable OpenMP support in samples" OFF)

# Set the host compiler C++ standard to C++ 14
# This should be set as a global property when we update the CMake version
set(CMAKE_CXX_STANDARD 14)

# Set include ComputeCpp directories
include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})

enable_testing()

add_subdirectory(samples)
if(COMPUTECPP_SDK_BUILD_TESTS)
  add_subdirectory(tests)
endif()
