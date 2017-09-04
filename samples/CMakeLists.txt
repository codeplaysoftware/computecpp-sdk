project(ComputeCpp-samples-sdk)

# Set minimum required version of CMake to 3.2.2
cmake_minimum_required(VERSION 3.2.2)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake/Modules)
include(FindComputeCpp)

# Include the path to the virtual pointer
set(COMPUTECPP_SDK_VIRTUAL_POINTER_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../utils/vptr/include)
include_directories(${COMPUTECPP_SDK_VIRTUAL_POINTER_PREFIX})

# Set the host compiler C++ standard to C++11
# This should be set as a global property, since it can be
# set as a target property, but this doesn't work in 3.2.2
# testing locally. Instead, we set this directly.
set(CMAKE_CXX_STANDARD 11)

# Set include ComputeCpp directories
include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})

# warnings for SDK software
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Enable testing
enable_testing()

# Add subdirectories for samples
file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
    if (EXISTS ${_FILE}/CMakeLists.txt)
     add_subdirectory(${_FILE})
    endif()
  endif()
endforeach()
