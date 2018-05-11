#.rst:
# FindComputeCpp
#---------------
#
#   Copyright 2016 Codeplay Software Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use these files except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#########################
#  FindComputeCpp.cmake
#########################
#
#  Tools for finding and building with ComputeCpp.
#
#  User must define COMPUTECPP_PACKAGE_ROOT_DIR pointing to the ComputeCpp
#   installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk

# Require CMake version 3.2.2 or higher
cmake_minimum_required(VERSION 3.2.2)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - Not found! (gcc version must be at least 4.8)")
    else()
      message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - Not found! (clang version must be at least 3.6)")
    else()
      message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

set(COMPUTECPP_64_BIT_DEFAULT ON)
option(COMPUTECPP_64_BIT_CODE "Compile device code in 64 bit mode"
        ${COMPUTECPP_64_BIT_DEFAULT})
mark_as_advanced(COMPUTECPP_64_BIT_CODE)

option(COMPUTECPP_DISABLE_GCC_DUAL_ABI "Compile with pre-5.1 ABI" OFF)
mark_as_advanced(COMPUTECPP_DISABLE_GCC_DUAL_ABI)

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
mark_as_advanced(COMPUTECPP_USER_FLAGS)

if(NOT MSVC)
  # The compiler driver is only available on Linux at the moment.
  option(COMPUTECPP_USE_COMPILER_DRIVER "Use ComputeCpp compiler driver" ON)
  mark_as_advanced(COMPUTECPP_USE_COMPILER_DRIVER)
endif()

# Platform-specific arguments
if(MSVC)
  # Workaround to an unfixed Clang bug, rationale:
  # https://github.com/codeplaysoftware/computecpp-sdk/pull/51#discussion_r139399093
  set (COMPUTECPP_PLATFORM_SPECIFIC_ARGS "-fno-ms-compatibility")
endif()

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Find ComputeCpp package

# Try to read the environment variable
if(DEFINED ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
  if(NOT COMPUTECPP_PACKAGE_ROOT_DIR)
    set(COMPUTECPP_PACKAGE_ROOT_DIR $ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
  endif()
endif()

if(NOT COMPUTECPP_PACKAGE_ROOT_DIR)
  message(FATAL_ERROR
    "ComputeCpp package - Not found! (please set COMPUTECPP_PACKAGE_ROOT_DIR)")
else()
  message(STATUS "ComputeCpp package - Found")
endif()

# Obtain the path to compute++
find_program(COMPUTECPP_DEVICE_COMPILER compute++ PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin)
if (EXISTS ${COMPUTECPP_DEVICE_COMPILER})
  mark_as_advanced(COMPUTECPP_DEVICE_COMPILER)
  message(STATUS "compute++ - Found")
else()
  message(FATAL_ERROR "compute++ - Not found! (${COMPUTECPP_DEVICE_COMPILER})")
endif()

# Obtain the path to computecpp_info
find_program(COMPUTECPP_INFO_TOOL computecpp_info PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin)
if (EXISTS ${COMPUTECPP_INFO_TOOL})
  mark_as_advanced(${COMPUTECPP_INFO_TOOL})
  message(STATUS "computecpp_info - Found")
else()
  message(FATAL_ERROR "computecpp_info - Not found! (${COMPUTECPP_INFO_TOOL})")
endif()

# Obtain the path to the ComputeCpp runtime library
find_library(COMPUTECPP_RUNTIME_LIBRARY
  NAMES ComputeCpp ComputeCpp_vs2015
  PATHS ${COMPUTECPP_PACKAGE_ROOT_DIR}
  HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}/lib PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${COMPUTECPP_RUNTIME_LIBRARY})
  mark_as_advanced(COMPUTECPP_RUNTIME_LIBRARY)
else()
  message(FATAL_ERROR "ComputeCpp Runtime Library - Not found!")
endif()

find_library(COMPUTECPP_RUNTIME_LIBRARY_DEBUG
  NAMES ComputeCpp ComputeCpp_vs2015_d
  PATHS ${COMPUTECPP_PACKAGE_ROOT_DIR}
  HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}/lib PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG})
  mark_as_advanced(COMPUTECPP_RUNTIME_LIBRARY_DEBUG)
else()
  message(FATAL_ERROR "ComputeCpp Debug Runtime Library - Not found!")
endif()

# NOTE: Having two sets of libraries is Windows specific, not MSVC specific.
# Compiling with Clang on Windows would still require linking to both of them.
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  message(STATUS "ComputeCpp runtime (Release): ${COMPUTECPP_RUNTIME_LIBRARY} - Found")
  message(STATUS "ComputeCpp runtime  (Debug) : ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} - Found")
else()
  message(STATUS "ComputeCpp runtime: ${COMPUTECPP_RUNTIME_LIBRARY} - Found")
endif()

# Obtain the ComputeCpp include directory
set(COMPUTECPP_INCLUDE_DIRECTORY ${COMPUTECPP_PACKAGE_ROOT_DIR}/include)
if (NOT EXISTS ${COMPUTECPP_INCLUDE_DIRECTORY})
  message(FATAL_ERROR "ComputeCpp includes - Not found!")
else()
  message(STATUS "ComputeCpp includes - Found")
endif()

# Obtain the package version
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-version"
  OUTPUT_VARIABLE COMPUTECPP_PACKAGE_VERSION
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "Package version - Error obtaining version!")
else()
  mark_as_advanced(COMPUTECPP_PACKAGE_VERSION)
  message(STATUS "Package version - ${COMPUTECPP_PACKAGE_VERSION}")
endif()

# Obtain the device compiler flags
set(USE_SPIRV "")
if (COMPUTECPP_USE_SPIRV)
  set(USE_SPIRV "--use-spirv")
endif()

set(USE_PTX "")
if (COMPUTECPP_USE_PTX)
  set(USE_PTX "--use-ptx")
endif()

execute_process(COMMAND ${COMPUTECPP_INFO_TOOL}
  ${USE_SPIRV} ${USE_PTX} "--dump-device-compiler-flags"
  OUTPUT_VARIABLE COMPUTECPP_DEVICE_COMPILER_FLAGS
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "compute++ flags - Error obtaining compute++ flags!")
else()
  mark_as_advanced(COMPUTECPP_COMPILER_FLAGS)
  message(STATUS "compute++ flags - ${COMPUTECPP_DEVICE_COMPILER_FLAGS}")
endif()

# Check if the platform is supported
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-is-supported"
  OUTPUT_VARIABLE COMPUTECPP_PLATFORM_IS_SUPPORTED
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "platform - Error checking platform support!")
else()
  mark_as_advanced(COMPUTECPP_PLATFORM_IS_SUPPORTED)
  if (COMPUTECPP_PLATFORM_IS_SUPPORTED)
    message(STATUS "platform - your system can support ComputeCpp")
  else()
    message(STATUS "platform - your system CANNOT support ComputeCpp")
  endif()
endif()

# This property allows targets to specify that their sources should be
# compiled with the integration header included after the user's
# sources, not before (e.g. when an enum is used in a kernel name, this
# is not technically valid SYCL code but can work with ComputeCpp)
define_property(
  TARGET PROPERTY COMPUTECPP_INCLUDE_AFTER
  BRIEF_DOCS "Include integration header after user source"
  FULL_DOCS "Changes compiler arguments such that the source file is
  actually the integration header, and the .cpp file is included on
  the command line so that it is seen by the compiler first. Enables
  non-standards-conformant SYCL code to compile with ComputeCpp."
)

# Enable the usage of the ComputeCpp compiler driver. This removes the need
# to explicitly generate the sycl integration header in a separate compiler pass.
if(COMPUTECPP_USE_COMPILER_DRIVER)
  # Check if the current system supports the compiler driver
  set(isCompilerDriverSupported NO)

  # For now, the compiler driver is only supported if the host compiler is either clang or gcc
  if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(isCompilerDriverSupported YES)
  endif()

  if(${isCompilerDriverSupported})
    # If the compiler driver is supported, change the CXX compiler to compute++ so the driver can work property.
    # When building a non-sycl application, compute++ acts like the host compiler, so no problems are expected.
    list(APPEND COMPUTECPP_COMPILER_FLAGS -sycl-driver)
  else()
    message(WARNING "The compiler driver is not supported by this system.")
    message(WARNING "COMPUTECPP_USE_COMPILER_DRIVER will be ignored.")
  endif()
endif()

####################
#   __build_sycl
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header.
#
#  targetName : Name of the target.
#  sourceFile : Source file to be compiled.
#  binaryDir : Intermediate directory to output the integration header.
#  fileCounter : Counter included in name of custom target. Different counter
#       values prevent duplicated names of custom target when source files with the same name,
#       but located in different directories, are used for the same target.
#
function(__build_spir targetName sourceFile binaryDir fileCounter)

  # Retrieve source file name.
  get_filename_component(sourceFileName ${sourceFile} NAME)

  # Set the path to the Sycl file.
  set(outputSyclFile ${binaryDir}/${sourceFileName}.sycl)

  # Add any user-defined include to the device compiler
  set(device_compiler_includes "")
  get_property(includeDirectories DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
    INCLUDE_DIRECTORIES)
  foreach(directory ${includeDirectories})
    set(device_compiler_includes "-I${directory}" ${device_compiler_includes})
  endforeach()
  get_target_property(targetIncludeDirectories ${targetName} INCLUDE_DIRECTORIES)
  foreach(directory ${targetIncludeDirectories})
    set(device_compiler_includes "-I${directory}" ${device_compiler_includes})
  endforeach()
  if (CMAKE_INCLUDE_PATH)
    foreach(directory ${CMAKE_INCLUDE_PATH})
      set(device_compiler_includes "-I${directory}"
        ${device_compiler_includes})
    endforeach()
  endif()

  # Obtain language standard of the file
  set(device_compiler_cxx_standard)
  get_target_property(targetCxxStandard ${targetName} CXX_STANDARD)
  if (targetCxxStandard MATCHES 17)
    set(device_compiler_cxx_standard "-std=c++1z")
  elseif (targetCxxStandard MATCHES 14)
    set(device_compiler_cxx_standard "-std=c++14")
  elseif (targetCxxStandard MATCHES 11)
    set(device_compiler_cxx_standard "-std=c++11")
  elseif (targetCxxStandard MATCHES 98)
    message(FATAL_ERROR "SYCL applications cannot be compiled using C++98")
  else ()
    set(device_compiler_cxx_standard "")
  endif()

  set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
    ${COMPUTECPP_USER_FLAGS})
  # Convert argument list format
  separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputSyclFile}
    COMMAND ${COMPUTECPP_DEVICE_COMPILER}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            -isystem ${COMPUTECPP_INCLUDE_DIRECTORY}
            ${COMPUTECPP_PLATFORM_SPECIFIC_ARGS}
            ${device_compiler_includes}
            -o ${outputSyclFile}
            -c ${sourceFile}
    DEPENDS ${sourceFile}
    IMPLICIT_DEPENDS CXX ${sourceFile}
    WORKING_DIRECTORY ${binaryDir}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name:
  # (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${targetName}_${sourceFileName}_${fileCounter}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputSyclFile})

    # Add a dependency on the integration header
    add_dependencies(${targetName} ${headerTargetName})
  endif()

  # This property can be set on a per-target basis to indicate that the
  # integration header should appear after the main source listing
  get_property(includeAfter TARGET ${targetName}
      PROPERTY COMPUTECPP_INCLUDE_AFTER)

  if(includeAfter)
    # Change the source file to the integration header - e.g.
    # g++ -c source_file_name.cpp.sycl
    set_property(TARGET ${targetName} PROPERTY SOURCES ${outputSyclFile})
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${sourceFile})
  else()
    set(includedFile ${outputSyclFile})
  endif()

  # Force inclusion of the integration header for the host compiler
  if(MSVC)
    # Group SYCL files inside Visual Studio
    source_group("SYCL" FILES ${outputSyclFile})

    if(includeAfter)
      # Allow the source file to be edited using Visual Studio.
      # It will be added as a header file so it won't be compiled.
      set_property(SOURCE ${sourceFile} PROPERTY HEADER_FILE_ONLY true)
    endif()

    # Add both source and the sycl files to the VS solution.
    target_sources(${targetName} PUBLIC ${sourceFile} ${outputSyclFile})

    # NOTE: The Visual Studio generators parse compile flags differently,
    # hence the different argument syntax
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      set(forceIncludeFlags "/FI\"${includedFile}\" /TP")
    else()
      set(forceIncludeFlags /FI ${includedFile} /TP)
    endif()
  else()
      set(forceIncludeFlags "-include ${includedFile} -x c++")
  endif()
  # target_compile_options removes duplicated flags, which does not work
  # for -include (it should appear once per included file - CMake bug #15826)
  #   target_compile_options(${targetName} BEFORE PUBLIC ${forceIncludeFlags})
  # To avoid the problem, we get the value of the property and manually append
  # it to the previous status.
  get_property(currentFlags TARGET ${targetName} PROPERTY COMPILE_FLAGS)
  set_property(TARGET ${targetName} PROPERTY COMPILE_FLAGS
                  "${forceIncludeFlags} ${currentFlags}")

  if(COMPUTECPP_DISABLE_GCC_DUAL_ABI)
    set_property(TARGET ${targetName} APPEND PROPERTY COMPILE_DEFINITIONS
      "_GLIBCXX_USE_CXX11_ABI=0")
  endif()

endfunction()

##############################
#  enable_sycl_compiler_driver
##############################
#
#  Helper macro that switches on the compiler driver to be used from this point onwards.
#  Since this call will replace the current C++ compiler, this needs to be at
#  the end of the CMake tree to avoid the cache being deleted and all flags lost.
#
#  Calling this macro should have no effect when compiling existing non-sycl code since
#  the compiler driver will the same as the host compiler.
#
macro(enable_sycl_compiler_driver)
  if(COMPUTECPP_USE_COMPILER_DRIVER AND isCompilerDriverSupported)
    set(CMAKE_CXX_COMPILER ${COMPUTECPP_DEVICE_COMPILER})
  endif()
endmacro(enable_sycl_compiler_driver)


#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  targetName : Name of the target to add a SYCL to.
#  binaryDir : Intermediate directory to output the integration header.
#  sourceFiles : Source files to be compiled for SYCL.
#
function(add_sycl_to_target)
  set(options)
  set(oneValueArgs TARGET BINARY_DIR)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(add_sycl_to_target "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(targetName ${add_sycl_to_target_TARGET})
  set(binaryDir ${add_sycl_to_target_BINARY_DIR})
  set(sourceFiles ${add_sycl_to_target_SOURCES})

  set(fileCounter 0)
  target_include_directories(
    ${targetName} SYSTEM
    PRIVATE ${OpenCL_INCLUDE_DIR}
    PRIVATE ${COMPUTECPP_INCLUDE_DIRECTORY}
  )

  if(COMPUTECPP_USE_COMPILER_DRIVER AND isCompilerDriverSupported)
    # Set all flags required to compile the target when using the compiler driver
    set(computeCppCompileOptions ${COMPUTECPP_COMPILER_FLAGS} -sycl-driver)
    get_property(includeAfter TARGET ${targetName} PROPERTY COMPUTECPP_INCLUDE_AFTER)
    if(includeAfter)
      list(APPEND computeCppCompileOptions -fsycl-ih-last)
    endif()
    target_compile_options(${targetName} PUBLIC ${computeCppCompileOptions})
  else()
    if(NOT binaryDir)
      message(WARNING " ComputeCpp Warning: BINARY_DIR was not specified. Using CMAKE_CUURENT_BINARY_DIR which is \"${CMAKE_CUURENT_BINARY_DIR}\"")
      set(binaryDir ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    if(NOT sourceFiles)
      message(FALTAL_ERROR " ComputeCpp Error: SOURCES was not specified. At least one source file must be provided.")
    endif()

    # Add custom target to run compute++ and generate the integration header
    foreach(sourceFile ${sourceFiles})
      __build_spir(${targetName} ${sourceFiles} ${binaryDir} ${fileCounter})
      MATH(EXPR fileCounter "${fileCounter} + 1")
    endforeach()
  endif()

  # Link with the ComputeCpp runtime library
  target_link_libraries(${targetName} PUBLIC $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}>
                                             $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:${COMPUTECPP_RUNTIME_LIBRARY}>
                                             ${OpenCL_LIBRARIES})

endfunction(add_sycl_to_target)
