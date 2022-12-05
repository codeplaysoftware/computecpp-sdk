#.rst:
# FindComputeCpp
#---------------
#
#   Copyright Codeplay Software Ltd.
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
#  User must define ComputeCpp_DIR pointing to the ComputeCpp
#  installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk

cmake_minimum_required(VERSION 3.10.2)
include(FindPackageHandleStandardArgs)

# These should match the types of IR output by compute++
set(IR_MAP_spir bc)
set(IR_MAP_spir64 bc)
set(IR_MAP_spir32 bc)
set(IR_MAP_spirv spv)
set(IR_MAP_spirv64 spv)
set(IR_MAP_spirv32 spv)
set(IR_MAP_aorta-x86_64 o)
set(IR_MAP_aorta-aarch64 o)
set(IR_MAP_aorta-rcar-cve o)
set(IR_MAP_custom-spir64 bc)
set(IR_MAP_custom-spir32 bc)
set(IR_MAP_custom-spirv64 spv)
set(IR_MAP_custom-spirv32 spv)
set(IR_MAP_ptx64 s)
set(IR_MAP_amdgcn s)

# Retrieves the filename extension of the IR output of compute++
function(get_sycl_target_extension output)
  set(syclExtension ${IR_MAP_${COMPUTECPP_BITCODE}})
  if(NOT syclExtension)
    # Needed when using multiple device targets
    set(syclExtension "bc")
  endif()
  set(${output} ${syclExtension} PARENT_SCOPE)
endfunction()

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
separate_arguments(COMPUTECPP_USER_FLAGS)
mark_as_advanced(COMPUTECPP_USER_FLAGS)

set(COMPUTECPP_BITCODE "" CACHE STRING "")
mark_as_advanced(COMPUTECPP_BITCODE)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
  # Policy enabling rewrites of paths in depfiles when using ninja
  cmake_policy(SET CMP0116 NEW)
endif()

set(SYCL_LANGUAGE_VERSION "2017" CACHE STRING "SYCL version to use. Defaults to 1.2.1.")

find_package(OpenCL REQUIRED)

if(TARGET OpenCL::Headers)
  get_target_property(OpenCL_INTERFACE_INCLUDE_DIRECTORIES OpenCL::Headers INTERFACE_INCLUDE_DIRECTORIES)
else()
  get_target_property(OpenCL_INTERFACE_INCLUDE_DIRECTORIES OpenCL::OpenCL INTERFACE_INCLUDE_DIRECTORIES)
endif()

# Find ComputeCpp package
set(computecpp_find_hint
  "${ComputeCpp_DIR}"
  "$ENV{COMPUTECPP_DIR}"
)

# Used for running executables on the host
set(computecpp_host_find_hint ${computecpp_find_hint})

if(CMAKE_CROSSCOMPILING)
  # ComputeCpp_HOST_DIR is used to find executables that are run on the host
  set(computecpp_host_find_hint
    "${ComputeCpp_HOST_DIR}"
    "$ENV{COMPUTECPP_HOST_DIR}"
    ${computecpp_find_hint}
  )
endif()

find_program(ComputeCpp_DEVICE_COMPILER_EXECUTABLE compute++
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  NO_SYSTEM_ENVIRONMENT_PATH)
find_program(ComputeCpp_INFO_EXECUTABLE computecpp_info
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  NO_SYSTEM_ENVIRONMENT_PATH)
find_library(COMPUTECPP_LIBRARY
  NAMES ComputeCpp
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library")

# Found the library, use only single hint from now on
get_filename_component(computecpp_library_path "${COMPUTECPP_LIBRARY}" DIRECTORY)
get_filename_component(computecpp_find_hint "${computecpp_library_path}/.." ABSOLUTE)

if(WIN32)
  set(DEBUG_POSTFIX "_d")
else()
  set(DEBUG_POSTFIX "")
endif()
find_library(COMPUTECPP_LIBRARY_DEBUG
  NAMES ComputeCpp${DEBUG_POSTFIX}
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library")

find_path(ComputeCpp_INCLUDE_DIRS
  NAMES "CL/sycl.hpp"
  HINTS ${computecpp_find_hint}/include
  DOC "The ComputeCpp include directory")
get_filename_component(ComputeCpp_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS} ABSOLUTE)

get_filename_component(computecpp_canonical_root_dir "${ComputeCpp_INCLUDE_DIRS}/.." ABSOLUTE)
set(ComputeCpp_ROOT_DIR "${computecpp_canonical_root_dir}" CACHE PATH
    "The root of the ComputeCpp install")

if(NOT ComputeCpp_INFO_EXECUTABLE)
  message(WARNING "Can't find computecpp_info - check ComputeCpp_DIR")
else()
  execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-version"
    OUTPUT_VARIABLE ComputeCpp_VERSION
    RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
    message(WARNING "Package version - Error obtaining version!")
  endif()

  execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-is-supported"
    OUTPUT_VARIABLE COMPUTECPP_PLATFORM_IS_SUPPORTED
    RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
    message(WARNING "platform - Error checking platform support!")
  else()
    if(NOT PREVIOUS_COMPUTECPP_PLATFORM_IS_SUPPORTED STREQUAL COMPUTECPP_PLATFORM_IS_SUPPORTED)
      mark_as_advanced(COMPUTECPP_PLATFORM_IS_SUPPORTED)
      if (COMPUTECPP_PLATFORM_IS_SUPPORTED)
        message(STATUS "platform - your system can support ComputeCpp")
      else()
        message(STATUS "platform - your system is not officially supported")
      endif()
      set(PREVIOUS_COMPUTECPP_PLATFORM_IS_SUPPORTED "${COMPUTECPP_PLATFORM_IS_SUPPORTED}" CACHE INTERNAL "Remember not to re-print when there's no change")
    endif()
  endif()
endif()

find_package_handle_standard_args(ComputeCpp
  REQUIRED_VARS ComputeCpp_ROOT_DIR
                ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                ComputeCpp_INFO_EXECUTABLE
                COMPUTECPP_LIBRARY
                COMPUTECPP_LIBRARY_DEBUG
                ComputeCpp_INCLUDE_DIRS
  VERSION_VAR ComputeCpp_VERSION)
mark_as_advanced(ComputeCpp_ROOT_DIR
                 ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                 ComputeCpp_INFO_EXECUTABLE
                 COMPUTECPP_LIBRARY
                 COMPUTECPP_LIBRARY_DEBUG
                 ComputeCpp_INCLUDE_DIRS
                 ComputeCpp_VERSION)

if(NOT ComputeCpp_FOUND)
  return()
endif()

list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -O2 -mllvm -inline-threshold=1000 -intelspirmetadata)
mark_as_advanced(COMPUTECPP_DEVICE_COMPILER_FLAGS)

if(CMAKE_CROSSCOMPILING)
  if(NOT COMPUTECPP_DONT_USE_TOOLCHAIN)
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --gcc-toolchain=${COMPUTECPP_TOOLCHAIN_DIR})
  endif()
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --sysroot=${COMPUTECPP_SYSROOT_DIR})
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -target ${COMPUTECPP_TARGET_TRIPLE})
endif()

list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -DSYCL_LANGUAGE_VERSION=${SYCL_LANGUAGE_VERSION})

foreach (bitcode IN ITEMS ${COMPUTECPP_BITCODE})
  if(NOT "${bitcode}" STREQUAL "")
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -sycl-target ${bitcode})
  endif()
endforeach()

if(NOT PREVIOUS_COMPUTECPP_DEVICE_COMPILER_FLAGS STREQUAL COMPUTECPP_DEVICE_COMPILER_FLAGS)
  message(STATUS "compute++ flags - ${COMPUTECPP_DEVICE_COMPILER_FLAGS}")
  set(PREVIOUS_COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS}" CACHE INTERNAL "Remember not to re-print when there's no change")
endif()


if(CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message(FATAL_ERROR "host compiler - gcc version must be > 4.8")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
    message(FATAL_ERROR "host compiler - clang version must be > 3.6")
  endif()
endif()

if(MSVC)
  set(ComputeCpp_STL_CHECK_SRC __STL_check)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
    "#include <CL/sycl.hpp>  \n"
    "int main() { return 0; }\n")
  set(_stl_test_command "${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}"
                        -sycl
                        ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
                        ${COMPUTECPP_USER_FLAGS}
                        -isystem ${ComputeCpp_INCLUDE_DIRS}
                        -isystem ${OpenCL_INTERFACE_INCLUDE_DIRECTORIES}
                        -o ${ComputeCpp_STL_CHECK_SRC}.sycl
                        -c ${ComputeCpp_STL_CHECK_SRC}.cpp)
  execute_process(
    COMMAND ${_stl_test_command}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
    ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
    OUTPUT_QUIET)
  if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
    # Try disabling compiler version checks
    execute_process(
      COMMAND ${_stl_test_command}
              -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
      ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
      OUTPUT_QUIET)
    if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
      # Try again with __CUDACC__ and _HAS_CONDITIONAL_EXPLICIT=0. This relaxes the restritions in the MSVC headers
      execute_process(
        COMMAND ${_stl_test_command}
                -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                -D_HAS_CONDITIONAL_EXPLICIT=0
                -D__CUDACC__
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
        ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
        OUTPUT_QUIET)
        if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
          message(FATAL_ERROR "compute++ cannot consume hosted STL headers. This means that compute++ can't \
                               compile a simple program in this platform and will fail when used in this system. \
                               \n ${ComputeCpp_STL_CHECK_ERROR_OUTPUT}")
        else()
          list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                                                       -D_HAS_CONDITIONAL_EXPLICIT=0
                                                       -D__CUDACC__)
        endif()
    else()
      list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
    endif()
  endif()
  file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
              ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp.sycl)
endif(MSVC)

if(NOT TARGET ComputeCpp::ComputeCpp)
  add_library(ComputeCpp::ComputeCpp SHARED IMPORTED)
  if(WIN32)
    string(REGEX REPLACE [[lib$]] [[dll]] COMPUTECPP_RUNTIME_LIBRARY "${COMPUTECPP_LIBRARY}")
    string(REGEX REPLACE [[lib$]] [[dll]] COMPUTECPP_RUNTIME_LIBRARY_DEBUG "${COMPUTECPP_LIBRARY_DEBUG}")
    set(EXTRA_IMPORTED_ARG IMPORTED_IMPLIB "${COMPUTECPP_LIBRARY}"
                           IMPORTED_IMPLIB_DEBUG "${COMPUTECPP_LIBRARY_DEBUG}")
  else()
    set(COMPUTECPP_RUNTIME_LIBRARY_DEBUG "${COMPUTECPP_LIBRARY_DEBUG}")
    set(COMPUTECPP_RUNTIME_LIBRARY "${COMPUTECPP_LIBRARY}")
    set(EXTRA_IMPORTED_ARG IMPORTED_SONAME ComputeCpp)
  endif()
  set_target_properties(ComputeCpp::ComputeCpp PROPERTIES
    IMPORTED_LOCATION_DEBUG          "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION                "${COMPUTECPP_RUNTIME_LIBRARY}"
    ${EXTRA_IMPORTED_ARG}
    INTERFACE_INCLUDE_DIRECTORIES    "${ComputeCpp_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES         "OpenCL::OpenCL"
  )
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
define_property(
  TARGET PROPERTY INTERFACE_COMPUTECPP_FLAGS
  BRIEF_DOCS "Interface compile flags to provide compute++"
  FULL_DOCS  "Set additional compile flags to pass to compute++ when compiling
  any target which links to this one."
)
define_property(
  SOURCE PROPERTY COMPUTECPP_SOURCE_FLAGS
  BRIEF_DOCS "Source file compile flags for compute++"
  FULL_DOCS  "Set additional compile flags for compiling the SYCL integration
  header for the given source file."
)

####################
#   __build_ir
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header and kernel binary.
#
#  TARGET : Name of the target.
#  SOURCE : Source file to be compiled.
#  COUNTER : Counter included in name of custom target. Different counter
#       values prevent duplicated names of custom target when source files with
#       the same name, but located in different directories, are used for the
#       same target.
#
function(__build_ir)
  set(options)
  set(one_value_args
    TARGET
    SOURCE
    COUNTER
  )
  set(multi_value_args)
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  get_filename_component(sourceFileName ${ARG_SOURCE} NAME)

  # Set the path to the integration header.
  # The .sycl filename must depend on the target so that different targets
  # using the same source file will be generated with a different rule.
  set(baseSyclName ${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_${sourceFileName})
  set(outputSyclFile ${baseSyclName}.sycl)
  get_sycl_target_extension(targetExtension)
  set(outputDeviceFile ${baseSyclName}.${targetExtension})
  set(depFileName ${baseSyclName}.sycl.d)

  set(include_directories "$<TARGET_PROPERTY:${ARG_TARGET},INCLUDE_DIRECTORIES>")
  set(compile_definitions "$<TARGET_PROPERTY:${ARG_TARGET},COMPILE_DEFINITIONS>")
  set(generated_include_directories
    $<$<BOOL:${include_directories}>:-I$<JOIN:${include_directories},;-I>>)
  set(generated_compile_definitions
    $<$<BOOL:${compile_definitions}>:-D$<JOIN:${compile_definitions},;-D>>)

  # Obtain language standard of the file
  set(device_compiler_cxx_standard
    "-std=c++$<TARGET_PROPERTY:${ARG_TARGET},CXX_STANDARD>")

  get_property(source_compile_flags
    SOURCE ${ARG_SOURCE}
    PROPERTY COMPUTECPP_SOURCE_FLAGS
  )
  separate_arguments(source_compile_flags)
  if(source_compile_flags)
    list(APPEND computecpp_source_flags ${source_compile_flags})
  endif()

  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${COMPUTECPP_USER_FLAGS}
    ${computecpp_source_flags}
  )

  set(ir_dependencies ${ARG_SOURCE})
  get_target_property(target_libraries ${ARG_TARGET} LINK_LIBRARIES)
  if(target_libraries)
    foreach(library ${target_libraries})
      if(TARGET ${library})
        list(APPEND ir_dependencies ${library})
      endif()
    endforeach()
  endif()

  # Depfile support was only added in CMake 3.7
  # CMake throws an error if it is unsupported by the generator (i. e. not ninja)
  if((NOT CMAKE_VERSION VERSION_LESS 3.7.0) AND
          CMAKE_GENERATOR MATCHES "Ninja")
    file(RELATIVE_PATH relOutputFile ${CMAKE_BINARY_DIR} ${outputDeviceFile})
    set(generate_depfile -MMD -MF ${depFileName} -MT ${relOutputFile})
    set(enable_depfile DEPFILE ${depFileName})
  endif()

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputDeviceFile} ${outputSyclFile}
    COMMAND "${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}"
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            "${generated_include_directories}"
            "${generated_compile_definitions}"
            -sycl-ih ${outputSyclFile}
            -o ${outputDeviceFile}
            -c ${ARG_SOURCE}
            ${generate_depfile}
    COMMAND_EXPAND_LISTS
    DEPENDS ${ir_dependencies}
    IMPLICIT_DEPENDS CXX ${ARG_SOURCE}
    ${enable_depfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name: (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${ARG_TARGET}_${sourceFileName}_${ARG_COUNTER}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputDeviceFile} ${outputSyclFile})
    add_dependencies(${ARG_TARGET} ${headerTargetName})
  endif()

  # This property can be set on a per-target basis to indicate that the
  # integration header should appear after the main source listing
  get_target_property(includeAfter ${ARG_TARGET} COMPUTECPP_INCLUDE_AFTER)

  if(includeAfter)
    # Change the source file to the integration header - e.g.
    # g++ -c source_file_name.cpp.sycl
    get_target_property(current_sources ${ARG_TARGET} SOURCES)
    # Remove absolute path to source file
    list(REMOVE_ITEM current_sources ${ARG_SOURCE})
    # Remove relative path to source file
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" ""
      rel_source_file ${ARG_SOURCE}
    )
    list(REMOVE_ITEM current_sources ${rel_source_file})
    # Add SYCL header to source list
    list(APPEND current_sources ${outputSyclFile})
    set_property(TARGET ${ARG_TARGET}
      PROPERTY SOURCES ${current_sources})
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${ARG_SOURCE})
    set(cppFile ${outputSyclFile})
  else()
    set_property(SOURCE ${outputSyclFile} PROPERTY HEADER_FILE_ONLY ON)
    set(includedFile ${outputSyclFile})
    set(cppFile ${ARG_SOURCE})
  endif()

  # Force inclusion of the integration header for the host compiler
  if(MSVC)
    # Group SYCL files inside Visual Studio
    source_group("SYCL" FILES ${outputSyclFile})

    if(includeAfter)
      # Allow the source file to be edited using Visual Studio.
      # It will be added as a header file so it won't be compiled.
      set_property(SOURCE ${ARG_SOURCE} PROPERTY HEADER_FILE_ONLY true)
    endif()

    # Add both source and the sycl files to the VS solution.
    target_sources(${ARG_TARGET} PUBLIC ${ARG_SOURCE} ${outputSyclFile})

    set(forceIncludeFlags "/FI${includedFile} /TP")
  else()
    set(forceIncludeFlags "-include ${includedFile} -x c++")
  endif()

  set_property(
    SOURCE ${cppFile}
    APPEND_STRING PROPERTY COMPILE_FLAGS "${forceIncludeFlags}"
  )

endfunction(__build_ir)

#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  TARGET : Name of the target to add SYCL to.
#  SOURCES : Source files to be compiled for SYCL.
#
function(add_sycl_to_target)
  set(options)
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    SOURCES
  )
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  if ("${ARG_SOURCES}" STREQUAL "")
    message(WARNING "No source files provided to add_sycl_to_target. "
                    "SYCL integration headers may not be generated.")
  endif()
  set_target_properties(${ARG_TARGET} PROPERTIES LINKER_LANGUAGE CXX)

  # If the CXX compiler is set to compute++ enable the driver.
  get_filename_component(cmakeCxxCompilerFileName "${CMAKE_CXX_COMPILER}" NAME)
  if("${cmakeCxxCompilerFileName}" STREQUAL "compute++")
    if(MSVC)
      message(FATAL_ERROR "The compiler driver is not supported by this system,
                           revert the CXX compiler to your default host compiler.")
    endif()

    get_target_property(includeAfter ${ARG_TARGET} COMPUTECPP_INCLUDE_AFTER)
    if(includeAfter)
      list(APPEND COMPUTECPP_USER_FLAGS -fsycl-ih-last)
    endif()
    list(INSERT COMPUTECPP_DEVICE_COMPILER_FLAGS 0 -sycl-driver)
    # Prepend COMPUTECPP_DEVICE_COMPILER_FLAGS and append COMPUTECPP_USER_FLAGS
    foreach(prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
      get_target_property(target_compile_options ${ARG_TARGET} ${prop})
      if(NOT target_compile_options)
        set(target_compile_options "")
      endif()
      set_property(
        TARGET ${ARG_TARGET}
        PROPERTY ${prop}
        ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
        ${target_compile_options}
        ${COMPUTECPP_USER_FLAGS}
      )
    endforeach()
  else()
    set(fileCounter 0)
    list(INSERT COMPUTECPP_DEVICE_COMPILER_FLAGS 0 -sycl)
    # Add custom target to run compute++ and generate the integration header
    foreach(sourceFile ${ARG_SOURCES})
      if(NOT IS_ABSOLUTE ${sourceFile})
        set(sourceFile "${CMAKE_CURRENT_SOURCE_DIR}/${sourceFile}")
      endif()
      __build_ir(
        TARGET     ${ARG_TARGET}
        SOURCE     ${sourceFile}
        COUNTER    ${fileCounter}
      )
      MATH(EXPR fileCounter "${fileCounter} + 1")
    endforeach()
  endif()

  set_property(TARGET ${ARG_TARGET}
    APPEND PROPERTY LINK_LIBRARIES ComputeCpp::ComputeCpp)
  set_property(TARGET ${ARG_TARGET}
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES ComputeCpp::ComputeCpp)
  target_compile_definitions(${ARG_TARGET} PUBLIC
    SYCL_LANGUAGE_VERSION=${SYCL_LANGUAGE_VERSION})
endfunction(add_sycl_to_target)
