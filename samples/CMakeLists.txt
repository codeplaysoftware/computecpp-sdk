if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

function(add_sample)
  set(options NO_TEST)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(SDK_ADD_SAMPLE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  add_executable(${SDK_ADD_SAMPLE_TARGET} ${SDK_ADD_SAMPLE_SOURCES})
  add_sycl_to_target(TARGET ${SDK_ADD_SAMPLE_TARGET}
    SOURCES ${SDK_ADD_SAMPLE_SOURCES})
  if(NOT SDK_ADD_SAMPLE_NO_TEST)
    add_test(NAME ${SDK_ADD_SAMPLE_TARGET} COMMAND ${SDK_ADD_SAMPLE_TARGET})
  endif()
  install(TARGETS ${SDK_ADD_SAMPLE_TARGET} RUNTIME DESTINATION bin)
endfunction(add_sample)

add_sample(TARGET accessors SOURCES accessors.cpp)
add_sample(TARGET async-handler SOURCES async-handler.cpp)
add_sample(TARGET builtin-kernel-example SOURCES builtin-kernel-example.cpp)
add_sample(TARGET custom-device-selector SOURCES custom-device-selector.cpp)
add_sample(TARGET example-sycl-application SOURCES example-sycl-application.cpp)
add_sample(TARGET gaussian-blur SOURCES gaussian-blur.cpp NO_TEST)
target_include_directories(gaussian-blur PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME gaussian-blur
  COMMAND gaussian-blur ${CMAKE_CURRENT_SOURCE_DIR}/Lenna.png)
add_sample(TARGET images SOURCES images.cpp)

add_sample(TARGET matrix-multiply SOURCES matrix-multiply.cpp NO_TEST)
if(COMPUTECPP_SDK_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(matrix-multiply PRIVATE OpenMP::OpenMP_CXX)
  else()
    # Workaround for CMake <3.9 - FindOpenMP does not provide the
    # linker flags and no imported target
    target_compile_options(matrix-multiply PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries( matrix-multiply PRIVATE ${OpenMP_CXX_FLAGS})
  endif()
  add_test(NAME matrix-multiply_omp COMMAND matrix-multiply 64 omp)
endif()
add_test(NAME matrix-multiply_sycl COMMAND matrix-multiply 64 sycl)

add_sample(TARGET ivka SOURCES ivka.cpp)
target_include_directories(ivka PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_sample(TARGET monte-carlo-pi SOURCES monte-carlo-pi.cpp)
add_sample(TARGET opencl-c-interop SOURCES opencl-c-interop.cpp)
add_sample(TARGET parallel-for SOURCES parallel-for.cpp)
add_sample(TARGET placeholder-accessors SOURCES placeholder-accessors.cpp)
add_sample(TARGET reduction SOURCES reduction.cpp)
add_sample(TARGET reinterpret SOURCES reinterpret.cpp)
add_sample(TARGET scan SOURCES scan.cpp)
add_sample(TARGET simple-example-of-vectors SOURCES simple-example-of-vectors.cpp)
add_sample(TARGET simple-local-barrier SOURCES simple-local-barrier.cpp)
add_sample(TARGET simple-private-memory SOURCES simple-private-memory.cpp)
add_sample(TARGET simple-vector-add SOURCES simple-vector-add.cpp)
add_sample(TARGET smart-pointer SOURCES smart-pointer.cpp)
add_sample(TARGET sync-handler SOURCES sync-handler.cpp)
add_sample(TARGET template-function-object SOURCES template-function-object.cpp)
add_sample(TARGET use-onchip-memory SOURCES use-onchip-memory.cpp)
add_sample(TARGET using-function-objects SOURCES using-function-objects.cpp)
# This property is not necessary here, but demonstrates how to use it.
# Forces the integration header to appear after the main code.
set_property(TARGET using-function-objects PROPERTY COMPUTECPP_INCLUDE_AFTER 1)
add_sample(TARGET vector-addition SOURCES vector-addition-examples.cpp)
add_sample(TARGET vptr SOURCES vptr.cpp)
target_include_directories(vptr PRIVATE ${PROJECT_SOURCE_DIR}/include)
