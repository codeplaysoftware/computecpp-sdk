set(SOURCE_NAME "matrix_multiply")

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp  )
target_compile_options(
  ${SOURCE_NAME}
  PUBLIC ${HOST_COMPILER_OPTIONS} -Wno-unknown-pragmas
  )

add_sycl_to_target(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME ${SOURCE_NAME}_omp COMMAND ${SOURCE_NAME} 64 omp)
add_test(NAME ${SOURCE_NAME}_sycl COMMAND ${SOURCE_NAME} 64 sycl)
