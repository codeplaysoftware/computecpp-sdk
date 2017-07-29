set(SOURCE_NAME "matrix_multiply")
add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
target_compile_options(
  ${SOURCE_NAME}
  PUBLIC -Wno-unknown-pragmas
)
add_sycl_to_target(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
add_test(
  NAME ${SOURCE_NAME}_omp
  COMMAND ${SOURCE_NAME} 64 omp
)
add_test(
  NAME ${SOURCE_NAME}_sycl
  COMMAND ${SOURCE_NAME} 64 sycl
)
install(
  TARGETS ${SOURCE_NAME}
  RUNTIME DESTINATION bin
)
