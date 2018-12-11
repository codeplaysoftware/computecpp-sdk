set(SOURCE_NAME "use-onchip-memory")

add_executable(
   ${SOURCE_NAME}
   ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)
add_sycl_to_target(
  TARGET ${SOURCE_NAME}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
add_test(
  NAME ${SOURCE_NAME}
  COMMAND ${SOURCE_NAME}
)
install(
  TARGETS ${SOURCE_NAME}
  RUNTIME DESTINATION bin
)
