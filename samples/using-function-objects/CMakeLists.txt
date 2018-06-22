set(SOURCE_NAME "using-function-objects")
add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
# This property is not necessary here, but demonstrates how to use it.
# Forces the integration header to appear after the main code.
set_property(
  TARGET ${SOURCE_NAME}
  PROPERTY COMPUTECPP_INCLUDE_AFTER 1
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
