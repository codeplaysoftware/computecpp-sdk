set(SOURCE_NAME "gaussian_blur")
add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
if(NOT MSVC)
  target_compile_options(
    ${SOURCE_NAME}
    PUBLIC -Wno-unused-but-set-variable
  )
endif()
add_sycl_to_target(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
# Copy the image Lenna.png into the binary directory. As there are no
# configuration directories on multi configuration generators (i.e. MSVC) if you
# use the configuration in the path it fails for them, so we copy to the
# explicit paths.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Lenna.png
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Lenna.png
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Lenna.png
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME ${SOURCE_NAME} COMMAND ${SOURCE_NAME} Lenna.png)
install(
  TARGETS ${SOURCE_NAME}
  RUNTIME DESTINATION bin
)
