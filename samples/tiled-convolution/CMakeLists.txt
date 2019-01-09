set(files
  "tiled-convolution-naive.cpp"
  "tiled-convolution-optimized.cpp"
  "tiled-convolution-standard.cpp"
)

foreach(SOURCE_NAME ${files})
  get_filename_component(EXEC_NAME ${SOURCE_NAME} NAME_WE)

  add_executable(
    ${EXEC_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}
  )

  add_sycl_to_target(
    TARGET ${EXEC_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}
  )

  add_test(
    NAME ${EXEC_NAME}
    COMMAND ${EXEC_NAME}
  )

  install(
    TARGETS ${EXEC_NAME}
    RUNTIME DESTINATION bin
  )
endforeach()
