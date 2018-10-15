set(SOURCE_NAME "tiled-convolution")

if(DEFINED TARGET)
message(STATUS "TARGET is defined")
  # If the user has specified  RCAR as a target backend
  #the local size will be set to 1 and 32
  if(${TARGET} STREQUAL "RCAR")
    message(STATUS "${TARGET} device is chosen")
    set(LOCAL_SIZE_M 1)
    set(LOCAL_SIZE_N 32)
  endif()
else()
  if (NOT DEFINED LOCAL_SIZE_M)
    set(LOCAL_SIZE_M 16)
  endif()

  if (NOT DEFINED LOCAL_SIZE_N)
    set(LOCAL_SIZE_N 16)
  endif()

endif()

add_definitions(-DLOCAL_SIZE_M=${LOCAL_SIZE_M})
add_definitions(-DLOCAL_SIZE_N=${LOCAL_SIZE_N})

add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
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