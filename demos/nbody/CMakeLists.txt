project(NBodyApp)

set(SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/sim.cpp")

ci_make_app(
  APP_NAME "NBody"
  SOURCES "${SOURCES}"
  LIBRARIES "${OpenCL_LIBRARIES}"
  LIBRARIES cinder_imgui
  CINDER_PATH "${CINDER_PATH}")

add_sycl_to_target(
  TARGET "NBody"
  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sim.cpp")
