project(GameOfLifeApp)

set(SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/sim.cpp")

ci_make_app(
  APP_NAME "GameOfLife"
  SOURCES "${SOURCES}"
  LIBRARIES "${OpenCL_LIBRARIES}"
  CINDER_PATH "${CINDER_PATH}")

add_sycl_to_target(
    TARGET "GameOfLife"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/sim.cpp")
