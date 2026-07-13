cmake_minimum_required(VERSION 3.5)
project(px4_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Message/service definitions are synced from the PX4 fork by
# scripts/sync_px4_msgs.sh; collect them by glob so a re-sync needs no edits here.
file(GLOB PX4_MSGS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "msg/*.msg")
file(GLOB PX4_SRVS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "srv/*.srv")

rosidl_generate_interfaces(${PROJECT_NAME}
  ${PX4_MSGS}
  ${PX4_SRVS}
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
