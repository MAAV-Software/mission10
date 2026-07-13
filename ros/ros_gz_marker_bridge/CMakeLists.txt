cmake_minimum_required(VERSION 3.16)
project(ros_gz_marker_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_executable(marker_bridge src/marker_bridge.cpp)
ament_target_dependencies(marker_bridge rclcpp visualization_msgs)
target_link_libraries(marker_bridge gz-transport13::core gz-msgs10::core)

install(TARGETS marker_bridge DESTINATION lib/${PROJECT_NAME})

ament_package()
