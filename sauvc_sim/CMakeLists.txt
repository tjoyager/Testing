cmake_minimum_required(VERSION 3.8)
project(sauvc_sim)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(gazebo_dev REQUIRED)

add_library(rov_controller SHARED src/rov_controller.cpp)
ament_target_dependencies(rov_controller rclcpp gazebo_ros geometry_msgs nav_msgs sensor_msgs tf2 tf2_ros gazebo_dev)

install(TARGETS rov_controller ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY launch description worlds DESTINATION share/${PROJECT_NAME})

ament_package()