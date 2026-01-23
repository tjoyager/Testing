cmake_minimum_required(VERSION 3.8)
project(sauvc_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sauvc_msgs REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_executable(yolo_node src/yolo_node.cpp src/yolo_engine.cpp)
ament_target_dependencies(yolo_node rclcpp sensor_msgs cv_bridge sauvc_msgs)
target_link_libraries(yolo_node ${OpenCV_LIBS} openvino::runtime)

install(TARGETS yolo_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()