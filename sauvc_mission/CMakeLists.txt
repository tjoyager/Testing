cmake_minimum_required(VERSION 3.8)
project(sauvc_mission)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sauvc_msgs REQUIRED)

# 2. Add Executable (INI HARUS DULUAN)
# "mission_node" adalah nama target. File sumbernya adalah "src/mission_node.cpp"
add_executable(mission_node src/mission_node.cpp)

# 3. Target Dependencies (NAMA TARGET HARUS SAMA DENGAN DI ATAS)
ament_target_dependencies(mission_node
  rclcpp
  geometry_msgs
  nav_msgs
  sauvc_msgs
)

# 4. Install Rules
install(TARGETS
  mission_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()