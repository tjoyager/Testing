cmake_minimum_required(VERSION 3.8)
project(sauvc_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- BAGIAN INI SERING TERLEWAT ---
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjDetection.msg"
  "msg/ObjArray.msg"
)
# ----------------------------------

ament_package()