import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'sauvc_sim'
    xacro_file = os.path.join(get_package_share_directory(pkg_name), 'description', 'robot.xacro')
    world_file = os.path.join(get_package_share_directory(pkg_name), 'worlds', 'underwater.world')

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': Command(['xacro ', xacro_file]), 'use_sim_time': True}]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world_file}.items()
    )

    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'submarine_robot', '-z', '-1.5'],
        output='screen'
    )

    return LaunchDescription([rsp, gazebo, spawn])