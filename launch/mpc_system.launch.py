import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Configuration Variables
    # The topic we found in Gazebo that provides Ground Truth Odometry
    gz_odom_topic = '/model/bluerov2_heavy/odometry'
    
    # MAVROS Connection (Matches your terminal 3)
    fcu_url = 'udp://:14552@' 

    # 2. Node: ROS-GZ Bridge
    # Bridges Gazebo Odometry -> ROS 2 Odometry
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Syntax: <GZ_TOPIC>@<ROS_TYPE>[<GZ_TYPE>
            f'{gz_odom_topic}@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ],
        output='screen'
    )

    # 3. Node: MAVROS
    # Communicates with ArduPilot SITL
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        output='screen',
        parameters=[{
            'fcu_url': fcu_url,
            'system_id': 1,
            'component_id': 1,
            'target_system_id': 1,
            'target_component_id': 1,
        }]
    )

    # 4. Node: Your MPC Controller
    # Make sure 'mpc_controller' matches your package name in setup.py
    mpc_node = Node(
        package='mpc_controller',
        executable='mpc_node',
        name='mpc_node',
        output='screen',
        # Setup environment variable for the python path if needed, 
        # though ideally this is handled by sourcing the workspace.
        additional_env={'PYTHONUNBUFFERED': '1'} 
    )

    return LaunchDescription([
        bridge,
        mavros_node,
        mpc_node
    ])