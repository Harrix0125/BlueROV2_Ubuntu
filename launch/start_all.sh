#!/bin/bash

# 1. Kill any lingering sessions (Safety first!)
killall -9 gz sim_vehicle.py ruby python3
echo "Cleaning up old processes..."
sleep 1

# 2. Start Gazebo (in a new tab or background)
echo "Starting Gazebo..."
# Load the underwater world
gnome-terminal --tab --title="Gazebo" -- bash -c "export GZ_SIM_RESOURCE_PATH=\$GZ_SIM_RESOURCE_PATH:\$HOME/models_ws/src/bluerov2_gz/models; gz sim -v4 -r ~/models_ws/worlds/simple_underwater.sdf; exec bash"

# Wait for Gazebo to load
sleep 5

# 3. Start ArduPilot SITL
echo "Starting ArduPilot SITL..."
# We pass the initialization commands automatically!
gnome-terminal --tab --title="ArduPilot" -- bash -c "sim_vehicle.py -v ArduSub -f vectored_6dof --model JSON --console --out=127.0.0.1:14552 --add-param-file=$HOME/ros2_ws/src/mpc_controller/params/bluerov_init.parm; exec bash"

# Wait for ArduPilot to initialize
sleep 5

# 4. Start ROS 2 System (Bridge + MAVROS + MPC)
echo "Starting ROS 2 Nodes..."
gnome-terminal --tab --title="ROS2 Control" -- bash -c "source ~/ros2_ws/install/setup.bash; export PYTHONPATH=\$PYTHONPATH:\$HOME/ros2_ws/src/mpc_controller/BlueROV2; ros2 launch mpc_controller mpc_system.launch.py; exec bash"

echo "System Started!"