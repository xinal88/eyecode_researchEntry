#!/usr/bin/env bash
# debug helper: start the world, then report what ROS 2 / Ignition see
source /opt/ros/humble/setup.bash; source /root/ros2_ws/install/setup.bash; source /root/demo/ws/typerc >/dev/null
export ROS_LOCALHOST_ONLY=1 DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 ROS_DOMAIN_ID=73
export IGN_GAZEBO_RESOURCE_PATH=/root/ros2_ws/install/rosorin_description/share:/root/ros2_ws/install/robot_gazebo/share
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99; Xvfb :99 -screen 0 1920x1080x24 +extension GLX >/dev/null 2>&1 &
for _ in $(seq 1 60); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 1; done
ros2 launch robot_gazebo room_worlds.launch.py > /root/demo/logs/diag_gazebo.log 2>&1 &
sleep 90
echo "--- ros2 topic info /scan"; timeout 30 ros2 topic info /scan 2>&1 | head -5
echo "--- ign topics"; timeout 20 ign topic -l 2>&1 | grep -E "scan|odom|clock|imu" | head
echo "--- ign scan echo"; timeout 30 ign topic -e -t /scan -n 1 2>&1 | head -5
echo "--- ros2 echo scan"; timeout 30 ros2 topic echo /scan --once 2>&1 | grep -E "frame_id|angle_min" | head -3
echo "--- clock"; timeout 20 ign topic -e -t /clock -n 1 2>&1 | head -4
