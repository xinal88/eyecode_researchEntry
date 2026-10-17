#!/usr/bin/env bash
# Runs one stage of the ROSOrin Pro simulation demo INSIDE the rosorin-sim container.
# Every stage starts Ignition Gazebo + RViz on a virtual display (Xvfb) and arranges the windows
# side by side. Set RECORD=1 to also screen-record each stage to /root/demo/media/<stage>.mp4
# (off by default; for a live demo run the same commands on a desktop with a real display).
#
#   stage.sh build      colcon build of the (patched) official packages
#   stage.sh teleop     drive the robot + visualise LiDAR / point cloud / depth camera / odom
#   stage.sh slam       slam_toolbox mapping with autonomous frontier exploration, save map
#   stage.sh nav        Nav2 + AMCL on the saved map: single goal + multi-point navigation
set -eo pipefail
STAGE=${1:?stage}
DEMO=/root/demo
MEDIA=$DEMO/media; RES=$DEMO/results; LOGS=$DEMO/logs
mkdir -p "$MEDIA" "$RES" "$LOGS"

source /opt/ros/humble/setup.bash
[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash
source $DEMO/ws/typerc >/dev/null
export ROS_DOMAIN_ID=${SIM_DOMAIN_ID:-73}  # own DDS domain on the shared server
export ROS_LOCALHOST_ONLY=1 DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1
export IGN_GAZEBO_RESOURCE_PATH=/root/ros2_ws/install/rosorin_description/share:/root/ros2_ws/install/robot_gazebo/share
export ROSORIN_MAP_DIR=$DEMO/maps
REC_FPS=${REC_FPS:-10}

start_display() {
  if ! xdpyinfo -display :99 >/dev/null 2>&1; then
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99  # stale lock of an Xvfb killed with a previous stage
    Xvfb :99 -screen 0 1920x1080x24 +extension GLX +render -noreset >"$LOGS/xvfb.log" 2>&1 &
  fi
  # on a loaded host Xvfb can take a while; starting Gazebo before it answers makes Qt fail
  # ("no Qt platform plugin") and Ogre2 fall back to EGL ("OpenGL 3.3 is not supported")
  for _ in $(seq 1 60); do xdpyinfo -display :99 >/dev/null 2>&1 && return 0; sleep 1; done
  echo "Xvfb did not come up"; return 1
}

wait_topic() {  # wait until a topic has a publisher
  for _ in $(seq 1 180); do
    # capture first: with pipefail, `| grep -q` would SIGPIPE ros2 and fail even on a match
    info=$(timeout 30 ros2 topic info "$1" 2>/dev/null || true)
    [[ "$info" == *"Publisher count: "[1-9]* ]] && return 0
    sleep 2
  done
  echo "timeout waiting for $1"; return 1
}

place() {  # place <window-name-regex> x y w h
  for _ in $(seq 1 90); do
    wid=$(xdotool search --onlyvisible --name "$1" 2>/dev/null | head -1 || true)
    if [ -n "$wid" ]; then
      xdotool windowmove "$wid" "$2" "$3" windowsize "$wid" "$4" "$5" || true
      return 0
    fi
    sleep 2
  done
  echo "window '$1' not found"
}

record_start() {
  [ "${RECORD:-0}" = 1 ] || return 0
  ffmpeg -y -loglevel error -f x11grab -video_size 1920x1080 -framerate "$REC_FPS" -i :99 \
    -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p "$MEDIA/$1.mp4" &
  REC_PID=$!
}

record_stop() {
  [ -n "${REC_PID:-}" ] || return 0
  kill -INT "$REC_PID" 2>/dev/null || true
  wait "$REC_PID" 2>/dev/null || true
}

cleanup() {
  record_stop 2>/dev/null || true
  pkill -INT -f "ros2 launch" || true; sleep 3
  pkill -9 -f "ign gazebo|ruby|rviz2|parameter_bridge|slam_toolbox|nav2|amcl|map_server|robot_state_publisher|joint_state_publisher|component_container|lifecycle_manager|static_transform" || true
}

launch_world() {  # extra args forwarded to room_worlds.launch.py
  ros2 launch robot_gazebo room_worlds.launch.py "$@" >"$LOGS/${STAGE}_gazebo.log" 2>&1 &
  wait_topic /scan
  wait_topic /odom
}

case "$STAGE" in
build)
  cd /root/ros2_ws && colcon build --symlink-install 2>&1 | tail -5
  ;;

teleop)
  trap cleanup EXIT
  start_display
  launch_world
  rviz2 -d $DEMO/rviz/sensors.rviz >"$LOGS/teleop_rviz.log" 2>&1 &
  place "Gazebo" 0 0 960 1080; place "RViz" 960 0 960 1080
  sleep 5
  ros2 topic hz /scan --window 50 >"$RES/scan_hz.txt" 2>&1 & HZ=$!
  record_start 01_teleop_sensors
  python3 $DEMO/scripts/teleop_demo.py --out "$RES/teleop.json" 2>&1 | tee "$LOGS/teleop_demo.log"
  record_stop; kill $HZ || true
  ros2 topic list > "$RES/topic_list.txt"
  ros2 run tf2_tools view_frames -o "$RES/tf_frames" >/dev/null 2>&1 || true
  ;;

slam)
  trap cleanup EXIT
  start_display
  launch_world nav:=true map_tf:=false  # nav:=true routes Nav2 /cmd_vel to the robot
  ros2 launch $DEMO/launch/slam_nav.launch.py >"$LOGS/slam_nav.log" 2>&1 &
  wait_topic /map
  place "Gazebo" 0 0 960 1080; place "RViz" 960 0 960 1080
  record_start 02_slam_mapping
  python3 $DEMO/scripts/frontier_explorer.py --budget "${BUDGET:-1200}" --bounds ${BOUNDS:--4.9 5.0 -3.95 3.95} --out "$RES/exploration.json" 2>&1 | tee "$LOGS/explorer.log"
  sleep 3
  record_stop
  # same command as the official tutorial (map_saver_cli with transient-local subscription)
  cd $DEMO/maps && ros2 run nav2_map_server map_saver_cli -f rosorin_house --ros-args -p map_subscribe_transient_local:=true
  ;;

nav)
  trap cleanup EXIT
  start_display
  launch_world nav:=true map_tf:=false
  ros2 launch robot_gazebo navigation.launch.py map:=rosorin_house >"$LOGS/navigation.log" 2>&1 &
  wait_topic /amcl_pose || true
  place "Gazebo" 0 0 960 1080; place "RViz" 960 0 960 1080
  sleep 5
  record_start 03_navigation
  python3 $DEMO/scripts/nav_demo.py --out "$RES/navigation.json" 2>&1 | tee "$LOGS/nav_demo.log"
  sleep 3
  record_stop
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "stage $STAGE done"
