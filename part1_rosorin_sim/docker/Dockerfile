# ROSOrin Pro simulation environment: ROS 2 Humble + Ignition Gazebo Fortress
# (same stack as Hiwonder's official Ubuntu 22.04 VM), headless-capable.
# ros-humble-ros-ign = the ros_ign_* compatibility packages the official launch files still use.
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
# dpkg fsyncs every unpacked file; on the host's btrfs storage that made the first build take
# ~3.8 h. Unsafe I/O is fine inside an image build (a crash just means rebuilding the layer).
RUN echo force-unsafe-io > /etc/dpkg/dpkg.cfg.d/02-unsafe-io \
 && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "60";\n' > /etc/apt/apt.conf.d/80-retries
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-ros-gz ros-humble-ros-ign \
      ros-humble-ign-ros2-control \
      ros-humble-ros2-control ros-humble-ros2-controllers \
      ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
      ros-humble-slam-toolbox \
      ros-humble-navigation2 \
      ros-humble-rviz2 ros-humble-tf2-tools ros-humble-tf-transformations \
      python3-colcon-common-extensions python3-pip \
      xvfb x11-utils xdotool ffmpeg mesa-utils libgl1-mesa-dri imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Hiwonder environment file expectations (see official .typerc)
ENV need_compile=False LIDAR_TYPE=MS200 MACHINE_TYPE=ROSOrin_Mecanum ROS_DOMAIN_ID=0 \
    IGN_GAZEBO_RESOURCE_PATH=/root/ros2_ws/install/rosorin_description/share:/root/ros2_ws/install/robot_gazebo/share
WORKDIR /root/ros2_ws
