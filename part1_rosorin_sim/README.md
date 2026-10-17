# Part 1 — ROSOrin Pro Gazebo simulation (ROS 2 Humble + Ignition Gazebo Fortress)

Follows the official Hiwonder tutorial (wiki.hiwonder.com → ROSOrin Pro → 9. Gazebo Simulation), using the
official `simulations.zip` feature package (`robot_gazebo`, `rosorin_description`) and `.typerc` from
Hiwonder's Google Drive. Instead of the 20+ GB VMware image, the same stack (Ubuntu 22.04, ROS 2 Humble,
Ignition Fortress, slam_toolbox, Nav2) runs in a Docker container, headless (Xvfb), with every stage
scripted (the demo video is recorded separately on a desktop).

## Layout
```
docker/Dockerfile          ROS 2 Humble + ros_gz + ign_ros2_control + slam_toolbox + Nav2 + Xvfb/ffmpeg
ws/src/simulations/        official robot_gazebo + rosorin_description, with the fixes below applied
ws/typerc                  official environment file (MACHINE_TYPE=ROSOrin_Mecanum, LIDAR_TYPE=MS200, ...)
patches/official_fixes.patch   diff of every change made to the official package
launch/slam_nav.launch.py  slam_toolbox + Nav2 (no AMCL) for autonomous exploration while mapping
scripts/teleop_demo.py     scripted teleop (forward/rotate/strafe/diagonal) + sensor statistics
scripts/frontier_explorer.py   frontier-based autonomous exploration for SLAM
scripts/nav_demo.py        2D Pose Estimate + single-goal (NavigateToPose) + multi-point (FollowWaypoints), measured
scripts/stage.sh           in-container orchestration of each stage (Gazebo + RViz, optional recording)
scripts/make_demo_video.py cuts the three stage recordings into media/part1_demo.mp4 (host, ffmpeg)
scripts/make_chroot.sh     same stack as the Dockerfile, as a chroot (fallback when Docker storage is too slow)
scripts/pull_rootfs.py     downloads ros:humble-ros-base-jammy from Docker Hub into a directory (no docker)
rviz/sensors.rviz          LiDAR scan / point cloud / depth image / odometry view
maps/                      map saved by SLAM (rosorin_house.pgm/.yaml)
media/part1_demo.mp4       4-minute demo video (the raw per-stage recordings are not committed)
results/                   JSON metrics, topic list, TF tree, logs
```

## Reproduce
```bash
./run.sh image     # docker build (≈1.5 GB of apt packages)
./run.sh build     # colcon build --symlink-install of the patched official packages
./run.sh teleop    # 1) run & control the robot, visualise LiDAR/IMU/camera     -> media/01_teleop_sensors.mp4
./run.sh slam      # 2) SLAM with autonomous exploration, save map                -> maps/rosorin_house.*
./run.sh nav       # 3) AMCL + Nav2 on the saved map: single goal + multi-point   -> media/03_navigation.mp4
./run.sh all       # everything
ENVS='-e RECORD=1' ./run.sh teleop   # same, screen-recorded to media/0N_*.mp4 (also slam, nav)
python3 scripts/make_demo_video.py   # -> media/part1_demo.mp4
python3 scripts/map_metrics.py maps/rosorin_house.yaml ws/src/simulations/robot_gazebo/maps/map_01.yaml
```
Without a usable Docker (what happened on the shared server used here), the identical stack runs in a chroot:
```bash
sudo bash scripts/make_chroot.sh /opt/rosorin-jammy   # ≈270 MB base image + ≈530 MB of apt packages
RUNTIME=chroot ./run.sh all                           # same stages; runs in its own PID/mount namespace
```
Interactive use with the exact tutorial commands (`./run.sh shell`, then e.g. with an X display):
```bash
ros2 launch robot_gazebo room_worlds.launch.py
ros2 launch robot_gazebo slam.launch.py
ros2 run robot_gazebo teleop_key_control
cd ~/demo/maps && ros2 run nav2_map_server map_saver_cli -f "map_01" --ros-args -p map_subscribe_transient_local:=true
ros2 launch robot_gazebo room_worlds.launch.py nav:=true map_tf:=false
ros2 launch robot_gazebo navigation.launch.py map:=map_01
```

## Results (Docker, 32-core host shared with other jobs, software rendering)
| Stage | Result | File |
|---|---|---|
| Teleop + sensors | all mecanum commands sent on `/controller/cmd_vel`; forward / rotate / diagonal / arc / backward move as commanded, **strafe does not** (see limitations). `/scan` 270 beams over 156°, `/scan/points`, `/imu`, `/odom` at 6 / 6 / 20 / 20 Hz wall clock (sim slower than real time), depth camera on `/depth_cam/*` | `results/teleop.json`, `topic_list.txt`, `tf_frames.pdf` |
| SLAM + autonomous exploration | 8 frontier goals, 7 reached, 35.5 m driven in 5.6 min; 74.1 m² mapped (reference 73.4 m²) | `results/exploration.json`, `maps/rosorin_house.*` |
| Map vs. Hiwonder reference `map_01` | occupied precision / recall @10 cm **0.996 / 0.998**, free-space IoU **0.965** (default `slam.yaml`: 0.34 / 0.41 / 0.81) | `results/map_metrics.json`, `map_comparison.png` |
| Single-goal navigation (AMCL + Nav2) | success, 2.06 m path for 2.19 m straight line, final error 0.15 m | `results/navigation.json` |
| Multi-point navigation (FollowWaypoints) | **5/5 waypoints**, 16.0 m in 90 s sim time, final error 0.14–0.23 m per waypoint, AMCL error 0.07–0.29 m | `results/navigation.json` |

## Known limitations
- **No lateral (strafe) motion.** The official `wheel.xacro` models the mecanum wheels as plain cylinders with isotropic
  friction (`mu1 = mu2`), so the 45° rollers are missing and the opposite wheel speeds of a strafe cancel out (`teleop.json`:
  ≈0.05 m for a 0.75 m strafe command). The official Nav2 config is differential (`max_vel_y: 0`), so SLAM/navigation are not
  affected. Fix: per-wheel anisotropic friction (`fdir1` at ±45°, `mu2 ≈ 0`) or roller links. Also the plugin's
  `wheel_radius` (0.05) differs from the URDF wheels (0.04), which scales `/wheel_odom`.
- An earlier `teleop.json` reported `/imu` at 0 Hz; that run predates fix P8 (`imu.gazebo.xacro` was never included).
  With P8 `/imu` is bridged and measured at ≈20 Hz wall clock (50 Hz sim).
- The top-left room (one ≈0.8 m door) was dropped from the waypoint route: under heavy host load DWB oscillated in that
  doorway for minutes ("Failed to make progress") while every other leg succeeded.

## Problems found in the official package and how they were fixed
| # | Problem (symptom) | Fix |
|---|---|---|
| P1 | `room_worlds.launch.py` / `worlds.launch.py` call `get_package_share_directory('holonomic_sim')`, a package not shipped → `PackageNotFoundError` at launch | removed the unused lookup |
| P2 | `nav:=true` is declared but never forwarded to the bridge, so Nav2's `/cmd_vel` is never remapped to `/controller/cmd_vel` → robot does not move during navigation | forward `nav` to `ros_ign_bridge.launch.py` |
| P3 | bridge always publishes a static identity `map→odom` TF, while slam_toolbox / AMCL also publish `map→odom` → two TF authorities, map "jumps" | new `map_tf` argument (default `true` = official behaviour); `map_tf:=false` for SLAM / navigation |
| P5 | `MecanumDrive` and `OdometryPublisher` both publish on `/odom` (different child frames) | wheel odometry moved to `/wheel_odom`; `/odom` = OdometryPublisher |
| P6 | malformed `<kp>1000000000.0.0</kp>` in `wheel.xacro` | `1000000000.0` |
| P7 | the tutorial's `navigation.launch.py map:=map_01` passes the bare name to map_server, which needs a yaml path | resolve bare names to `$ROSORIN_MAP_DIR/<name>.yaml` (fallback `robot_gazebo/maps`) |
| P8 | `imu.gazebo.xacro` is shipped but never included in `robot.gazebo.xacro` → `/imu` has no publisher | include it |
| P9 | default `slam.yaml` (±20° coarse angle search, low loop-closure thresholds) mis-matched rotations with the 156° MS200 FOV facing a single wall → map rotated ≈10° w.r.t. the reference, occupied precision 0.34 (see `results/slam_attempt1_default_params/`) | narrower angle search, stricter loop closure, higher odometry trust |

The patched workspace can be rebuilt from the official download at any time:
`cd ws/src && unzip simulations.zip && cd simulations && patch -p1 < ../../../patches/official_fixes.patch`
(`../ws/typerc` is the official `.typerc`).

Problems in our own tooling found on the second host (fixed in `scripts/`):
`frontier_explorer.py` sent its first goal before Nav2's lifecycle manager had activated the costmaps, the goal was rejected
and black-listed and, since a 156° scan from the spawn pose shows a single frontier, exploration "finished" after 4 s →
wait for `lifecycle_manager_navigation/is_active`, spin 360° first, retry instantly-failing goals. `nav_demo.py` waited for
navigation to be active *before* setting the initial pose, but the costmaps only activate once AMCL publishes `map→odom` (a
300 s deadlock) → initial pose first, sim-time stamps. Recordings had 1–3 frames: libx264 with one thread per core stalled on
the memory-pressured host → 2 threads + `ultrafast`, and wait for the first encoded data before starting a demo. Leftover
`ros2 launch`/`spawner`/`create` processes piled up between stages → killed in `cleanup()`.

Environment issues: no VMware/X server on the shared Linux server → Docker + Xvfb + software OpenGL
(llvmpipe); Docker's storage on that server was so I/O-starved that the image build took 3.8 h (and was lost in a
reboot), so the same packages were installed into a chroot instead (`make_chroot.sh`); the server is itself an
Incus container whose `/dev` nodes are individual mounts, so the chroot needs `mount --rbind /dev` (a plain
`--bind` leaves empty placeholder files and apt fails with "gpgv required"); the official launch files read `need_compile` from `.typerc` (sourced in
`stage.sh`); keyboard teleop needs a TTY, so a scripted publisher on the same topic is used for the recording.
