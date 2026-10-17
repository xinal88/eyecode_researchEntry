#!/usr/bin/env bash
# Host-side entry point for Part 1 (ROSOrin Pro Gazebo simulation).
#   ./run.sh image            build the docker image (ROS 2 Humble + Ignition Fortress + Nav2)
#   ./run.sh up               start the container (mounts this folder at /root/demo)
#   ./run.sh build            colcon build the patched official packages
#   ./run.sh teleop|slam|nav  run one demo stage (ENVS='-e RECORD=1' to screen-record)
#   ./run.sh all              build + teleop + slam + nav
#   ./run.sh shell            interactive shell in the container (manual tutorial commands)
#   ./run.sh down             remove the container
# RUNTIME=chroot runs the same stages in a chroot made by scripts/make_chroot.sh (needs sudo);
# used on the shared server where Docker's storage was too slow to build the image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME=rosorin_sim
IMAGE=rosorin-sim:humble
RUNTIME=${RUNTIME:-docker}
CHROOT=${CHROOT:-/opt/rosorin-jammy}

up() {
  if [ "$RUNTIME" = chroot ]; then
    sudo mkdir -p "$CHROOT/root/demo" "$CHROOT/root/ros2_ws/src"
    # rbind: device nodes are separate mounts on this host. --make-rslave is MANDATORY: with shared
    # propagation every later bind multiplies (55k mounts) and an unmount inside the chroot
    # propagates back and removes the HOST's /dev/null etc.
    for d in sys dev; do
      mountpoint -q "$CHROOT/$d" || { sudo mount --rbind /$d "$CHROOT/$d" && sudo mount --make-rslave "$CHROOT/$d"; }
    done
    mountpoint -q "$CHROOT/root/demo" || sudo mount --bind "$HERE" "$CHROOT/root/demo"
    mountpoint -q "$CHROOT/root/ros2_ws/src" || sudo mount --bind "$HERE/ws/src" "$CHROOT/root/ros2_ws/src"
    return
  fi
  docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null | grep -q true && return
  docker rm -f $NAME >/dev/null 2>&1 || true
  docker run -d --name $NAME --shm-size=2g \
    -v "$HERE":/root/demo \
    -v "$HERE/ws/src":/root/ros2_ws/src \
    $IMAGE sleep infinity >/dev/null
}

# run a command inside the runtime; ENVS = extra "-e KEY=VAL" pairs
inside() {
  if [ "$RUNTIME" = chroot ]; then
    local envs=()
    for e in ${ENVS:-}; do [ "$e" = -e ] || envs+=("$e"); done
    # own PID + mount namespace: stage.sh's pkill cleanup only sees the simulation's processes,
    # and everything started inside dies with the namespace
    sudo unshare --pid --fork --mount chroot "$CHROOT" /usr/bin/env -i HOME=/root \
      PATH=/usr/sbin:/usr/bin:/sbin:/bin TERM=xterm need_compile=False "${envs[@]}" \
      bash -c 'mount -t proc proc /proc && exec "$@"' _ "$@"
  else
    docker exec ${ENVS:-} $NAME "$@"
  fi
}

case "${1:-all}" in
image) docker build -t $IMAGE "$HERE/docker" ;;
up) up ;;
down)
  if [ "$RUNTIME" = chroot ]; then
    # never `umount -l` here and always cut propagation first, or the host's mounts go with it
    for d in root/ros2_ws/src root/demo dev sys; do
      mountpoint -q "$CHROOT/$d" && sudo mount --make-rprivate "$CHROOT/$d" && sudo umount -R "$CHROOT/$d"
    done
  else docker rm -f $NAME; fi ;;
shell) up; if [ "$RUNTIME" = chroot ]; then inside bash; else docker exec -it $NAME bash; fi ;;
all) up; for s in build teleop slam nav; do inside bash /root/demo/scripts/stage.sh $s; done ;;
*) up; inside bash /root/demo/scripts/stage.sh "$1" ;;
esac
