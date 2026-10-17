#!/usr/bin/env bash
# Fallback to Docker when /var/lib/docker is I/O-starved: download the same base image
# (ros:humble-ros-base-jammy) from Docker Hub into a plain directory on a fast disk, install the
# packages from docker/Dockerfile there with apt inside a chroot.
#   sudo bash scripts/make_chroot.sh /opt/rosorin-jammy
set -euo pipefail
ROOT=${1:-/opt/rosorin-jammy}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -x "$ROOT/bin/bash" ]; then
  python3 "$HERE/scripts/pull_rootfs.py" ros:humble-ros-base-jammy "$ROOT"
fi
cp /etc/resolv.conf "$ROOT/etc/resolv.conf"
# --make-rslave is mandatory (see run.sh): shared rbinds multiply and unmounts leak to the host
for d in sys dev; do mountpoint -q "$ROOT/$d" || { mount --rbind /$d "$ROOT/$d" && mount --make-rslave "$ROOT/$d"; }; done
# the same package list as docker/Dockerfile
PKGS=$(sed -n '/apt-get install/,/rm -rf/p' "$HERE/docker/Dockerfile" | grep -v "apt-get install\|rm -rf" | tr -d '\\' | xargs)
unshare --pid --fork --mount chroot "$ROOT" /usr/bin/env DEBIAN_FRONTEND=noninteractive bash -c "mount -t proc proc /proc; apt-get update && apt-get install -y --no-install-recommends $PKGS && apt-get clean"
echo "chroot ready: $ROOT"
