#!/usr/bin/env python3
"""Download a Docker Hub image's layers and unpack them into a directory (no docker needed).

Used to create the ROS 2 Humble rootfs for the chroot runtime when the local Docker storage is
too slow to export from:   sudo python3 pull_rootfs.py ros:humble-ros-base-jammy /opt/rosorin-jammy
"""
import json
import os
import shutil
import sys
import tarfile
import urllib.request

image, dest = sys.argv[1], sys.argv[2]
repo, tag = image.split(":")
repo = repo if "/" in repo else f"library/{repo}"
REG = "https://registry-1.docker.io/v2"


def get(url, token, accept=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", **({"Accept": accept} if accept else {})})
    return urllib.request.urlopen(req, timeout=120)


token = json.load(urllib.request.urlopen(
    f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]
idx = json.load(get(f"{REG}/{repo}/manifests/{tag}", token, ", ".join([
    "application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"])))
if "manifests" in idx:  # multi-arch index -> linux/amd64 manifest
    d = next(m["digest"] for m in idx["manifests"]
             if m.get("platform", {}).get("architecture") == "amd64" and m["platform"].get("os") == "linux")
    man = json.load(get(f"{REG}/{repo}/manifests/{d}", token, "application/vnd.oci.image.manifest.v1+json, "
                        "application/vnd.docker.distribution.manifest.v2+json"))
else:
    man = idx
os.makedirs(dest, exist_ok=True)
total = sum(l["size"] for l in man["layers"])
print(f"{len(man['layers'])} layers, {total / 1e6:.0f} MB compressed", flush=True)
for i, layer in enumerate(man["layers"]):
    tmp = os.path.join(dest, f".layer{i}.tar.gz")
    with get(f"{REG}/{repo}/blobs/{layer['digest']}", token) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    with tarfile.open(tmp) as tf:
        for m in tf.getmembers():  # OCI whiteouts: .wh.<name> deletes, .wh..wh..opq empties a dir
            base, name = os.path.split(m.name)
            if name == ".wh..wh..opq":
                p = os.path.join(dest, base)
                for c in os.listdir(p) if os.path.isdir(p) else []:
                    q = os.path.join(p, c)
                    shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
            elif name.startswith(".wh."):
                p = os.path.join(dest, base, name[4:])
                if os.path.lexists(p):
                    shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
        tf.extractall(dest, members=[m for m in tf.getmembers() if not os.path.basename(m.name).startswith(".wh.")],
                      filter="fully_trusted")
    os.remove(tmp)
    print(f"layer {i + 1}/{len(man['layers'])} done ({layer['size'] / 1e6:.0f} MB)", flush=True)
print("rootfs ready:", dest)
