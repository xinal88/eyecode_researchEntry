#!/usr/bin/env python3
"""Autonomous frontier exploration for SLAM (replaces manual keyboard driving).

Loop: read the live slam_toolbox /map -> find frontier cells (known-free next to unknown)
-> cluster them -> pick the best cluster (size / distance) -> send a Nav2 NavigateToPose
goal to a free cell near it with obstacle clearance -> repeat until no frontier is left
(or the time budget is spent). Failed goals are black-listed.

  python3 frontier_explorer.py --budget 900 --out /root/demo/results/exploration.json
"""
import argparse
import json
import math
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import Buffer, TransformListener

MIN_CLUSTER = 10      # cells (0.05 m) -> ignore tiny frontier specks
CLEARANCE = 0.30      # m from occupied cells for a goal
BLACKLIST_R = 0.6     # m


def clusters(mask):
    """8-connected components of a boolean grid (pure python BFS; maps are ~200x200)."""
    lab = np.zeros(mask.shape, np.int32)
    out, cur = [], 0
    H, W = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if lab[r, c]:
            continue
        cur += 1
        q, cells = deque([(r, c)]), []
        lab[r, c] = cur
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = cur
                        q.append((yy, xx))
        out.append(np.array(cells))
    return out


def dilate(m, k):
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dy * dy + dx * dx <= k * k:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


class Explorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map = None
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.odom_path = 0.0
        self.last_xy = None
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def on_map(self, m):
        self.map = m

    def on_odom(self, m):
        p = m.pose.pose.position
        if self.last_xy is not None:
            self.odom_path += math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
        self.last_xy = (p.x, p.y)

    def robot_xy(self):
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def pick_goal(self, blacklist, bounds=None):
        m = self.map
        g = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
        res, ox, oy = m.info.resolution, m.info.origin.position.x, m.info.origin.position.y
        free, unknown, occ = g == 0, g == -1, g >= 50
        near_unknown = dilate(unknown, 1)
        frontier = free & near_unknown
        safe = free & ~dilate(occ, int(round(CLEARANCE / res)))
        rxy = self.robot_xy()
        if rxy is None:
            return None, 0
        best, best_score, n_ok = None, -1, 0
        sy, sx = np.nonzero(safe)
        if len(sy) == 0:
            return None, 0
        for cells in clusters(frontier):
            if len(cells) < MIN_CLUSTER:
                continue
            cy, cx = cells.mean(0)
            # nearest safe cell to the cluster centroid
            d2 = (sy - cy) ** 2 + (sx - cx) ** 2
            i = int(np.argmin(d2))
            if d2[i] > (1.0 / res) ** 2:  # no safe spot within 1 m of this frontier
                continue
            gx, gy = ox + (sx[i] + 0.5) * res, oy + (sy[i] + 0.5) * res
            if bounds and not (bounds[0] <= gx <= bounds[1] and bounds[2] <= gy <= bounds[3]):
                continue  # geofence: the house has open doors, the ground plane outside is endless
            if any(math.hypot(gx - bx, gy - by) < BLACKLIST_R for bx, by in blacklist):
                continue
            n_ok += 1
            dist = math.hypot(gx - rxy[0], gy - rxy[1])
            score = len(cells) / (dist + 0.5)
            if score > best_score:
                best, best_score = (gx, gy, math.atan2(gy - rxy[1], gx - rxy[0])), score
        return best, n_ok

    def go(self, x, y, yaw, timeout):
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = x, y
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        goal.pose = ps
        fut = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10)
        h = fut.result()
        if h is None or not h.accepted:
            return False
        res = h.get_result_async()
        t0 = time.time()
        while not res.done():
            rclpy.spin_once(self, timeout_sec=0.2)
            if time.time() - t0 > timeout:
                h.cancel_goal_async()
                return False
        return res.result().status == 4  # STATUS_SUCCEEDED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=900.0, help='wall-clock seconds')
    ap.add_argument('--out', default=None)
    ap.add_argument("--bounds", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="only explore frontiers inside this box (map frame, m)")
    ap.add_argument("--goal_timeout", type=float, default=240.0, help="wall seconds per goal (sim runs slower than real time on a loaded host)")
    args = ap.parse_args()
    rclpy.init()
    n = Explorer()
    n.get_logger().info('waiting for /map and navigate_to_pose ...')
    while n.map is None or n.robot_xy() is None:
        rclpy.spin_once(n, timeout_sec=0.5)
    n.nav.wait_for_server()
    t_start, blacklist, log, empty_rounds = time.time(), [], [], 0
    while time.time() - t_start < args.budget:
        for _ in range(10):
            rclpy.spin_once(n, timeout_sec=0.1)
        goal, n_frontiers = n.pick_goal(blacklist, args.bounds)
        if goal is None:
            empty_rounds += 1
            if empty_rounds >= 3:
                n.get_logger().info('no reachable frontier left -> exploration complete')
                break
            time.sleep(2.0)
            continue
        empty_rounds = 0
        n.get_logger().info(f'goal {goal[0]:+.2f},{goal[1]:+.2f}  ({n_frontiers} frontier clusters)')
        ok = n.go(*goal, timeout=args.goal_timeout)
        log.append({'goal': goal, 'ok': ok, 't': time.time() - t_start, 'frontiers': n_frontiers})
        # visited or failed, never pick the same spot twice: a frontier that stays open after the
        # robot stood on it (e.g. behind furniture the LiDAR cannot see past) would loop forever
        blacklist.append(goal[:2])
    m = n.map
    g = np.array(m.data).reshape(m.info.height, m.info.width)
    summary = {'wall_time_s': time.time() - t_start, 'goals': log, 'n_goals': len(log),
               'n_success': sum(l['ok'] for l in log), 'odom_path_m': n.odom_path,
               'known_area_m2': float(((g >= 0).sum()) * m.info.resolution ** 2),
               'free_area_m2': float(((g == 0).sum()) * m.info.resolution ** 2)}
    n.get_logger().info(f'exploration summary: {json.dumps({k: v for k, v in summary.items() if k != "goals"})}')
    if args.out:
        json.dump(summary, open(args.out, 'w'), indent=2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
