#!/usr/bin/env python3
"""Scripted teleoperation of the simulated ROSOrin Pro (mecanum) + sensor sanity report.

Drives the same topic the official keyboard teleop uses (/controller/cmd_vel) through a
sequence that shows every mecanum degree of freedom (forward/back, strafe, rotate, diagonal),
and prints live statistics of the simulated sensors (/scan LiDAR, /scan/points, /imu, /odom).

  python3 teleop_demo.py [--out /root/demo/results/teleop.json]
"""
import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, PointCloud2

# (label, vx, vy, wz, seconds)
SEQUENCE = [
    ("forward", 0.3, 0.0, 0.0, 3.0),
    ("rotate left", 0.0, 0.0, 0.8, 3.0),
    ("strafe left (mecanum)", 0.0, 0.25, 0.0, 3.0),
    ("strafe right (mecanum)", 0.0, -0.25, 0.0, 3.0),
    ("diagonal", 0.2, 0.2, 0.0, 3.0),
    ("arc", 0.25, 0.0, -0.6, 4.0),
    ("backward", -0.25, 0.0, 0.0, 3.0),
    ("stop", 0.0, 0.0, 0.0, 1.0),
]


class TeleopDemo(Node):
    def __init__(self):
        super().__init__('teleop_demo')
        self.pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.stats = {'scan': 0, 'points': 0, 'imu': 0, 'odom': 0}
        self.last_scan = None
        self.odom = None
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/scan/points', lambda m: self.count('points'), qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', lambda m: self.count('imu'), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

    def count(self, k):
        self.stats[k] += 1

    def on_scan(self, m):
        self.count('scan')
        self.last_scan = m

    def on_odom(self, m):
        self.count('odom')
        self.odom = m

    def pose(self):
        if self.odom is None:
            return None
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return round(p.x, 3), round(p.y, 3), round(math.degrees(yaw), 1)

    def spin_for(self, sec, twist=None):
        end = time.time() + sec
        while time.time() < end:
            if twist is not None:
                self.pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rclpy.init()
    n = TeleopDemo()
    n.get_logger().info('waiting for sensors ...')
    t0 = time.time()
    while (n.last_scan is None or n.odom is None) and time.time() - t0 < 120:
        n.spin_for(0.5)
    report = {'segments': []}
    for name, vx, vy, wz, sec in SEQUENCE:
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
        before = n.pose()
        n.spin_for(sec, tw)
        after = n.pose()
        n.get_logger().info(f'{name:24s} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})  pose {before} -> {after}')
        report['segments'].append({'name': name, 'cmd': [vx, vy, wz], 'pose_before': before, 'pose_after': after})
    # sensor rates over a 5 s window
    for k in n.stats:
        n.stats[k] = 0
    n.spin_for(5.0)
    rates = {k: v / 5.0 for k, v in n.stats.items()}
    s = n.last_scan
    valid = [r for r in s.ranges if s.range_min < r < s.range_max] if s else []
    report['sensor_rates_hz_wall'] = rates
    if s:
        report['scan'] = {'frame': s.header.frame_id, 'n_beams': len(s.ranges), 'valid': len(valid),
                          'fov_deg': math.degrees(s.angle_max - s.angle_min),
                          'min_range': min(valid) if valid else None, 'max_range_cfg': s.range_max}
    n.get_logger().info(f'sensor rates (wall-clock Hz): {rates}')
    n.get_logger().info(f"scan: {report.get('scan')}")
    if args.out:
        json.dump(report, open(args.out, 'w'), indent=2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
