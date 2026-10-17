#!/usr/bin/env python3
"""Autonomous navigation demo on the saved SLAM map (Nav2 + AMCL).

What the official tutorial does with RViz clicks, done programmatically and measured:
  1. "2D Pose Estimate"  -> publish /initialpose
  2. "2D Nav Goal"       -> NavigateToPose (single goal)
  3. "Waypoint mode"     -> FollowWaypoints (multi-point navigation)
Metrics per goal: success, sim-time duration, travelled path length (ground-truth /odom from
the Ignition OdometryPublisher), straight-line distance, final position error w.r.t. the goal
in the ground-truth frame, and AMCL localisation error at arrival.

  python3 nav_demo.py --out /root/demo/results/navigation.json
"""
import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import FollowWaypoints, NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

# goals in the map frame (= world frame: SLAM started at the spawn pose 0,0,0);
# chosen in open space of the house with >= 0.55 m clearance
SINGLE_GOAL = ('dining room', 2.11, 0.58, 0.0)
WAYPOINTS = [('top-left room', -2.89, 2.18, math.pi / 2),
             ('bottom-left room', -2.44, -1.27, math.pi),
             ('bottom corridor', 0.01, -2.92, 0.0),
             ('start', 0.0, 0.0, 0.0)]


def pose(x, y, yaw, stamp):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp = stamp
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


class NavDemo(Node):
    def __init__(self):
        super().__init__('nav_demo')
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints', qos)
        self.gt = None
        self.amcl = None
        self.path_len = 0.0
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl, 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.wp = ActionClient(self, FollowWaypoints, 'follow_waypoints')

    def on_odom(self, m):
        p = m.pose.pose.position
        if self.gt is not None:
            self.path_len += math.hypot(p.x - self.gt[0], p.y - self.gt[1])
        self.gt = (p.x, p.y)

    def on_amcl(self, m):
        self.amcl = (m.pose.pose.position.x, m.pose.pose.position.y)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def set_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.05
        for _ in range(5):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.init_pub.publish(msg)
            self.spin(0.5)

    def show_markers(self, goals):
        arr = MarkerArray()
        for i, (name, x, y, _) in enumerate(goals):
            m = Marker()
            m.header.frame_id = 'map'
            m.id, m.type, m.action = i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.4, 0.0, 0.9
            arr.markers.append(m)
            t = Marker()
            t.header.frame_id = 'map'
            t.id, t.type, t.action = 100 + i, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, 0.5
            t.scale.z = 0.3
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f'{i + 1}: {name}'
            arr.markers.append(t)
        self.marker_pub.publish(arr)

    def wait_result(self, send_future, timeout):
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=20)
        h = send_future.result()
        if h is None or not h.accepted:
            return None, None
        res = h.get_result_async()
        t0 = time.time()
        while not res.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                h.cancel_goal_async()
                return None, None
        return res.result().status, res.result().result

    def measure(self, name, gx, gy, t_start, start_xy, path0, status):
        err = math.hypot(self.gt[0] - gx, self.gt[1] - gy)
        loc = math.hypot(self.amcl[0] - self.gt[0], self.amcl[1] - self.gt[1]) if self.amcl else None
        r = {'goal': name, 'target': [gx, gy], 'success': status == 4, 'status': status,
             'sim_time_s': round(self.now_s() - t_start, 2),
             'path_length_m': round(self.path_len - path0, 3),
             'straight_line_m': round(math.hypot(gx - start_xy[0], gy - start_xy[1]), 3),
             'final_error_m': round(err, 3), 'amcl_error_m': None if loc is None else round(loc, 3)}
        self.get_logger().info(f'RESULT {json.dumps(r)}')
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--timeout', type=float, default=600.0)
    args = ap.parse_args()
    rclpy.init()
    n = NavDemo()
    while n.gt is None:
        n.spin(0.5)
    n.get_logger().info('setting initial pose (2D Pose Estimate) ...')
    n.set_initial_pose()
    n.nav.wait_for_server()
    n.wp.wait_for_server()
    n.spin(3.0)
    results = {'single_goal': None, 'multi_point': []}

    # ---- single goal (2D Nav Goal) ----
    name, gx, gy, yaw = SINGLE_GOAL
    n.show_markers([SINGLE_GOAL])
    goal = NavigateToPose.Goal()
    goal.pose = pose(gx, gy, yaw, n.get_clock().now().to_msg())
    t0, s0, p0 = n.now_s(), n.gt, n.path_len
    n.get_logger().info(f'single-goal navigation -> {name} ({gx}, {gy})')
    status, _ = n.wait_result(n.nav.send_goal_async(goal), args.timeout)
    results['single_goal'] = n.measure(name, gx, gy, t0, s0, p0, status)
    n.spin(2.0)

    # ---- multi-point (waypoint following) ----
    n.show_markers(WAYPOINTS)
    goal = FollowWaypoints.Goal()
    stamp = n.get_clock().now().to_msg()
    goal.poses = [pose(x, y, yw, stamp) for _, x, y, yw in WAYPOINTS]
    n.get_logger().info(f'multi-point navigation through {len(WAYPOINTS)} waypoints')
    # follow progress via feedback-free polling of ground truth to time each leg
    t0, s0, p0 = n.now_s(), n.gt, n.path_len
    fut = n.wp.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=20)
    h = fut.result()
    res = h.get_result_async()
    leg, leg_t0, leg_s0, leg_p0 = 0, t0, s0, p0
    t_wall = time.time()
    while not res.done() and time.time() - t_wall < args.timeout * len(WAYPOINTS):
        rclpy.spin_once(n, timeout_sec=0.1)
        if leg < len(WAYPOINTS):
            wn, wx, wy, _ = WAYPOINTS[leg]
            if math.hypot(n.gt[0] - wx, n.gt[1] - wy) < 0.3:
                results['multi_point'].append(n.measure(wn, wx, wy, leg_t0, leg_s0, leg_p0, 4))
                leg += 1
                leg_t0, leg_s0, leg_p0 = n.now_s(), n.gt, n.path_len
    missed = list(res.result().result.missed_waypoints) if res.done() else None
    results['multi_point_summary'] = {
        'status': res.result().status if res.done() else 'timeout', 'missed_waypoints': missed,
        'reached': leg, 'total': len(WAYPOINTS), 'sim_time_s': round(n.now_s() - t0, 2),
        'path_length_m': round(n.path_len - p0, 3)}
    n.get_logger().info(f"multi-point summary {json.dumps(results['multi_point_summary'])}")
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
