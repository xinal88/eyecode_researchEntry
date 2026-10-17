"""SLAM + Nav2 (no map_server / AMCL) so the robot can explore autonomously while mapping.

Re-uses the official robot_gazebo slam.launch.py (slam_toolbox + RViz with slam.rviz) and the
official nav2_params.yaml; the Nav2 node list mirrors robot_gazebo/launch/include/navigation.launch.py
minus the localization group (slam_toolbox already provides map->odom and /map).

  ros2 launch /root/demo/launch/slam_nav.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('robot_gazebo')
    params = RewrittenYaml(source_file=os.path.join(pkg, 'config', 'nav2_params.yaml'),
                           param_rewrites={'use_sim_time': 'true', 'autostart': 'true'},
                           convert_types=True)
    remap = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    def nav_node(package, executable, extra_remap=()):
        return Node(package=package, executable=executable, name=executable, output='screen',
                    parameters=[params], remappings=remap + list(extra_remap))

    nodes = [
        nav_node('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        nav_node('nav2_smoother', 'smoother_server'),
        nav_node('nav2_planner', 'planner_server'),
        nav_node('nav2_behaviors', 'behavior_server'),
        nav_node('nav2_bt_navigator', 'bt_navigator'),
        nav_node('nav2_waypoint_follower', 'waypoint_follower'),
        nav_node('nav2_velocity_smoother', 'velocity_smoother',
                 [('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['controller_server', 'smoother_server', 'planner_server',
                                         'behavior_server', 'bt_navigator', 'waypoint_follower',
                                         'velocity_smoother']}]),
    ]
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'include', 'slam.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items())
    return LaunchDescription([SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'), slam] + nodes)
