"""
Launch file for ree_exploration_dqn.

Starts one dqn_agent and one dqn_trainer.  Both load hyperparameters from
config/dqn_params.yaml (installed into share/ree_exploration_dqn/config/).

Usage:
    ros2 launch ree_exploration_dqn ree_dqn.launch.py
    ros2 launch ree_exploration_dqn ree_dqn.launch.py robot_id:=2
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('ree_exploration_dqn')
    params_file = os.path.join(pkg_share, 'config', 'dqn_params.yaml')

    robot_id_arg = DeclareLaunchArgument(
        'robot_id',
        default_value='0',
        description='Integer robot identifier (0–3)',
    )

    agent_node = Node(
        package='ree_exploration_dqn',
        executable='dqn_agent',
        name='dqn_agent',
        parameters=[
            params_file,
            {'robot_id': LaunchConfiguration('robot_id')},
        ],
        output='screen',
        emulate_tty=True,
    )

    trainer_node = Node(
        package='ree_exploration_dqn',
        executable='dqn_trainer',
        name='dqn_trainer',
        parameters=[params_file],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        robot_id_arg,
        agent_node,
        trainer_node,
    ])
