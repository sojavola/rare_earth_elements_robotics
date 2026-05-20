"""
Multi-robot launch file — ree_exploration_dqn.

Starts 4 independent (dqn_agent, dqn_trainer) pairs (robot_id 0–3).
Each pair is fully isolated: its own topics, model dir, TensorBoard dir, CSV dir,
and replay buffer — no shared weights, no shared replay memory (DTDE).

Usage:
    ros2 launch ree_exploration_dqn ree_dqn_multi.launch.py

Per-robot topics (example for robot_id=1):
    /robot_1/agent_experience    — agent → trainer (experience tuples)
    /robot_1/dqn/weight_update   — trainer → agent (updated weights)
    /robot_1/agent/epsilon       — trainer → agent (current epsilon)

Per-robot paths (example for robot_id=1):
    models/dqn/robot_1/latest.pt
    models/dqn/robot_1/replay_buffer.pkl.gz
    tensorboard_logs/dqn/robot_1/
    logs/dqn/robot_1/episodes.csv
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _robot_pair(robot_id: int, params_file: str) -> list:
    """Return [agent_node, trainer_node] for one robot."""
    rid = str(robot_id)
    agent = Node(
        package='ree_exploration_dqn',
        executable='dqn_agent',
        name=f'dqn_agent_{rid}',
        parameters=[
            params_file,
            {'robot_id': robot_id},
        ],
        output='screen',
        emulate_tty=True,
    )
    trainer = Node(
        package='ree_exploration_dqn',
        executable='dqn_trainer',
        name=f'dqn_trainer_{rid}',
        parameters=[
            params_file,
            {'robot_id': robot_id},
        ],
        output='screen',
        emulate_tty=True,
    )
    return [agent, trainer]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('ree_exploration_dqn')
    params_file = os.path.join(pkg_share, 'config', 'dqn_params.yaml')

    nodes = []
    for rid in range(4):
        nodes.extend(_robot_pair(rid, params_file))

    return LaunchDescription(nodes)
