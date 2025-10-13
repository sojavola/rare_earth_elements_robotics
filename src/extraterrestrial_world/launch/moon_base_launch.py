from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Chemin vers le monde Gazebo
    world_path = os.path.join(
        get_package_share_directory('extraterrestrial_world'),
        'worlds',
        'lunar_landscape.world'
    )
    
    return LaunchDescription([
        # Lancer Gazebo avec le monde lunaire
        ExecuteProcess(
            cmd=['gazebo', '--verbose', world_path],
            output='screen'
        ),
        
        # Spawn le premier robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'robot1',
                '-topic', 'robot_description',  # Utiliser topic pour le modèle
                '-x', '0.0', '-y', '0.0', '-z', '0.5'
            ],
            output='screen'
        ),
    ])