from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Chemin vers le package
    pkg_path = get_package_share_directory('extraterrestrial_world')
    world_file = os.path.join(pkg_path, 'worlds', 'lunar_landscape.world')
    
    # Vérifier que le fichier monde existe
    if not os.path.exists(world_file):
        raise FileNotFoundError(f"World file not found: {world_file}")
    
    return LaunchDescription([
        # Lancer Gazebo Classic avec le monde lunaire
        ExecuteProcess(
            cmd=[
                'gazebo', 
                '--verbose',
                world_file,
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so'
            ],
            output='screen'
        ),
        
        # Noeud de contrôle manuel (optionnel)
        Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop',
            output='screen',
            prefix='xterm -e'
        ),
    ])