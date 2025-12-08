import rclpy
from rclpy.node import Node
import numpy as np
import json
from threading import Lock

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
import struct

class FoxgloveVisualizationNode(Node):
    def __init__(self):
        super().__init__('foxglove_visualization_node')
        
        # Configuration
        self.map_width = 100
        self.map_height = 100
        
        # État actuel
        self.robot_positions = {}
        self.mineral_map = None
        self.obstacle_map = None
        self.lock = Lock()
        
        # Couleurs des robots
        self.robot_colors = [
            (1.0, 0.0, 0.0, 1.0),   # Rouge - Robot 0
            (0.0, 1.0, 0.0, 1.0),   # Vert - Robot 1
            (0.0, 0.0, 1.0, 1.0),   # Bleu - Robot 2
            (1.0, 0.65, 0.0, 1.0)   # Orange - Robot 3
        ]
        
        # Setup des communications
        self.setup_subscribers()
        self.setup_publishers()
        
        # Timers
        self.viz_timer = self.create_timer(0.5, self.publish_foxglove_visualization)  # 2 Hz
        self.tf_timer = self.create_timer(0.1, self.publish_tf_transforms)  # 10 Hz
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        self.get_logger().info('🦊 Foxglove Visualization Node initialized')
    
    def setup_subscribers(self):
        """Configure tous les subscribers"""
        # Positions des robots
        for i in range(4):
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self.create_position_callback(i),
                10
            )
        
        # Cartes du serveur
        self.create_subscription(
            Float32MultiArray, '/mineral_map',
            self.mineral_callback, 10
        )
        
        self.create_subscription(
            OccupancyGrid, '/obstacle_map',
            self.obstacle_callback, 10
        )
    
    def setup_publishers(self):
        """Configure tous les publishers pour Foxglove"""
        # Markers pour la visualisation 3D
        self.robot_markers_pub = self.create_publisher(MarkerArray, '/foxglove/robot_markers', 10)
        self.mineral_markers_pub = self.create_publisher(MarkerArray, '/foxglove/mineral_markers', 10)
        self.high_value_markers_pub = self.create_publisher(MarkerArray, '/foxglove/high_value_markers', 10)
        
        # PointCloud pour les concentrations minérales
        self.mineral_cloud_pub = self.create_publisher(PointCloud2, '/foxglove/mineral_cloud', 10)
        
        # Données structurées pour le panel Table
        self.robot_data_pub = self.create_publisher(String, '/foxglove/robot_data', 10)
        self.system_stats_pub = self.create_publisher(String, '/foxglove/system_stats', 10)
    
    def create_position_callback(self, robot_id):
        """Callback pour les positions des robots"""
        def callback(msg):
            with self.lock:
                self.robot_positions[robot_id] = (msg.x, msg.y)
        return callback
    
    def mineral_callback(self, msg):
        """Callback pour la carte minérale"""
        with self.lock:
            if len(msg.data) == self.map_height * self.map_width * 4:
                self.mineral_map = np.array(msg.data).reshape(
                    self.map_height, self.map_width, 4
                )
    
    def obstacle_callback(self, msg):
        """Callback pour la carte d'obstacles"""
        with self.lock:
            if len(msg.data) == self.map_height * self.map_width:
                self.obstacle_map = np.array(msg.data).reshape(
                    self.map_height, self.map_width
                )
    
    def publish_foxglove_visualization(self):
        """Publie toutes les données de visualisation pour Foxglove"""
        self.publish_robot_markers()
        self.publish_mineral_visualization()
        self.publish_robot_data()
        self.publish_system_stats()
    
    def publish_robot_markers(self):
        """Publie les markers pour les robots"""
        marker_array = MarkerArray()
        
        for robot_id, position in self.robot_positions.items():
            if position is not None:
                # Marker principal du robot
                marker = self.create_robot_marker(robot_id, position)
                marker_array.markers.append(marker)
                
                # Zone de détection
                detection_marker = self.create_detection_zone(robot_id, position)
                marker_array.markers.append(detection_marker)
                
                # Flèche de direction
                direction_marker = self.create_direction_arrow(robot_id, position)
                marker_array.markers.append(direction_marker)
        
        self.robot_markers_pub.publish(marker_array)
    
    def create_robot_marker(self, robot_id, position):
        """Crée un marker visuel pour un robot"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robots"
        marker.id = robot_id * 3  # Multiplier pour éviter les conflits d'ID
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.3  # Légèrement au-dessus du sol
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 2.5  # Diamètre
        marker.scale.y = 2.5  # Diamètre  
        marker.scale.z = 0.8  # Hauteur
        
        marker.color.r = self.robot_colors[robot_id][0]
        marker.color.g = self.robot_colors[robot_id][1]
        marker.color.b = self.robot_colors[robot_id][2]
        marker.color.a = 0.9
        
        marker.lifetime.nanosec = int(1e9)  # 1 seconde
        
        return marker
    
    def create_detection_zone(self, robot_id, position):
        """Crée une zone de détection autour du robot"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detection_zones"
        marker.id = robot_id * 3 + 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.05  # Au niveau du sol
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 12.0  # Diamètre de détection
        marker.scale.y = 12.0  # Diamètre de détection
        marker.scale.z = 0.1   # Épaisseur
        
        marker.color.r = 0.3
        marker.color.g = 0.3
        marker.color.b = 1.0
        marker.color.a = 0.2  # Très transparent
        
        marker.lifetime.nanosec = int(1e9)
        
        return marker
    
    def create_direction_arrow(self, robot_id, position):
        """Crée une flèche indiquant la direction"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "direction_arrows"
        marker.id = robot_id * 3 + 2
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.8  # Au-dessus du robot
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 3.0  # Longueur
        marker.scale.y = 0.5  # Largeur
        marker.scale.z = 0.5  # Hauteur
        
        marker.color.r = self.robot_colors[robot_id][0]
        marker.color.g = self.robot_colors[robot_id][1]
        marker.color.b = self.robot_colors[robot_id][2]
        marker.color.a = 0.8
        
        marker.lifetime.nanosec = int(1e9)
        
        return marker
    
    def publish_mineral_visualization(self):
        """Publie la visualisation des minéraux"""
        if self.mineral_map is None:
            return
        
        self.publish_mineral_markers()
        self.publish_mineral_pointcloud()
    
    def publish_mineral_markers(self):
        """Publie les markers pour les minéraux importants"""
        marker_array = MarkerArray()
        marker_id = 1000  # Commencer à un ID élevé
        
        with self.lock:
            for y in range(0, self.map_height, 2):  # Échantillonnage pour performance
                for x in range(0, self.map_width, 2):
                    mineral_concentrations = self.mineral_map[y, x, :]
                    max_concentration = np.max(mineral_concentrations)
                    
                    # Seulement les minéraux significatifs
                    if max_concentration > 0.3:
                        marker = self.create_mineral_marker(x, y, max_concentration, marker_id)
                        marker_array.markers.append(marker)
                        marker_id += 1
        
        self.mineral_markers_pub.publish(marker_array)
    
    def create_mineral_marker(self, x, y, concentration, marker_id):
        """Crée un marker pour un dépôt minéral"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "minerals"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        
        # Taille basée sur la concentration
        size = 0.3 + concentration * 2.0
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size * 0.5
        
        # Couleur basée sur la concentration et le type
        mineral_type = np.argmax(self.mineral_map[y, x, :])
        color = self.get_mineral_color(mineral_type, concentration)
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = min(0.8, concentration + 0.3)
        
        marker.lifetime.nanosec = int(2e9)  # 2 secondes
        
        return marker
    
    def get_mineral_color(self, mineral_type, concentration):
        """Retourne la couleur basée sur le type de minéral"""
        colors = [
            (1.0, 0.0, 0.0),   # Rouge - REE_Oxides
            (0.0, 1.0, 0.0),   # Vert - REE_Silicates
            (0.0, 0.0, 1.0),   # Bleu - REE_Phosphates
            (1.0, 1.0, 0.0)    # Jaune - REE_Carbonates
        ]
        
        base_color = colors[mineral_type % len(colors)]
        intensity = min(1.0, concentration * 1.5)
        
        return (
            base_color[0] * intensity,
            base_color[1] * intensity,
            base_color[2] * intensity
        )
    
    def publish_mineral_pointcloud(self):
        """Publie un PointCloud2 pour la visualisation des concentrations"""
        if self.mineral_map is None:
            return
        
        points = []
        
        with self.lock:
            for y in range(0, self.map_height, 3):  # Échantillonnage pour performance
                for x in range(0, self.map_width, 3):
                    mineral_concentrations = self.mineral_map[y, x, :]
                    max_concentration = np.max(mineral_concentrations)
                    
                    if max_concentration > 0.1:
                        points.append((x, y, max_concentration * 2.0, max_concentration))
        
        if points:
            cloud_msg = self.create_pointcloud2(points)
            self.mineral_cloud_pub.publish(cloud_msg)
    
    def create_pointcloud2(self, points):
        """Crée un message PointCloud2 à partir des points"""
        msg = PointCloud2()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.height = 1
        msg.width = len(points)
        
        # Définir les champs
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.fields = fields
        
        msg.is_bigendian = False
        msg.point_step = 16  # 4 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Serialiser les données
        msg.data = []
        for point in points:
            msg.data.extend(struct.pack('4f', *point))
        
        return msg
    
    def publish_tf_transforms(self):
        """Publie les transformations TF pour les robots"""
        transforms = []
        
        for robot_id, position in self.robot_positions.items():
            if position is not None:
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = "map"
                t.child_frame_id = f"robot_{robot_id}_base"
                t.transform.translation.x = float(position[0])
                t.transform.translation.y = float(position[1])
                t.transform.translation.z = 0.3  # Hauteur du robot
                t.transform.rotation.w = 1.0
                transforms.append(t)
        
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)
    
    def publish_robot_data(self):
        """Publie les données structurées des robots pour le panel Table"""
        robot_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'robots': {}
        }
        
        for robot_id, position in self.robot_positions.items():
            if position is not None:
                robot_data['robots'][robot_id] = {
                    'position': {'x': float(position[0]), 'y': float(position[1])},
                    'status': 'active',
                    'color': self.robot_colors[robot_id],
                    'last_update': self.get_clock().now().nanoseconds
                }
        
        msg = String()
        msg.data = json.dumps(robot_data)
        self.robot_data_pub.publish(msg)
    
    def publish_system_stats(self):
        """Publie les statistiques du système"""
        active_robots = len([p for p in self.robot_positions.values() if p is not None])
        
        stats = {
            'timestamp': self.get_clock().now().nanoseconds,
            'active_robots': active_robots,
            'total_robots': 4,
            'mineral_data_available': self.mineral_map is not None,
            'obstacle_data_available': self.obstacle_map is not None
        }
        
        # Ajouter les stats minérales si disponibles
        if self.mineral_map is not None:
            total_minerals = np.sum(self.mineral_map)
            high_value_count = np.sum(self.mineral_map > 0.7)
            
            stats['mineral_stats'] = {
                'total_concentration': float(total_minerals),
                'high_value_deposits': int(high_value_count),
                'average_concentration': float(np.mean(self.mineral_map))
            }
        
        msg = String()
        msg.data = json.dumps(stats)
        self.system_stats_pub.publish(msg)

def main():
    rclpy.init()
    
    try:
        node = FoxgloveVisualizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down Foxglove visualization node...')
    except Exception as e:
        node.get_logger().error(f'❌ Error in visualization node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()