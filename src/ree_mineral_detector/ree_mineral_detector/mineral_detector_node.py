import rclpy
from rclpy.node import Node
import numpy as np
import json
import time
import math
from threading import Lock
from collections import deque

from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

class MineralDetectorNode(Node):
    def __init__(self):
        super().__init__('mineral_detector_node')
        
        # Configuration
        self.map_width = 100
        self.map_height = 100
        
        # État des robots et carte
        self.robot_positions = {}
        self.mineral_map = None
        self.lock = Lock()
        
        # Système de détection avancé
        self.active_detections = []
        self.detection_history = deque(maxlen=100)  # Historique des 100 dernières détections
        self.scan_radius = 4  # Rayon de scan en cases
        self.min_detection_threshold = 0.3
        
        # Types de terres rares avec propriétés uniques
        self.ree_types = {
            0: {
                'name': 'REE_Oxides', 
                'color': (0.9, 0.1, 0.1), 
                'symbol': '🔴', 
                'rarity': 1.0,
                'emissive_color': (1.0, 0.3, 0.3)
            },
            1: {
                'name': 'REE_Silicates', 
                'color': (0.1, 0.8, 0.1), 
                'symbol': '🟢', 
                'rarity': 1.3,
                'emissive_color': (0.3, 1.0, 0.3)
            },
            2: {
                'name': 'REE_Phosphates', 
                'color': (0.1, 0.3, 0.9), 
                'symbol': '🔵', 
                'rarity': 1.7,
                'emissive_color': (0.3, 0.5, 1.0)
            },
            3: {
                'name': 'REE_Carbonates', 
                'color': (0.9, 0.8, 0.1), 
                'symbol': '🟡', 
                'rarity': 2.0,
                'emissive_color': (1.0, 1.0, 0.3)
            }
        }
        
        # Setup des communications
        self.setup_subscribers()
        self.setup_publishers()
        
        # Timers
        self.detection_timer = self.create_timer(0.4, self.process_detections)  # 2.5 Hz
        self.analysis_timer = self.create_timer(2.0, self.publish_advanced_analysis)  # 0.5 Hz
        self.visualization_timer = self.create_timer(0.5, self.publish_advanced_visualization)  # 2 Hz
        
        # Statistiques
        self.robot_stats = {i: {'detections': 0, 'total_value': 0.0} for i in range(4)}
        
        self.get_logger().info('🎯 Advanced Mineral Detector Node started!')
        self.get_logger().info('   └── Scanning radius: {} units'.format(self.scan_radius))
        self.get_logger().info('   └── Detection threshold: {}'.format(self.min_detection_threshold))
    
    def setup_subscribers(self):
        """Configure tous les subscribers"""
        # Positions des robots
        for i in range(4):
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self.create_robot_callback(i),
                10
            )
        
        # Carte minérale
        self.create_subscription(
            Float32MultiArray, '/mineral_map',
            self.mineral_callback, 10
        )
    
    def setup_publishers(self):
        """Configure tous les publishers avancés"""
        # Détections individuelles
        self.detection_pub = self.create_publisher(String, '/mineral_detections', 10)
        
        # Analyse avancée
        self.analysis_pub = self.create_publisher(String, '/mineral_analysis', 10)
        self.robot_stats_pub = self.create_publisher(String, '/robot_mineral_stats', 10)
        
        # Visualisation avancée
        self.markers_pub = self.create_publisher(MarkerArray, '/advanced_mineral_markers', 10)
        self.hotspot_pub = self.create_publisher(MarkerArray, '/mineral_hotspots', 10)
        self.scanning_pub = self.create_publisher(MarkerArray, '/scanning_zones', 10)
        
        # Alertes spéciales
        self.alert_pub = self.create_publisher(String, '/mineral_alerts', 10)
    
    def create_robot_callback(self, robot_id):
        def callback(msg):
            with self.lock:
                self.robot_positions[robot_id] = {
                    'x': msg.x, 
                    'y': msg.y, 
                    'theta': msg.theta,
                    'timestamp': time.time()
                }
        return callback
    
    def mineral_callback(self, msg):
        with self.lock:
            if len(msg.data) == self.map_height * self.map_width * 4:
                self.mineral_map = np.array(msg.data).reshape(
                    self.map_height, self.map_width, 4
                )
    
    def process_detections(self):
        """Processus principal de détection avancée"""
        if self.mineral_map is None:
            return
        
        with self.lock:
            current_time = time.time()
            
            # Nettoyer les anciennes détections
            self.active_detections = [
                d for d in self.active_detections 
                if current_time - d['timestamp'] < 10.0  # 10 secondes de durée de vie
            ]
            
            # Scanner pour chaque robot
            for robot_id, robot_data in self.robot_positions.items():
                if robot_data is None:
                    continue
                
                x, y = int(robot_data['x']), int(robot_data['y'])
                
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Scanner la zone autour du robot
                    detections = self.advanced_scan(x, y, robot_id)
                    
                    for detection in detections:
                        if self.is_new_detection(detection):
                            self.register_detection(detection)
    
    def advanced_scan(self, center_x, center_y, robot_id):
        """Scan avancé avec analyse de signal"""
        detections = []
        
        for dy in range(-self.scan_radius, self.scan_radius + 1):
            for dx in range(-self.scan_radius, self.scan_radius + 1):
                x, y = center_x + dx, center_y + dy
                
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    # Calculer la distance et l'angle
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance > self.scan_radius:
                        continue
                    
                    # Obtenir les concentrations
                    concentrations = self.mineral_map[y, x, :]
                    max_conc = np.max(concentrations)
                    
                    if max_conc > self.min_detection_threshold:
                        mineral_type = np.argmax(concentrations)
                        ree_info = self.ree_types[mineral_type]
                        
                        # Calculs avancés
                        signal_strength = self.calculate_signal_strength(max_conc, distance)
                        value_score = max_conc * ree_info['rarity']
                        confidence = self.calculate_confidence(signal_strength, distance)
                        
                        detection = {
                            'id': f"{robot_id}_{x}_{y}_{mineral_type}_{int(time.time()*1000)}",
                            'robot_id': robot_id,
                            'position': (x, y),
                            'mineral_type': mineral_type,
                            'concentration': max_conc,
                            'distance': distance,
                            'signal_strength': signal_strength,
                            'value_score': value_score,
                            'confidence': confidence,
                            'timestamp': time.time(),
                            'ree_info': ree_info
                        }
                        detections.append(detection)
        
        return detections
    
    def calculate_signal_strength(self, concentration, distance):
        """Calcule la force du signal avec atténuation"""
        base_signal = concentration * 2.0
        attenuation = 1.0 / (1.0 + distance * 0.3)  # Atténuation avec la distance
        return base_signal * attenuation
    
    def calculate_confidence(self, signal_strength, distance):
        """Calcule le niveau de confiance de la détection"""
        distance_penalty = 1.0 / (1.0 + distance * 0.2)
        return min(1.0, signal_strength * distance_penalty)
    
    def is_new_detection(self, detection):
        """Vérifie si c'est une nouvelle détection significative"""
        for existing in self.active_detections:
            if (existing['position'] == detection['position'] and 
                existing['mineral_type'] == detection['mineral_type'] and
                abs(existing['timestamp'] - detection['timestamp']) < 5.0):
                return False
        return True
    
    def register_detection(self, detection):
        """Enregistre une nouvelle détection"""
        self.active_detections.append(detection)
        self.detection_history.append(detection)
        
        # Mettre à jour les stats du robot
        robot_id = detection['robot_id']
        self.robot_stats[robot_id]['detections'] += 1
        self.robot_stats[robot_id]['total_value'] += detection['value_score']
        
        # Publier la détection
        self.publish_detection_alert(detection)
        
        # Logger coloré
        self.log_detection(detection)
    
    def publish_detection_alert(self, detection):
        """Publie une alerte de détection"""
        alert_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'type': 'MINERAL_DETECTION',
            'severity': 'HIGH' if detection['value_score'] > 1.5 else 'MEDIUM' if detection['value_score'] > 1.0 else 'LOW',
            'detection': {
                'robot_id': detection['robot_id'],
                'position': {'x': detection['position'][0], 'y': detection['position'][1]},
                'mineral_type': detection['ree_info']['name'],
                'symbol': detection['ree_info']['symbol'],
                'concentration': float(detection['concentration']),
                'signal_strength': float(detection['signal_strength']),
                'value_score': float(detection['value_score']),
                'confidence': float(detection['confidence']),
                'distance_from_robot': float(detection['distance'])
            }
        }
        
        msg = String()
        msg.data = json.dumps(alert_data)
        self.detection_pub.publish(msg)
    
    def log_detection(self, detection):
        """Log coloré des détections"""
        ree_info = detection['ree_info']
        
        # Émojis de statut basés sur la valeur
        if detection['value_score'] > 1.5:
            status_emoji = '💎✨'  # Très haute valeur
        elif detection['value_score'] > 1.0:
            status_emoji = '💎'    # Haute valeur
        elif detection['value_score'] > 0.7:
            status_emoji = '🔶'    # Valeur moyenne
        else:
            status_emoji = '🔹'    # Faible valeur
        
        self.get_logger().info(
            f'{status_emoji} {ree_info["symbol"]} ROBOT_{detection["robot_id"]} '
            f'→ {ree_info["name"]} '
            f'| Score: {detection["value_score"]:.2f} '
            f'| Signal: {detection["signal_strength"]:.2f} '
            f'| Confiance: {detection["confidence"]:.1%} '
            f'| Position: ({detection["position"][0]}, {detection["position"][1]})'
        )
    
    def publish_advanced_analysis(self):
        """Publie l'analyse avancée du système"""
        if not self.active_detections:
            return
        
        # Calculs statistiques avancés
        total_detections = len(self.active_detections)
        value_scores = [d['value_score'] for d in self.active_detections]
        signal_strengths = [d['signal_strength'] for d in self.active_detections]
        
        avg_value = np.mean(value_scores)
        max_value = np.max(value_scores)
        avg_signal = np.mean(signal_strengths)
        
        # Analyse par type
        type_analysis = {}
        for mineral_type, info in self.ree_types.items():
            type_detections = [d for d in self.active_detections if d['mineral_type'] == mineral_type]
            if type_detections:
                type_analysis[info['name']] = {
                    'count': len(type_detections),
                    'avg_value': float(np.mean([d['value_score'] for d in type_detections])),
                    'max_value': float(np.max([d['value_score'] for d in type_detections]))
                }
        
        # Niveau d'activité
        activity_level = 'EXTREME' if total_detections > 15 else 'HIGH' if total_detections > 10 else 'MODERATE' if total_detections > 5 else 'LOW'
        
        analysis_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'system_analysis': {
                'total_active_detections': total_detections,
                'activity_level': activity_level,
                'average_value_score': float(avg_value),
                'peak_value_score': float(max_value),
                'average_signal_strength': float(avg_signal),
                'detection_quality': 'EXCELLENT' if avg_signal > 0.6 else 'GOOD' if avg_signal > 0.4 else 'FAIR'
            },
            'mineral_breakdown': type_analysis,
            'performance_metrics': {
                'scan_efficiency': min(100, int((total_detections / 20) * 100)),  # Efficacité en %
                'value_density': float(avg_value * total_detections / 100)  # Densité de valeur
            }
        }
        
        msg = String()
        msg.data = json.dumps(analysis_data)
        self.analysis_pub.publish(msg)
        
        # Publier les stats des robots
        self.publish_robot_stats()
    
    def publish_robot_stats(self):
        """Publie les statistiques individuelles des robots"""
        stats_data = {
            'timestamp': self.get_clock().now().nanoseconds,
            'robot_statistics': {}
        }
        
        for robot_id, stats in self.robot_stats.items():
            detections = stats['detections']
            avg_value = stats['total_value'] / detections if detections > 0 else 0
            
            stats_data['robot_statistics'][robot_id] = {
                'total_detections': detections,
                'total_value_collected': float(stats['total_value']),
                'average_detection_value': float(avg_value),
                'efficiency_rating': 'HIGH' if avg_value > 1.2 else 'MEDIUM' if avg_value > 0.8 else 'LOW',
                'productivity_score': detections * avg_value
            }
        
        msg = String()
        msg.data = json.dumps(stats_data)
        self.robot_stats_pub.publish(msg)
    
    def publish_advanced_visualization(self):
        """Publie la visualisation avancée"""
        self.publish_mineral_markers()
        self.publish_scanning_zones()
        self.publish_hotspot_markers()
    
    def publish_mineral_markers(self):
        """Publie les marqueurs minéraux avancés"""
        marker_array = MarkerArray()
        
        for i, detection in enumerate(self.active_detections):
            # Marqueur principal
            main_marker = self.create_advanced_mineral_marker(detection, i)
            marker_array.markers.append(main_marker)
            
            # Effets spéciaux pour les hautes valeurs
            if detection['value_score'] > 1.0:
                # Effet de halo
                halo_marker = self.create_halo_effect(detection, i + 1000)
                marker_array.markers.append(halo_marker)
                
                # Texte de valeur
                if detection['value_score'] > 1.5:
                    text_marker = self.create_value_text(detection, i + 2000)
                    marker_array.markers.append(text_marker)
        
        if marker_array.markers:
            self.markers_pub.publish(marker_array)
    
    def create_advanced_mineral_marker(self, detection, marker_id):
        """Crée un marqueur minéral avancé"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "advanced_minerals"
        marker.id = marker_id
        
        # Forme basée sur la valeur
        if detection['value_score'] > 1.5:
            marker.type = Marker.SPHERE  # Forme premium
        elif detection['value_score'] > 1.0:
            marker.type = Marker.CYLINDER  # Forme importante
        else:
            marker.type = Marker.CUBE  # Forme standard
        
        marker.action = Marker.ADD
        
        x, y = detection['position']
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        
        # Hauteur dynamique basée sur la valeur
        base_height = 0.5 + (detection['value_score'] * 0.8)
        marker.pose.position.z = base_height
        marker.pose.orientation.w = 1.0
        
        # Taille dynamique
        base_size = 1.0 + (detection['value_score'] * 1.2)
        marker.scale.x = base_size
        marker.scale.y = base_size
        marker.scale.z = base_size * 0.6
        
        # Couleur avancée avec effets
        ree_info = detection['ree_info']
        r, g, b = ree_info['color']
        
        # Intensité basée sur le signal
        intensity = min(1.0, detection['signal_strength'] * 1.5)
        
        marker.color.r = r * intensity
        marker.color.g = g * intensity
        marker.color.b = b * intensity
        marker.color.a = 0.9 - (detection['distance'] * 0.1)  # Transparence basée sur la distance
        
        # Animation de pulsation pour les valeurs moyennes à hautes
        if detection['value_score'] > 0.8:
            pulse = (math.sin(time.time() * 4) + 1) * 0.15
            marker.scale.x += pulse
            marker.scale.y += pulse
        
        marker.lifetime.nanosec = int(4e9)  # 4 secondes
        
        return marker
    
    def create_halo_effect(self, detection, marker_id):
        """Crée un effet de halo pour les dépôts importants"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "halo_effects"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        x, y = detection['position']
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.1  # Au niveau du sol
        marker.pose.orientation.w = 1.0
        
        # Taille du halo (plus grand que le marqueur principal)
        halo_size = 2.5 + (detection['value_score'] * 1.0)
        marker.scale.x = halo_size
        marker.scale.y = halo_size
        marker.scale.z = 0.1  # Très fin
        
        # Couleur du halo (couleur émissive)
        ree_info = detection['ree_info']
        r, g, b = ree_info['emissive_color']
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.3  # Très transparent
        
        # Animation de pulsation du halo
        halo_pulse = (math.sin(time.time() * 3) + 1) * 0.2
        marker.scale.x += halo_pulse
        marker.scale.y += halo_pulse
        
        marker.lifetime.nanosec = int(3e9)
        
        return marker
    
    def create_value_text(self, detection, marker_id):
        """Crée un texte flottant avec la valeur"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "value_text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        x, y = detection['position']
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 2.5 + (detection['value_score'] * 0.5)  # Hauteur basée sur la valeur
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = 0.8  # Taille du texte
        
        # Couleur du texte (or pour les très hautes valeurs)
        if detection['value_score'] > 2.0:
            marker.color.r = 1.0
            marker.color.g = 0.84
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        
        marker.color.a = 0.9
        
        ree_info = detection['ree_info']
        marker.text = f"{ree_info['symbol']} {detection['value_score']:.2f}"
        
        marker.lifetime.nanosec = int(4e9)
        
        return marker
    
    def publish_scanning_zones(self):
        """Publie les zones de scan des robots"""
        marker_array = MarkerArray()
        
        for robot_id, robot_data in self.robot_positions.items():
            if robot_data is not None:
                zone_marker = self.create_scanning_zone(robot_data, robot_id)
                if zone_marker:
                    marker_array.markers.append(zone_marker)
        
        if marker_array.markers:
            self.scanning_pub.publish(marker_array)
    
    def create_scanning_zone(self, robot_data, robot_id):
        """Crée une zone de scan visible autour du robot"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "scanning_zones"
        marker.id = robot_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(robot_data['x'])
        marker.pose.position.y = float(robot_data['y'])
        marker.pose.position.z = 0.05  # Au niveau du sol
        marker.pose.orientation.w = 1.0
        
        # Taille de la zone de scan
        marker.scale.x = self.scan_radius * 2.2  # Diamètre
        marker.scale.y = self.scan_radius * 2.2  # Diamètre
        marker.scale.z = 0.05  # Très fin
        
        # Couleur de la zone (bleu transparent)
        marker.color.r = 0.1
        marker.color.g = 0.3
        marker.color.b = 0.8
        marker.color.a = 0.15  # Très transparent
        
        marker.lifetime.nanosec = int(2e9)  # 2 secondes
        
        return marker
    
    def publish_hotspot_markers(self):
        """Publie les marqueurs de hotspots (zones à haute concentration)"""
        if self.mineral_map is None:
            return
        
        marker_array = MarkerArray()
        hotspot_id = 3000
        
        # Analyser la carte pour trouver les hotspots
        with self.lock:
            for y in range(0, self.map_height, 5):  # Échantillonnage
                for x in range(0, self.map_width, 5):
                    concentrations = self.mineral_map[y, x, :]
                    total_concentration = np.sum(concentrations)
                    
                    if total_concentration > 2.0:  # Seuil de hotspot
                        hotspot_marker = self.create_hotspot_marker(x, y, total_concentration, hotspot_id)
                        marker_array.markers.append(hotspot_marker)
                        hotspot_id += 1
        
        if marker_array.markers:
            self.hotspot_pub.publish(marker_array)
    
    def create_hotspot_marker(self, x, y, concentration, marker_id):
        """Crée un marqueur de hotspot"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mineral_hotspots"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        
        # Taille basée sur la concentration
        size = 1.0 + (concentration * 0.3)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size * 0.3
        
        # Couleur rouge pour les hotspots
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.4  # Transparent
        
        marker.lifetime.nanosec = int(10e9)  # 10 secondes
        
        return marker

def main():
    rclpy.init()
    
    try:
        node = MineralDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Advanced Mineral Detector Node shutting down...')
    except Exception as e:
        node.get_logger().error(f'❌ Error in detector node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()