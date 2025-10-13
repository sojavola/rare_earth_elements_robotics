import rclpy
from rclpy.node import Node
import threading
import json
import time
from typing import Dict, Any, Optional

class ROS2Connector(Node):
    def __init__(self):
        super().__init__('dashboard_connector')
        
        self.connected = False
        self.last_data_update = None
        self.cached_data = {
            'mission_data': {},
            'mineral_data': None,
            'robot_data': {},
            'ai_analysis': None,
            'spectral_data': None
        }
        
        self.setup_subscribers()
        self.connection_thread = threading.Thread(target=self._monitor_connection)
        self.connection_thread.daemon = True
        self.connection_thread.start()
    
    def setup_subscribers(self):
        """Configure les subscribers ROS2"""
        try:
            # Subscriber pour les données de mission
            self.mission_sub = self.create_subscription(
                String, '/mission_data', self.mission_data_callback, 10
            )
            
            # Subscriber pour les données minérales
            self.mineral_sub = self.create_subscription(
                Float32MultiArray, '/mineral_map', self.mineral_data_callback, 10
            )
            
            # Subscriber pour le statut des robots
            self.robot_sub = self.create_subscription(
                String, '/robot_status', self.robot_data_callback, 10
            )
            
            self.connected = True
            self.get_logger().info('Dashboard ROS2 connector initialized')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize ROS2 connector: {e}')
            self.connected = False
    
    def mission_data_callback(self, msg):
        """Callback pour les données de mission"""
        try:
            self.cached_data['mission_data'] = json.loads(msg.data)
            self.last_data_update = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid mission data format')
    
    def mineral_data_callback(self, msg):
        """Callback pour les données minérales"""
        # Convertir les données minérales
        import numpy as np
        if len(msg.data) > 0:
            # Supposons une forme 100x100x4 pour la carte minérale
            mineral_map = np.array(msg.data).reshape(100, 100, 4)
            self.cached_data['mineral_data'] = mineral_map
            self.last_data_update = time.time()
    
    def robot_data_callback(self, msg):
        """Callback pour le statut des robots"""
        try:
            robot_data = json.loads(msg.data)
            self.cached_data['robot_data'].update(robot_data)
            self.last_data_update = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid robot data format')
    
    def _monitor_connection(self):
        """Surveille la connexion ROS2"""
        while True:
            if self.last_data_update and (time.time() - self.last_data_update) > 10:
                self.connected = False
            else:
                self.connected = True
            time.sleep(5)
    
    def is_connected(self) -> bool:
        """Vérifie si la connexion ROS2 est active"""
        return self.connected
    
    def get_mission_data(self) -> Dict[str, Any]:
        """Récupère les données de mission"""
        return self.cached_data['mission_data']
    
    def get_mineral_data(self):
        """Récupère les données minérales"""
        return self.cached_data['mineral_data']
    
    def get_robot_data(self) -> Dict[str, Any]:
        """Récupère les données des robots"""
        return self.cached_data['robot_data']
    
    def get_ai_analysis(self) -> Optional[str]:
        """Récupère l'analyse IA"""
        return self.cached_data['ai_analysis']
    
    def get_spectral_data(self):
        """Récupère les données spectrales"""
        return self.cached_data['spectral_data']
    
    def send_command(self, command: str, robot_id: str = None):
        """Envoie une commande aux robots"""
        if not self.connected:
            return False
        
        try:
            # Implémentation de l'envoi de commande
            self.get_logger().info(f'Sending command: {command} to {robot_id}')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to send command: {e}')
            return False

class DummyROS2Connector:
    """Connecteur factice pour le développement sans ROS2"""
    
    def __init__(self):
        self.connected = True
        self.cached_data = self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        """Génère des données factices pour le développement"""
        import numpy as np
        
        return {
            'mission_data': {
                'minerals_discovered': 45,
                'area_explored': 65,
                'high_value_samples': 8,
                'science_score': 1245,
                'active_robots': 4
            },
            'mineral_data': np.random.rand(100, 100, 4),
            'robot_data': {
                'Robot_1': {'battery': 85, 'status': 'Exploring', 'samples': 12, 'distance': 245},
                'Robot_2': {'battery': 72, 'status': 'Analyzing', 'samples': 8, 'distance': 189},
                'Robot_3': {'battery': 91, 'status': 'Exploring', 'samples': 15, 'distance': 312},
                'Robot_4': {'battery': 68, 'status': 'Returning', 'samples': 10, 'distance': 267}
            },
            'ai_analysis': "L'analyse des données actuelles indique un fort potentiel dans la région nord-est...",
            'spectral_data': np.random.rand(100, 4)
        }
    
    def is_connected(self) -> bool:
        return True
    
    def get_mission_data(self):
        return self.cached_data['mission_data']
    
    def get_mineral_data(self):
        return self.cached_data['mineral_data']
    
    def get_robot_data(self):
        return self.cached_data['robot_data']
    
    def get_ai_analysis(self):
        return self.cached_data['ai_analysis']
    
    def get_spectral_data(self):
        return self.cached_data['spectral_data']
    
    def send_command(self, command: str, robot_id: str = None):
        print(f"[DUMMY] Command sent: {command} to {robot_id}")
        return True