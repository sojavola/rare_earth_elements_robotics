
import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
from scipy.ndimage import gaussian_filter

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import OccupancyGrid

class MineralMapGenerator:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        
    def generate_mineral_map(self):
        """Génère une carte minérale procédurale"""
        mineral_map = np.zeros((self.height, self.width, 4))
        
        # Générer des dépôts pour chaque type de minéral
        for mineral_idx in range(4):
            # Créer un bruit de base
            base = np.random.rand(self.height, self.width)
            
            # Appliquer un filtre gaussien pour créer des clusters
            filtered = gaussian_filter(base, sigma=3.0)
            
            # Seuillage pour créer des dépôts
            threshold = 0.6 + np.random.random() * 0.3
            deposits = np.where(filtered > threshold, filtered, 0)
            
            # Normaliser
            if np.max(deposits) > 0:
                deposits = deposits / np.max(deposits)
            
            mineral_map[:, :, mineral_idx] = deposits
        
        return mineral_map

class REEExplorationServer(Node):
    def __init__(self):
        super().__init__('ree_exploration_server')
        
        # Configuration
        self.map_width = 100
        self.map_height = 100
        self.num_robots = 4
        
        # Générateur de carte
        self.mineral_generator = MineralMapGenerator(self.map_width, self.map_height)
        
        # État du système
        self.mineral_map = None
        self.robot_positions = {}
        self.exploration_map = np.zeros((self.map_height, self.map_width))
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        
        # Verrou pour accès thread-safe
        self.lock = threading.Lock()
        
        # Publishers
        self.mineral_pub = self.create_publisher(Float32MultiArray, '/mineral_map', 10)
        self.obstacle_pub = self.create_publisher(OccupancyGrid, '/obstacle_map', 10)
        self.science_pub = self.create_publisher(Float32MultiArray, '/science_targets', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        
        # Subscribers pour positions des robots
        for i in range(self.num_robots):
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self.create_position_callback(i),
                10
            )
        
        # Subscriber pour actions de nettoyage
        for i in range(self.num_robots):
            self.create_subscription(
                Float32MultiArray,
                f'/robot_{i}/cleaning_action',
                self.create_cleaning_callback(i),
                10
            )
        
        # Timers
        self.map_timer = self.create_timer(2.0, self.publish_maps)  # 0.5 Hz
        self.status_timer = self.create_timer(5.0, self.publish_status)
        self.update_timer = self.create_timer(1.0, self.update_system)
        
        # Initialisation
        self.initialize_system()
        
        self.get_logger().info('🚀 REE Exploration Server Node initialized')
        self.get_logger().info(f'📊 Map size: {self.map_width}x{self.map_height}')
        self.get_logger().info(f'🤖 Number of robots: {self.num_robots}')
    
    def create_position_callback(self, robot_id):
        """Crée un callback pour la position d'un robot spécifique"""
        def callback(msg):
            with self.lock:
                self.robot_positions[robot_id] = (msg.x, msg.y)
                self.update_exploration_map(msg.x, msg.y)
                self.get_logger().debug(f'🤖 Robot {robot_id} position: ({msg.x}, {msg.y})')
        return callback
    
    def create_cleaning_callback(self, robot_id):
        """Crée un callback pour les actions de nettoyage"""
        def callback(msg):
            if len(msg.data) >= 2:
                x, y = int(msg.data[0]), int(msg.data[1])
                self.clean_area(x, y)
                self.get_logger().info(f'🧹 Robot {robot_id} explored area: ({x}, {y})')
        return callback
    
    def initialize_system(self):
        """Initialise le système"""
        with self.lock:
            # Générer la carte minérale initiale
            self.mineral_map = self.mineral_generator.generate_mineral_map()
            
            # Générer la carte d'obstacles
            self.generate_obstacles()
            
            # Initialiser les positions des robots
            for i in range(self.num_robots):
                self.robot_positions[i] = self.get_valid_start_position()
        
        self.get_logger().info('✅ System initialized')
    
    def generate_obstacles(self):
        """Génère des obstacles réalistes"""
        # Créer quelques obstacles fixes
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        
        # Ajouter des bords
        self.obstacle_map[0, :] = 100  # Bord supérieur
        self.obstacle_map[-1, :] = 100  # Bord inférieur
        self.obstacle_map[:, 0] = 100  # Bord gauche
        self.obstacle_map[:, -1] = 100  # Bord droit
        
        # Ajouter quelques obstacles internes
        num_obstacles = 10
        for _ in range(num_obstacles):
            x = np.random.randint(10, self.map_width - 10)
            y = np.random.randint(10, self.map_height - 10)
            size = np.random.randint(3, 8)
            
            for i in range(max(0, y-size), min(self.map_height, y+size+1)):
                for j in range(max(0, x-size), min(self.map_width, x+size+1)):
                    if np.sqrt((i-y)**2 + (j-x)**2) <= size:
                        self.obstacle_map[i, j] = 100
    
    def get_valid_start_position(self):
        """Retourne une position de départ valide"""
        while True:
            x = np.random.randint(10, self.map_width - 10)
            y = np.random.randint(10, self.map_height - 10)
            
            if self.obstacle_map[y, x] == 0:
                return (x, y)
    
    def update_exploration_map(self, x, y):
        """Met à jour la carte d'exploration"""
        exploration_radius = 5
        x_int = int(x)
        y_int = int(y)
        
        for i in range(max(0, y_int-exploration_radius), min(self.map_height, y_int+exploration_radius+1)):
            for j in range(max(0, x_int-exploration_radius), min(self.map_width, x_int+exploration_radius+1)):
                distance = np.sqrt((i-y_int)**2 + (j-x_int)**2)
                if distance <= exploration_radius:
                    self.exploration_map[i, j] = 1.0
    
    def clean_area(self, x, y):
        """Nettoie une zone autour de la position spécifiée"""
        cleaning_radius = 3
        x_int = int(x)
        y_int = int(y)
        
        with self.lock:
            for i in range(max(0, y_int-cleaning_radius), min(self.map_height, y_int+cleaning_radius+1)):
                for j in range(max(0, x_int-cleaning_radius), min(self.map_width, x_int+cleaning_radius+1)):
                    distance = np.sqrt((i-y_int)**2 + (j-x_int)**2)
                    if distance <= cleaning_radius:
                        # Réduire les concentrations minérales dans cette zone
                        for mineral_idx in range(4):
                            reduction = 1.0 - (distance / cleaning_radius) * 0.8
                            self.mineral_map[i, j, mineral_idx] *= reduction
    
    def update_system(self):
        """Met à jour l'état du système"""
        with self.lock:
            # Simuler une légère évolution des dépôts minéraux
            if np.random.random() < 0.1:  # 10% de chance à chaque mise à jour
                evolution = np.random.normal(0, 0.01, self.mineral_map.shape)
                self.mineral_map = np.clip(self.mineral_map + evolution, 0, 1)
    
    def publish_maps(self):
        """Publie les cartes mises à jour"""
        with self.lock:
            # Publier la carte minérale
            mineral_msg = Float32MultiArray()
            
            # Définir le layout pour la carte minérale
            mineral_msg.layout.dim.append(self.create_multi_array_dimension("height", self.map_height, self.map_height * self.map_width * 4))
            mineral_msg.layout.dim.append(self.create_multi_array_dimension("width", self.map_width, self.map_width * 4))
            mineral_msg.layout.dim.append(self.create_multi_array_dimension("channels", 4, 4))
            
            mineral_msg.data = self.mineral_map.flatten().tolist()
            self.mineral_pub.publish(mineral_msg)
            
            # Publier la carte d'obstacles
            obstacle_msg = OccupancyGrid()
            obstacle_msg.header.stamp = self.get_clock().now().to_msg()
            obstacle_msg.header.frame_id = "map"
            obstacle_msg.info.width = self.map_width
            obstacle_msg.info.height = self.map_height
            obstacle_msg.info.resolution = 0.1
            obstacle_msg.data = self.obstacle_map.flatten().astype(np.int8).tolist()
            self.obstacle_pub.publish(obstacle_msg)
            
            # Publier les cibles scientifiques
            science_msg = Float32MultiArray()
            science_targets = self.calculate_science_targets()
            
            # Définir le layout pour les cibles scientifiques
            science_msg.layout.dim.append(self.create_multi_array_dimension("height", self.map_height, self.map_height * self.map_width))
            science_msg.layout.dim.append(self.create_multi_array_dimension("width", self.map_width, self.map_width))
            
            science_msg.data = science_targets.flatten().tolist()
            self.science_pub.publish(science_msg)
        
        self.get_logger().debug('🗺️ Maps published')
    
    def create_multi_array_dimension(self, label, size, stride):
        """Crée une dimension pour MultiArray message"""
        from std_msgs.msg import MultiArrayDimension
        dim = MultiArrayDimension()
        dim.label = label
        dim.size = size
        dim.stride = stride
        return dim
    
    def calculate_science_targets(self):
        """Calcule les cibles scientifiques prioritaires"""
        # Combinaison de zones inexplorées et de forts potentiels minéraux
        unexplored = 1.0 - self.exploration_map
        mineral_potential = np.max(self.mineral_map, axis=2)
        
        # Cibles = zones inexplorées avec fort potentiel minéral
        targets = unexplored * mineral_potential
        return targets
    
    def publish_status(self):
        """Publie le statut du système"""
        status_msg = String()
        
        with self.lock:
            coverage = np.mean(self.exploration_map) * 100
            mineral_density = np.mean(self.mineral_map) * 100
            active_robots = len(self.robot_positions)
            
            status_text = (f"System Status: "
                          f"Coverage: {coverage:.1f}%, "
                          f"Mineral Density: {mineral_density:.1f}%, "
                          f"Active Robots: {active_robots}")
        
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        
        # Log moins fréquent pour éviter le spam
        if hasattr(self, 'status_counter'):
            self.status_counter += 1
        else:
            self.status_counter = 0
            
        if self.status_counter % 3 == 0:  # Log toutes les 3 publications
            self.get_logger().info(f'📊 {status_text}')

def main():
    rclpy.init()
    
    try:
        node = REEExplorationServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down server...')
    except Exception as e:
        node.get_logger().error(f'❌ Error in server node: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()    