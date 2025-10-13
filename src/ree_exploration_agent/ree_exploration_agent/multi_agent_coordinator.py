import rclpy
from rclpy.node import Node
import numpy as np
import threading
from collections import defaultdict, deque
import networkx as nx

from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D

class MultiAgentCoordinator(Node):
    def __init__(self, num_agents=4):
        super().__init__('multi_agent_coordinator')
        
        self.num_agents = num_agents
        self.agent_positions = [None] * num_agents
        self.agent_status = ['idle'] * num_agents
        self.shared_knowledge = SharedKnowledgeMap()
        self.communication_graph = CommunicationGraph(num_agents)
        
        # Coordination stratégique
        self.exploration_strategy = ExplorationStrategy(num_agents)
        self.task_allocator = TaskAllocator(num_agents)
        
        # Publishers pour chaque agent
        self.assignment_pubs = []
        for i in range(num_agents):
            pub = self.create_publisher(
                String, 
                f'/robot_{i}/task_assignment', 
                10
            )
            self.assignment_pubs.append(pub)
        
        # Subscribers pour les positions des agents
        self.position_subs = []
        for i in range(num_agents):
            sub = self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                lambda msg, idx=i: self.agent_position_callback(msg, idx),
                10
            )
            self.position_subs.append(sub)
        
        # Subscriber pour les découvertes partagées
        self.discovery_sub = self.create_subscription(
            Float32MultiArray,
            '/shared_discoveries',
            self.discovery_callback,
            10
        )
        
        # Timer pour la coordination
        self.coordination_timer = self.create_timer(2.0, self.coordinate_agents)  # 2 Hz
        
        self.get_logger().info(f'Multi-Agent Coordinator initialized with {num_agents} agents')
    
    def agent_position_callback(self, msg, agent_id):
        """Met à jour la position d'un agent"""
        self.agent_positions[agent_id] = (msg.x, msg.y)
        self.shared_knowledge.update_agent_position(agent_id, (msg.x, msg.y))
    
    def discovery_callback(self, msg):
        """Traite les découvertes partagées entre agents"""
        if len(msg.data) >= 3:
            agent_id = int(msg.data[0])
            x, y = int(msg.data[1]), int(msg.data[2])
            mineral_data = msg.data[3:]
            
            # Mettre à jour la connaissance partagée
            self.shared_knowledge.add_discovery(agent_id, (x, y), mineral_data)
            
            # Mettre à jour le graphe de communication
            self.communication_graph.add_communication(agent_id, 'discovery')
    
    def coordinate_agents(self):
        """Coordonne les agents selon la stratégie d'exploration"""
        if not all(pos is not None for pos in self.agent_positions):
            return  # Attendre que toutes les positions soient connues
        
        # Analyser l'état actuel de l'exploration
        exploration_state = self.analyze_exploration_state()
        
        # Allouer les tâches selon la stratégie
        task_assignments = self.task_allocator.allocate_tasks(
            self.agent_positions, 
            self.shared_knowledge,
            exploration_state
        )
        
        # Envoyer les assignations aux agents
        for agent_id, assignment in task_assignments.items():
            self.send_task_assignment(agent_id, assignment)
        
        # Optimiser la communication
        self.optimize_communication()
    
    def analyze_exploration_state(self):
        """Analyse l'état actuel de l'exploration"""
        state = {
            'coverage_rate': self.shared_knowledge.get_coverage_rate(),
            'mineral_hotspots': self.shared_knowledge.get_mineral_hotspots(),
            'frontier_regions': self.shared_knowledge.get_frontier_regions(),
            'agent_distribution': self.calculate_agent_distribution(),
            'communication_efficiency': self.communication_graph.get_efficiency()
        }
        return state
    
    def calculate_agent_distribution(self):
        """Calcule la distribution spatiale des agents"""
        positions = [pos for pos in self.agent_positions if pos is not None]
        if not positions:
            return {'uniformity': 0.0, 'clustering': 0.0}
        
        # Calculer l'uniformité de la distribution
        from scipy.spatial import distance
        distances = distance.pdist(positions)
        uniformity = np.std(distances) if len(distances) > 0 else 0.0
        
        # Calculer le degré de clustering
        clustering = 1.0 / (1.0 + uniformity)  # Métrique simplifiée
        
        return {
            'uniformity': uniformity,
            'clustering': clustering,
            'min_distance': np.min(distances) if len(distances) > 0 else 0.0
        }
    
    def optimize_communication(self):
        """Optimise le graphe de communication entre agents"""
        # Analyser la connectivité actuelle
        connectivity = self.communication_graph.analyze_connectivity()
        
        # Si la connectivité est faible, ajuster les positions
        if connectivity['efficiency'] < 0.7:
            self.adjust_agent_positions_for_communication()
    
    def adjust_agent_positions_for_communication(self):
        """Ajuste les positions des agents pour améliorer la communication"""
        # Stratégie simple: rapprocher les agents les plus isolés
        central_positions = self.find_central_positions()
        
        for agent_id, position in enumerate(self.agent_positions):
            if position is not None:
                # Calculer la distance au centre le plus proche
                distances = [np.linalg.norm(np.array(position) - np.array(central_pos)) 
                           for central_pos in central_positions]
                min_distance = min(distances)
                
                # Si trop éloigné, suggérer un rapprochement
                if min_distance > 50:  # Seuil arbitraire
                    nearest_center = central_positions[np.argmin(distances)]
                    self.suggest_position_adjustment(agent_id, nearest_center)
    
    def find_central_positions(self):
        """Trouve les positions centrales basées sur la distribution des agents"""
        valid_positions = [pos for pos in self.agent_positions if pos is not None]
        if not valid_positions:
            return [(0, 0)]  # Position par défaut
        
        # Utiliser K-means simplifié pour trouver les centres
        if len(valid_positions) >= 3:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(3, len(valid_positions)))
            kmeans.fit(valid_positions)
            return kmeans.cluster_centers_.tolist()
        else:
            # Retourner la position moyenne
            avg_x = np.mean([pos[0] for pos in valid_positions])
            avg_y = np.mean([pos[1] for pos in valid_positions])
            return [(avg_x, avg_y)]
    
    def suggest_position_adjustment(self, agent_id, target_position):
        """Suggère un ajustement de position à un agent"""
        assignment_msg = String()
        assignment_msg.data = f"ADJUST_POSITION:{target_position[0]}:{target_position[1]}"
        self.assignment_pubs[agent_id].publish(assignment_msg)
    
    def send_task_assignment(self, agent_id, assignment):
        """Envoie une assignation de tâche à un agent"""
        assignment_msg = String()
        
        if assignment['type'] == 'explore':
            target = assignment['target']
            assignment_msg.data = f"EXPLORE:{target[0]}:{target[1]}"
        elif assignment['type'] == 'analyze':
            target = assignment['target']
            assignment_msg.data = f"ANALYZE:{target[0]}:{target[1]}"
        elif assignment['type'] == 'sample':
            target = assignment['target']
            assignment_msg.data = f"SAMPLE:{target[0]}:{target[1]}"
        else:
            assignment_msg.data = "CONTINUE_EXPLORATION"
        
        self.assignment_pubs[agent_id].publish(assignment_msg)

class SharedKnowledgeMap:
    def __init__(self, map_width=172, map_height=100):
        self.map_width = map_width
        self.map_height = map_height
        self.exploration_map = np.zeros((map_height, map_width))
        self.mineral_map = np.zeros((map_height, map_width, 4))  # 4 types de minéraux
        self.agent_positions = {}
        self.discoveries = defaultdict(list)
        self.communication_log = deque(maxlen=1000)
    
    def update_agent_position(self, agent_id, position):
        """Met à jour la position d'un agent"""
        self.agent_positions[agent_id] = position
        x, y = int(position[0]), int(position[1])
        
        # Mettre à jour la carte d'exploration
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            exploration_radius = 3
            for i in range(max(0, y-exploration_radius), min(self.map_height, y+exploration_radius+1)):
                for j in range(max(0, x-exploration_radius), min(self.map_width, x+exploration_radius+1)):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance <= exploration_radius:
                        self.exploration_map[i, j] = 1.0
    
    def add_discovery(self, agent_id, position, mineral_data):
        """Ajoute une découverte à la connaissance partagée"""
        x, y = position
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            # Mettre à jour la carte minérale
            for mineral_idx, concentration in enumerate(mineral_data[:4]):
                self.mineral_map[y, x, mineral_idx] = max(
                    self.mineral_map[y, x, mineral_idx], concentration
                )
            
            # Enregistrer la découverte
            self.discoveries[agent_id].append({
                'position': position,
                'mineral_data': mineral_data,
                'timestamp': rclpy.clock.Clock().now()
            })
    
    def get_coverage_rate(self):
        """Calcule le taux de couverture de l'exploration"""
        return np.mean(self.exploration_map)
    
    def get_mineral_hotspots(self, threshold=0.7):
        """Identifie les hotspots minéraux"""
        hotspots = []
        total_mineral_concentration = np.sum(self.mineral_map, axis=2)
        
        for y in range(self.map_height):
            for x in range(self.map_width):
                if total_mineral_concentration[y, x] > threshold:
                    hotspots.append({
                        'position': (x, y),
                        'intensity': total_mineral_concentration[y, x],
                        'mineral_composition': self.mineral_map[y, x]
                    })
        
        return hotspots
    
    def get_frontier_regions(self, frontier_width=2):
        """Identifie les régions frontières (bords entre exploré et inexploré)"""
        frontiers = []
        
        # Utiliser un filtre de détection de bord
        from scipy.ndimage import sobel
        exploration_gradient = np.sqrt(sobel(self.exploration_map, axis=0)**2 + 
                                     sobel(self.exploration_map, axis=1)**2)
        
        frontier_points = exploration_gradient > 0.1
        frontier_coords = np.argwhere(frontier_points)
        
        for y, x in frontier_coords:
            frontiers.append((x, y))
        
        return frontiers

class CommunicationGraph:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_agents))
        self.communication_history = defaultdict(list)
    
    def add_communication(self, source_agent, target_agent, message_type):
        """Enregistre une communication entre agents"""
        self.graph.add_edge(source_agent, target_agent)
        self.communication_history[(source_agent, target_agent)].append({
            'type': message_type,
            'timestamp': rclpy.clock.Clock().now()
        })
    
    def analyze_connectivity(self):
        """Analyse la connectivité du graphe de communication"""
        if self.graph.number_of_nodes() == 0:
            return {'connected': False, 'efficiency': 0.0}
        
        # Vérifier la connectivité
        is_connected = nx.is_connected(self.graph)
        
        # Calculer l'efficiency du réseau (simplifié)
        try:
            efficiency = nx.global_efficiency(self.graph)
        except:
            efficiency = 0.0
        
        # Degré moyen de connectivité
        avg_degree = np.mean([deg for _, deg in self.graph.degree()])
        
        return {
            'connected': is_connected,
            'efficiency': efficiency,
            'average_degree': avg_degree,
            'diameter': nx.diameter(self.graph) if is_connected else float('inf')
        }
    
    def get_communication_partners(self, agent_id):
        """Retourne les partenaires de communication d'un agent"""
        return list(self.graph.neighbors(agent_id))

class ExplorationStrategy:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.strategies = {
            'frontier_based': self.frontier_based_exploration,
            'mineral_focused': self.mineral_focused_exploration,
            'coordinated_sweep': self.coordinated_sweep_exploration
        }
        self.current_strategy = 'frontier_based'
    
    def frontier_based_exploration(self, agent_positions, shared_knowledge):
        """Stratégie d'exploration basée sur les frontières"""
        frontiers = shared_knowledge.get_frontier_regions()
        assignments = {}
        
        for agent_id, position in enumerate(agent_positions):
            if position is not None and frontiers:
                # Trouver la frontière la plus proche
                closest_frontier = min(frontiers, 
                    key=lambda frontier: np.linalg.norm(np.array(position) - np.array(frontier)))
                
                assignments[agent_id] = {
                    'type': 'explore',
                    'target': closest_frontier,
                    'priority': 1.0
                }
            else:
                assignments[agent_id] = {
                    'type': 'continue',
                    'target': position,
                    'priority': 0.5
                }
        
        return assignments
    
    def mineral_focused_exploration(self, agent_positions, shared_knowledge):
        """Stratégie focalisée sur les minéraux"""
        hotspots = shared_knowledge.get_mineral_hotspots()
        assignments = {}
        
        # Assigner les hotspots aux agents les plus proches
        unassigned_agents = set(range(len(agent_positions)))
        assigned_hotspots = set()
        
        for hotspot in sorted(hotspots, key=lambda h: h['intensity'], reverse=True):
            if not unassigned_agents:
                break
            
            # Trouver l'agent le plus proche non assigné
            closest_agent = min(unassigned_agents, 
                key=lambda agent_id: np.linalg.norm(
                    np.array(agent_positions[agent_id]) - np.array(hotspot['position'])
                ) if agent_positions[agent_id] is not None else float('inf'))
            
            if agent_positions[closest_agent] is not None:
                assignments[closest_agent] = {
                    'type': 'analyze',
                    'target': hotspot['position'],
                    'priority': hotspot['intensity']
                }
                unassigned_agents.remove(closest_agent)
                assigned_hotspots.add(hotspot['position'])
        
        # Assigner les agents restants à l'exploration de frontière
        frontier_assignments = self.frontier_based_exploration(
            [agent_positions[i] for i in unassigned_agents], 
            shared_knowledge
        )
        
        for i, agent_id in enumerate(unassigned_agents):
            assignments[agent_id] = frontier_assignments[i]
        
        return assignments
    
    def coordinated_sweep_exploration(self, agent_positions, shared_knowledge):
        """Stratégie de balayage coordonné"""
        # Implémentation d'un balayage systématique
        assignments = {}
        map_width, map_height = shared_knowledge.map_width, shared_knowledge.map_height
        
        # Diviser la carte en secteurs pour chaque agent
        sectors = self.divide_into_sectors(len(agent_positions), map_width, map_height)
        
        for agent_id, position in enumerate(agent_positions):
            if position is not None and agent_id < len(sectors):
                sector = sectors[agent_id]
                target = self.get_sector_center(sector)
                
                assignments[agent_id] = {
                    'type': 'explore',
                    'target': target,
                    'priority': 0.8
                }
            else:
                assignments[agent_id] = {
                    'type': 'continue',
                    'target': position,
                    'priority': 0.5
                }
        
        return assignments
    
    def divide_into_sectors(self, num_agents, width, height):
        """Divise la carte en secteurs pour les agents"""
        sectors = []
        
        if num_agents == 1:
            sectors.append((0, 0, width, height))
        elif num_agents == 2:
            sectors.append((0, 0, width//2, height))
            sectors.append((width//2, 0, width, height))
        elif num_agents >= 3:
            # Division en grille
            rows = int(np.ceil(np.sqrt(num_agents)))
            cols = int(np.ceil(num_agents / rows))
            
            sector_width = width // cols
            sector_height = height // rows
            
            for i in range(rows):
                for j in range(cols):
                    if len(sectors) < num_agents:
                        sectors.append((
                            j * sector_width,
                            i * sector_height,
                            (j + 1) * sector_width,
                            (i + 1) * sector_height
                        ))
        
        return sectors
    
    def get_sector_center(self, sector):
        """Retourne le centre d'un secteur"""
        x1, y1, x2, y2 = sector
        return ((x1 + x2) // 2, (y1 + y2) // 2)

class TaskAllocator:
    def __init__(self, num_agents):
        self.num_agents = num_agents
    
    def allocate_tasks(self, agent_positions, shared_knowledge, exploration_state):
        """Alloue les tâches aux agents selon l'état d'exploration"""
        
        # Choisir la stratégie basée sur l'état
        if exploration_state['mineral_hotspots']:
            # Si des hotspots sont détectés, prioriser l'analyse minérale
            strategy = 'mineral_focused'
        elif exploration_state['coverage_rate'] < 0.3:
            # Début de mission: balayage coordonné
            strategy = 'coordinated_sweep'
        else:
            # Exploration générale: frontières
            strategy = 'frontier_based'
        
        # Obtenir les assignations de la stratégie choisie
        exploration_strategy = ExplorationStrategy(self.num_agents)
        assignments = exploration_strategy.strategies[strategy](
            agent_positions, shared_knowledge
        )
        
        # Ajuster les priorités basées sur l'état de communication
        communication_efficiency = exploration_state['communication_efficiency']
        for agent_id, assignment in assignments.items():
            # Réduire la priorité si la communication est mauvaise
            if communication_efficiency < 0.5:
                assignment['priority'] *= 0.7
        
        return assignments

def main():
    rclpy.init()
    coordinator = MultiAgentCoordinator(num_agents=4)
    
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()