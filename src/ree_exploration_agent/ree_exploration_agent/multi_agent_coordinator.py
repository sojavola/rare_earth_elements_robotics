import rclpy
from rclpy.node import Node
import numpy as np
import threading
import json
import time
from collections import defaultdict
import os

from std_msgs.msg import String, Float32MultiArray, Int32, Float32
from geometry_msgs.msg import Pose2D

class MultiAgentCoordinator(Node):
    def __init__(self, num_agents=4):
        super().__init__('multi_agent_coordinator')
        
        self.num_agents = num_agents
        self.agent_positions = {}
        self.agent_status = {}
        self.agent_performance = defaultdict(float)
        self.shared_discoveries = []
        self.shared_model_updates = []
        
        # ✅ NOUVEAU: Gestion du modèle partagé
        self.global_step_count = 0
        self.global_episode_count = 0
        self.model_version = 0
        self.last_model_update = time.time()
        
        # ✅ NOUVEAU: Synchronisation DQN
        self.dqn_sync_lock = threading.Lock()
        
        # ✅ NOUVEAU: Publishers pour synchronisation DQN
        self.model_update_pub = self.create_publisher(
            String, 
            '/global/model_update', 
            10
        )
        
        self.global_step_pub = self.create_publisher(
            Int32,
            '/global/step_count', 
            10
        )
        
        self.epsilon_pub = self.create_publisher(
            Float32,
            '/global/epsilon',
            10
        )
        
        # ✅ NOUVEAU: Subscribers pour DQN
        self.create_subscription(
            Float32MultiArray,
            '/agent/dqn_update',
            self.dqn_update_callback,
            10
        )
        
        self.create_subscription(
            Int32,
            '/agent/step_completed',
            self.step_completed_callback,
            10
        )
        
        self.create_subscription(
            Float32,
            '/agent/epsilon',
            self.epsilon_callback,
            10
        )
        
        # Publishers pour chaque agent (existants)
        self.task_publishers = []
        for i in range(num_agents):
            publisher = self.create_publisher(
                String, 
                f'/robot_{i}/task_assignment', 
                10
            )
            self.task_publishers.append(publisher)
        
        # Subscribers pour les positions (existants)
        self.position_subscribers = []
        for i in range(num_agents):
            self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                self.create_position_callback(i),
                10
            )
        
        # Subscriber pour les découvertes (existant)
        self.create_subscription(
            Float32MultiArray,
            '/shared_discoveries',
            self.discovery_callback,
            10
        )
        
        # Timers
        self.coordination_timer = self.create_timer(3.0, self.coordinate_agents)
        
        # ✅ NOUVEAU: Timer pour synchronisation DQN
        self.dqn_sync_timer = self.create_timer(1.0, self.synchronize_dqn)
        
        # ✅ NOUVEAU: Timer pour statistiques globales
        self.stats_timer = self.create_timer(5.0, self.publish_global_stats)
        
        # ✅ NOUVEAU: Répertoire pour modèle partagé
        self.shared_model_dir = "models/shared"
        os.makedirs(self.shared_model_dir, exist_ok=True)
        self.shared_model_path = f"{self.shared_model_dir}/latest_shared_model.pth"
        
        self.get_logger().info(f'🚀 Multi-Agent Coordinator initialized with {num_agents} agents')
        self.get_logger().info(f'📊 DQN Synchronization ENABLED')
    
    def create_position_callback(self, agent_id):
        """Crée un callback pour la position d'un agent"""
        def callback(msg):
            self.agent_positions[agent_id] = (msg.x, msg.y)
            self.agent_status[agent_id] = 'active'
        return callback
    
    def discovery_callback(self, msg):
        """Traite les découvertes partagées"""
        if len(msg.data) >= 3:
            agent_id = int(msg.data[0])
            x, y = msg.data[1], msg.data[2]
            discovery_data = msg.data[3:]
            
            self.shared_discoveries.append({
                'agent_id': agent_id,
                'position': (x, y),
                'data': discovery_data,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
    
    # ✅ NOUVEAU: Callbacks pour synchronisation DQN
    def dqn_update_callback(self, msg):
        """Reçoit les mises à jour DQN des agents"""
        if len(msg.data) >= 5:
            agent_id = int(msg.data[0])
            loss = msg.data[1]
            epsilon = msg.data[2]
            memory_size = int(msg.data[3])
            step_count = int(msg.data[4])
            
            with self.dqn_sync_lock:
                self.agent_performance[agent_id] = loss
                
                # Mettre à jour le compteur global
                if step_count > self.global_step_count:
                    self.global_step_count = step_count
                
                # Sauvegarder la mise à jour
                self.shared_model_updates.append({
                    'agent_id': agent_id,
                    'loss': loss,
                    'epsilon': epsilon,
                    'timestamp': time.time(),
                    'step_count': step_count
                })
                
                # Publier le compteur global mis à jour
                self.publish_global_step()
                
                # Mettre à jour epsilon global
                self.publish_global_epsilon(epsilon)
    
    def step_completed_callback(self, msg):
        """Reçoit la notification de step complété"""
        with self.dqn_sync_lock:
            self.global_step_count += 1
            self.publish_global_step()
    
    def epsilon_callback(self, msg):
        """Reçoit les valeurs d'epsilon des agents"""
        self.publish_global_epsilon(msg.data)
    
    def publish_global_step(self):
        """Publie le compteur global à tous les agents"""
        msg = Int32()
        msg.data = self.global_step_count
        self.global_step_pub.publish(msg)
        
        if self.global_step_count % 10 == 0:
            self.get_logger().info(f'📊 Global Step: {self.global_step_count}')
    
    def publish_global_epsilon(self, epsilon):
        """Publie la valeur d'epsilon à tous les agents"""
        msg = Float32()
        msg.data = float(epsilon)
        self.epsilon_pub.publish(msg)
    
    def synchronize_dqn(self):
        """Synchronise le modèle DQN entre tous les agents"""
        with self.dqn_sync_lock:
            # Vérifier si une mise à jour du modèle est nécessaire
            current_time = time.time()
            
            # Mettre à jour toutes les 30 secondes ou après 100 steps
            if (current_time - self.last_model_update > 30 or 
                len(self.shared_model_updates) > 100):
                
                self.model_version += 1
                
                # Créer le message de mise à jour
                update_msg = String()
                update_data = {
                    'model_version': self.model_version,
                    'global_step_count': self.global_step_count,
                    'global_episode_count': self.global_episode_count,
                    'timestamp': current_time,
                    'num_agents': len(self.agent_status),
                    'update_type': 'full_sync'
                }
                update_msg.data = json.dumps(update_data)
                
                # Publier la mise à jour
                self.model_update_pub.publish(update_msg)
                self.last_model_update = current_time
                
                self.get_logger().info(f'🔄 DQN Synchronization: Version {self.model_version}, Step {self.global_step_count}')
                
                # Nettoyer les anciennes mises à jour
                if len(self.shared_model_updates) > 100:
                    self.shared_model_updates = self.shared_model_updates[-50:]
    
    def coordinate_agents(self):
        """Coordonne les agents avec prise en compte DQN"""
        if len(self.agent_positions) < self.num_agents:
            self.get_logger().warn('⏳ Waiting for all agents to report positions...')
            return
        
        # Stratégie de coordination améliorée avec DQN
        for agent_id in range(self.num_agents):
            if agent_id in self.agent_positions:
                task = self.assign_smart_task(agent_id)
                self.send_task(agent_id, task)
    
    def assign_smart_task(self, agent_id):
        """Assigne une tâche intelligente basée sur les performances DQN"""
        current_pos = self.agent_positions.get(agent_id, (0, 0))
        
        # Performance de l'agent (plus bas = mieux)
        agent_perf = self.agent_performance.get(agent_id, 1.0)
        
        # Déterminer la difficulté de la tâche basée sur les performances
        if agent_perf < 0.1:  # Agent performant
            # Donner une tâche plus difficile (exploration lointaine)
            task_type = "EXPLORE_DEEP"
            exploration_range = 50
            priority = 1.0
        elif agent_perf < 0.5:  # Agent moyen
            # Tâche standard
            task_type = "EXPLORE"
            exploration_range = 30
            priority = 0.8
        else:  # Agent en difficulté
            # Tâche facile (exploration locale)
            task_type = "EXPLORE_LOCAL"
            exploration_range = 15
            priority = 0.6
        
        # Baser la cible sur les découvertes partagées
        if self.shared_discoveries:
            # Aller vers la zone de découverte la plus récente
            latest_discovery = self.shared_discoveries[-1]
            target = latest_discovery['position']
            
            # Ajouter un peu d'aléatoire
            target = (
                target[0] + np.random.randint(-10, 10),
                target[1] + np.random.randint(-10, 10)
            )
        else:
            # Exploration basée sur le quadrant
            quadrant = agent_id % 4
            base_targets = {
                0: (75, 75),  # NE
                1: (25, 75),  # NW
                2: (25, 25),  # SW
                3: (75, 25)   # SE
            }
            base_target = base_targets[quadrant]
            
            # Ajouter de l'aléatoire selon la portée d'exploration
            target = (
                base_target[0] + np.random.randint(-exploration_range, exploration_range),
                base_target[1] + np.random.randint(-exploration_range, exploration_range)
            )
        
        # Assurer que la cible est dans les limites (0-100)
        target = (
            max(0, min(99, target[0])),
            max(0, min(99, target[1]))
        )
        
        return {
            'type': task_type,
            'target': target,
            'priority': priority,
            'exploration_range': exploration_range,
            'agent_performance': agent_perf
        }
    
    def send_task(self, agent_id, task):
        """Envoie une tâche à un agent avec info DQN"""
        msg = String()
        
        task_data = {
            'task_type': task['type'],
            'target_x': task['target'][0],
            'target_y': task['target'][1],
            'priority': task['priority'],
            'global_step': self.global_step_count,
            'model_version': self.model_version,
            'exploration_range': task['exploration_range']
        }
        
        msg.data = json.dumps(task_data)
        self.task_publishers[agent_id].publish(msg)
    
    def publish_global_stats(self):
        """Publie les statistiques globales"""
        active_agents = len(self.agent_status)
        avg_performance = np.mean(list(self.agent_performance.values())) if self.agent_performance else 0
        
        stats_msg = f"📊 Global Stats - Active: {active_agents}/{self.num_agents}, "
        stats_msg += f"Global Step: {self.global_step_count}, "
        stats_msg += f"Model Version: {self.model_version}, "
        stats_msg += f"Avg Loss: {avg_performance:.4f}, "
        stats_msg += f"Discoveries: {len(self.shared_discoveries)}"
        
        self.get_logger().info(stats_msg)

def main():
    rclpy.init()
    
    try:
        node = MultiAgentCoordinator(num_agents=4)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error in coordinator: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()