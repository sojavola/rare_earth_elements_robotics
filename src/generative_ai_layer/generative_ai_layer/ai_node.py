#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose2D
import numpy as np
import json
import threading
from datetime import datetime

from .science_ai_system import ScienceAISystem
#from .api_config import config_manager

class AISystemNode(Node):
    def __init__(self):
        super().__init__('ai_system_node')
        
        # Configuration
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('mineral_threshold', 0.3)
        self.declare_parameter('enable_voice', True)
        
        # Système IA
        self.ai_system = ScienceAISystem()
        self.get_logger().info(self.ai_system.get_system_status())
        
        # État actuel
        self.current_mineral_data = None
        self.current_positions = {}
        self.mission_stats = {
            'start_time': datetime.now(),
            'minerals_found': 0,
            'high_value_samples': 0,
            'areas_explored': set()
        }
        
        # Publishers
        self.analysis_pub = self.create_publisher(String, '/ai/analysis', 10)
        self.advice_pub = self.create_publisher(String, '/ai/advice', 10)
        self.report_pub = self.create_publisher(String, '/ai/reports', 10)
        self.insights_pub = self.create_publisher(String, '/ai/science_insights', 10)
        
        # Subscribers
        self.mineral_sub = self.create_subscription(
            Float32MultiArray,
            '/mineral_discoveries',
            self.mineral_callback,
            10
        )
        
        self.position_subs = []
        for i in range(4):  # 4 robots
            sub = self.create_subscription(
                Pose2D,
                f'/robot_{i}/position',
                lambda msg, idx=i: self.position_callback(msg, idx),
                10
            )
            self.position_subs.append(sub)
        
        self.mission_sub = self.create_subscription(
            String,
            '/mission/status',
            self.mission_callback,
            10
        )
        
        # Timers
        publish_rate = self.get_parameter('publish_rate').value
        self.analysis_timer = self.create_timer(publish_rate, self.publish_analysis)
        self.advice_timer = self.create_timer(publish_rate * 2, self.publish_advice)
        
        self.get_logger().info("🤖 Node AI System démarré")
    
    def mineral_callback(self, msg):
        """Traite les nouvelles découvertes minérales"""
        try:
            # Conversion des données ROS2 vers numpy
            if len(msg.data) > 0:
                size = int(np.sqrt(len(msg.data) // 4))
                if size > 0:
                    self.current_mineral_data = np.array(msg.data).reshape(size, size, 4)
                    
                    # Détection de minéraux importants
                    threshold = self.get_parameter('mineral_threshold').value
                    high_value_count = np.sum(self.current_mineral_data > threshold)
                    
                    if high_value_count > 0:
                        self.mission_stats['minerals_found'] += high_value_count
                        self.mission_stats['high_value_samples'] += np.sum(
                            self.current_mineral_data > 0.7
                        )
                        
                        self.get_logger().info(
                            f"🔍 {high_value_count} points minéraux détectés "
                            f"(dont {self.mission_stats['high_value_samples']} haute valeur)"
                        )
                
        except Exception as e:
            self.get_logger().error(f"❌ Erreur traitement minéral: {e}")
    
    def position_callback(self, msg, robot_id):
        """Met à jour les positions des robots"""
        self.current_positions[robot_id] = (msg.x, msg.y)
        
        # Enregistrement des zones explorées
        area_x, area_y = int(msg.x // 10), int(msg.y // 10)  # Zones de 10x10
        area_key = f"{area_x}_{area_y}"
        self.mission_stats['areas_explored'].add(area_key)
    
    def mission_callback(self, msg):
        """Traite les mises à jour de mission"""
        try:
            mission_data = json.loads(msg.data)
            
            # Génération de conseils en temps réel
            advice = self.ai_system.get_real_time_advice(
                current_state=f"Robots: {len(self.current_positions)}, "
                            f"Zones: {len(self.mission_stats['areas_explored'])}",
                recent_discoveries=f"{self.mission_stats['minerals_found']} minéraux détectés",
                mission_goals=mission_data.get('goals', 'Exploration optimale')
            )
            
            # Publication des conseils
            advice_msg = String()
            advice_msg.data = advice
            self.advice_pub.publish(advice_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Erreur traitement mission: {e}")
    
    def publish_analysis(self):
        """Publie l'analyse IA périodique"""
        if self.current_mineral_data is not None:
            # Préparation des données d'exploration
            exploration_data = {
                'area_covered': len(self.mission_stats['areas_explored']) * 5,  # Estimation
                'active_robots': len(self.current_positions),
                'positions_visited': sum(len(areas) for areas in self.mission_stats['areas_explored']),
                'mission_duration': (datetime.now() - self.mission_stats['start_time']).total_seconds() / 60
            }
            
            # Statistiques scientifiques
            science_stats = {
                'discovered_minerals': self.mission_stats['minerals_found'],
                'high_value_samples': self.mission_stats['high_value_samples'],
                'visited_regions': len(self.mission_stats['areas_explored']),
                'unique_minerals': 4,  # Estimation
                'total_science_value': self.mission_stats['minerals_found'] * 10
            }
            
            # Analyse IA
            analysis = self.ai_system.analyze_mission_data(
                self.current_mineral_data,
                exploration_data,
                science_stats
            )
            
            # Publication
            analysis_msg = String()
            analysis_msg.data = analysis
            self.analysis_pub.publish(analysis_msg)
    
    def publish_advice(self):
        """Publie des conseils stratégiques"""
        if self.current_positions:
            advice = self.ai_system.get_real_time_advice(
                current_state=f"{len(self.current_positions)} robots actifs",
                recent_discoveries=f"{self.mission_stats['minerals_found']} découvertes",
                mission_goals="Optimisation exploration"
            )
            
            advice_msg = String()
            advice_msg.data = advice
            self.advice_pub.publish(advice_msg)
    
    def generate_mission_report(self):
        """Génère un rapport de mission complet"""
        mission_data = {
            'duration': str(datetime.now() - self.mission_stats['start_time']),
            'minerals_found': self.mission_stats['minerals_found'],
            'areas_explored': len(self.mission_stats['areas_explored']),
            'robots_used': len(self.current_positions),
            'completion_estimate': min(100, len(self.mission_stats['areas_explored']) * 2)
        }
        
        discoveries = f"""
        Découvertes majeures:
        - {self.mission_stats['minerals_found']} points minéraux
        - {self.mission_stats['high_value_samples']} échantillons haute valeur
        - {len(self.mission_stats['areas_explored'])} zones explorées
        """
        
        recommendations = "Poursuite exploration, optimisation coordination robots"
        
        report = self.ai_system.generate_mission_report(
            mission_data, discoveries, recommendations
        )
        
        report_msg = String()
        report_msg.data = report
        self.report_pub.publish(report_msg)
        
        return report
    
    def emergency_protocol(self, emergency_type, situation):
        """Active un protocole d'urgence"""
        resources = f"Robots: {list(self.current_positions.keys())}"
        
        protocol = self.ai_system.emergency_protocol(
            emergency_type, situation, resources
        )
        
        self.get_logger().warn(f"🚨 PROTOCOLE URGENCE: {protocol}")
        return protocol

def main():
    rclpy.init()
    node = AISystemNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Génération du rapport final
        node.generate_mission_report()
        node.get_logger().info("📊 Rapport de mission généré")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()