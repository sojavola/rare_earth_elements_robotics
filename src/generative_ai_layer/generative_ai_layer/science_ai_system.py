#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose2D
import numpy as np
import json
import requests
from datetime import datetime
import threading
import time
import random

class ScienceAISystem(Node):
    def __init__(self):
        super().__init__('science_ai_system')
        
        # Configuration
        self.mistral_api_key = self._get_api_key()
        self.api_available = bool(self.mistral_api_key and self.mistral_api_key != 'xiIZeHKEkdJHh72eLzDQvFuQnKgDg5im')
        
        # État de la mission
        self.mission_data = {
            'mineral_discoveries': [],
            'robot_positions': {},
            'exploration_stats': {
                'areas_covered': 0,
                'minerals_found': 0,
                'high_value_samples': 0,
                'unique_minerals': set()
            },
            'mission_start': datetime.now()
        }
        
        # Cache pour les réponses
        self.response_cache = {}
        
        # Setup ROS2
        self._setup_communication()
        
        status = "🟢 Mistral AI" if self.api_available else "🟡 Mode Local"
        self.get_logger().info(f"🧠 Science AI System initialisé - {status}")

    def _get_api_key(self):
        """Récupère la clé API depuis les variables d'environnement"""
        import os
        return os.getenv('MISTRAL_API_KEY', 'YOUR_API_KEY_HERE')

    def _setup_communication(self):
        """Configure la communication ROS2"""
        # Publishers
        self.analysis_pub = self.create_publisher(String, '/ai/analysis', 10)
        self.advice_pub = self.create_publisher(String, '/ai/advice', 10)
        self.report_pub = self.create_publisher(String, '/ai/reports', 10)
        self.insights_pub = self.create_publisher(String, '/ai/science_insights', 10)
        
        # Subscribers
        self.mineral_sub = self.create_subscription(
            Float32MultiArray, '/mineral_map', self._mineral_callback, 10
        )
        
        # Positions des robots
        for i in range(4):
            self.create_subscription(
                Pose2D, f'/robot_{i}/position',
                lambda msg, idx=i: self._position_callback(msg, idx), 10
            )
        
        # Timers
        self.analysis_timer = self.create_timer(30.0, self._publish_analysis)
        self.advice_timer = self.create_timer(45.0, self._publish_advice)
        self.report_timer = self.create_timer(120.0, self._publish_report)

    def _mineral_callback(self, msg):
        """Traite les données minérales"""
        try:
            if len(msg.data) > 0:
                mineral_data = np.array(msg.data)
                
                # Détection de minéraux
                threshold = 0.1
                high_value_threshold = 0.7
                
                mineral_count = np.sum(mineral_data > threshold)
                high_value_count = np.sum(mineral_data > high_value_threshold)
                
                if mineral_count > 0:
                    self.mission_data['exploration_stats']['minerals_found'] += mineral_count
                    self.mission_data['exploration_stats']['high_value_samples'] += high_value_count
                    
                    # Détection de types de minéraux (simplifiée)
                    if np.max(mineral_data) > 0.8:
                        self.mission_data['exploration_stats']['unique_minerals'].add('REE_Oxides')
                    if np.mean(mineral_data) > 0.3:
                        self.mission_data['exploration_stats']['unique_minerals'].add('REE_Silicates')
                    
                    self.get_logger().info(
                        f"🔍 Minéraux: {mineral_count} total, {high_value_count} haute valeur"
                    )
                
        except Exception as e:
            self.get_logger().error(f"❌ Erreur minéral: {e}")

    def _position_callback(self, msg, robot_id):
        """Met à jour les positions des robots"""
        self.mission_data['robot_positions'][robot_id] = {
            'x': msg.x, 
            'y': msg.y,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mise à jour des zones explorées
        area_x, area_y = int(msg.x // 20), int(msg.y // 20)
        self.mission_data['exploration_stats']['areas_covered'] = max(
            self.mission_data['exploration_stats']['areas_covered'],
            len(self.mission_data['robot_positions']) * 10
        )

    def _publish_analysis(self):
        """Publie l'analyse des données"""
        try:
            analysis = self.analyze_mission_data()
            
            msg = String()
            msg.data = analysis
            self.analysis_pub.publish(msg)
            
            self.get_logger().info("📊 Analyse publiée")
            
        except Exception as e:
            self.get_logger().error(f"❌ Erreur analyse: {e}")

    def _publish_advice(self):
        """Publie des conseils stratégiques"""
        try:
            advice = self.generate_strategic_advice()
            
            msg = String()
            msg.data = advice
            self.advice_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Erreur conseils: {e}")

    def _publish_report(self):
        """Publie un rapport périodique"""
        try:
            report = self.generate_mission_report()
            
            msg = String()
            msg.data = report
            self.report_pub.publish(msg)
            
            self.get_logger().info("📋 Rapport de mission publié")
            
        except Exception as e:
            self.get_logger().error(f"❌ Erreur rapport: {e}")

    def analyze_mission_data(self):
        """Analyse les données de mission"""
        if self.api_available:
            return self._call_mistral_api(self._create_analysis_prompt())
        else:
            return self._local_analysis()

    def _call_mistral_api(self, prompt):
        """Appelle l'API Mistral directement"""
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.mistral_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()['choices'][0]['message']['content']
            return result
            
        except Exception as e:
            self.get_logger().warning(f"⚠️ API Mistral échouée: {e}")
            return self._local_analysis()

    def _create_analysis_prompt(self):
        """Crée le prompt pour l'analyse"""
        stats = self.mission_data['exploration_stats']
        robots = self.mission_data['robot_positions']
        
        return f"""
        RÔLE: Expert en géologie planétaire et exploration robotique
        MISSION: Détection de terres rares sur corps céleste
        
        DONNÉES ACTUELLES:
        - Minéraux détectés: {stats['minerals_found']}
        - Échantillons haute valeur: {stats['high_value_samples']} 
        - Types minéraux: {len(stats['unique_minerals'])}
        - Robots actifs: {len(robots)}
        - Zones explorées: {stats['areas_covered']}%
        - Durée mission: {(datetime.now() - self.mission_data['mission_start']).total_seconds() / 60:.1f} min
        
        TÂCHE: Fournir une analyse scientifique complète incluant:
        1. Évaluation du potentiel en terres rares
        2. Recommandations d'exploration prioritaires  
        3. Stratégie d'optimisation robotique
        4. Implications pour l'extraction spatiale
        
        FORMAT: Rapport scientifique en français, structuré et actionnable.
        """

    def _local_analysis(self):
        """Analyse locale sans API"""
        stats = self.mission_data['exploration_stats']
        robots = len(self.mission_data['robot_positions'])
        
        # Évaluation du potentiel
        if stats['high_value_samples'] > 15:
            potential = "EXCEPTIONNEL 💎"
            confidence = "Très élevée"
        elif stats['high_value_samples'] > 8:
            potential = "ÉLEVÉ ✅" 
            confidence = "Élevée"
        elif stats['minerals_found'] > 30:
            potential = "PROMETTEUR ⚠️"
            confidence = "Modérée"
        else:
            potential = "MODESTE 📈"
            confidence = "Faible"
        
        # Efficacité d'exploration
        efficiency = "EXCELLENTE 🚀" if stats['minerals_found'] > robots * 15 else "BONNE ✅"
        
        analysis = f"""
        🤖 ANALYSE SCIENTIFIQUE - SYSTÈME LOCAL
        =====================================
        
        📊 DONNÉES MISSION:
        • Minéraux détectés: {stats['minerals_found']}
        • Échantillons haute valeur: {stats['high_value_samples']}
        • Types minéraux uniques: {len(stats['unique_minerals'])}
        • Robots déployés: {robots}
        • Zones explorées: {stats['areas_covered']}%
        
        🎯 ÉVALUATION POTENTIEL:
        Niveau: {potential}
        Confiance: {confidence}
        Efficacité exploration: {efficiency}
        
        🔬 RECOMMANDATIONS STRATÉGIQUES:
        
        1. EXPLORATION PRIORITAIRE:
        {self._get_exploration_recommendations()}
        
        2. OPTIMISATION ROBOTIQUE:
        {self._get_robot_optimization()}
        
        3. STRATÉGIE SCIENTIFIQUE:
        {self._get_science_strategy()}
        
        📈 PERSPECTIVES:
        {self._get_prospects()}
        
        ⏱️ Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return analysis

    def _get_exploration_recommendations(self):
        """Génère des recommandations d'exploration"""
        stats = self.mission_data['exploration_stats']
        
        if stats['high_value_samples'] > 10:
            return "- Cibler les zones adjacentes aux détections existantes\n- Intensifier l'échantillonnage dans les cratères\n- Prioriser les formations géologiques anciennes"
        elif stats['minerals_found'] > 20:
            return "- Explorer systématiquement les secteurs inexplorés\n- Échantillonnage régulier toutes les 5 unités\n- Coordination multi-robots pour couverture optimale"
        else:
            return "- Élargir le périmètre d'exploration\n- Augmenter la densité d'analyse\n- Cibler les zones à fort gradient géologique"

    def _get_robot_optimization(self):
        """Génère des recommandations d'optimisation robotique"""
        robots = len(self.mission_data['robot_positions'])
        
        if robots >= 3:
            return "- Formation en éventail pour couverture maximale\n- Rotation des zones d'exploration\n- Communication optimisée pour éviter les chevauchements"
        elif robots == 2:
            return "- Exploration en binôme avec zones complémentaires\n- Synchronisation des analyses spectrales\n- Partage des données en temps réel"
        else:
            return "- Exploration spirale depuis le point central\n- Analyses fréquentes tous les 3 points\n- Conservation d'énergie pour mission longue"

    def _get_science_strategy(self):
        """Génère la stratégie scientifique"""
        stats = self.mission_data['exploration_stats']
        
        strategies = [
            "- Cartographie systématique des concentrations",
            "- Analyse spectrale des échantillons prometteurs", 
            "- Échantillonnage stratifié par zone géologique",
            "- Surveillance continue des paramètres environnementaux"
        ]
        
        if stats['high_value_samples'] > 5:
            strategies.append("- Analyse approfondie des échantillons >0.7 concentration")
            strategies.append("- Priorisation des zones à fort potentiel")
        
        return "\n".join(strategies)

    def _get_prospects(self):
        """Génère les perspectives"""
        stats = self.mission_data['exploration_stats']
        
        if stats['high_value_samples'] > 20:
            return "Potentiel d'extraction commerciale identifié. Recommander phase d'exploitation."
        elif stats['high_value_samples'] > 10:
            return "Ressources significatives détectées. Poursuite de l'exploration justifiée."
        else:
            return "Exploration en cours. Données insuffisantes pour conclusion définitive."

    def generate_strategic_advice(self):
        """Génère des conseils stratégiques en temps réel"""
        stats = self.mission_data['exploration_stats']
        robots = self.mission_data['robot_positions']
        
        # Conseils basés sur l'état actuel
        if not robots:
            priority = "CRITIQUE 🚨"
            action = "Déployer les robots et établir la communication"
        elif stats['minerals_found'] == 0:
            priority = "ÉLEVÉE 🔍" 
            action = "Ajuster les paramètres de détection et explorer de nouvelles zones"
        elif stats['high_value_samples'] > 5:
            priority = "OPTIMISATION ✅"
            action = "Concentrer les ressources sur les zones à haut potentiel"
        else:
            priority = "ROUTINE 📊"
            action = "Maintenir l'exploration systématique"
        
        advice = f"""
        💡 CONSEILS STRATÉGIQUES - {datetime.now().strftime('%H:%M')}
        =================================
        
        PRIORITÉ: {priority}
        ACTION: {action}
        
        DÉTAILS OPÉRATIONNELS:
        
        🤖 GESTION ROBOTS:
        {self._get_robot_management_advice()}
        
        🔧 OPTIMISATION CAPTEURS:
        {self._get_sensor_advice()}
        
        📡 COMMUNICATION:
        {self._get_communication_advice()}
        
        ⚡ ÉNERGIE:
        {self._get_energy_advice()}
        
        🔬 SCIENCE:
        {self._get_science_advice()}
        """
        
        return advice

    def _get_robot_management_advice(self):
        """Conseils de gestion des robots"""
        robots = len(self.mission_data['robot_positions'])
        
        if robots >= 3:
            return "- Coordonner en formation triangulaire\n- Assigner des secteurs spécifiques\n- Rotation périodique des positions"
        elif robots == 2:
            return "- Explorer en miroir pour couverture\n- Synchroniser les analyses\n- Maintenir distance optimale"
        else:
            return "- Exploration spirale centrée\n- Analyses fréquentes\n- Surveillance autonomie"

    def _get_sensor_advice(self):
        """Conseils d'optimisation des capteurs"""
        stats = self.mission_data['exploration_stats']
        
        if stats['high_value_samples'] > 0:
            return "- Augmenter résolution spectrale zones chaudes\n- Calibration fine détecteurs REE\n- Surveillance continue qualité données"
        else:
            return "- Vérifier calibration capteurs\n- Ajuster sensibilité détection\n- Tests périodiques performance"

    def _get_communication_advice(self):
        """Conseils de communication"""
        robots = len(self.mission_data['robot_positions'])
        
        if robots > 1:
            return "- Synchronisation données temps réel\n- Partage cartes minérales\n- Alertes coordonnées"
        else:
            return "- Transmission données régulières\n- Sauvegarde locale données\n- Surveillance signal"

    def _get_energy_advice(self):
        """Conseils de gestion d'énergie"""
        return "- Surveillance niveaux batterie\n- Optimisation trajets\n- Planification recharge"

    def _get_science_advice(self):
        """Conseils scientifiques"""
        stats = self.mission_data['exploration_stats']
        
        if stats['high_value_samples'] > 10:
            return "- Analyse approfondie échantillons\n- Cartographie 3D veines\n- Étude composition minérale"
        else:
            return "- Exploration systématique\n- Échantillonnage représentatif\n- Documentation géologique"

    def generate_mission_report(self):
        """Génère un rapport de mission complet"""
        stats = self.mission_data['exploration_stats']
        duration = datetime.now() - self.mission_data['mission_start']
        
        report = f"""
        📋 RAPPORT DE MISSION REE EXPLORATION
        =====================================
        
        📅 INFORMATIONS GÉNÉRALES:
        • Début: {self.mission_data['mission_start'].strftime('%Y-%m-%d %H:%M:%S')}
        • Durée: {duration.total_seconds() / 60:.1f} minutes
        • Statut: En cours
        • Système: {"Mistral AI" if self.api_available else "Local"}
        
        📊 RÉSULTATS SCIENTIFIQUES:
        • Minéraux détectés: {stats['minerals_found']}
        • Échantillons haute valeur: {stats['high_value_samples']}
        • Types minéraux: {', '.join(stats['unique_minerals']) if stats['unique_minerals'] else 'Aucun'}
        • Zones explorées: {stats['areas_covered']}%
        
        🤖 DÉPLOIEMENT ROBOTIQUE:
        • Robots actifs: {len(self.mission_data['robot_positions'])}
        • Couverture: {self._calculate_coverage()}%
        • Efficacité: {self._calculate_efficiency()}%
        
        🎯 ANALYSE STRATÉGIQUE:
        {self._get_strategic_analysis()}
        
        🔮 RECOMMANDATIONS FUTURES:
        {self._get_future_recommendations()}
        
        📈 CONCLUSION:
        {self._get_conclusion()}
        
        ⏱️ Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report

    def _calculate_coverage(self):
        """Calcule la couverture d'exploration"""
        stats = self.mission_data['exploration_stats']
        return min(100, stats['areas_covered'])

    def _calculate_efficiency(self):
        """Calcule l'efficacité d'exploration"""
        stats = self.mission_data['exploration_stats']
        robots = len(self.mission_data['robot_positions'])
        
        if robots == 0:
            return 0
        return min(100, (stats['minerals_found'] / max(1, robots)) * 3)

    def _get_strategic_analysis(self):
        """Génère l'analyse stratégique"""
        stats = self.mission_data['exploration_stats']
        
        if stats['high_value_samples'] > 15:
            return "• Potentiel exceptionnel identifié\n• Recommander expansion mission\n• Préparer phase extraction"
        elif stats['high_value_samples'] > 8:
            return "• Ressources significatives confirmées\n• Poursuite exploration justifiée\n• Optimisation stratégique requise"
        else:
            return "• Exploration en phase initiale\n• Données supplémentaires nécessaires\n• Stratégie adaptative recommandée"

    def _get_future_recommendations(self):
        """Génère les recommandations futures"""
        stats = self.mission_data['exploration_stats']
        
        recommendations = [
            "• Maintenir l'exploration systématique",
            "• Optimiser la coordination multi-robots",
            "• Améliorer la résolution des analyses spectrales"
        ]
        
        if stats['high_value_samples'] > 10:
            recommendations.extend([
                "• Déployer des capteurs spécialisés",
                "• Planifier une mission d'extraction",
                "• Étendre la zone d'exploration de 50%"
            ])
        
        return "\n".join(recommendations)

    def _get_conclusion(self):
        """Génère la conclusion du rapport"""
        stats = self.mission_data['exploration_stats']
        
        if stats['minerals_found'] > 50:
            return "Mission très réussie. Potentiel commercial identifié. Recommander poursuite et expansion."
        elif stats['minerals_found'] > 20:
            return "Mission réussie. Ressources détectées. Justification poursuite exploration."
        else:
            return "Mission en cours. Données prometteuses. Nécessité continuation exploration."

    def emergency_protocol(self, emergency_type, situation):
        """Active un protocole d'urgence"""
        protocols = {
            'communication': """
            🚨 PROTOCOLE URGENCE - COMMUNICATION
            • Activer mode sauvegarde données
            • Tentative reconnexion automatique
            • Transmission prioritaire données critiques
            • Mode autonomie robots activé
            """,
            'power': """
            🚨 PROTOCOLE URGENCE - ÉNERGIE  
            • Réduction consommation énergie
            • Priorisation fonctions essentielles
            • Planification retour base
            • Transmission état urgence
            """,
            'sensor': """
            🚨 PROTOCOLE URGENCE - CAPTEURS
            • Recalibration automatique
            • Activation capteurs secondaires
            • Diagnostic en temps réel
            • Transmission rapports erreur
            """,
            'environment': """
            🚨 PROTOCOLE URGENCE - ENVIRONNEMENT
            • Activation boucliers protection
            • Adaptation paramètres opération
            • Surveillance conditions extrêmes
            • Planification mise à l'abri
            """
        }
        
        return protocols.get(emergency_type, "🚨 Protocole urgence standard activé")

def main():
    rclpy.init()
    
    try:
        node = ScienceAISystem()
        node.get_logger().info("🚀 Démarrage du Science AI System...")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        node.get_logger().error(f"❌ Erreur critique: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()