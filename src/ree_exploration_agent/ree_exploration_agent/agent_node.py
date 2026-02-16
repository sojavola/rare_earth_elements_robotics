#!/usr/bin/env python3
# agent_node.py — Version CORRIGÉE avec garantie de mouvement

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import random
import sys
import os
import json
import re
from datetime import datetime

import pandas as pd
from std_msgs.msg import Float32MultiArray, String, Int32, Float32
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import OccupancyGrid

# Import des agents
from .advanced_dqn_agent import RobustDQNAgent
from .science_reward_system import RealMineralRewardSystem

class REEExplorationAgent(Node):
    def __init__(self, robot_id: int = 0):
        super().__init__(f'ree_exploration_agent_{robot_id}')

        # --- Identification du robot ---
        self.robot_id = int(robot_id)
        self.agent_name = f'robot_{self.robot_id}'
        self.get_logger().info(f'🎯 ROBOT ID DÉFINI: {self.robot_id}')

        # --- Configuration ---
        #self.num_channels = 8  # Canaux minéraux

        self.state_channels = 8
        #self.mineral_channels = 4  # Canaux minéraux
        self.map_width = 100
        self.map_height = 100
        self.num_actions = 8
        self.state_shape = (self.map_height, self.map_width,self.state_channels) 
        self.decision_hz = 1.0

        # --- États / cartes ---
        self.mineral_map = np.zeros((self.map_height, self.map_width, self.state_channels), dtype=np.float32)
        self.obstacle_map = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.science_targets = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        self.current_position = self.get_initial_position()
        self.last_position = self.current_position

        # --- Statistiques ---
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.minerals_collected = 0
        self.episode_memory = []
        self.latest_loss = 0.0
        
        ###
        self.unique_minerals_found = set()
        ###
        # --- Force démarrage ---
        self.force_start_executed = False

        # --- Thread lock ---
        self.lock = threading.Lock()

        # ============================================
        # 🎯 INITIALISATION DU SYSTÈME DE RÉCOMPENSES SCIENTIFIQUE
        # ============================================
        self.get_logger().info('🧪 Initializing Scientific Reward System...')
        
        # Initialiser UNIQUEMENT le système scientifique
        self.reward_system = RealMineralRewardSystem(
            grid_size=(self.map_width, self.map_height),
            robot_id=self.robot_id
        )
        self.visited_positions = set()
        self.cleaned_positions = set()  # Pour suivi académique
        self.steps_without_mineral = 0

        
        # ============================================
        # 🤖 INITIALISATION DE L'AGENT DQN
        # ============================================
        self.get_logger().info('🧠 Initializing DQN agent...')
        
        self.agent = RobustDQNAgent(
            state_shape=self.state_shape,
            num_actions=self.num_actions,
            robot_id=self.robot_id,
            learning_rate=0.00025,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.998,
            memory_size=1000,
            batch_size=32,
            target_update=1000,
            load_latest_model=True,
            enable_logging=True,
            use_real_reward_system=True,
            use_shared_model=False
        )

        
        self.get_logger().info(f'🧠 DQN initialized with scientific reward system')
        self.get_logger().info(f'🤖 Agent epsilon: {self.agent.epsilon:.3f}')
        
        ####
        xx, yy = np.meshgrid(np.arange(self.map_width), np.arange(self.map_height))
        self.coord_map = np.stack([
            xx.astype(np.float32) / self.map_width,
            yy.astype(np.float32) / self.map_height
        ], axis=-1)
        
        self.visited_map = np.zeros((self.map_height, self.map_width, 1), dtype=np.float32)
        ####


        # --- Publishers ---
        self.dqn_update_pub = self.create_publisher(Float32MultiArray, '/agent/dqn_update', 10)
        self.step_completed_pub = self.create_publisher(Int32, '/agent/step_completed', 10)
        self.epsilon_pub = self.create_publisher(Float32, '/agent/epsilon', 10)
        
        self.position_pub = self.create_publisher(Pose2D, f'/{self.agent_name}/position', 10)
        self.velocity_pub = self.create_publisher(Twist, f'/{self.agent_name}/cmd_vel', 10)
        self.cleaning_pub = self.create_publisher(Float32MultiArray, f'/{self.agent_name}/cleaning_action', 10)
        self.status_pub = self.create_publisher(String, f'/{self.agent_name}/status', 10)
        self.discovery_pub = self.create_publisher(Float32MultiArray, '/shared_discoveries', 10)

        # --- Subscribers ---
        self.mineral_sub = self.create_subscription(Float32MultiArray, '/mineral_map', self.mineral_callback, 10)
        self.obstacle_sub = self.create_subscription(OccupancyGrid, '/obstacle_map', self.obstacle_callback, 10)
        self.science_sub = self.create_subscription(Float32MultiArray, '/science_targets', self.science_callback, 10)

        # ============================================
        # ⏰ TIMERS - SOLUTION GARANTIE
        # ============================================
        self.get_logger().info('⏰ Configuration des timers...')
        
        # 1. Timer de test pour vérifier ROS2
        self.test_timer = self.create_timer(1.0, self.test_timer_callback)
        self.get_logger().info('✅ Timer de test créé')
        
        # 2. Timer FORCÉ pour le démarrage
        self.force_timer = self.create_timer(0.5, self.force_first_decision)
        self.get_logger().info('✅ Timer forcé créé')
        
        # 3. Timer principal (peut ne pas fonctionner, d'où la solution de secours)
        try:
            self.decision_timer = self.create_timer(1.0 / self.decision_hz, self.make_decision)
            self.get_logger().info(f'✅ Timer principal créé à {self.decision_hz} Hz')
        except Exception as e:
            self.get_logger().error(f'❌ Erreur création timer principal: {e}')
            self.decision_timer = None
        
        # 4. Autres timers
        self.status_timer = self.create_timer(5.0, self.publish_status)
        self.position_timer = self.create_timer(0.5, self.publish_position)

        # --- Dirs de sauvegarde ---
        self.save_dir = os.path.join("models", f"robot_{self.robot_id}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info(f'🚀 Scientific DQN Agent Node {self.robot_id} initialized')
        self.get_logger().info(f'📍 Position initiale: {self.current_position}')


        self.training_logs = {
            "episode": [],
            "reward": [],
            "steps": [],
            "minerals": [],
            "loss": [],
            "epsilon": []
        }

        self.get_logger().info(f'✅ Coord map shape: {self.coord_map.shape if hasattr(self, "coord_map") else "NON INITIALISÉ"}')
        self.get_logger().info(f'✅ Visited map shape: {self.visited_map.shape if hasattr(self, "visited_map") else "NON INITIALISÉ"}')
        self.get_logger().info(f'✅ Training logs: {"INITIALISÉ" if hasattr(self, "training_logs") else "NON INITIALISÉ"}')

        # Test get_current_state
        try:
            test_state = self.get_current_state()
            self.get_logger().info(f'✅ Test state shape: {test_state.shape}')
        except Exception as e:
            self.get_logger().error(f'❌ Test state failed: {e}')
                

    # ============================================
    # ⏰ CALLBACKS DES TIMERS
    # ============================================
    def test_timer_callback(self):
        """Vérifie que ROS2 fonctionne"""
        self.get_logger().info('✅ TEST: ROS2 timer fonctionnel')

    def force_first_decision(self):
        """FORCE la première décision - solution de secours"""
        if not self.force_start_executed:
            self.get_logger().info('🚨 FORCE FIRST DECISION - Solution de secours')
            self.force_start_executed = True
            
            # Appeler make_decision immédiatement
            try:
                self.make_decision()
            except Exception as e:
                self.get_logger().error(f'❌ Erreur force_first_decision: {e}')
            
            # Si le timer principal n'est pas créé, créer un thread manuel
            if self.decision_timer is None:
                self.get_logger().warn('⚠️ Timer principal non disponible, démarrage thread manuel')
                self.start_manual_control()

    def start_manual_control(self):
        """Démarre un contrôle manuel si ROS2 timers échouent"""
        import threading
        
        def manual_control_loop():
            self.get_logger().info('🎮 DÉMARRAGE CONTRÔLE MANUEL')
            while rclpy.ok():
                try:
                    self.make_decision()
                    time.sleep(1.0 / self.decision_hz)
                except Exception as e:
                    self.get_logger().error(f'❌ Erreur contrôle manuel: {e}')
                    time.sleep(1.0)
        
        self.manual_thread = threading.Thread(target=manual_control_loop, daemon=True)
        self.manual_thread.start()
        self.get_logger().info('✅ Thread de contrôle manuel démarré')

    # ============================================
    # 🎯 MÉTHODE PRINCIPALE CORRIGÉE
    # ============================================

    def make_decision(self):
        """Prend une décision - VERSION GARANTIE"""
        try:
            """
            self.get_logger().info('='*50)
            self.get_logger().info(f'🚀 MAKE_DECISION() - Step {self.steps}')
            self.get_logger().info('='*50)
            
            # DEBUG: Vérifier l'état
            mineral_max = np.max(self.mineral_map)
            self.get_logger().info(f'📊 Mineral map max: {mineral_max:.4f}')
            self.get_logger().info(f'📍 Position actuelle: {self.current_position}')
            self.get_logger().info(f'🤖 Agent ε: {self.agent.epsilon:.3f}')
            """
            # 1. Obtenir l'état
            state = self.get_current_state()
            
            # 2. Choix d'action
            if self.steps < 5 or random.random() < self.agent.epsilon:
                action = random.randint(0, self.num_actions - 1)
                #self.get_logger().info(f'🎲 Exploration: action {action}')
            else:
                try:
                    action = self.agent.choose_action(state)
                    #self.get_logger().info(f'🎯 Exploitation: action {action}')
                except Exception as e:
                    #self.get_logger().error(f'❌ Erreur choose_action: {e}')
                    action = random.randint(0, self.num_actions - 1)
                    #self.get_logger().info(f'🎲 Fallback: action {action}')
            
            # 3. Exécuter l'action
            #self.get_logger().info(f'▶️  Exécution action {action}...')
            reward, done = self.execute_action(action)
            
            #self.get_logger().info(f'💰 Reward: {reward:.2f}')
            #self.get_logger().info(f'📍 Nouvelle position: {self.current_position}')
            
            # 4. Stocker l'expérience
            try:
                self.agent.store_experience(
                    state=state, 
                    action=action, 
                    reward=reward, 
                    next_state=self.get_current_state(), 
                    done=done, 
                    position=self.current_position
                )
                self.get_logger().info(f'💾 Expérience stockée')
            except Exception as e:
                self.get_logger().error(f'❌ store_experience: {e}')
            
            # 5. Mettre à jour les statistiques
            self.episode_reward += reward
            self.total_reward += reward
            self.steps += 1
            
            self.get_logger().info(f'📈 Episode reward: {self.episode_reward:.1f}')
            
            # 6. Entraîner périodiquement
            if len(self.agent.memory) >= self.agent.batch_size and self.steps % 2 == 0:
                try:
                    loss = self.agent.train()
                    if loss is not None:
                        self.latest_loss = float(loss)
                        self.publish_dqn_update(self.latest_loss)
                        self.get_logger().info(f'📚 Training: loss={loss:.4f}')
                except Exception as e:
                    self.get_logger().error(f'❌ Training failed: {e}')
            
            # 7. Fin d'épisode
            if done or self.steps >= 50:  # Réduit pour test
                self.get_logger().info(f'🏁 Fin épisode: steps={self.steps}')
                
                # Mettre à jour le compteur d'épisodes
                self.episodes += 1
                
                # Enregistrer les métriques pour l'épisode complété
                self.training_logs["episode"].append(self.episodes)
                self.training_logs["reward"].append(self.episode_reward)
                self.training_logs["steps"].append(self.steps)
                
                
                # ✅ RÉINITIALISER LE SYSTÈME DE RÉCOMPENSES
                self.reset_reward_system()
                
                # Sauvegarde périodique des logs
                if self.episodes % 20 == 0:
                    self.save_training_logs()
                
                # Sauvegarde périodique du modèle
                if self.episodes % 5 == 0:
                    self.save_model()
                
                # Afficher les statistiques de l'épisode
                self.get_logger().info(
                    f'📊 Episode {self.episodes} completed: '
                    f'Steps={self.steps}, Reward={self.episode_reward:.1f}, '
                    f'Minerals={self.minerals_collected}, ε={self.agent.epsilon:.3f}'
                )
                
                # Réinitialiser pour le prochain épisode
                self.current_position = self.get_initial_position()
                self.last_position = self.current_position
                self.episode_reward = 0.0
                self.steps = 0
                self.minerals_collected = 0  # Déjà fait dans reset_reward_system, mais au cas où
                self.episode_memory = []
                
                # Démarrer un nouvel épisode dans l'agent
                self.agent.start_episode()
                # Dans make_decision(), dans le bloc "Fin d'épisode"

                self.agent.end_episode()
                self.reset_reward_system()  # ← CET APPEL
                
                self.get_logger().info(f'🔄 Starting new episode at {self.current_position}')
                return  # Retourner après réinitialisation
            
            self.get_logger().info(f'✅ Decision complétée\n')
            
        except Exception as e:
            self.get_logger().error(f'❌ ERREUR CRITIQUE dans make_decision: {e}')
            import traceback
            traceback.print_exc()

    # ============================================
    # 🎯 EXÉCUTION DES ACTIONS
    # ============================================
    def execute_action(self, action: int):
        """Exécute une action avec logs détaillés"""
        x, y = self.current_position
        
        self.get_logger().info(f'  🎯 Début execute_action: ({x},{y}), action {action}')
        
        direction_vectors = [
            (0, 1),    # 0: Nord
            (0, -1),   # 1: Sud
            (-1, 0),   # 2: Ouest
            (1, 0),    # 3: Est
            (-1, 1),   # 4: Nord-Ouest
            (1, 1),    # 5: Nord-Est
            (-1, -1),  # 6: Sud-Ouest
            (1, -1)    # 7: Sud-Est
        ]
        
        dx, dy = direction_vectors[int(action) % len(direction_vectors)]
        new_x = max(0, min(self.map_width - 1, x + dx))
        new_y = max(0, min(self.map_height - 1, y + dy))
        
        self.get_logger().info(f'  ➕ Calcul: ({x},{y}) + ({dx},{dy}) → ({new_x},{new_y})')
        
        if self.is_valid_position(new_x, new_y):
            self.last_position = self.current_position
            self.current_position = (new_x, new_y)
            
            self.get_logger().info(f'  ✅ Déplacement: {self.last_position} → {self.current_position}')
            
            # Calculer la récompense
            reward = self.calculate_reward(new_x, new_y)
            done = self.is_episode_done()
            
            # Actions secondaires
            self.publish_cleaning_action(new_x, new_y)
            self.publish_velocity(linear_x=0.15, angular_z=0.0)
            
            if reward > 50.0:
                self.get_logger().info(f'  💎 Découverte majeure!')
                self.publish_discovery(new_x, new_y, reward)
            
            return reward, done
        else:
            self.get_logger().warn(f'  ❌ Position invalide')
            return -2.0, False

    # ============================================
    # 🧠 CALCUL DES RÉCOMPENSES
    # ============================================
    def calculate_reward(self, x, y):
        """Calcule la récompense ACADÉMIQUE basée sur les minéraux et la heatmap"""
        try:
            with self.lock:
                # 1. Obtenir les concentrations minérales RÉELLES
                mineral_concentrations = self.mineral_map[y, x, :].tolist()
                
                # 2. Vérifier si c'est une nouvelle position
                position_key = (int(x), int(y))
                is_new_position = position_key not in self.visited_positions
                
                # 3. Vérifier les collisions
                has_collision = int(self.obstacle_map[y, x]) != 0
                
                # 4. Calculer la récompense ACADÉMIQUE HYBRIDE
                reward = self.reward_system.calculate_reward(
                    mineral_concentrations=mineral_concentrations,
                    position=(x, y),
                    is_new_position=is_new_position,
                    has_collision=has_collision,
                    step_count=self.steps,
                    sensor_data=None
                )
                
                # 5. Mettre à jour le suivi local
                self.visited_positions.add(position_key)
                
                # 6. Log des événements significatifs
                if reward > 50.0:  # Détection de minéral ou zone prioritaire
                    max_concentration = max(mineral_concentrations) if mineral_concentrations else 0.0
                    
                    # Obtenir la breakdown académique
                    breakdown = self.reward_system.get_reward_breakdown((x, y), mineral_concentrations)
                    
                    self.get_logger().info(
                        f'🎯 DÉTECTION à ({x},{y}):\n'
                        f'   Reward: {reward:.1f}\n'
                        f'   Concentration: {max_concentration:.3f}\n'
                        f'   Heatmap: {breakdown["heatmap_value"]:.3f}\n'
                        f'   Académique: {breakdown["academic_potential"]:.1f}\n'
                        f'   Réel: {breakdown["real_potential"]:.1f}'
                    )
                    
                    if max_concentration > 0.3:  # Vrai minéral
                        mineral_type = np.argmax(mineral_concentrations)
                        self.minerals_collected += 1
                        self.unique_minerals_found.add(mineral_type)
                        self.publish_discovery(x, y)
                        ###
                        self.publish_discovery(x, y, reward)
                        ###
                return reward
                
        except Exception as e:
            self.get_logger().error(f'❌ Error in calculate_reward: {e}')
            return 0.0
    
    def reset_reward_system(self):
        """Réinitialise le système de récompenses académique pour un nouvel épisode"""
        try:
            # Conserver les informations d'apprentissage avant reset
            minerals_before = self.reward_system.minerals_collected
            stats_before = self.reward_system.get_statistics()
            
            # Réinitialiser le système académique
            self.reward_system.reset_episode()
            
            # Réinitialiser les ensembles de suivi local
            self.visited_positions.clear()
            self.steps_without_mineral = 0
            
            # Log détaillé
            self.get_logger().info(
                f'🔄 Système académique réinitialisé (Robot {self.robot_id}):\n'
                f'   Minéraux collectés ce run: {self.minerals_collected}\n'
                f'   Minéraux totaux (historique): {minerals_before}\n'
                f'   Couverture précédente: {stats_before.get("coverage_percentage", 0):.1f}%\n'
                f'   Propreté précédente: {stats_before.get("cleanliness_percentage", 0):.1f}%\n'
                f'   Updates gaussiens: {stats_before.get("gaussian_updates", 0)}'
            )
            
            # Réinitialiser aussi le compteur local de minéraux pour l'épisode
            self.minerals_collected = 0
            self.unique_minerals_found.clear()
            
        except Exception as e:
            self.get_logger().error(f'❌ Error in reset_reward_system: {e}')
    
    def print_reward_report(self):
        """Affiche un rapport détaillé du système académique"""
        try:
            stats = self.reward_system.get_statistics()
            
            # Rapport compact pour logs ROS
            self.get_logger().info(
                f'📊 ACADÉMIQUE Robot {self.robot_id} (Step {self.steps}):\n'
                f'   Minéraux collectés: {stats["minerals_collected"]}\n'
                f'   Reward Total: {stats["total_reward"]:.1f}\n'
                f'   → Académique: {stats.get("academic_reward", 0):.1f}\n'
                f'   → Réel: {stats.get("real_reward", 0):.1f}\n'
                f'   Positions visitées: {stats["visited_positions"]}\n'
                f'   Couverture: {stats["coverage_percentage"]:.1f}%\n'
                f'   Propreté (c_perc): {stats.get("cleanliness_percentage", 0):.1f}%'
            )
            
            # Rapport détaillé tous les 5 épisodes ou tous les 200 steps
            if self.episodes % 5 == 0 or (self.steps % 200 == 0 and self.steps > 0):
                self.reward_system.print_detailed_report()
                
        except Exception as e:
            self.get_logger().error(f'❌ Error in print_reward_report: {e}')


    def save_training_logs(self):
        """Sauvegarde les logs d'entraînement"""
        try:
            if hasattr(self, 'training_logs') and self.training_logs["episode"]:
                df = pd.DataFrame(self.training_logs)
                log_file = os.path.join(self.save_dir, "training_logs.csv")
                df.to_csv(log_file, index=False)
                self.get_logger().info(f'📊 Logs sauvegardés: {log_file}')
        except Exception as e:
            self.get_logger().error(f'❌ Erreur save_training_logs: {e}')
    
    def save_model(self):
        """Sauvegarde le modèle"""
        try:
            if hasattr(self.agent, 'save'):
                self.agent.save()
                self.get_logger().info(f'💾 Modèle sauvegardé pour robot {self.robot_id}')
        except Exception as e:
            self.get_logger().error(f'❌ Erreur save_model: {e}')
    

    def check_nearby_obstacles(self, x, y, radius=2):
        """Vérifie les obstacles à proximité"""
        with self.lock:
            x_min = max(0, x - radius)
            x_max = min(self.map_width - 1, x + radius)
            y_min = max(0, y - radius)
            y_max = min(self.map_height - 1, y + radius)
            
            region = self.obstacle_map[y_min:y_max+1, x_min:x_max+1]
            return np.any(region != 0)

    # ============================================
    # 📊 GESTION DES ÉPISODES
    # ============================================
    def end_episode(self):
        """Termine un épisode"""
        self.episodes += 1
        
        self.get_logger().info(
            f'📊 Episode {self.episodes} terminé: '
            f'Steps={self.steps}, Reward={self.episode_reward:.1f}, '
            f'Minerals={self.minerals_collected}'
        )
        
        # Réinitialiser
        self.current_position = self.get_initial_position()
        self.last_position = self.current_position
        self.episode_reward = 0.0
        self.steps = 0
        self.minerals_collected = 0
        self.episode_memory = []
        
        # Réinitialiser le système scientifique
        self.scientific_reward_system.reset_episode()
        
        self.get_logger().info(f'🔄 Nouvel épisode à {self.current_position}')

    # ============================================
    # 🎯 MÉTHODES UTILITAIRES
    # ============================================
    def get_initial_position(self):
        pos = (random.randint(5, self.map_width - 6), random.randint(5, self.map_height - 6))
        self.get_logger().info(f'📍 Position initiale: {pos}')
        return pos
    
    def get_current_state(self):
        """
        État optimisé selon les principes DQN
        Shape: (100, 100, 8)
        """
        with self.lock:
            x, y = self.current_position
            
            # 1. Minéraux normalisés (4 canaux)
            mineral_state = self.mineral_map[:, :, :4].copy()
            max_val = np.max(mineral_state)
            if max_val > 1e-8:
                mineral_state /= max_val
            
            # 2. Position actuelle (1 canal binaire)
            pos_map = np.zeros((self.map_height, self.map_width, 1), dtype=np.float32)
            if 0 <= y < self.map_height and 0 <= x < self.map_width:
                pos_map[y, x, 0] = 1.0
            
            # 3. Obstacles normalisés (1 canal)
            obstacle_state = self.obstacle_map.reshape(self.map_height, self.map_width, 1).astype(np.float32)
            obstacle_state = np.clip(obstacle_state / 100.0, 0, 1)
            
            # 4. Historique visitées (1 canal décroissant)
            visited_map = np.zeros((self.map_height, self.map_width, 1), dtype=np.float32)
            for vx, vy in self.visited_positions:
                if 0 <= vy < self.map_height and 0 <= vx < self.map_width:
                    visited_map[vy, vx, 0] = 0.5  # Valeur plus faible que position actuelle
            
            # 5. Heatmap scientifique normalisée (1 canal)
            science_state = self.science_targets.reshape(self.map_height, self.map_width, 1)
            max_sci = np.max(science_state)
            if max_sci > 1e-8:
                science_state /= max_sci
            
            # Concaténation finale: 4+1+1+1+1 = 8 canaux
            state = np.concatenate([
                mineral_state,    # (100, 100, 4)
                pos_map,          # (100, 100, 1)
                obstacle_state,   # (100, 100, 1)
                visited_map,      # (100, 100, 1)
                science_state     # (100, 100, 1)
            ], axis=-1)
            
            # Validation
            assert state.shape == (self.map_height, self.map_width, 8), \
                f"État invalide: {state.shape}"
            
            return state

    """
    def is_valid_position(self, x, y):
        with self.lock:
            valid = (0 <= x < self.map_width and 
                     0 <= y < self.map_height and 
                     int(self.obstacle_map[y, x]) == 0)
            if not valid:
                self.get_logger().debug(f'  ❌ Position ({x},{y}) invalide')
            return valid
    """  
    def is_valid_position(self, x, y):
        """Vérifie si une position est valide avec buffer de sécurité"""
        with self.lock:
            # DEBUG initial
            self.get_logger().debug(f'🔍 is_valid_position({x},{y}) appelée')
            
            # 1. Vérifier les limites strictes
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                self.get_logger().warn(f'❌ Position ({x},{y}) hors limites: map={self.map_width}x{self.map_height}')
                return False
            
            # 2. Obtenir la valeur d'obstacle
            try:
                obstacle_value = int(self.obstacle_map[y, x])
            except Exception as e:
                self.get_logger().error(f'❌ Erreur lecture obstacle_map: {e}')
                # Par défaut, considérer comme valide
                return True
            
            # 3. Log détaillé pour debugging
            if obstacle_value != 0:
                self.get_logger().debug(f'📊 Position ({x},{y}): obstacle={obstacle_value}')
            
            # 4. Définir ce qui est considéré comme obstacle
            # ROS standard: 0=libre, 100=obstacle, -1=inconnu
            OBSTACLE_THRESHOLD = 50  # Tout ce qui est > 50 = obstacle
            
            if obstacle_value > OBSTACLE_THRESHOLD:
                self.get_logger().info(f'🚫 Position ({x},{y}) bloquée: obstacle={obstacle_value}')
                return False
            
            # 5. Vérifier aussi les cases adjacentes pour éviter de se coincer
            # (optionnel mais recommandé)
            SAFE_MARGIN = 1  # Vérifier 1 case autour
            
            for dx in range(-SAFE_MARGIN, SAFE_MARGIN + 1):
                for dy in range(-SAFE_MARGIN, SAFE_MARGIN + 1):
                    if dx == 0 and dy == 0:
                        continue  # On a déjà vérifié la case centrale
                        
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        try:
                            nearby_obstacle = int(self.obstacle_map[ny, nx])
                            if nearby_obstacle > 80:  # Obstacle très proche
                                self.get_logger().debug(f'⚠️ Obstacle proche à ({nx},{ny}): {nearby_obstacle}')
                        except:
                            pass
            
            return True  
        
    def is_episode_done(self):
        if self.steps > 30 and self.episode_reward < 2.0:
            return True
        if self.minerals_collected >= 2:
            return True
        if self.steps >= 50:
            return True
        return False
    # ============================================
    # 📞 CALLBACKS
    # ============================================
    def mineral_callback(self, msg: Float32MultiArray):
        """
        Reçoit la carte minérale (4 canaux : Nd, Eu, Tb, Dy)
        """
        with self.lock:
            data = np.array(msg.data, dtype=np.float32)
            
            # ✅ VÉRIFICATION : Doit être 4 canaux
            expected_size = self.map_height * self.map_width * 4
            
            if data.size == expected_size:
                self.mineral_map = data.reshape(self.map_height, self.map_width, 4)
                max_val = np.max(self.mineral_map)
                self.get_logger().info(f'📊 Mineral map reçu: Max={max_val:.3f}')
            else:
                self.get_logger().warn(
                    f'⚠️ Mineral map size mismatch: '
                    f'expected {expected_size}, got {data.size}'
                )

    def obstacle_callback(self, msg: OccupancyGrid):
        with self.lock:
            data = np.array(msg.data, dtype=np.int8)
            if data.size == self.map_height * self.map_width:
                self.obstacle_map = data.reshape(self.map_height, self.map_width)

    def science_callback(self, msg: Float32MultiArray):
        with self.lock:
            data = np.array(msg.data, dtype=np.float32)
            if data.size == self.map_height * self.map_width:
                self.science_targets = data.reshape(self.map_height, self.map_width)

    # ============================================
    # 📤 PUBLISHERS
    # ============================================
    def publish_position(self):
        msg = Pose2D()
        msg.x = float(self.current_position[0])
        msg.y = float(self.current_position[1])
        msg.theta = 0.0
        self.position_pub.publish(msg)

    def publish_cleaning_action(self, x, y):
        msg = Float32MultiArray()
        msg.data = [float(x), float(y)]
        self.cleaning_pub.publish(msg)

    def publish_velocity(self, linear_x=0.1, angular_z=0.0):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.velocity_pub.publish(msg)

    def publish_discovery(self, x, y, reward):
        msg = Float32MultiArray()
        mineral_data = self.mineral_map[y, x, :].tolist()
        
        msg.data = [
            float(self.robot_id),
            float(x),
            float(y),
            float(reward)
        ] + mineral_data
        
        self.discovery_pub.publish(msg)
        self.get_logger().info(f'🔬 Discovery publiée à ({x},{y})')
    """ 
    def publish_status(self):
        status_text = (
            f"Robot {self.robot_id} - Episode: {self.episodes}, Steps: {self.steps}, "
            f"Reward: {self.episode_reward:.1f}, Minerals: {self.minerals_collected}, "
            f"Pos: {self.current_position}, ε: {self.agent.epsilon:.3f}"
        )
        
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
    """
    
    def publish_status(self):
        status_text = (f"Robot {self.robot_id} - Episode: {self.episodes}, Steps: {self.steps}, "
                    f"EpisodeReward: {self.episode_reward:.1f}, "
                    f"Minerals: {self.minerals_collected}, "
                    f"ε: {getattr(self.agent, 'epsilon', 0):.3f}")
        
        # Afficher le rapport académique tous les 100 steps
        if self.steps % 100 == 0 and self.steps > 0:
            self.print_reward_report()
        
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        
        # Log périodique
        if self.steps % 20 == 0:
            self.get_logger().info(f'📈 {status_text}')


    def publish_dqn_update(self, loss: float = 0.0):
        try:
            update_msg = Float32MultiArray()
            update_msg.data = [
                float(self.robot_id),
                float(loss),
                float(self.agent.epsilon),
                float(len(self.agent.memory))
            ]
            self.dqn_update_pub.publish(update_msg)
        except Exception as e:
            self.get_logger().warn(f'⚠️ publish_dqn_update failed: {e}')

def main(argv=None):
    rclpy.init(args=argv)
    argv = argv if argv is not None else sys.argv
    
    # Parser robot ID
    robot_id = 0
    if len(argv) > 1:
        try:
            robot_id = int(argv[1])
        except ValueError:
            nums = re.findall(r'\d+', argv[1])
            if nums:
                robot_id = int(nums[0])
    
    node = REEExplorationAgent(robot_id=robot_id)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()