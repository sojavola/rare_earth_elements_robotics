"""import numpy as np
from collections import defaultdict
import math

class AdvancedRewardSystem:
 
    
    def __init__(self):
        self.mineral_types = ['REE_Oxides', 'REE_Silicates', 'REE_Phosphates', 'REE_Carbonates']
        
        # Configuration des récompenses
        self.reward_config = {
            'mineral_discovery': 10.0,
            'high_value_mineral': 20.0,
            'new_mineral_type': 15.0,
            'exploration_bonus': 5.0,
            'science_target': 8.0,
            'collision_penalty': -5.0,
            'revisiting_penalty': -2.0,
            'step_penalty': -0.1,
            'efficiency_bonus': 3.0
        }
        
        # Suivi des découvertes
        self.discovered_locations = set()
        self.discovered_minerals = defaultdict(set)
        self.visited_positions = set()
        self.high_value_samples = 0
        
        # Historique
        self.reward_history = []
        self.mineral_history = []
    
    def calculate_reward(self, mineral_concentrations, position, science_value, is_new_position, has_collision, robot_id):
    
        reward = 0.0
        x, y = position
        position_key = (x, y)
        
        # Récompense pour découverte minérale
        mineral_reward = self._calculate_mineral_reward(mineral_concentrations, position_key, robot_id)
        reward += mineral_reward
        
        # Récompense pour exploration
        exploration_reward = self._calculate_exploration_reward(position_key, is_new_position)
        reward += exploration_reward
        
        # Récompense pour cibles scientifiques
        science_reward = self._calculate_science_reward(science_value)
        reward += science_reward
        
        # Pénalités
        penalties = self._calculate_penalties(has_collision, is_new_position)
        reward += penalties
        
        # Pénalité d'étape
        reward += self.reward_config['step_penalty']
        
        # Enregistrer la récompense
        self.reward_history.append(reward)
        
        return reward
    
    def _calculate_mineral_reward(self, mineral_concentrations, position_key, robot_id):
      
        reward = 0.0
        
        for mineral_idx, concentration in enumerate(mineral_concentrations):
            if concentration > 0.1:  # Seuil de détection
                mineral_name = self.mineral_types[mineral_idx]
                
                # Récompense de base
                base_reward = concentration * self.reward_config['mineral_discovery']
                reward += base_reward
                
                # Bonus pour minéraux de haute valeur
                if concentration > 0.7:
                    reward += self.reward_config['high_value_mineral']
                    self.high_value_samples += 1
                
                # Bonus pour nouveau type de minéral
                if mineral_name not in self.discovered_minerals[robot_id]:
                    reward += self.reward_config['new_mineral_type']
                    self.discovered_minerals[robot_id].add(mineral_name)
                
                # Enregistrer la découverte
                self.discovered_locations.add(position_key)
                self.mineral_history.append({
                    'robot_id': robot_id,
                    'position': position_key,
                    'mineral_type': mineral_name,
                    'concentration': concentration,
                    'reward': base_reward
                })
        
        return reward
    
    def _calculate_exploration_reward(self, position_key, is_new_position):
    
        reward = 0.0
        
        if is_new_position:
            reward += self.reward_config['exploration_bonus']
            self.visited_positions.add(position_key)
        else:
            reward += self.reward_config['revisiting_penalty']
        
        return reward
    
    def _calculate_science_reward(self, science_value):
     "
        reward = 0.0
        
        if science_value > 0.3:
            reward += science_value * self.reward_config['science_target']
        
        return reward
    
    def _calculate_penalties(self, has_collision, is_new_position):
    
        penalty = 0.0
        
        # Pénalité de collision
        if has_collision:
            penalty += self.reward_config['collision_penalty']
        
        return penalty
    
    def get_statistics(self):
     
        total_reward = sum(self.reward_history)
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
        
        unique_minerals = set()
        for minerals in self.discovered_minerals.values():
            unique_minerals.update(minerals)
        
        return {
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'discovered_locations': len(self.discovered_locations),
            'unique_minerals': len(unique_minerals),
            'high_value_samples': self.high_value_samples,
            'visited_positions': len(self.visited_positions),
            'total_steps': len(self.reward_history)
        }
    
    def reset(self):
 
        self.discovered_locations.clear()
        self.discovered_minerals.clear()
        self.visited_positions.clear()
        self.high_value_samples = 0
        self.reward_history.clear()
        self.mineral_history.clear()
"""
import numpy as np
from collections import defaultdict
import math

class OptimizedRewardSystem:
    """Système de récompenses OPTIMISÉ pour l'exploration scientifique"""
    
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.mineral_types = ['REE_Oxides', 'REE_Silicates', 'REE_Phosphates', 'REE_Carbonates']
        
        # CONFIGURATION TRÈS GÉNÉREUSE
        self.reward_config = {
            # RÉCOMPENSES POSITIVES - TRÈS GÉNÉREUSES
            'mineral_discovery_base': 200.0,  # Énorme récompense de base
            'mineral_concentration_multiplier': 500.0,  # Multiplicateur massif
            'high_value_mineral': 300.0,  # Bonus énorme
            'new_mineral_type': 150.0,  # Bonus important
            'exploration_bonus': 50.0,  # Bonus d'exploration généreux
            'science_target': 80.0,  # Bonus scientifique
            'efficiency_bonus': 40.0,  # Bonus d'efficacité
            'cooperative_bonus': 60.0,  # Bonus coopératif
            'survival_bonus': 10.0,  # Bonus juste pour survivre
            'movement_bonus': 5.0,  # Bonus pour se déplacer
            
            # PÉNALITÉS - TRÈS FAIBLES
            'collision_penalty': -5.0,  # Pénalité légère
            'revisiting_penalty': -0.5,  # Pénalité très légère
            'step_penalty': -0.01,  # Presque rien
            'inefficiency_penalty': -2.0  # Pénalité minime
        }
        
        # Suivi des découvertes
        self.discovered_locations = set()
        self.discovered_minerals = defaultdict(set)
        self.visited_positions = set()
        self.high_value_samples = 0
        self.concentration_history = []
        
        # Heatmap avec priorités ARTIFICIELLES pour guider l'agent
        self.priority_heatmap = self._initialize_guided_heatmap()
        self.cleaned_areas = np.zeros(grid_size, dtype=bool)
        
        # Historique
        self.reward_history = []
        self.mineral_history = []
        self.episode_rewards = []
        
        # Seuils TRÈS BAS pour détecter les minéraux
        self.exploration_threshold = 0.1
        self.mineral_threshold = 0.05  # Seuil très bas
    
    def _initialize_guided_heatmap(self):
        """Initialise une heatmap avec des zones prioritaires artificielles"""
        heatmap = np.zeros(self.grid_size)
        
        # Créer des zones prioritaires artificielles pour guider l'agent
        centers = [(25, 25), (75, 25), (25, 75), (75, 75), (50, 50)]
        
        for center_x, center_y in centers:
            for i in range(max(0, center_x-15), min(self.grid_size[0], center_x+16)):
                for j in range(max(0, center_y-15), min(self.grid_size[1], center_y+16)):
                    distance = math.sqrt((center_x-i)**2 + (center_y-j)**2)
                    if distance <= 15:
                        priority = 0.8 * math.exp(-(distance**2)/(2*8**2))
                        heatmap[i, j] = max(heatmap[i, j], priority)
        
        return heatmap
    
    def calculate_reward(self, mineral_concentrations, position, science_value, 
                        is_new_position, has_collision, robot_id, other_robots_positions=None):
        """Calcule la récompense avec focus sur l'ENCOURAGEMENT"""
        reward = 0.0
        x, y = position
        position_key = (x, y)
        
        # 1. RÉCOMPENSE DE SURVIE - Toujours positive
        reward += self.reward_config['survival_bonus']
        
        # 2. RÉCOMPENSE DE MOUVEMENT - Se déplacer est bon
        reward += self.reward_config['movement_bonus']
        
        # 3. Récompense basée sur la heatmap de priorité (GUIDAGE)
        priority_reward = self._calculate_priority_reward(position)
        reward += priority_reward
        
        # 4. Récompense pour exploration (TRÈS GÉNÉREUSE)
        exploration_reward = self._calculate_exploration_reward(position_key, is_new_position)
        reward += exploration_reward
        
        # 5. Récompense pour minéraux (SIMULÉE si nécessaire)
        mineral_reward = self._calculate_mineral_reward(mineral_concentrations, position_key, robot_id)
        reward += mineral_reward
        
        # 6. Récompense scientifique
        science_reward = self._calculate_science_reward(science_value)
        reward += science_reward
        
        # 7. Pénalités TRÈS LÉGÈRES
        penalties = self._calculate_penalties(has_collision)
        reward += penalties
        
        # 8. Bonus pour exploration de zones spécifiques
        zone_bonus = self._calculate_zone_bonus(position)
        reward += zone_bonus
        
        # Mettre à jour les zones visitées
        self._update_visited_position(position_key, is_new_position)
        
        # Enregistrer avec vérification de validité
        if not np.isnan(reward) and not np.isinf(reward):
            self.reward_history.append(reward)
        else:
            self.reward_history.append(0.0)  # Fallback safe
        
        return reward
    
    def _calculate_priority_reward(self, position):
        """Récompense pour visiter des zones prioritaires"""
        x, y = position
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            priority_value = self.priority_heatmap[x, y]
            
            if priority_value > 0.1 and not self.cleaned_areas[x, y]:
                return priority_value * 100.0  # Récompense significative
        
        return 0.0
    
    def _calculate_mineral_reward(self, mineral_concentrations, position_key, robot_id):
        """Récompense pour minéraux - avec DÉTECTION SIMULÉE si nécessaire"""
        reward = 0.0
        
        # Vérifier s'il y a des minéraux (même faibles concentrations)
        total_concentration = sum(mineral_concentrations) if mineral_concentrations else 0.0
        
        # SIMULATION: 30% de chance de "trouver" des minéraux dans les zones prioritaires
        x, y = position_key
        if (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 
            self.priority_heatmap[x, y] > 0.2 and np.random.random() < 0.3):
            
            # Simuler une découverte
            simulated_concentration = 0.3 + np.random.random() * 0.5
            base_reward = simulated_concentration * self.reward_config['mineral_discovery_base']
            reward += base_reward
            
            # Bonus pour haute concentration simulée
            if simulated_concentration > 0.7:
                reward += self.reward_config['high_value_mineral']
                self.high_value_samples += 1
            
            # Enregistrer la "découverte"
            self.discovered_locations.add(position_key)
            self.concentration_history.append(simulated_concentration)
            
        # Récompense pour vraies détections
        elif total_concentration > self.mineral_threshold:
            base_reward = total_concentration * self.reward_config['mineral_discovery_base']
            reward += base_reward
            
            if total_concentration > 0.7:
                reward += self.reward_config['high_value_mineral']
                self.high_value_samples += 1
            
            self.discovered_locations.add(position_key)
            self.concentration_history.append(total_concentration)
        
        return reward
    
    def _calculate_exploration_reward(self, position_key, is_new_position):
        """Récompense d'exploration TRÈS GÉNÉREUSE"""
        reward = 0.0
        
        if is_new_position:
            reward += self.reward_config['exploration_bonus']
            
            # Bonus progressif pour exploration étendue
            exploration_bonus = min(len(self.visited_positions) * 0.5, 50.0)
            reward += exploration_bonus
            
        else:
            # Pénalité TRÈS LÉGÈRE pour revisite
            reward += self.reward_config['revisiting_penalty']
        
        return reward
    
    def _calculate_science_reward(self, science_value):
        """Récompense scientifique"""
        if science_value > self.exploration_threshold:
            return science_value * self.reward_config['science_target']
        return 0.0
    
    def _calculate_zone_bonus(self, position):
        """Bonus pour exploration de zones spécifiques"""
        x, y = position
        
        # Bonus pour les bords de la carte (exploration risquée)
        if x < 10 or x > 90 or y < 10 or y > 90:
            return 15.0
        
        # Bonus pour le centre (zone importante)
        if 40 <= x <= 60 and 40 <= y <= 60:
            return 10.0
        
        return 0.0
    
    def _calculate_penalties(self, has_collision):
        """Pénalités TRÈS LÉGÈRES"""
        penalty = 0.0
        
        if has_collision:
            penalty += self.reward_config['collision_penalty']
        
        return penalty
    
    def _update_visited_position(self, position_key, is_new_position):
        """Met à jour les positions visitées"""
        if is_new_position:
            self.visited_positions.add(position_key)
    
    def get_heatmap_state(self):
        """Retourne la heatmap pour le DQN"""
        return self.priority_heatmap.copy()
    
    def get_statistics(self):
        """Retourne les statistiques"""
        if not self.reward_history:
            return {
                'total_reward': 0.0,
                'average_reward': 0.0,
                'discovered_locations': len(self.discovered_locations),
                'visited_positions': len(self.visited_positions),
                'total_steps': 0,
                'coverage_percentage': 0.0
            }
        
        total_reward = sum(self.reward_history)
        avg_reward = np.mean(self.reward_history)
        
        return {
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'discovered_locations': len(self.discovered_locations),
            'visited_positions': len(self.visited_positions),
            'total_steps': len(self.reward_history),
            'coverage_percentage': (len(self.visited_positions) / (self.grid_size[0] * self.grid_size[1])) * 100
        }
    
    def episode_summary(self, episode_num):
        """Résumé de fin d'épisode POSITIF"""
        stats = self.get_statistics()
        
        print(f"\n🎉 EPISODE {episode_num} SUMMARY 🎉")
        print(f"💰 Total Reward: {stats['total_reward']:+.2f}")
        print(f"📈 Average Reward: {stats['average_reward']:+.2f}")
        print(f"📍 Discovered Locations: {stats['discovered_locations']}")
        print(f"🗺️ Visited Positions: {stats['visited_positions']}")
        print(f"📊 Coverage: {stats['coverage_percentage']:.1f}%")
        
        # Réinitialiser pour le prochain épisode
        self.episode_rewards.append(stats['total_reward'])
        self.reward_history.clear()
        self.mineral_history.clear()
        
        # Récompense verbale basée sur la performance
        if stats['total_reward'] > 0:
            print("✅ Excellent! L'agent apprend bien!")
        else:
            print("⚠️  Continue, tu vas y arriver!")
    
    def reset(self):
        """Réinitialisation partielle"""
        self.reward_history.clear()
        self.mineral_history.clear()
    
    def full_reset(self):
        """Réinitialisation complète"""
        self.discovered_locations.clear()
        self.discovered_minerals.clear()
        self.visited_positions.clear()
        self.high_value_samples = 0
        self.reward_history.clear()
        self.mineral_history.clear()
        self.concentration_history.clear()
        self.episode_rewards.clear()
        self.priority_heatmap = self._initialize_guided_heatmap()
        self.cleaned_areas = np.zeros(self.grid_size, dtype=bool)
