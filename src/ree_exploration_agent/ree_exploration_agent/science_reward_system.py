import numpy as np
from collections import defaultdict

class ScienceRewardSystem:
    def __init__(self):
        self.mineral_discovery_bonus = 50.0
        self.high_value_sample_bonus = 30.0
        self.new_region_bonus = 20.0
        self.collision_penalty = -10.0
        self.revisiting_penalty = -5.0
        self.step_penalty = -0.1
        self.exploration_bonus = 2.0
        
        # Suivi des découvertes
        self.discovered_minerals = set()
        self.visited_positions = set()
        self.visited_regions = set()
        self.high_value_samples = 0
        
        # Historique des récompenses
        self.reward_history = []
        
    def calculate_reward(self, current_state, action, next_state, done):
        """Calcule la récompense scientifique complète"""
        reward = 0.0
        
        # Position actuelle
        x, y = next_state['position']
        region = self._get_region(x, y)
        
        # Récompense de découverte minérale
        mineral_reward = self._calculate_mineral_reward(next_state)
        reward += mineral_reward
        
        # Récompense d'exploration
        exploration_reward = self._calculate_exploration_reward(x, y, region)
        reward += exploration_reward
        
        # Récompense pour échantillons de haute valeur
        high_value_reward = self._calculate_high_value_reward(next_state)
        reward += high_value_reward
        
        # Pénalités
        penalty = self._calculate_penalties(x, y, action, next_state)
        reward += penalty
        
        # Pénalité d'étape
        reward += self.step_penalty
        
        # Enregistrer la récompense
        self.reward_history.append(reward)
        
        return reward
    
    def _calculate_mineral_reward(self, state):
        """Calcule la récompense pour la détection minérale"""
        reward = 0.0
        mineral_concentrations = state['mineral_concentrations']
        
        for mineral_idx, concentration in enumerate(mineral_concentrations):
            if concentration > 0.1:  # Seuil de détection
                mineral_name = state['mineral_types'][mineral_idx]
                
                # Récompense de base proportionnelle à la concentration
                reward += concentration * 10.0
                
                # Bonus pour nouvelle découverte
                if mineral_name not in self.discovered_minerals:
                    reward += self.mineral_discovery_bonus
                    self.discovered_minerals.add(mineral_name)
        
        return reward
    
    def _calculate_exploration_reward(self, x, y, region):
        """Calcule la récompense pour l'exploration de nouvelles zones"""
        reward = 0.0
        position = (x, y)
        
        # Bonus pour nouvelle position
        if position not in self.visited_positions:
            reward += self.exploration_bonus
            self.visited_positions.add(position)
        
        # Bonus pour nouvelle région
        if region not in self.visited_regions:
            reward += self.new_region_bonus
            self.visited_regions.add(region)
        
        return reward
    
    def _calculate_high_value_reward(self, state):
        """Calcule la récompense pour les échantillons de haute valeur"""
        reward = 0.0
        mineral_concentrations = state['mineral_concentrations']
        
        for concentration in mineral_concentrations:
            if concentration > 0.9:  # Échantillon de très haute valeur
                reward += self.high_value_sample_bonus
                self.high_value_samples += 1
        
        return reward
    
    def _calculate_penalties(self, x, y, action, state):
        """Calcule les pénalités"""
        penalty = 0.0
        position = (x, y)
        
        # Pénalité pour revisite
        if position in self.visited_positions:
            penalty += self.revisiting_penalty
        
        # Pénalité pour collision (si détectée)
        if state.get('collision', False):
            penalty += self.collision_penalty
        
        return penalty
    
    def _get_region(self, x, y, region_size=10):
        """Détermine la région basée sur la position"""
        region_x = x // region_size
        region_y = y // region_size
        return (region_x, region_y)
    
    def get_science_stats(self):
        """Retourne les statistiques scientifiques"""
        return {
            'discovered_minerals': len(self.discovered_minerals),
            'visited_positions': len(self.visited_positions),
            'visited_regions': len(self.visited_regions),
            'high_value_samples': self.high_value_samples,
            'total_reward': sum(self.reward_history),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0
        }
    
    def reset(self):
        """Réinitialise le système de récompenses"""
        self.discovered_minerals.clear()
        self.visited_positions.clear()
        self.visited_regions.clear()
        self.high_value_samples = 0
        self.reward_history.clear()