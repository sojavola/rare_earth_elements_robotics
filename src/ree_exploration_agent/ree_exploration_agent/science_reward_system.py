import numpy as np
import math
from collections import defaultdict
import time
from scipy.ndimage import gaussian_filter


class RealMineralRewardSystem:
    """
    Système de récompenses RÉEL basé sur la logique académique
    avec heatmap, diffusion gaussienne et équation de Bellman
    """
    
    def __init__(self, grid_size=(100, 100), robot_id=0):
        self.grid_size = grid_size
        self.robot_id = robot_id
        
        # === CONFIGURATION ACADÉMIQUE OPTIMISÉE ===
        self.reward_config = {
            # === PARAMÈTRES ACADÉMIQUES (comme dans l'article) ===
            'academic_penalty': -2.0,           # Pénalité standard académique
            'clean_threshold': 0.1,             # Seuil de propreté
            'gaussian_sigma': 0.9,              # σ pour diffusion gaussienne
            'gaussian_update_freq': 15,         # ψ = 15 pas de temps
            'discount_factor': 0.99,            # γ pour Bellman
            
            # === RÉCOMPENSES MINÉRALES (adaptées) ===
            'mineral_base_reward': 100.0,       # Base par minéral détecté
            'concentration_multiplier': 50.0,   # Multiplicateur académique
            'high_concentration_bonus': 50.0,   # Bonus pour > 0.7
            
            # === EXPLORATION ACADÉMIQUE ===
            'exploration_bonus': 1.0,           # Très faible comme dans l'article
            'new_zone_bonus': 2.0,              # Pour nouvelles zones
            
            # === PÉNALITÉS ACADÉMIQUES ===
            'step_penalty': -0.05,              # Pénalité par pas (légère)
            'collision_penalty': -5.0,          # Pour collisions
            'revisiting_penalty': -0.5,         # Pour zones revisitées
            
            # === BONUS STRATÉGIQUES ===
            'coverage_bonus': 0.02,             # Par % de carte exploré
            'efficiency_bonus': 0.3,            # Pour minéraux/steps
        }
        
        # === HEATMAP ACADÉMIQUE (simulée) ===
        self.academic_heatmap = self._initialize_academic_heatmap()
        
        # === SUIVI DES MINÉRAUX RÉELS ===
        self.minerals_collected = 0
        self.steps_without_mineral = 0
        self.last_mineral_position = None
        self.mineral_positions = []
        self.concentration_history = []
        
        # === SUIVI EXPLORATION ===
        self.visited_positions = set()
        self.cleaned_positions = set()  # Positions "nettoyées" académiquement
        self.unique_positions_count = 0
        
        # === COMPTEURS GAUSSIENS ===
        self.gaussian_step = 0
        self.total_steps = 0
        
        # === STATISTIQUES ===
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        self.episode_rewards = []
        
        print(f"🤖 Robot {robot_id}: Système de récompenses ACADÉMIQUE initialisé")
        print(f"   - Heatmap: {grid_size[0]}x{grid_size[1]}")
        print(f"   - Diffusion gaussienne: σ={self.reward_config['gaussian_sigma']}, ψ={self.reward_config['gaussian_update_freq']}")
        print(f"   - Équation Bellman: γ={self.reward_config['discount_factor']}")
    
    def _initialize_academic_heatmap(self):
        """Initialise la heatmap académique avec bruit réaliste"""
        height, width = self.grid_size
        heatmap = np.random.rand(height, width)
        
        # Appliquer un filtre gaussien pour créer des clusters
        heatmap = gaussian_filter(heatmap, sigma=3.0)
        
        # Normaliser entre 0 et 1
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        # Ajouter quelques zones très prioritaires
        for _ in range(10):
            x = np.random.randint(10, width - 10)
            y = np.random.randint(10, height - 10)
            radius = np.random.randint(5, 15)
            
            for i in range(max(0, y - radius), min(height, y + radius)):
                for j in range(max(0, x - radius), min(width, x + radius)):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance < radius:
                        concentration = 0.8 * (1.0 - distance / radius)
                        heatmap[i, j] = max(heatmap[i, j], concentration)
        
        return np.clip(heatmap, 0, 1)
    
    def calculate_reward(self, mineral_concentrations, position, 
                        is_new_position=False, has_collision=False,
                        step_count=0, sensor_data=None):
        """
        Calcule la récompense ACADÉMIQUE hybride
        Combine: heatmap simulée + données minérales réelles
        """
        x, y = position
        position_key = (int(x), int(y))
        self.total_steps += 1
        
        # === 1. RÉCOMPENSE ACADÉMIQUE (heatmap) ===
        academic_reward = self._calculate_academic_reward(position_key, has_collision)
        
        # === 2. RÉCOMPENSE RÉELLE (minéraux) ===
        real_reward = self._calculate_real_mineral_reward(mineral_concentrations, position_key)
        
        # === 3. COMBINAISON HYBRIDE (70% réel, 30% académique) ===
        hybrid_reward = 0.7 * real_reward + 0.3 * academic_reward
        
        # === 4. BONUS STRATÉGIQUES ===
        strategic_bonus = self._calculate_strategic_bonus(position_key, step_count)
        hybrid_reward += strategic_bonus
        
        # === 5. DIFFUSION GAUSSIENNE (simulation académique) ===
        self._update_gaussian_diffusion()
        
        # === 6. MISE À JOUR ===
        self._update_tracking(position_key, is_new_position, real_reward > 0)
        self.total_reward += hybrid_reward
        self.academic_reward_total += academic_reward
        self.real_reward_total += real_reward
        
        # Validation
        if np.isnan(hybrid_reward) or np.isinf(hybrid_reward):
            print(f"⚠️  Reward invalide: {hybrid_reward}, correction à 0.0")
            return 0.0
        
        return hybrid_reward
    
    def _calculate_academic_reward(self, position_key, has_collision):
        """
        Calcule la récompense académique: cpi = Σ(s(j,l) × xi(j,l))
        Selon la logique: ri = { cpi si cpi > 0, penalty sinon }
        """
        x, y = position_key
        
        # Vérifier les limites
        if not (0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]):
            return self.reward_config['academic_penalty']
        
        # Pénalité de collision
        if has_collision:
            return self.reward_config['collision_penalty']
        
        # Vérifier si position déjà nettoyée
        if position_key in self.cleaned_positions:
            return self.reward_config['revisiting_penalty']
        
        # Obtenir la valeur de la heatmap: s(j,l)
        heatmap_value = self.academic_heatmap[y, x]
        
        # Seuil de propreté académique
        if heatmap_value < self.reward_config['clean_threshold']:
            # Zone déjà "propre" → pénalité académique
            return self.reward_config['academic_penalty']
        
        # RÉCOMPENSE ACADÉMIQUE: cpi = s(j,l) × xi(j,l)
        # où xi(j,l) = 1 car le robot "nettoie" cette cellule
        academic_reward = heatmap_value * self.reward_config['mineral_base_reward']
        
        # "Nettoyer" la cellule dans la heatmap (simuler l'effet de nettoyage)
        reduction_factor = 0.1  # Réduire de 90%
        self.academic_heatmap[y, x] *= reduction_factor
        
        # Marquer comme nettoyée
        self.cleaned_positions.add(position_key)
        
        return academic_reward
    
    def _calculate_real_mineral_reward(self, concentrations, position_key):
        """Calcule la récompense pour minéraux réels"""
        if not concentrations:
            return 0.0
        
        # Concentration maximale détectée
        max_concentration = max(concentrations) if concentrations else 0.0
        detection_threshold = 0.1
        
        if max_concentration < detection_threshold:
            self.steps_without_mineral += 1
            return 0.0
        
        # MINÉRAL DÉTECTÉ !
        mineral_reward = 0.0
        self.steps_without_mineral = 0
        
        # Récompense de base proportionnelle à la concentration
        base_reward = max_concentration * self.reward_config['mineral_base_reward']
        mineral_reward += base_reward
        
        # Bonus concentration (académique)
        concentration_bonus = max_concentration * self.reward_config['concentration_multiplier']
        mineral_reward += concentration_bonus
        
        # Bonus haute concentration
        if max_concentration > 0.7:
            mineral_reward += self.reward_config['high_concentration_bonus']
        
        # Mettre à jour les statistiques
        self.minerals_collected += 1
        self.mineral_positions.append(position_key)
        self.concentration_history.append(max_concentration)
        self.last_mineral_position = position_key
        
        # Log des découvertes importantes
        if max_concentration > 0.5:
            print(f"🎯 Robot {self.robot_id}: Minéral détecté à {position_key}")
            print(f"   Concentration: {max_concentration:.3f}, Reward: {mineral_reward:.1f}")
        
        return mineral_reward
    
    def _calculate_strategic_bonus(self, position_key, step_count):
        """Bonus stratégiques pour comportements intelligents"""
        bonus = 0.0
        
        # Bonus exploration de nouvelles zones
        if position_key not in self.visited_positions:
            bonus += self.reward_config['exploration_bonus']
            
            # Bonus supplémentaire pour exploration rapide
            if step_count < 50:  # Début d'épisode
                bonus += self.reward_config['new_zone_bonus']
        
        # Bonus de couverture (académique)
        coverage = len(self.visited_positions) / (self.grid_size[0] * self.grid_size[1])
        bonus += coverage * self.reward_config['coverage_bonus'] * 1000
        
        # Bonus d'efficacité (minéraux par step)
        if step_count > 10 and self.minerals_collected > 0:
            efficiency = self.minerals_collected / max(step_count, 1)
            bonus += efficiency * self.reward_config['efficiency_bonus'] * 100
        
        # Pénalité par step (encourage l'efficacité)
        bonus += self.reward_config['step_penalty']
        
        return bonus
    
    def _update_gaussian_diffusion(self):
        """Applique la diffusion gaussienne académique"""
        self.gaussian_step += 1
        
        if self.gaussian_step % self.reward_config['gaussian_update_freq'] == 0:
            # Appliquer le filtre gaussien: N(μ, σ²) avec μ=0, σ=0.9
            sigma = self.reward_config['gaussian_sigma']
            self.academic_heatmap = gaussian_filter(self.academic_heatmap, sigma=sigma)
            
            # Régénérer légèrement les zones nettoyées
            regeneration_rate = 0.05
            mask = np.random.rand(*self.academic_heatmap.shape) < regeneration_rate
            self.academic_heatmap[mask] = np.minimum(1.0, self.academic_heatmap[mask] + 0.1)
            
            # Effacer certaines positions nettoyées (regénération)
            positions_to_remove = []
            for pos in self.cleaned_positions:
                x, y = pos
                if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
                    if np.random.random() < 0.1:  # 10% de chance de regénération
                        positions_to_remove.append(pos)
                        self.academic_heatmap[y, x] = np.random.random() * 0.5
            
            for pos in positions_to_remove:
                self.cleaned_positions.remove(pos)
            
            print(f"🔄 Robot {self.robot_id}: Diffusion gaussienne appliquée")
    
    def _update_tracking(self, position_key, is_new_position, found_mineral):
        """Met à jour le suivi"""
        if is_new_position and position_key not in self.visited_positions:
            self.visited_positions.add(position_key)
            self.unique_positions_count += 1
    
    def get_statistics(self):
        """Retourne les statistiques complètes académiques"""
        if self.minerals_collected > 0:
            avg_concentration = np.mean(self.concentration_history)
        else:
            avg_concentration = 0.0
        
        # Couverture exploration
        coverage = (len(self.visited_positions) / 
                   (self.grid_size[0] * self.grid_size[1]) * 100)
        
        # Métrique académique: c_perc = ((xtot - scurr) / xtot) × 100
        total_cells = self.grid_size[0] * self.grid_size[1]
        current_priority_sum = np.sum(self.academic_heatmap)
        cleanliness_percentage = 100 * (1.0 - current_priority_sum / total_cells)
        
        return {
            'robot_id': self.robot_id,
            'minerals_collected': self.minerals_collected,
            'total_reward': self.total_reward,
            'academic_reward': self.academic_reward_total,
            'real_reward': self.real_reward_total,
            'visited_positions': len(self.visited_positions),
            'coverage_percentage': coverage,
            'cleanliness_percentage': cleanliness_percentage,
            'avg_mineral_concentration': avg_concentration,
            'steps_without_mineral': self.steps_without_mineral,
            'gaussian_updates': self.gaussian_step // self.reward_config['gaussian_update_freq'],
            'heatmap_mean': np.mean(self.academic_heatmap),
            'heatmap_std': np.std(self.academic_heatmap),
            'bellman_discount': self.reward_config['discount_factor']
        }
    
    def print_detailed_report(self):
        """Affiche un rapport détaillé académique"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"📊 RAPPORT ACADÉMIQUE - Robot {self.robot_id}")
        print(f"{'='*60}")
        
        print(f"🎯 PERFORMANCE:")
        print(f"   Minéraux collectés: {stats['minerals_collected']}")
        print(f"   Récompense totale: {stats['total_reward']:.1f}")
        print(f"     → Académique: {stats['academic_reward']:.1f}")
        print(f"     → Réelle: {stats['real_reward']:.1f}")
        
        print(f"\n🗺️  EXPLORATION:")
        print(f"   Positions visitées: {stats['visited_positions']}")
        print(f"   Couverture: {stats['coverage_percentage']:.1f}%")
        print(f"   Propreté (c_perc): {stats['cleanliness_percentage']:.1f}%")
        
        print(f"\n⚙️  SYSTÈME:")
        print(f"   Updates gaussiens: {stats['gaussian_updates']}")
        print(f"   Heatmap: μ={stats['heatmap_mean']:.3f}, σ={stats['heatmap_std']:.3f}")
        print(f"   Bellman: γ={stats['bellman_discount']}")
        
        if stats['minerals_collected'] > 0:
            print(f"\n💎 MINÉRAUX:")
            print(f"   Concentration moyenne: {stats['avg_mineral_concentration']:.3f}")
            print(f"   Derniers minéraux: {self.mineral_positions[-3:]}")
        
        print(f"{'='*60}")
    
    def reset_episode(self):
        """Réinitialise pour un nouvel épisode (conserve l'apprentissage)"""
        # Conserver l'apprentissage
        minerals_before = self.minerals_collected
        
        # Sauvegarder les récompenses
        self.episode_rewards.append(self.total_reward)
        
        # Réinitialiser les compteurs d'épisode
        self.total_reward = 0.0
        self.academic_reward_total = 0.0
        self.real_reward_total = 0.0
        self.visited_positions.clear()
        self.cleaned_positions.clear()
        self.unique_positions_count = 0
        self.steps_without_mineral = 0
        self.total_steps = 0
        self.gaussian_step = 0
        
        # Régénérer partiellement la heatmap
        regeneration = np.random.rand(*self.academic_heatmap.shape) * 0.3
        self.academic_heatmap = np.minimum(1.0, self.academic_heatmap + regeneration)
        
        print(f"\n🔄 Robot {self.robot_id}: Nouvel épisode académique")
        print(f"   Minéraux collectés (total): {minerals_before}")
        print(f"   Heatmap régénérée")
    
    def get_reward_breakdown(self, position, concentrations):
        """Retourne la décomposition détaillée des récompenses"""
        x, y = position
        position_key = (int(x), int(y))
        
        heatmap_value = 0.0
        if 0 <= y < self.grid_size[0] and 0 <= x < self.grid_size[1]:
            heatmap_value = self.academic_heatmap[y, x]
        
        max_concentration = max(concentrations) if concentrations else 0.0
        
        return {
            'position': position_key,
            'heatmap_value': heatmap_value,
            'max_concentration': max_concentration,
            'academic_potential': heatmap_value * self.reward_config['mineral_base_reward'],
            'real_potential': max_concentration * self.reward_config['mineral_base_reward'],
            'is_cleaned': position_key in self.cleaned_positions,
            'is_visited': position_key in self.visited_positions,
            'gaussian_updates': self.gaussian_step // self.reward_config['gaussian_update_freq']
        }


# ============================================================================
# 🎯 INTÉGRATION SIMPLIFIÉE
# ============================================================================

if __name__ == "__main__":
    # Test du système académique
    print("🧪 TEST DU SYSTÈME ACADÉMIQUE")
    print("="*50)
    
    # Créer le système
    reward_system = RealMineralRewardSystem(grid_size=(50, 50), robot_id=0)
    
    # Simuler quelques steps
    for step in range(100):
        # Données simulées
        if step % 20 == 0:
            concentrations = [0.6]  # Minéral occasionnel
        else:
            concentrations = [0.05]  # Pas de minéral
        
        position = (np.random.randint(0, 50), np.random.randint(0, 50))
        is_new = np.random.random() > 0.5
        has_collision = np.random.random() > 0.9
        
        # Calculer la récompense
        reward = reward_system.calculate_reward(
            mineral_concentrations=concentrations,
            position=position,
            is_new_position=is_new,
            has_collision=has_collision,
            step_count=step
        )
        
        if step % 25 == 0:
            print(f"Step {step}: position={position}, reward={reward:.1f}")
    
    # Fin de l'épisode
    reward_system.print_detailed_report()
    
    # Afficher les statistiques
    stats = reward_system.get_statistics()
    print(f"\n✅ TEST TERMINÉ:")
    print(f"   Minéraux: {stats['minerals_collected']}")
    print(f"   Couverture: {stats['coverage_percentage']:.1f}%")
    print(f"   Propreté: {stats['cleanliness_percentage']:.1f}%")