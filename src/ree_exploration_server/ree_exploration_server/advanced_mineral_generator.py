import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import cv2
from sklearn.cluster import DBSCAN

class AdvancedMineralGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Types de minéraux de terres rares avec propriétés géologiques
        self.mineral_types = {
            'REE_Oxides': {
                'color': [1.0, 0.0, 0.0],  # Rouge
                'density': 0.05,
                'cluster_tendency': 0.8,
                'depth_bias': 0.3
            },
            'REE_Silicates': {
                'color': [0.0, 1.0, 0.0],  # Vert
                'density': 0.07,
                'cluster_tendency': 0.6,
                'depth_bias': 0.5
            },
            'REE_Phosphates': {
                'color': [0.0, 0.0, 1.0],  # Bleu
                'density': 0.04,
                'cluster_tendency': 0.9,
                'depth_bias': 0.2
            },
            'REE_Carbonates': {
                'color': [1.0, 1.0, 0.0],  # Jaune
                'density': 0.06,
                'cluster_tendency': 0.7,
                'depth_bias': 0.4
            }
        }
    
    def generate_geological_map(self):
        """Génère une carte géologique réaliste avec veines minérales"""
        mineral_map = np.zeros((self.height, self.width, len(self.mineral_types)))
        
        # Générer une carte de base avec bruit géologique
        base_geology = self._generate_geological_base()
        
        # Générer des veines pour chaque type minéral
        for mineral_idx, (mineral_name, properties) in enumerate(self.mineral_types.items()):
            mineral_layer = self._generate_mineral_veins(base_geology, properties)
            mineral_map[:, :, mineral_idx] = mineral_layer
        
        # Appliquer des transformations géologiques
        mineral_map = self._apply_geological_processes(mineral_map)
        
        return mineral_map
    
    def _generate_geological_base(self):
        """Génère une base géologique réaliste"""
        # Combinaison de bruits multi-échelles pour simuler la géologie
        base = np.zeros((self.height, self.width))
        
        # Bruit à grande échelle (formations géologiques majeures)
        large_scale = self._generate_perlin_noise(scale=50.0, octaves=1)
        base += large_scale * 0.3
        
        # Bruit à moyenne échelle (veines)
        medium_scale = self._generate_perlin_noise(scale=20.0, octaves=2)
        base += medium_scale * 0.4
        
        # Bruit à petite échelle (texture)
        small_scale = self._generate_perlin_noise(scale=5.0, octaves=4)
        base += small_scale * 0.3
        
        return (base - np.min(base)) / (np.max(base) - np.min(base))
    
    def _generate_mineral_veins(self, base_geology, mineral_properties):
        """Génère des veines minérales réalistes"""
        # Seuillage adaptatif basé sur la géologie de base
        threshold = 0.5 + (mineral_properties['density'] * 0.3)
        
        # Créer des veines à partir du bruit de base
        veins = np.zeros_like(base_geology)
        vein_mask = base_geology > threshold
        
        # Amplifier les zones de veines
        veins[vein_mask] = (base_geology[vein_mask] - threshold) / (1 - threshold)
        
        # Appliquer la tendance au clustering
        if mineral_properties['cluster_tendency'] > 0.5:
            veins = gaussian_filter(veins, sigma=2.0 * mineral_properties['cluster_tendency'])
        
        # Seuillage final
        veins = np.clip(veins * (1.0 + mineral_properties['density'] * 2), 0, 1)
        
        return veins
    
    def _generate_perlin_noise(self, scale=10.0, octaves=4, persistence=0.5, lacunarity=2.0):
        """Génère du bruit de Perlin amélioré"""
        # Implémentation simplifiée du bruit de Perlin
        base = np.random.rand(self.height, self.width)
        result = np.zeros_like(base)
        
        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave
            
            # Redimensionner et ajouter l'octave
            octave_noise = cv2.resize(
                np.random.rand(
                    int(self.height / (scale / frequency)) + 1,
                    int(self.width / (scale / frequency)) + 1
                ),
                (self.width, self.height)
            )
            
            result += octave_noise * amplitude
        
        return (result - np.min(result)) / (np.max(result) - np.min(result))
    
    def _apply_geological_processes(self, mineral_map):
        """Applique des processus géologiques réalistes"""
        processed_map = mineral_map.copy()
        
        # Érosion simulée
        for i in range(mineral_map.shape[2]):
            layer = mineral_map[:, :, i]
            
            # Gradient pour simuler l'érosion
            gradient_x = sobel(layer, axis=1)
            gradient_y = sobel(layer, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Réduire les concentrations sur les pentes fortes
            erosion_factor = 1.0 - (gradient_magnitude * 0.3)
            processed_map[:, :, i] = layer * np.clip(erosion_factor, 0.3, 1.0)
        
        # Dépôts sédimentaires simulés
        sediment_map = self._generate_sedimentary_deposits()
        for i in range(mineral_map.shape[2]):
            processed_map[:, :, i] = np.maximum(processed_map[:, :, i], sediment_map * 0.2)
        
        return np.clip(processed_map, 0, 1)
    
    def _generate_sedimentary_deposits(self):
        """Génère des dépôts sédimentaires"""
        # Simuler l'accumulation dans les dépressions
        topography = self._generate_perlin_noise(scale=30.0, octaves=2)
        
        # Inverser pour avoir les dépressions en haut
        depressions = 1.0 - topography
        
        # Amplifier les zones basses
        sedimentary_map = np.power(depressions, 2)
        
        return sedimentary_map
    
    def generate_underground_layers(self, surface_mineral_map, num_layers=3):
        """Génère des couches souterraines avec gradients de concentration"""
        underground_layers = []
        
        for depth in range(num_layers):
            depth_factor = (depth + 1) / num_layers
            
            # Modifier la carte de surface pour chaque profondeur
            layer_map = surface_mineral_map.copy()
            
            for mineral_idx, (mineral_name, properties) in enumerate(self.mineral_types.items()):
                depth_bias = properties['depth_bias']
                
                # Ajuster la concentration selon la profondeur et le biais
                if depth_bias < 0.5:
                    # Minéraux de surface - diminuer avec la profondeur
                    reduction = (1.0 - depth_bias) * depth_factor
                    layer_map[:, :, mineral_idx] *= (1.0 - reduction)
                else:
                    # Minéraux profonds - augmenter avec la profondeur
                    enhancement = (depth_bias - 0.5) * depth_factor * 2
                    layer_map[:, :, mineral_idx] *= (1.0 + enhancement)
            
            underground_layers.append(np.clip(layer_map, 0, 1))
        
        return underground_layers
    
    def detect_mineral_clusters(self, mineral_map, mineral_idx, min_samples=5, eps=0.1):
        """Détecte les clusters minéraux using DBSCAN"""
        mineral_layer = mineral_map[:, :, mineral_idx]
        
        # Seuiller pour obtenir les points d'intérêt
        threshold = 0.3
        points = np.argwhere(mineral_layer > threshold)
        
        if len(points) < min_samples:
            return []
        
        # Appliquer DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        
        clusters = []
        for label in set(clustering.labels_):
            if label != -1:  # Ignorer le bruit
                cluster_points = points[clustering.labels_ == label]
                clusters.append({
                    'points': cluster_points,
                    'center': np.mean(cluster_points, axis=0),
                    'size': len(cluster_points),
                    'max_concentration': np.max(mineral_layer[cluster_points[:, 0], cluster_points[:, 1]])
                })
        
        return clusters