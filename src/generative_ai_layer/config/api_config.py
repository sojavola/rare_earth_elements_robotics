#!/usr/bin/env python3

import os
import yaml
from pathlib import Path

class APIConfig:
    """Gestionnaire de configuration pour l'API Mistral"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self):
        """Retourne le chemin par défaut du fichier de configuration"""
        # Essayer plusieurs emplacements
        possible_paths = [
            Path.home() / '.ree_exploration' / 'mistral_config.yaml',
            Path('/etc/ree_exploration/mistral_config.yaml'),
            Path(__file__).parent / 'mistral_config.yaml',
            Path('config/mistral_config.yaml')
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Créer un fichier de configuration par défaut
        default_path = Path.home() / '.ree_exploration' / 'mistral_config.yaml'
        default_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            'mistral': {
                'api_key': 'YOUR_API_KEY_HERE',
                'model': 'mistral-large-latest',
                'temperature': 0.7,
                'max_tokens': 2000
            }
        }
        
        with open(default_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"📁 Fichier de configuration créé: {default_path}")
        print("🔑 Veuillez y ajouter votre clé API Mistral")
        
        return default_path
    
    def _load_config(self):
        """Charge la configuration depuis le fichier YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Erreur chargement configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Retourne une configuration par défaut"""
        return {
            'mistral': {
                'api_key': os.getenv('MISTRAL_API_KEY', ''),
                'model': 'mistral-large-latest',
                'temperature': 0.7,
                'max_tokens': 2000
            }
        }
    
    def get_api_key(self):
        """Récupère la clé API (priorité: variable env > fichier config)"""
        # Priorité 1: Variable d'environnement
        env_key = os.getenv('MISTRAL_API_KEY')
        if env_key and env_key != 'YOUR_API_KEY_HERE':
            return env_key
        
        # Priorité 2: Fichier de configuration
        config_key = self.config.get('mistral', {}).get('api_key', '')
        if config_key and config_key != 'YOUR_API_KEY_HERE':
            return config_key
        
        return None
    
    def get_model_config(self):
        """Récupère la configuration du modèle"""
        return self.config.get('mistral', {})
    
    def update_api_key(self, new_api_key):
        """Met à jour la clé API dans le fichier de configuration"""
        if 'mistral' not in self.config:
            self.config['mistral'] = {}
        
        self.config['mistral']['api_key'] = new_api_key
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"✅ Clé API mise à jour dans: {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur mise à jour clé API: {e}")
            return False

# Instance globale de configuration
config_manager = APIConfig()