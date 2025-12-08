import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import glob
import threading



class SharedDQNManager:
    """Manager central pour le modèle DQN partagé entre tous les robots"""
    
    _instance = None
    _lock = threading.Lock()
     
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._init_attributes()
        return cls._instance
    
    def _init_attributes(self):
        """Initialise les attributs (appelé une seule fois)"""
        if not hasattr(self, 'training_lock'):
            self.training_lock = threading.Lock()
            self.shared_policy_net = None
            self.shared_target_net = None
            self.shared_optimizer = None
            self.global_step_count = 0
            self.global_episode_count = 0
            self.robot_ids = set()
            print(f"✅ [GLOBAL] Attributs initialisés")
    
    def initialize(self, state_shape, num_actions, device='cpu'):
        """Initialise le modèle partagé (à appeler une seule fois)"""

        with self._lock:  # Verrou pour l'initialisation
            if not self._initialized:
                print(f"🧠 [GLOBAL] Initialisation du modèle partagé...")

                self.device = torch.device(device)
                self.state_shape = state_shape
                self.num_actions = num_actions
                
                # ✅ UN SEUL MODÈLE POUR TOUS LES ROBOTS
                self.shared_policy_net = CNN_DQN(state_shape, num_actions).to(self.device)
                self.shared_target_net = CNN_DQN(state_shape, num_actions).to(self.device)
                self.shared_target_net.load_state_dict(self.shared_policy_net.state_dict())
                self.shared_target_net.eval()
                
                # Optimiseur partagé
                self.shared_optimizer = optim.Adam(
                    self.shared_policy_net.parameters(), 
                    lr=0.00025  # ✅ Comme dans le papier
                )

                self.global_step_count = 0
                self.global_episode_count = 0
                self.robot_ids = set()
                
                self.training_lock = threading.Lock()

                self._initialized = True
                print(f"🧠 [GLOBAL] Modèle DQN PARTAGÉ initialisé") 
                print(f"🔒 Verrou d'entraînement créé: {id(self.training_lock)}")

            else:
                print(f"ℹ️  [GLOBAL] Modèle déjà initialisé - {len(self.robot_ids)} robots enregistrés") 


    def get_shared_networks(self):
        """Retourne les réseaux partagés"""
        if not self._initialized:
            raise RuntimeError("SharedDQNManager non initialisé!")
        
        return self.shared_policy_net, self.shared_target_net, self.shared_optimizer

    def register_robot(self, robot_id):
        """Enregistre un robot dans le système partagé"""
        # ✅ VÉRIFIER SI INITIALISÉ
        if not self._initialized:
            print(f"⚠️  [GLOBAL] Manager non initialisé, initialisation automatique...")
            # Vous devrez peut-être passer les paramètres ici
        
        self.robot_ids.add(robot_id)
        print(f"🤖 [GLOBAL] Robot {robot_id} enregistré - Total: {len(self.robot_ids)} robots")


    def get_stats(self):
        """Retourne les statistiques globales"""
        return {
            'global_step_count': self.global_step_count,
            'global_episode_count': self.global_episode_count,
            'active_robots': len(self.robot_ids),
            'robot_ids': list(self.robot_ids)
        }

        
    
    def get_shared_networks(self):
        """Retourne les réseaux partagés"""
        return self.shared_policy_net, self.shared_target_net, self.shared_optimizer
    
    def get_training_lock(self):
        """Garantit que le verrou existe"""
        if not hasattr(self, 'training_lock') or self.training_lock is None:
            print(f"⚠️  [GLOBAL] Création d'urgence du verrou d'entraînement...")
            self.training_lock = threading.Lock()
            print(f"✅ [GLOBAL] Verrou d'urgence créé: {id(self.training_lock)}")
        return self.training_lock



class CNN_DQN(nn.Module):
    """DQN avec architecture CNN comme décrite dans le papier"""
    
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        self.input_shape = input_shape  # (height, width, channels)
        self.num_actions = num_actions
        
        # Couches convolutionnelles exactement comme dans le papier
        self.conv_layers = nn.Sequential(
            # Première couche: 32 filtres 8x8, stride 4
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Deuxième couche: 64 filtres 4x4, stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Troisième couche: 64 filtres 3x3, stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculer la taille des features après les conv layers
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Couches denses comme dans le papier
        self.dense_layers = nn.Sequential(
            # Flatten + Dense 512
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            
            # Couche de sortie: 8 neurones avec activation linéaire
            nn.Linear(512, num_actions)
            # Pas d'activation pour la sortie (linéaire comme dans le papier)
        )
        
    def _get_conv_output_size(self, shape):
        """Calcule automatiquement la taille après les couches convolutionnelles"""
        with torch.no_grad():
            # Créer un tensor dummy pour calculer la taille
            dummy_input = torch.zeros(1, shape[2], shape[0], shape[1])
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)
    
    def forward(self, state):
        try:
            mineral_map, position = state
            
            # Convertir en tensor si nécessaire
            if isinstance(mineral_map, np.ndarray):
                mineral_map = torch.FloatTensor(mineral_map)
            
            # Réorganiser les dimensions: (H, W, C) -> (C, H, W)
            if mineral_map.dim() == 3:
                mineral_map = mineral_map.permute(2, 0, 1)  # (C, H, W)
            elif mineral_map.dim() == 4:
                mineral_map = mineral_map.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Ajouter dimension batch si nécessaire
            if mineral_map.dim() == 3:
                mineral_map = mineral_map.unsqueeze(0)  # (1, C, H, W)
            
            # Passer à travers les couches convolutionnelles
            conv_output = self.conv_layers(mineral_map)
            
            # Passer à travers les couches denses
            q_values = self.dense_layers(conv_output)
            
            return q_values
            
        except Exception as e:
            print(f"❌ CNN Forward error: {e}")
            print(f"State shapes: mineral_map={mineral_map.shape if hasattr(mineral_map, 'shape') else 'N/A'}")
            return torch.zeros(1, self.num_actions)

# Configuration de l'optimiseur comme dans le papier
def create_optimizer(model, learning_rate=0.00025):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons une carte 84x84 avec 2 canaux comme dans le papier
    input_shape = (84, 84, 2)  # (height, width, channels)
    num_actions = 8
    
    # Créer le modèle
    model = CNN_DQN(input_shape, num_actions)
    optimizer = create_optimizer(model)
    
    print("✅ Architecture CNN créée avec succès!")
    print(f"📊 Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")


class RobustDQNAgent:
    """Agent DQN ultra-robuste avec gestion d'erreurs complète"""
    def __init__(self, 
                 state_shape, 
                 num_actions,
                 robot_id=0,
                 learning_rate=0.00025, 
                 gamma=0.99,
                 epsilon=1.0, 
                 epsilon_min=0.1, 
                 epsilon_decay=0.999, 
                 memory_size=5000,  # Réduit pour debug
                 batch_size=32, 
                 target_update=1000,
                 load_latest_model=True,# Réduit pour debug
                 use_shared_model=True
                 ):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Robot {robot_id} démarré sur device: {self.device}")

        import sys
        if len(sys.argv) > 1:
            try:
                robot_id = int(sys.argv[1])
                print(f"🎯 FORCÉ: Robot ID = {robot_id} (depuis ligne de commande)")
            except:
                pass
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.robot_id = robot_id  # ✅ MAINTENANT DÉFINI

        print(f"🤖 Robot {self.robot_id} démarré sur device: {self.device}")
        print(f"🎯 Robot ID final: {robot_id}")
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_shared_model = use_shared_model


        self.step_count = 0
        self.episode_count = 0
        self.local_step_count = 0

        self.training_lock = threading.Lock()
        self.global_step_count = 0
        

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Using device: {self.device}")
        
        # Réseaux
        if use_shared_model:
            # Obtenir les réseaux partagés
            self.shared_manager = SharedDQNManager.get_instance()
            self.shared_manager.initialize(state_shape, num_actions, self.device)
            self.shared_manager.register_robot(self.robot_id)
            self.policy_net, self.target_net, self.optimizer = self.shared_manager.get_shared_networks()
            print(f"🔗 Robot {self.robot_id} utilisant le modèle DQN PARTAGÉ")
            print(f"📊 ID Policy net: {id(self.policy_net)}")
            print(f"📊 ID Target net: {id(self.target_net)}")

        else:
            # Réseaux individuels (ancienne approche)
            self.policy_net = CNN_DQN(state_shape, num_actions).to(self.device)
            self.target_net = CNN_DQN(state_shape, num_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            print(f"🔧 Robot {self.robot_id} utilisant son propre modèle DQN")
        
        self.loss_fn = nn.MSELoss()
        
        # ✅ MEMOIRE EXPÉRIENCE INDIVIDUELLE (comme dans le papier)
        self.memory = deque(maxlen=memory_size)
        self.loss_history = []  

        # CONTINUOUS LEARNING: Charger le dernier modèle si disponible
        if load_latest_model:
            self.auto_load_latest_model()
        
        print(f"🧠 DQN initialized: state_shape={state_shape}, actions={num_actions}")
        print(f"📊 Continuous Learning: ε={self.epsilon:.3f}, step_count={self.step_count}")
    
    def find_model(self):
        """Trouve le modèle approprié (partagé ou individuel)"""
        try:
            if self.use_shared_model:
                # ✅ MODÈLE PARTAGÉ
                model_dir = "models/shared"
                preferred_file = "latest_shared_model.pth"
            else:
                # ✅ MODÈLE INDIVIDUEL (rétrocompatibilité)
                model_dir = f"models/robot_{self.robot_id}"
                preferred_file = "latest_model.pth"
            
            if not os.path.exists(model_dir):
                return None
            
            # Préférer le fichier latest
            preferred_path = f"{model_dir}/{preferred_file}"
            if os.path.exists(preferred_path):
                return preferred_path
            
            # Sinon chercher n'importe quel .pth
            model_files = glob.glob(f"{model_dir}/*.pth")
            if model_files:
                return max(model_files, key=os.path.getmtime)
            
            return None
        
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Erreur recherche modèle: {e}")
            return None

    def auto_load_latest_model(self):
        """Charge automatiquement le dernier modèle sauvegardé"""
        try:
            model_path = self.find_model()  # ✅ Utilise la nouvelle méthode
            
            if model_path is None:
                model_type = "partagé" if self.use_shared_model else "individuel"
                print(f"📁 Robot {self.robot_id}: Aucun modèle {model_type} trouvé - nouvel entraînement")
                return False
            
            # Charger le modèle
            self.load_model(model_path)
            
            # Ajuster epsilon (plus conservateur pour modèle partagé)
            if self.use_shared_model:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.8)  # Moins de réduction
            else:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.7)
            
            model_type = "PARTAGÉ" if self.use_shared_model else "individuel"
            print(f"✅ Robot {self.robot_id}: Modèle {model_type} chargé → {os.path.basename(model_path)}")
            print(f"🔄 Robot {self.robot_id}: ε={self.epsilon:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Erreur chargement: {e}")
            return False

    def choose_action(self, state):
        """Choisit une action de manière robuste"""
        if random.random() <= self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        try:
            with torch.no_grad():
                q_values = self.policy_net(state)
                
                # Vérifier les dimensions
                if q_values.dim() == 2 and q_values.size(0) == 1:
                    return q_values.squeeze(0).argmax().item()
                else:
                    return q_values.argmax().item()
                    
        except Exception as e:
            print(f"❌ Action selection error: {e}")
            return random.randint(0, self.num_actions - 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stocke une expérience"""
        try:
            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"❌ Store experience error: {e}")
  
    def train(self):
        """Entraîne le réseau avec verrouillage global pour modèle partagé"""
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # ✅ CORRECTION: Récupérer le verrou correctement
        if self.use_shared_model and hasattr(self, 'shared_manager'):
            lock = self.shared_manager.training_lock
            if lock is None:
                print(f"⚠️  Robot {self.robot_id}: Verrou non initialisé, création d'urgence")
                lock = threading.Lock()
        else:
            lock = threading.Lock()
        
        with lock:
            try:
                # Échantillonner un batch
                batch = random.sample(self.memory, self.batch_size)
                
                # Préparer les données batch
                batch_states = []
                batch_actions = []
                batch_rewards = []
                batch_next_states = []
                batch_dones = []
                
                for experience in batch:
                    state, action, reward, next_state, done = experience
                    batch_states.append(state)
                    batch_actions.append(action)
                    batch_rewards.append(reward)
                    batch_next_states.append(next_state)
                    batch_dones.append(done)
                
                # Calculer Q-values actuelles
                current_q_values_list = []
                for state in batch_states:
                    try:
                        q_values = self.policy_net(state)
                        current_q_values_list.append(q_values.squeeze(0))
                    except Exception as e:
                        print(f"❌ Robot {self.robot_id}: Current Q-values error: {e}")
                        current_q_values_list.append(torch.zeros(self.num_actions))
                
                # Stack pour avoir (batch_size, num_actions)
                current_q_values = torch.stack(current_q_values_list).to(self.device)
                
                # Sélectionner les Q-values des actions choisies
                actions_tensor = torch.LongTensor(batch_actions).to(self.device)
                current_q_selected = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Calculer Q-values cibles
                next_q_values_list = []
                for next_state in batch_next_states:
                    try:
                        next_q = self.target_net(next_state)
                        next_q_values_list.append(next_q.squeeze(0))
                    except Exception as e:
                        print(f"❌ Robot {self.robot_id}: Next Q-values error: {e}")
                        next_q_values_list.append(torch.zeros(self.num_actions))
                
                next_q_values = torch.stack(next_q_values_list).to(self.device)
                max_next_q_values = next_q_values.max(1)[0]
                
                # Calculer les targets
                rewards_tensor = torch.FloatTensor(batch_rewards).to(self.device)
                dones_tensor = torch.BoolTensor(batch_dones).to(self.device)
                target_q_values = rewards_tensor + (self.gamma * max_next_q_values * ~dones_tensor)
                
                # Vérifier les dimensions
                if current_q_selected.dim() == 0:
                    current_q_selected = current_q_selected.unsqueeze(0)
                if target_q_values.dim() == 0:
                    target_q_values = target_q_values.unsqueeze(0)
                
                # Calculer la loss
                loss = self.loss_fn(current_q_selected, target_q_values)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                
                # Mettre à jour epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # ✅ INITIALISER loss_value ICI
                loss_value = loss.item()
                self.loss_history.append(loss_value)
                
                # ✅ CORRECTION: Mise à jour du compteur global
                if self.use_shared_model and hasattr(self, 'shared_manager'):
                    # Incrémenter le compteur global
                    self.shared_manager.global_step_count += 1
                    self.step_count = self.shared_manager.global_step_count
                    
                    # Afficher avec le bon robot_id
                    if self.shared_manager.global_step_count % 5 == 0:  # Tous les 5 steps
                        print(f"📚 [GLOBAL] Step: {self.shared_manager.global_step_count:6d}, "
                            f"Robot: {self.robot_id}, Loss: {loss_value:6.4f}, "
                            f"ε: {self.epsilon:.3f}, Memory: {len(self.memory)}")
                    
                    # Mettre à jour le réseau cible
                    if self.shared_manager.global_step_count % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        print(f"🔄 [GLOBAL] Target network updated at step {self.shared_manager.global_step_count}")
                else:
                    # Mode individuel
                    self.step_count += 1
                    if self.step_count % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                    if self.step_count % 10 == 0:
                        print(f"📚 Robot {self.robot_id}: Step {self.step_count}, "
                            f"Loss: {loss_value:.4f}, ε: {self.epsilon:.3f}")
                
                return loss_value
                
            except Exception as e:
                print(f"❌ Robot {self.robot_id}: Training error: {e}")
                import traceback
                traceback.print_exc()
                return 0.0  # ✅ Retourner 0.0 en cas d'erreur   
    



    def verify_model_sharing(self):
        """Vérifie que le modèle est bien partagé"""
        if not self.use_shared_model:
            print(f"🔧 Robot {self.robot_id}: Modèle individuel - pas de vérification nécessaire")
            return
        
        if not hasattr(self, 'shared_manager'):
            print(f"❌ Robot {self.robot_id}: Pas de shared_manager!")
            return
        
        print(f"🔍 Robot {self.robot_id} - Vérification partage:")
        print(f"   ID Policy net: {id(self.policy_net)}")
        print(f"   ID Target net: {id(self.target_net)}")
        print(f"   ID Optimizer: {id(self.optimizer)}")
        print(f"   Robots enregistrés: {self.shared_manager.robot_ids}")
        
        # Vérifier si c'est le même objet
        if hasattr(self.shared_manager, '_reference_net_id'):
            if id(self.policy_net) == self.shared_manager._reference_net_id:
                print(f"✅ MODÈLE PARTAGÉ CONFIRMÉ!")
            else:
                print(f"❌ ERREUR: Modèles différents!")
        else:
            self.shared_manager._reference_net_id = id(self.policy_net)
            print(f"✅ Référence établie par Robot {self.robot_id}")


    def save_model(self, filepath=None):
        """Sauvegarde le modèle avec gestion automatique des noms"""
        try:
            if filepath is None:
                # ✅ DÉFINIR model_dir AVANT de l'utiliser
                if self.use_shared_model:
                    model_dir = "models/shared"
                else:
                    model_dir = f"models/robot_{self.robot_id}"
                
                os.makedirs(model_dir, exist_ok=True)
                
                if self.use_shared_model:
                    filepath = f"{model_dir}/shared_model_episode_{self.episode_count}.pth"
                else:
                    filepath = f"{model_dir}/model_episode_{self.episode_count}.pth"
            
            # ✅ S'assurer que le dossier existe
            model_dir = os.path.dirname(filepath)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            # Sauvegarder le modèle
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'is_shared_model': self.use_shared_model,
                'robot_id': self.robot_id
            }, filepath)
            
            # Sauvegarder aussi comme latest
            if self.use_shared_model:
                latest_path = f"models/shared/latest_shared_model.pth"
            else:
                latest_path = f"models/robot_{self.robot_id}/latest_model.pth"
            
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'is_shared_model': self.use_shared_model,
                'robot_id': self.robot_id
            }, latest_path)
            
            model_type = "PARTAGÉ" if self.use_shared_model else "individuel"
            print(f"💾 Robot {self.robot_id}: Modèle {model_type} sauvegardé → {os.path.basename(filepath)}")
            print(f"🔁 Latest model: {os.path.basename(latest_path)}")
            
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Error saving model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_model(self, filepath):
        """Charge le modèle"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_count = checkpoint.get('step_count', self.step_count)
            self.episode_count = checkpoint.get('episode_count', self.episode_count)
            self.loss_history = checkpoint.get('loss_history', [])
            print(f"📂 Model loaded: {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def update_episode_count(self):
        """Met à jour le compteur d'épisodes et sauvegarde périodiquement"""
        self.episode_count += 1
        
        # Sauvegarder tous les 5 épisodes
        if self.episode_count % 5 == 0:
            self.save_model()
        
        # Sauvegarder final_model à la fin
        if self.episode_count % 25 == 0:
            final_path = f"models/robot_{self.robot_id}/final_model.pth"
            self.save_model(final_path)

# Fonction utilitaire pour créer l'optimiseur
def create_optimizer(model, learning_rate=0.00025):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test du continuous learning
    input_shape = (100, 100, 4)
    num_actions = 8
    
    # Créer l'agent avec continuous learning
    agent = RobustDQNAgent(
        state_shape=input_shape,
        num_actions=num_actions,
        robot_id=0,
        load_latest_model=True  # Active le continuous learning
    )
    
    print("✅ Agent avec Continuous Learning créé avec succès!")

    