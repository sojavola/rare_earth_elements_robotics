import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import glob
import threading
from datetime import datetime


try:
    from .data_logger import EnhancedDataLogger
    DATA_LOGGER_AVAILABLE = True
    print("✅ EnhancedDataLogger importé avec succès")
except ImportError as e:
    DATA_LOGGER_AVAILABLE = False
    print(f"⚠️  EnhancedDataLogger non disponible: {e}")
    print("   Le logging sera désactivé")


class SharedDQNManager:
  
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SharedDQNManager, cls).__new__(cls)
                cls._instance._initialized = False
                
                # ✅ INITIALISATION DES ATTRIBUTS DANS __new__
                cls._instance.training_lock = threading.Lock()
                cls._instance.shared_policy_net = None
                cls._instance.shared_target_net = None
                cls._instance.shared_optimizer = None
                cls._instance.global_step_count = 0
                cls._instance.global_episode_count = 0
                cls._instance.robot_ids = set()

            return cls._instance
    
    def initialize(self, state_shape, num_actions, device='cpu'):
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
            print(f"🔒 Verrou d'entraînement créé") 

    def get_shared_networks(self):
        if not self._initialized:
            raise RuntimeError("SharedDQNManager non initialisé!")
        
        return self.shared_policy_net, self.shared_target_net, self.shared_optimizer

    def register_robot(self, robot_id):
        # ✅ VÉRIFIER SI INITIALISÉ
        if not self._initialized:
            print(f"⚠️  [GLOBAL] Manager non initialisé, initialisation automatique...")
            # Vous devrez peut-être passer les paramètres ici
        
        self.robot_ids.add(robot_id)
        print(f"🤖 [GLOBAL] Robot {robot_id} enregistré - Total: {len(self.robot_ids)} robots")


    def get_stats(self):
        return {
            'global_step_count': self.global_step_count,
            'global_episode_count': self.global_episode_count,
            'active_robots': len(self.robot_ids),
            'robot_ids': list(self.robot_ids)
        }

        
    def get_shared_networks(self):
        return self.shared_policy_net, self.shared_target_net, self.shared_optimizer
    
    def get_training_lock(self):
        if not hasattr(self, 'training_lock') or self.training_lock is None:
            print(f"⚠️  [GLOBAL] Création d'urgence du verrou d'entraînement...")
            self.training_lock = threading.Lock()
            print(f"✅ [GLOBAL] Verrou d'urgence créé: {id(self.training_lock)}")
        return self.training_lock

class CNN_DQN(nn.Module):
    """
    DQN adapté pour grilles spatiales 100×100
    Préserve la géométrie tout en réduisant progressivement
    """
    def __init__(self, input_shape, num_actions):
        super(CNN_DQN, self).__init__()
        
        self.input_shape = input_shape  # (100, 100, 8)
        self.num_actions = num_actions
        
        # ✅ ARCHITECTURE OPTIMISÉE POUR GRILLES
        self.conv_layers = nn.Sequential(
            # Conv1: Extraction features locales
            nn.Conv2d(input_shape[2], 32, kernel_size=5, stride=2, padding=2),  # 100→50
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Stabilité
            
            # Conv2: Features régionales
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 50→25
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Conv3: Features globales
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 25→13
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Conv4: Compression finale
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 13→13
            nn.ReLU()
        )
        
        # Calculer taille automatiquement
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Couches denses
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Régularisation
            nn.Linear(512, num_actions)
        )
    
    def _get_conv_output_size(self, shape):
        """Calcule automatiquement la taille après conv"""
        with torch.no_grad():
            # (C, H, W) pour PyTorch
            dummy = torch.zeros(1, shape[2], shape[0], shape[1])
            output = self.conv_layers(dummy)
            return output.view(1, -1).size(1)
    
    def forward(self, state):
        """
        Input: (H, W, C) numpy ou (B, H, W, C)
        Output: (B, num_actions) Q-values
        """
        try:
            # Conversion et permutation des dimensions
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            
            # (H, W, C) → (C, H, W)
            if state.dim() == 3:
                state = state.permute(2, 0, 1).unsqueeze(0)
            # (B, H, W, C) → (B, C, H, W)
            elif state.dim() == 4:
                state = state.permute(0, 3, 1, 2)
            
            # Forward pass
            features = self.conv_layers(state)
            q_values = self.dense_layers(features)
            
            return q_values
            
        except Exception as e:
            print(f"❌ Forward error: {e}")
            print(f"   State shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
            return torch.zeros(
                (state.size(0), self.num_actions),
                device=state.device,
                requires_grad=True
            )


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
                 use_shared_model=False,
                 enable_logging=True,
                 use_real_reward_system=True
                 ):  


        self.state_shape = state_shape
        self.num_actions = num_actions
        self.robot_id = robot_id  # ✅ MAINTENANT DÉFINI
        print(f"🎯 DEBUG: Robot ID reçu = {robot_id}, stocké = {self.robot_id}")


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
            self.shared_manager = SharedDQNManager()
            self.shared_manager.initialize(state_shape, num_actions, self.device)
            self.shared_manager.register_robot(self.robot_id)
            self.policy_net, self.target_net, self.optimizer = self.shared_manager.get_shared_networks()
            print(f"🔗 Robot {self.robot_id} utilisant le modèle DQN PARTAGÉ")
        else:
            # Réseaux individuels (ancienne approche)
            self.shared_manager = None  # Explicitement None
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


#Logger mettrics.
# 
#    
        self.enable_logging = enable_logging and DATA_LOGGER_AVAILABLE
        
        if self.enable_logging:
            try:
                self.logger = EnhancedDataLogger(robot_id=robot_id)
                print(f"📊 Logging CSV/Plots activé pour Robot {robot_id}")
                print(f"   📁 Données sauvegardées dans: logs/robot_{robot_id}/")
            except Exception as e:
                print(f"⚠️  Erreur initialisation logger: {e}")
                self.enable_logging = False
                self.logger = None
        else:
            self.logger = None
            if enable_logging and not DATA_LOGGER_AVAILABLE:
                print(f"ℹ️  Robot {robot_id}: DataLogger non disponible, logging désactivé")
        
        # Variables pour le tracking d'épisode
        self._current_episode_data = {
            'episode': 0,
            'total_reward': 0,
            'total_steps': 0,
            'minerals_collected': 0,
            'loss_sum': 0,
            'loss_count': 0,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        
        # Charger le modèle si demandé
#        if load_latest_model:
#            self.auto_load_latest_model()
        
        print(f"🧠 Robot {self.robot_id} initialisé - ε={self.epsilon:.3f}") 

           

    ###
       
    def log_step(self, state, action, reward, next_state, position):
        """Loggue les données d'un step"""
        if not self.enable_logging or self.logger is None:
            return
        
        try:
            step_data = {
                'step': self.step_count,
                'episode': self.episode_count,
                'action': action,
                'reward': reward,
                'position_x': position[0] if position else 0,
                'position_y': position[1] if position else 0,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_step(step_data)
            
        except Exception as e:
            print(f"⚠️  Robot {self.robot_id}: Erreur log_step: {e}")
    
    def log_training_step(self, loss):
        """Loggue les données d'entraînement"""
        if not self.enable_logging or self.logger is None:
            return
        
        try:
            training_data = {
                'training_step': self.step_count,
                'loss': loss.item() if hasattr(loss, 'item') else loss,
                'epsilon': self.epsilon,
                'memory_size': len(self.memory),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_training(training_data)
            
        except Exception as e:
            print(f"⚠️  Robot {self.robot_id}: Erreur log_training_step: {e}")
    
    def log_mineral_discovery(self, position, mineral_type, concentration, reward):
        """Loggue une découverte minérale"""
        if not self.enable_logging or self.logger is None:
            return
        
        try:
            mineral_data = {
                'episode': self.episode_count,
                'step': self.step_count,
                'position_x': position[0],
                'position_y': position[1],
                'mineral_type': mineral_type,
                'concentration': concentration,
                'reward_earned': reward,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_mineral(mineral_data)
            
            # Mettre à jour le compteur d'épisode
            self._current_episode_data['minerals_collected'] += 1
            
        except Exception as e:
            print(f"⚠️  Robot {self.robot_id}: Erreur log_mineral_discovery: {e}")
    
    def start_episode(self):
        """Démarre un nouvel épisode de logging"""
        self._current_episode_data = {
            'episode': self.episode_count,
            'total_reward': 0,
            'total_steps': 0,
            'minerals_collected': 0,
            'loss_sum': 0,
            'loss_count': 0,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
    
    def update_episode_data(self, reward, loss=None):
        """Met à jour les données de l'épisode en cours"""
        self._current_episode_data['total_reward'] += reward
        self._current_episode_data['total_steps'] += 1
        self._current_episode_data['epsilon'] = self.epsilon
        self._current_episode_data['memory_size'] = len(self.memory)
        
        if loss is not None:
            self._current_episode_data['loss_sum'] += loss
            self._current_episode_data['loss_count'] += 1
    
    def end_episode(self):
        """Termine l'épisode et loggue les données"""
        if not self.enable_logging or self.logger is None:
            # Juste incrémenter le compteur
            self.episode_count += 1
            return
        
        try:
            # Calculer les moyennes
            avg_loss = (self._current_episode_data['loss_sum'] / 
                       self._current_episode_data['loss_count'] 
                       if self._current_episode_data['loss_count'] > 0 else 0)
            
            avg_reward = (self._current_episode_data['total_reward'] / 
                         self._current_episode_data['total_steps'] 
                         if self._current_episode_data['total_steps'] > 0 else 0)
            
            # Calculer le taux d'exploration (simplifié)
            exploration_rate = 100 * (1.0 - self.epsilon)
            
            # Préparer les données d'épisode
            episode_data = {
                'episode': self.episode_count,
                'total_steps': self._current_episode_data['total_steps'],
                'total_reward': self._current_episode_data['total_reward'],
                'average_reward': avg_reward,
                'minerals_collected': self._current_episode_data['minerals_collected'],
                'exploration_rate': exploration_rate,
                'epsilon': self._current_episode_data['epsilon'],
                'memory_size': self._current_episode_data['memory_size'],
                'loss_avg': avg_loss,
                'timestamp': datetime.now().isoformat()
            }
            
            # Logguer l'épisode - Vérifier la méthode disponible
            if hasattr(self.logger, 'log_episode'):
                self.logger.log_episode(episode_data)
            elif hasattr(self.logger, 'add_episode_data'):
                self.logger.add_episode_data(episode_data)
            else:
                print(f"⚠️  Robot {self.robot_id}: Logger n'a pas de méthode log_episode")
            
            # ⭐ CORRECTION: Vérifier si la méthode existe
            if self.episode_count % 10 == 0:
                print(f"📊 Robot {self.robot_id}: Génération rapport épisode {self.episode_count}")
                
                if hasattr(self.logger, 'generate_comprehensive_report'):
                    self.logger.generate_comprehensive_report()
                elif hasattr(self.logger, 'save_data'):
                    self.logger.save_data()
                    print(f"💾 Données sauvegardées pour l'épisode {self.episode_count}")
                else:
                    print(f"ℹ️  Aucune méthode de sauvegarde disponible dans le logger")
            
            # Incrémenter le compteur d'épisodes
            self.episode_count += 1
            
            # Réinitialiser pour le prochain épisode
            self.start_episode()
        
        except Exception as e:
                print(f"⚠️  Robot {self.robot_id}: Erreur end_episode: {e}")
                # Incrémenter quand même le compteur
                self.episode_count += 1

                
    # ==========================================================
    """
    def auto_load_latest_model(self):
        
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
    """
    def auto_load_latest_model(self):
        """Charge automatiquement le dernier modèle sauvegardé"""
        try:
            # Appeler load_model() sans paramètre pour qu'il trouve le modèle automatiquement
            success = self.load_model()
            
            if success:
                # Ajuster epsilon après chargement
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.7)
                print(f"✅ Robot {self.robot_id}: Model loaded successfully")
                print(f"🔄 ε adjusted to: {self.epsilon:.3f}")
                return True
            else:
                print(f"📁 Robot {self.robot_id}: Starting fresh - no model found")
                return False
                
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Auto-load error: {e}")
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
 ######
 # ###
 # ####
 # ###   
 
    def store_experience(self, state, action, reward, next_state, done,position=None):
        """Stocke une expérience"""
        try:
            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"❌ Store experience error: {e}")
        
        if position is not None:
            self.log_step(state, action, reward, next_state, position)
        
        # Mettre à jour les données d'épisode
        self.update_episode_data(reward)

        if reward > 50.0 and self.enable_logging:
            self.log_mineral_discovery(
                position=position,
                mineral_type=-1,  # À déterminer depuis les données
                concentration=reward / 100.0,  # Approximation
                reward=reward
            )
            
####
# ##
# ##    
    def train(self):
        """Entraîne le réseau (version pour modèles individuels)"""
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # ✅ CORRECTION : SIMPLIFIER LE VERRouILLAGE
        # Puisque chaque robot a sa propre DQN, on utilise juste un verrou local
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
                
                # Logging
                self.log_training_step(loss)
                self.update_episode_data(0, loss)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                
                # Mettre à jour epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                    self.epsilon = max(self.epsilon, self.epsilon_min)

                # ✅ CORRECTION : TOUJOURS MODE INDIVIDUEL
                # Chaque robot a sa propre DQN, donc on incrémente son propre compteur
                self.step_count += 1
                
                # Mettre à jour le réseau cible
                if self.step_count % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"🔄 Robot {self.robot_id}: Target network updated at step {self.step_count}")
                
                # Enregistrer la loss
                loss_value = loss.item()
                self.loss_history.append(loss_value)
                
                # ✅ AFFICHAGE INDIVIDUEL SEULEMENT
                if self.step_count % 10 == 0:
                    print(f"📚 Robot {self.robot_id}: Step {self.step_count}, "
                        f"Loss: {loss_value:.4f}, ε: {self.epsilon:.3f}, "
                        f"Memory: {len(self.memory)}")
                
                return loss_value
                
            except Exception as e:
                print(f"❌ Robot {self.robot_id}: Training error: {e}")
                import traceback
                traceback.print_exc()
                return 0.0
    ####

    ####
    """
    def save_model(self, filepath=None):
 
        try:
            if filepath is None:
                # Générer un nom de fichier automatique
                model_dir = f"models/"
                os.makedirs(model_dir, exist_ok=True)
                
                filepath = f"{model_dir}/shared_model_episode_{self.episode_count}.pth"
            
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'is_shared_model':True,
                'robots_id': self.robot_id
            }, filepath)
            
            # Sauvegarder aussi comme latest_model
            latest_path = f"{model_dir}/latest_shared_model.pth"
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'robot_id': self.robot_id  
            }, latest_path)
            
            print(f"💾 Model saved: {filepath}")
            print(f"🔁 Latest model updated: {latest_path}")
            print(f"💾 Robot {self.robot_id}: Modèle PARTAGÉ sauvegardé")
            
            
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Error saving model: {e}")
   
    def load_model(self, filepath):
      
        try:
            model_dir = f"models/robot_{self.robot_id}"
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
    """

    def save_model(self, filepath=None):
        """Sauvegarde le modèle individuel pour chaque robot"""
        try:
            # Dossier pour ce robot
            model_dir = f"models/robot_{self.robot_id}"
            os.makedirs(model_dir, exist_ok=True)
            
            if filepath is None:
                # Générer un nom de fichier automatique
                filepath = f"{model_dir}/model_episode_{self.episode_count}.pth"
            elif not "/" in filepath and not "\\" in filepath:
                # Si c'est juste un nom de fichier, le mettre dans le dossier du robot
                filepath = f"{model_dir}/{filepath}"
            
            # Sauvegarder le modèle principal
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'is_shared_model': False,  # ← False pour modèle individuel
                'robot_id': self.robot_id,   # ← robot_id au singulier
                'training_steps': self.step_count,  # ✅ Doublon pour sécurité
                'total_steps': self.step_count,  # ✅ Encore un autre nom
                'timestamp': datetime.now().isoformat()
            }, filepath)
            
            # Sauvegarder aussi comme latest_model
            latest_path = f"{model_dir}/latest_model.pth"
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'loss_history': self.loss_history,
                'robot_id': self.robot_id,
                'training_steps': self.step_count,
                'timestamp': datetime.now().isoformat()
            }, latest_path)
            
            print(f"💾 Robot {self.robot_id}: Model saved")
            print(f"   Episode: {self.episode_count}, Steps: {self.step_count}, ε={self.epsilon:.3f}")
            print(f"   Path: {filepath}")
            
            return True
            
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False
        

        
    def load_model(self, filepath=None):
        """Charge le modèle - paramètre optionnel"""
        try:
            if filepath is None:
                # Trouver automatiquement le modèle
                if self.use_shared_model:
                    model_dir = "models/shared"
                else:
                    model_dir = f"models/robot_{self.robot_id}"
                
                # Chercher latest_model.pth d'abord
                latest_path = f"{model_dir}/latest_model.pth"
                if os.path.exists(latest_path):
                    filepath = latest_path
                    print(f"🔍 Robot {self.robot_id}: Found latest model")

                else:
                    # Chercher n'importe quel modèle
                    import glob
                    models = glob.glob(f"{model_dir}/*.pth")
                    if models:
                        models.sort(key=os.path.getmtime, reverse=True)
                        filepath = models[0]
                        print(f"🔍 Robot {self.robot_id}: Found model: {os.path.basename(filepath)}")
              
                    else:
                        print(f"⚠️ Robot {self.robot_id}: No model file found")
                        return False
            
            if not os.path.exists(filepath):
                print(f"⚠️ Robot {self.robot_id}: Model file not found: {filepath}")
                return False
            
            # Charger le checkpoint
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Charger les états des réseaux
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            
            # Charger l'optimizer si disponible
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            saved_step_count = 0
            
            # Restaurer les hyperparamètres
            if 'epsilon' in checkpoint:
                self.epsilon = float(checkpoint['epsilon'])
            
            if 'step_count' in checkpoint:
                self.step_count = int(checkpoint['step_count'])
            
            if 'episode_count' in checkpoint:
                self.episode_count = int(checkpoint['episode_count'])
            
            if 'loss_history' in checkpoint:
                self.loss_history = checkpoint['loss_history']
            
            print(f"✅ Robot {self.robot_id}: Model loaded from {filepath}")
            print(f"   Episode: {self.episode_count}, Steps: {self.step_count}, ε={self.epsilon:.3f}")
            
            return True
        
        except Exception as e:
            print(f"❌ Robot {self.robot_id}: Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
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
