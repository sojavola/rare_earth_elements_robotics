import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class AdvancedDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AdvancedDQN, self).__init__()
        
        self.input_shape = input_shape  # (height, width, channels)
        self.num_actions = num_actions
        
        # CNN pour traiter les cartes spatiales
        self.conv_layers = nn.Sequential(
            # Couche 1
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Couche 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Couche 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Couche 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # Calcul de la taille après les convolutions
        conv_out_size = self._get_conv_output(input_shape)
        
        # Réseau de valeur (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        # Réseau d'avantage (Advantage Stream) - Dueling DQN
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_actions)
        )
        
    def _get_conv_output(self, shape):
        """Calcule la taille de sortie des couches convolutives"""
        with torch.no_grad():
            input = torch.zeros(1, *shape).permute(0, 3, 1, 2)  # (batch, channels, height, width)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))
    
    def forward(self, x):
        # x shape: (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Passage through CNN
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Dueling DQN: Value + Advantage
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class AdvancedDQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate=0.00025, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000,
                 batch_size=32, target_update=1000):
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = AdvancedDQN(state_shape, num_actions).to(self.device)
        self.target_net = AdvancedDQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Memory
        self.memory = deque(maxlen=memory_size)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
    def choose_action(self, state):
        """Choix d'action avec stratégie epsilon-greedy"""
        if np.random.random() <= self.epsilon:
            # Exploration
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Entraîne le réseau sur un batch d'expériences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Échantillonnage du batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Conversion en tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Q-values actuelles
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Q-values cibles
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcul de la loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Mise à jour de l'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Mise à jour du réseau cible
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_model(self, filepath):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath):
        """Charge le modèle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']