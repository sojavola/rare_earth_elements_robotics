import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import sys

class SpectralCNN1D(nn.Module):
    def __init__(self, input_length, num_classes, dropout_rate=0.3):
        super(SpectralCNN1D, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # Couches convolutives 1D pour les signatures spectrales
        self.conv_layers = nn.Sequential(
            # Première couche
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(dropout_rate),
            
            # Deuxième couche
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(dropout_rate),
            
            # Troisième couche
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pooling global
            nn.Dropout(dropout_rate)
        )
        
        # Couches fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, 1, input_length)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class SpectralClassifier:
    def __init__(self, input_length=100, num_classes=4):
        self.input_length = input_length
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modèle et préprocesseur
        self.model = SpectralCNN1D(input_length, num_classes).to(self.device)
        self.scaler = StandardScaler()
        
        # Classes de minéraux
        self.mineral_classes = [
            'REE_Oxides',
            'REE_Silicates', 
            'REE_Phosphates',
            'REE_Carbonates'
        ]
        
        # Historique d'entraînement
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def prepare_spectral_data(self, spectral_data, labels):
        """Prépare les données spectrales pour l'entraînement"""
        # Normalisation des données
        spectral_data_normalized = self.scaler.fit_transform(spectral_data)
        
        # Redimensionnement pour CNN 1D
        spectral_data_reshaped = spectral_data_normalized.reshape(
            -1, 1, self.input_length
        )
        
        return train_test_split(
            spectral_data_reshaped, labels, test_size=0.2, random_state=42, stratify=labels
        )
    
    def train(self, spectral_data, labels, epochs=100, batch_size=32, learning_rate=0.001):
        """Entraîne le classificateur spectral"""
        X_train, X_val, y_train, y_val = self.prepare_spectral_data(spectral_data, labels)
        
        # Conversion en tensors PyTorch
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Loss et optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Boucle d'entraînement
        for epoch in range(epochs):
            # Mode entraînement
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), batch_size):
                    batch_X = X_val_tensor[i:i+batch_size]
                    batch_y = y_val_tensor[i:i+batch_size]
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calcul des métriques
            train_loss_avg = train_loss / len(X_train_tensor)
            train_acc = train_correct / len(X_train_tensor)
            val_loss_avg = val_loss / len(X_val_tensor)
            val_acc = val_correct / len(X_val_tensor)
            
            # Mise à jour du scheduler
            scheduler.step(val_loss_avg)
            
            # Enregistrement de l'historique
            self.training_history['train_loss'].append(train_loss_avg)
            self.training_history['val_loss'].append(val_loss_avg)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}: '
                      f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}')
    
    def predict(self, spectral_data):
        """Prédit la classe minérale pour de nouvelles données spectrales"""
        self.model.eval()
        
        # Prétraitement
        spectral_data_normalized = self.scaler.transform(spectral_data)
        spectral_data_reshaped = spectral_data_normalized.reshape(-1, 1, self.input_length)
        
        # Prédiction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(spectral_data_reshaped).to(self.device)
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
        
        # Conversion en résultats interprétables
        results = []
        for i in range(len(spectral_data)):
            mineral_class = self.mineral_classes[predicted_classes[i].item()]
            confidence = probabilities[i][predicted_classes[i]].item()
            
            results.append({
                'mineral_type': mineral_class,
                'confidence': confidence,
                'all_probabilities': {
                    self.mineral_classes[j]: probabilities[i][j].item()
                    for j in range(self.num_classes)
                }
            })
        
        return results
    
    def save_model(self, model_path, scaler_path):
        """Sauvegarde le modèle et le scaler"""
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Modèle sauvegardé: {model_path}")
        print(f"Scaler sauvegardé: {scaler_path}")
    
    def load_model(self, model_path, scaler_path):
        """Charge le modèle et le scaler"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.scaler = joblib.load(scaler_path)
        print(f"Modèle chargé: {model_path}")
        print(f"Scaler chargé: {scaler_path}")

# NOUVEAU : Classe Node ROS2 pour le classificateur spectral
class SpectralClassifierNode(Node):
    def __init__(self):
        super().__init__('spectral_classifier_node')
        
        # Initialiser le classificateur
        self.classifier = SpectralClassifier()
        
        # Publishers
        self.prediction_pub = self.create_publisher(String, '/mineral_predictions', 10)
        
        # Subscribers
        self.spectral_sub = self.create_subscription(
            Float32MultiArray,
            '/spectral_data',
            self.spectral_data_callback,
            10
        )
        
        # Charger le modèle entraîné (si disponible)
        try:
            self.classifier.load_model('models/spectral_model.pth', 'models/scaler.joblib')
            self.get_logger().info('Modèle spectral chargé avec succès')
        except Exception as e:
            self.get_logger().warn(f'Modèle non trouvé, utilisation en mode simulation: {e}')
            # Générer des données d'entraînement simulées pour la démo
            self._generate_demo_model()
        
        self.get_logger().info('Spectral Classifier Node initialized')

    def spectral_data_callback(self, msg):
        """Traite les données spectrales reçues"""
        try:
            # Convertir les données ROS2 en numpy array
            spectral_data = np.array(msg.data).reshape(1, -1)
            
            # Faire une prédiction
            predictions = self.classifier.predict(spectral_data)
            
            # Publier les résultats
            result_msg = String()
            if predictions:
                pred = predictions[0]
                result_msg.data = f"{pred['mineral_type']} (confiance: {pred['confidence']:.2f})"
                self.prediction_pub.publish(result_msg)
                
                self.get_logger().info(f'Minéral détecté: {result_msg.data}')
            
        except Exception as e:
            self.get_logger().error(f'Erreur lors de la classification: {e}')

    def _generate_demo_model(self):
        """Génère un modèle de démonstration avec des données simulées"""
        from .spectral_data_generator import SpectralDataGenerator
        
        self.get_logger().info('Génération de données d\'entraînement simulées...')
        
        # Générer des données d'entraînement
        generator = SpectralDataGenerator()
        spectra, labels, classes = generator.generate_spectral_dataset(num_samples_per_class=100)
        
        # Entraîner le modèle
        self.classifier.train(spectra, labels, epochs=10, batch_size=16)
        
        # Sauvegarder le modèle
        import os
        os.makedirs('models', exist_ok=True)
        self.classifier.save_model('models/spectral_model.pth', 'models/scaler.joblib')
        
        self.get_logger().info('Modèle de démonstration entraîné et sauvegardé')

def main(args=None):
    """Point d'entrée principal du nœud ROS2"""
    rclpy.init(args=args)
    
    try:
        node = SpectralClassifierNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def train_main():
    """Point d'entrée pour l'entraînement du modèle"""
    print("🧪 Entraînement du classificateur spectral...")
    
    # Générer des données d'entraînement
    from spectral_data_generator import SpectralDataGenerator
    
    generator = SpectralDataGenerator()
    spectra, labels, classes = generator.generate_spectral_dataset(num_samples_per_class=1000)
    
    # Créer et entraîner le classificateur
    classifier = SpectralClassifier()
    classifier.train(spectra, labels, epochs=50)
    
    # Sauvegarder le modèle
    import os
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/spectral_model.pth', 'models/scaler.joblib')
    
    print("✅ Entraînement terminé et modèle sauvegardé!")

if __name__ == '__main__':
    # Si appelé directement, lancer le nœud ROS2
    main()