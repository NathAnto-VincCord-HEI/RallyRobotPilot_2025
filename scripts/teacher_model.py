"""
Modèle "Professeur" pour la conduite autonome
Utilise les capteurs auxiliaires (raycast, vitesse, angle) pour apprentissage
"""

import numpy as np
import pickle


class TeacherNetwork:
    """
    Réseau de neurones simple pour le modèle professeur
    Architecture: [input] -> [hidden1] -> [hidden2] -> [output]
    """
    
    def __init__(self, input_size=17, hidden1_size=32, hidden2_size=16, output_size=4):
        """
        Args:
            input_size: 15 raycasts + vitesse + angle = 17
            hidden1_size: Taille de la première couche cachée
            hidden2_size: Taille de la deuxième couche cachée
            output_size: 4 actions (Forward, Backward, Left, Right)
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        
        # Initialisation des poids (Xavier/Glorot)
        self.weights1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.bias1 = np.zeros(hidden1_size)
        
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.bias2 = np.zeros(hidden2_size)
        
        self.weights3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.bias3 = np.zeros(output_size)
        
    def relu(self, x):
        """Fonction d'activation ReLU"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Fonction d'activation Sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def normalize_inputs(self, raycast_distances, car_speed, car_angle):
        """
        Normalise les entrées pour améliorer l'apprentissage
        
        Args:
            raycast_distances: Liste de 15 distances (0-100)
            car_speed: Vitesse de la voiture (-15 à 50)
            car_angle: Angle en degrés (0-360)
        
        Returns:
            np.array: Vecteur d'entrée normalisé
        """
        # Normaliser les raycasts (0-100 -> 0-1)
        normalized_raycasts = np.array(raycast_distances) / 100.0
        
        # Normaliser la vitesse (-15 à 50 -> -1 à 1 approximativement)
        normalized_speed = car_speed / 50.0
        
        # Convertir l'angle en composantes sin/cos pour continuité
        angle_rad = np.radians(car_angle)
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)
        
        # Concaténer toutes les entrées
        # 15 raycasts + 1 vitesse + 2 composantes d'angle = 18 features
        # Mais on garde 17 pour correspondre à la taille attendue
        inputs = np.concatenate([
            normalized_raycasts,
            [normalized_speed],
            [angle_sin]
        ])
        
        return inputs
    
    def forward(self, inputs):
        """
        Propagation avant
        
        Args:
            inputs: Vecteur d'entrée normalisé (17 dimensions)
        
        Returns:
            np.array: Probabilités des 4 actions
        """
        # Couche 1
        hidden1 = self.relu(np.dot(inputs, self.weights1) + self.bias1)
        
        # Couche 2
        hidden2 = self.relu(np.dot(hidden1, self.weights2) + self.bias2)
        
        # Couche de sortie avec sigmoid pour obtenir des probabilités
        output = self.sigmoid(np.dot(hidden2, self.weights3) + self.bias3)
        
        return output
    
    def predict(self, raycast_distances, car_speed, car_angle, threshold=0.5):
        """
        Prédit les actions à prendre
        
        Args:
            raycast_distances: Liste de 15 distances
            car_speed: Vitesse actuelle
            car_angle: Angle actuel
            threshold: Seuil pour activer une action (0-1)
        
        Returns:
            tuple: (Forward, Backward, Left, Right) - Booléens
        """
        inputs = self.normalize_inputs(raycast_distances, car_speed, car_angle)
        output = self.forward(inputs)
        
        # Convertir les probabilités en actions binaires
        actions = output > threshold
        
        return tuple(actions.astype(int))
    
    def get_genome(self):
        """
        Extrait tous les poids du réseau en un seul vecteur (génome)
        Utilisé pour l'algorithme génétique
        
        Returns:
            np.array: Vecteur de tous les poids
        """
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten(),
            self.weights3.flatten(),
            self.bias3.flatten()
        ])
    
    def set_genome(self, genome):
        """
        Charge un génome (vecteur de poids) dans le réseau
        
        Args:
            genome: Vecteur de poids
        """
        idx = 0
        
        # Weights1
        size = self.input_size * self.hidden1_size
        self.weights1 = genome[idx:idx+size].reshape(self.input_size, self.hidden1_size)
        idx += size
        
        # Bias1
        size = self.hidden1_size
        self.bias1 = genome[idx:idx+size]
        idx += size
        
        # Weights2
        size = self.hidden1_size * self.hidden2_size
        self.weights2 = genome[idx:idx+size].reshape(self.hidden1_size, self.hidden2_size)
        idx += size
        
        # Bias2
        size = self.hidden2_size
        self.bias2 = genome[idx:idx+size]
        idx += size
        
        # Weights3
        size = self.hidden2_size * self.output_size
        self.weights3 = genome[idx:idx+size].reshape(self.hidden2_size, self.output_size)
        idx += size
        
        # Bias3
        size = self.output_size
        self.bias3 = genome[idx:idx+size]
    
    def get_genome_size(self):
        """Retourne la taille totale du génome"""
        return (self.input_size * self.hidden1_size + self.hidden1_size +
                self.hidden1_size * self.hidden2_size + self.hidden2_size +
                self.hidden2_size * self.output_size + self.output_size)
    
    def copy(self):
        """Crée une copie profonde du réseau"""
        new_network = TeacherNetwork(
            self.input_size,
            self.hidden1_size,
            self.hidden2_size,
            self.output_size
        )
        new_network.set_genome(self.get_genome().copy())
        return new_network
    
    def save(self, filename):
        """Sauvegarde le modèle"""
        data = {
            'input_size': self.input_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'output_size': self.output_size,
            'genome': self.get_genome()
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Modèle sauvegardé dans {filename}")
    
    @staticmethod
    def load(filename):
        """Charge un modèle sauvegardé"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        network = TeacherNetwork(
            data['input_size'],
            data['hidden1_size'],
            data['hidden2_size'],
            data['output_size']
        )
        network.set_genome(data['genome'])
        print(f"Modèle chargé depuis {filename}")
        return network


class ImitationLearningInitializer:
    """
    Initialise un réseau en utilisant l'apprentissage par imitation
    depuis des données de conduite humaine
    """
    
    def __init__(self, network):
        """
        Args:
            network: Instance de TeacherNetwork à initialiser
        """
        self.network = network
    
    def load_human_data(self, npz_file):
        """
        Charge les données de conduite humaine
        
        Args:
            npz_file: Chemin vers le fichier .npz avec les données
        
        Returns:
            tuple: (inputs, targets) - Arrays numpy
        """
        import lzma
        
        print(f"Chargement des données depuis {npz_file}...")
        
        with lzma.open(npz_file, 'rb') as f:
            snapshots = pickle.load(f)
        
        print(f"Nombre de snapshots: {len(snapshots)}")
        
        inputs = []
        targets = []
        
        for snapshot in snapshots:
            # Extraire les features
            raycast_distances = snapshot.raycast_distances
            car_speed = snapshot.car_speed
            car_angle = snapshot.car_angle
            
            # Normaliser les entrées
            input_vector = self.network.normalize_inputs(
                raycast_distances, car_speed, car_angle
            )
            
            # Extraire les actions
            # current_controls = (Forward, Backward, Left, Right)
            target_vector = np.array(snapshot.current_controls, dtype=float)
            
            inputs.append(input_vector)
            targets.append(target_vector)
        
        return np.array(inputs), np.array(targets)
    
    def train_supervised(self, inputs, targets, epochs=100, learning_rate=0.01, batch_size=32):
        """
        Entraîne le réseau par descente de gradient avec backpropagation
        
        Args:
            inputs: Matrice des entrées (N, 17)
            targets: Matrice des sorties désirées (N, 4)
            epochs: Nombre d'époques
            learning_rate: Taux d'apprentissage
            batch_size: Taille des mini-batches
        """
        n_samples = len(inputs)
        n_batches = max(1, n_samples // batch_size)
        
        print(f"\nEntraînement par imitation learning:")
        print(f"  Échantillons: {n_samples}")
        print(f"  Époques: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(n_samples)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]
            
            total_loss = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_inputs = inputs_shuffled[start_idx:end_idx]
                batch_targets = targets_shuffled[start_idx:end_idx]
                
                # Accumuler les gradients pour le batch
                grad_w1 = np.zeros_like(self.network.weights1)
                grad_b1 = np.zeros_like(self.network.bias1)
                grad_w2 = np.zeros_like(self.network.weights2)
                grad_b2 = np.zeros_like(self.network.bias2)
                grad_w3 = np.zeros_like(self.network.weights3)
                grad_b3 = np.zeros_like(self.network.bias3)
                
                batch_loss = 0
                
                for i in range(len(batch_inputs)):
                    # Forward pass
                    input_data = batch_inputs[i]
                    target = batch_targets[i]
                    
                    # Layer 1
                    z1 = np.dot(input_data, self.network.weights1) + self.network.bias1
                    a1 = self.network.relu(z1)
                    
                    # Layer 2
                    z2 = np.dot(a1, self.network.weights2) + self.network.bias2
                    a2 = self.network.relu(z2)
                    
                    # Output layer
                    z3 = np.dot(a2, self.network.weights3) + self.network.bias3
                    output = self.network.sigmoid(z3)
                    
                    # Loss (MSE)
                    loss = np.mean((output - target) ** 2)
                    batch_loss += loss
                    
                    # Backpropagation
                    # Output layer
                    delta3 = (output - target) * output * (1 - output)  # Dérivée sigmoid
                    grad_w3 += np.outer(a2, delta3)
                    grad_b3 += delta3
                    
                    # Hidden layer 2
                    delta2 = np.dot(delta3, self.network.weights3.T)
                    delta2[z2 <= 0] = 0  # Dérivée ReLU
                    grad_w2 += np.outer(a1, delta2)
                    grad_b2 += delta2
                    
                    # Hidden layer 1
                    delta1 = np.dot(delta2, self.network.weights2.T)
                    delta1[z1 <= 0] = 0  # Dérivée ReLU
                    grad_w1 += np.outer(input_data, delta1)
                    grad_b1 += delta1
                
                # Moyenne des gradients sur le batch
                batch_len = len(batch_inputs)
                grad_w1 /= batch_len
                grad_b1 /= batch_len
                grad_w2 /= batch_len
                grad_b2 /= batch_len
                grad_w3 /= batch_len
                grad_b3 /= batch_len
                
                # Mise à jour des poids
                self.network.weights1 -= learning_rate * grad_w1
                self.network.bias1 -= learning_rate * grad_b1
                self.network.weights2 -= learning_rate * grad_w2
                self.network.bias2 -= learning_rate * grad_b2
                self.network.weights3 -= learning_rate * grad_w3
                self.network.bias3 -= learning_rate * grad_b3
                
                total_loss += batch_loss
            
            avg_loss = total_loss / n_samples
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping à l'époque {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"  Époque {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        print("✓ Entraînement terminé!")
        print(f"  Loss finale: {avg_loss:.6f}")
    
    def initialize_from_data(self, npz_file, epochs=50, learning_rate=0.01):
        """
        Pipeline complet: charge les données et initialise le réseau
        
        Args:
            npz_file: Fichier de données humaines
            epochs: Nombre d'époques d'entraînement
            learning_rate: Taux d'apprentissage
        """
        inputs, targets = self.load_human_data(npz_file)
        self.train_supervised(inputs, targets, epochs, learning_rate)
        return self.network


if __name__ == "__main__":
    # Test du modèle
    print("Test du modèle TeacherNetwork")
    print("=" * 60)
    
    # Créer un réseau
    network = TeacherNetwork()
    print(f"✓ Réseau créé")
    print(f"  Taille du génome: {network.get_genome_size()}")
    print(f"  Architecture: {network.input_size} -> {network.hidden1_size} -> {network.hidden2_size} -> {network.output_size}")
    
    # Test de prédiction
    dummy_raycasts = [50.0] * 15  # 15 raycasts à 50 unités
    dummy_speed = 25.0
    dummy_angle = 45.0
    
    actions = network.predict(dummy_raycasts, dummy_speed, dummy_angle)
    print(f"\n✓ Prédiction test:")
    print(f"  Forward: {actions[0]}, Backward: {actions[1]}, Left: {actions[2]}, Right: {actions[3]}")
    
    # Test de sauvegarde/chargement
    network.save("test_model.pkl")
    loaded_network = TeacherNetwork.load("test_model.pkl")
    print(f"\n✓ Sauvegarde/chargement OK")
    
    print("\n" + "=" * 60)
    print("Tous les tests passés! ✓")