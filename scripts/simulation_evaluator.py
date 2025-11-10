"""
Système d'évaluation pour tester les réseaux dans le simulateur
Permet d'évaluer la fitness de chaque individu en le faisant conduire
"""

import numpy as np
import time
import requests
import json
from pathlib import Path


class SimulationEvaluator:
    """
    Évalue un réseau en le faisant conduire dans le simulateur
    Utilise l'API HTTP Flask pour communiquer avec le jeu
    """
    
    def __init__(self, 
                 game_host="127.0.0.1",
                 game_port=5000,
                 max_simulation_time=60,
                 checkpoint_data_file=None):
        """
        Args:
            game_host: Adresse du serveur du jeu
            game_port: Port du serveur Flask
            max_simulation_time: Temps maximum de simulation (secondes)
            checkpoint_data_file: Fichier JSON avec les positions des checkpoints
        """
        self.base_url = f"http://{game_host}:{game_port}"
        self.max_simulation_time = max_simulation_time
        
        # Charger les données de checkpoints si disponibles
        self.checkpoint_data = None
        if checkpoint_data_file and Path(checkpoint_data_file).exists():
            with open(checkpoint_data_file, 'r') as f:
                self.checkpoint_data = json.load(f)
            print(f"✓ Checkpoints chargés: {len(self.checkpoint_data['checkpoints'])} points")
    
    def reset_simulation(self):
        """Réinitialise la voiture dans le jeu"""
        try:
            response = requests.post(
                f"{self.base_url}/command",
                json={"command": "release all;"},
                timeout=2
            )
            
            time.sleep(0.1)
            
            response = requests.post(
                f"{self.base_url}/command",
                json={"command": "reset;"},
                timeout=2
            )
            
            time.sleep(0.5)  # Laisser le temps au reset
            
            return response.status_code == 200
        except Exception as e:
            print(f"Erreur lors du reset: {e}")
            return False
    
    def get_sensing_data(self):
        """Récupère les données de capteurs depuis le jeu"""
        try:
            response = requests.get(
                f"{self.base_url}/sensing",
                timeout=2
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Erreur lors de la lecture des capteurs: {e}")
            return None
    
    def send_controls(self, forward, backward, left, right):
        """
        Envoie les commandes de contrôle au jeu
        
        Args:
            forward, backward, left, right: Booléens pour chaque action
        """
        try:
            # Construire les commandes
            commands = []
            
            # D'abord, libérer toutes les touches
            commands.append("release all")
            
            # Puis, activer les touches nécessaires
            if forward:
                commands.append("push forward")
            if backward:
                commands.append("push back")
            if left:
                commands.append("push left")
            if right:
                commands.append("push right")
            
            # Envoyer les commandes
            for cmd in commands:
                response = requests.post(
                    f"{self.base_url}/command",
                    json={"command": cmd + ";"},
                    timeout=1
                )
            
            return True
        except Exception as e:
            print(f"Erreur lors de l'envoi des contrôles: {e}")
            return False
    
    def evaluate_network(self, network, verbose=False):
        """
        Évalue un réseau en simulation
        
        Args:
            network: TeacherNetwork à évaluer
            verbose: Afficher les détails
        
        Returns:
            float: Score de fitness
        """
        if verbose:
            print("\n  Démarrage de l'évaluation...")
        
        # Reset de la simulation
        if not self.reset_simulation():
            print("  ✗ Échec du reset")
            return 0.0
        
        # Métriques de performance
        total_distance = 0.0
        max_speed_reached = 0.0
        frames_survived = 0
        checkpoints_passed = 0
        collision_count = 0
        last_position = None
        stuck_counter = 0
        
        # Checkpoint tracking
        checkpoint_progress = [False] * (len(self.checkpoint_data['checkpoints']) if self.checkpoint_data else 0)
        
        start_time = time.time()
        last_update_time = start_time
        
        try:
            while (time.time() - start_time) < self.max_simulation_time:
                # Récupérer l'état actuel
                sensing_data = self.get_sensing_data()
                
                if sensing_data is None:
                    print("  ✗ Perte de connexion")
                    break
                
                # Extraire les données
                raycast_distances = [
                    sensing_data[f'raycast_distances {i}'] 
                    for i in range(15)
                ]
                car_speed = sensing_data['car_speed']
                car_angle = sensing_data['car_angle']
                car_pos = (
                    sensing_data['car_position x'],
                    sensing_data['car_position y'],
                    sensing_data['car_position z']
                )
                
                # Prédiction du réseau
                actions = network.predict(raycast_distances, car_speed, car_angle, threshold=0.5)
                forward, backward, left, right = actions
                
                # Envoyer les contrôles
                self.send_controls(forward, backward, left, right)
                
                # Calculer les métriques
                if last_position is not None:
                    distance = np.sqrt(
                        (car_pos[0] - last_position[0])**2 +
                        (car_pos[2] - last_position[2])**2
                    )
                    total_distance += distance
                    
                    # Détection de blocage
                    if distance < 0.1 and car_speed < 1.0:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0
                    
                    # Arrêter si bloqué trop longtemps
                    if stuck_counter > 50:  # ~5 secondes à 10 FPS
                        if verbose:
                            print("  ⚠ Voiture bloquée")
                        break
                
                last_position = car_pos
                max_speed_reached = max(max_speed_reached, abs(car_speed))
                frames_survived += 1
                
                # Vérifier les checkpoints si disponibles
                if self.checkpoint_data:
                    checkpoints_passed = self._check_checkpoints_passed(
                        car_pos, checkpoint_progress
                    )
                
                # Détecter les collisions (vitesse chute brutalement)
                if len(raycast_distances) > 0 and min(raycast_distances) < 1.0:
                    collision_count += 1
                
                # Contrôle de la fréquence (environ 10 Hz)
                elapsed = time.time() - last_update_time
                sleep_time = max(0, 0.1 - elapsed)
                time.sleep(sleep_time)
                last_update_time = time.time()
        
        except KeyboardInterrupt:
            print("\n  Évaluation interrompue par l'utilisateur")
        except Exception as e:
            print(f"  ✗ Erreur durant l'évaluation: {e}")
        
        # Calcul du score de fitness
        fitness = self._calculate_fitness(
            total_distance=total_distance,
            max_speed=max_speed_reached,
            frames_survived=frames_survived,
            checkpoints_passed=checkpoints_passed,
            collision_count=collision_count
        )
        
        if verbose:
            print(f"\n  Résultats de l'évaluation:")
            print(f"    Distance parcourue: {total_distance:.2f}")
            print(f"    Vitesse max: {max_speed_reached:.2f}")
            print(f"    Temps de survie: {frames_survived * 0.1:.1f}s")
            print(f"    Checkpoints: {checkpoints_passed}")
            print(f"    Collisions: {collision_count}")
            print(f"    FITNESS FINALE: {fitness:.2f}")
        
        # Libérer les touches avant de quitter
        self.send_controls(False, False, False, False)
        
        return fitness
    
    def _check_checkpoints_passed(self, car_pos, checkpoint_progress):
        """Vérifie quels checkpoints ont été franchis"""
        if not self.checkpoint_data:
            return 0
        
        passed_count = 0
        
        for i, checkpoint in enumerate(self.checkpoint_data['checkpoints']):
            if checkpoint_progress[i]:
                passed_count += 1
                continue
            
            # Vérifier si la voiture est dans le checkpoint
            cp_pos = checkpoint['position']
            cp_scale = checkpoint['scale']
            
            # AABB simple
            if (abs(car_pos[0] - cp_pos[0]) < cp_scale[0] and
                abs(car_pos[1] - cp_pos[1]) < cp_scale[1] and
                abs(car_pos[2] - cp_pos[2]) < cp_scale[2]):
                checkpoint_progress[i] = True
                passed_count += 1
        
        return passed_count
    
    def _calculate_fitness(self, total_distance, max_speed, frames_survived, 
                          checkpoints_passed, collision_count):
        """
        Calcule le score de fitness global
        
        Args:
            total_distance: Distance totale parcourue
            max_speed: Vitesse maximale atteinte
            frames_survived: Nombre de frames de survie
            checkpoints_passed: Nombre de checkpoints franchis
            collision_count: Nombre de collisions détectées
        
        Returns:
            float: Score de fitness (plus haut = meilleur)
        """
        # Pondérations des différents critères
        DISTANCE_WEIGHT = 1.5
        CHECKPOINT_WEIGHT = 150.0  
        SPEED_WEIGHT = 2.0
        SURVIVAL_WEIGHT = 0.1
        COLLISION_PENALTY = 10.0
        
        # Calcul du score
        fitness = (
            total_distance * DISTANCE_WEIGHT +
            checkpoints_passed * CHECKPOINT_WEIGHT +
            max_speed * SPEED_WEIGHT +
            frames_survived * SURVIVAL_WEIGHT -
            collision_count * COLLISION_PENALTY
        )
        
        # Bonus si tous les checkpoints sont passés
        if self.checkpoint_data and checkpoints_passed >= len(self.checkpoint_data['checkpoints']):
            fitness += 500  # Gros bonus pour compléter le tour
        
        return max(0, fitness)  # Jamais négatif


class BatchEvaluator:
    """
    Évalue plusieurs réseaux de manière séquentielle
    """
    
    def __init__(self, simulation_evaluator):
        """
        Args:
            simulation_evaluator: Instance de SimulationEvaluator
        """
        self.evaluator = simulation_evaluator
    
    def evaluate_population(self, population, verbose_interval=5):
        """
        Évalue toute une population
        
        Args:
            population: Liste de TeacherNetwork
            verbose_interval: Afficher les détails tous les N individus
        
        Returns:
            list: Scores de fitness
        """
        fitness_scores = []
        
        print(f"\n{'='*60}")
        print(f"ÉVALUATION DE {len(population)} INDIVIDUS")
        print(f"{'='*60}")
        
        for i, network in enumerate(population):
            print(f"\n[{i+1}/{len(population)}] Évaluation en cours...", end='', flush=True)
            
            verbose = (i % verbose_interval == 0)
            fitness = self.evaluator.evaluate_network(network, verbose=verbose)
            fitness_scores.append(fitness)
            
            if not verbose:
                print(f" Fitness: {fitness:.2f}")
            
            # Petite pause entre les évaluations
            time.sleep(0.5)
        
        print(f"\n{'='*60}")
        print(f"ÉVALUATION TERMINÉE")
        print(f"  Meilleur: {max(fitness_scores):.2f}")
        print(f"  Moyenne: {np.mean(fitness_scores):.2f}")
        print(f"  Pire: {min(fitness_scores):.2f}")
        print(f"{'='*60}\n")
        
        return fitness_scores


if __name__ == "__main__":
    # Test de l'évaluateur
    print("Test du SimulationEvaluator")
    print("=" * 60)
    print("⚠ ATTENTION: Le jeu doit être lancé avec main_with_checkpoints.py")
    print("=" * 60)
    
    input("\nAppuyez sur Entrée quand le jeu est prêt...")
    
    from teacher_model import TeacherNetwork
    
    # Créer un réseau de test
    network = TeacherNetwork()
    
    # Créer l'évaluateur
    evaluator = SimulationEvaluator(
        checkpoint_data_file="checkpoints_SimpleTrack.json"
    )
    
    # Tester une évaluation
    print("\nTest d'évaluation d'un réseau aléatoire...")
    fitness = evaluator.evaluate_network(network, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"Test terminé! Fitness obtenue: {fitness:.2f}")
    print(f"{'='*60}")
