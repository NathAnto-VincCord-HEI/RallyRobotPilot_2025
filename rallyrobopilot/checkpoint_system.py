"""
Système de checkpoints pour circuits de course
Permet de mesurer la progression précise et les temps intermédiaires
"""

from ursina import *
import time
import json
from pathlib import Path


class Checkpoint(Entity):
    """Un checkpoint individuel sur le circuit"""
    
    def __init__(self, position, rotation, scale, checkpoint_id, **kwargs):
        super().__init__(
            model='cube',
            position=position,
            rotation=rotation,
            scale=scale,
            color=color.rgba(0, 255, 0, 50),  # Vert transparent
            visible=False,  # Invisible en production, visible pour debug
            collider=None,
            **kwargs
        )
        self.checkpoint_id = checkpoint_id
        self.passed = False
        
    def reset(self):
        """Réinitialise l'état du checkpoint"""
        self.passed = False


class CheckpointManager(Entity):
    """Gestionnaire de tous les checkpoints d'un circuit"""
    
    def __init__(self, car, track_name, **kwargs):
        super().__init__(**kwargs)
        self.car = car
        self.track_name = track_name
        
        # Liste ordonnée des checkpoints
        self.checkpoints = []
        
        # Progression actuelle
        self.current_checkpoint_index = 0
        self.lap_count = 0
        self.total_checkpoints_passed = 0
        
        # Historique des temps
        self.checkpoint_times = []  # Liste des timestamps de passage
        self.lap_times = []  # Liste des temps de tours complets
        
        # Temps de départ
        self.start_time = None
        self.last_checkpoint_time = None
        
        # Callbacks optionnels
        self.on_checkpoint_passed = None  # Callback(checkpoint_id, split_time)
        self.on_lap_completed = None  # Callback(lap_time, lap_number)
        
        # Statistiques
        self.best_lap_time = float('inf')
        self.best_checkpoint_times = {}  # {checkpoint_id: best_time}
        
    def load_checkpoints_from_metadata(self):
        """Charge les checkpoints depuis le fichier metadata du circuit"""
        root_dir = Path(__file__).resolve().parent.parent
        
        if ".json" not in self.track_name:
            metadata_path = root_dir / f"assets/{self.track_name}/track_metadata.json"
        else:
            metadata_path = root_dir / f"assets/{self.track_name}"
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        if "checkpoints" in metadata:
            for i, cp_data in enumerate(metadata["checkpoints"]):
                checkpoint = Checkpoint(
                    position=tuple(cp_data["position"]),
                    rotation=tuple(cp_data["rotation"]),
                    scale=tuple(cp_data["scale"]),
                    checkpoint_id=i
                )
                self.checkpoints.append(checkpoint)
        else:
            print(f"Attention : Aucun checkpoint défini dans {metadata_path}")
    
    def create_checkpoints_manually(self, checkpoint_data):
        """
        Crée des checkpoints manuellement
        
        Args:
            checkpoint_data: Liste de dictionnaires contenant position, rotation, scale
            Exemple:
            [
                {"position": (10, 0, 5), "rotation": (0, 45, 0), "scale": (5, 10, 1)},
                {"position": (20, 0, 15), "rotation": (0, 90, 0), "scale": (5, 10, 1)},
                ...
            ]
        """
        for i, cp_data in enumerate(checkpoint_data):
            checkpoint = Checkpoint(
                position=cp_data["position"],
                rotation=cp_data.get("rotation", (0, 0, 0)),
                scale=cp_data.get("scale", (5, 10, 1)),
                checkpoint_id=i
            )
            self.checkpoints.append(checkpoint)
    
    def generate_checkpoints_from_spline(self, spline_points, checkpoint_spacing=10, 
                                        checkpoint_width=8, checkpoint_height=10):
        """
        Génère automatiquement des checkpoints le long d'une spline/courbe
        
        Args:
            spline_points: Liste de points (x, y, z) définissant la trajectoire
            checkpoint_spacing: Espacement approximatif entre checkpoints
            checkpoint_width: Largeur du checkpoint
            checkpoint_height: Hauteur du checkpoint
        """
        if len(spline_points) < 2:
            return
        
        checkpoint_id = 0
        accumulated_distance = 0
        
        for i in range(len(spline_points) - 1):
            p1 = Vec3(*spline_points[i])
            p2 = Vec3(*spline_points[i + 1])
            
            segment_length = distance(p1, p2)
            accumulated_distance += segment_length
            
            if accumulated_distance >= checkpoint_spacing:
                # Position au milieu du segment
                position = (p1 + p2) / 2
                
                # Rotation pour être perpendiculaire à la direction
                direction = (p2 - p1).normalized()
                angle = math.degrees(math.atan2(direction.x, direction.z))
                
                checkpoint = Checkpoint(
                    position=position,
                    rotation=(0, angle, 0),
                    scale=(checkpoint_width, checkpoint_height, 1),
                    checkpoint_id=checkpoint_id
                )
                self.checkpoints.append(checkpoint)
                checkpoint_id += 1
                accumulated_distance = 0
    
    def start_timing(self):
        """Démarre le chronométrage"""
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.checkpoint_times = []
        self.current_checkpoint_index = 0
        self.total_checkpoints_passed = 0
        
        # Reset tous les checkpoints
        for cp in self.checkpoints:
            cp.reset()
    
    def update(self):
        """Vérifie si la voiture passe par un checkpoint"""
        if not self.start_time or len(self.checkpoints) == 0:
            return
        
        # Vérifie le prochain checkpoint attendu
        next_checkpoint = self.checkpoints[self.current_checkpoint_index]
        
        if not next_checkpoint.passed and self.car.simple_intersects(next_checkpoint):
            self._on_checkpoint_reached(next_checkpoint)
    
    def _on_checkpoint_reached(self, checkpoint):
        """Appelé quand un checkpoint est franchi"""
        current_time = time.time()
        split_time = current_time - self.last_checkpoint_time
        total_time = current_time - self.start_time
        
        # Marque le checkpoint comme passé
        checkpoint.passed = True
        self.checkpoint_times.append({
            'checkpoint_id': checkpoint.checkpoint_id,
            'split_time': split_time,
            'total_time': total_time,
            'timestamp': current_time
        })
        
        # Met à jour le meilleur temps pour ce checkpoint
        if checkpoint.checkpoint_id not in self.best_checkpoint_times:
            self.best_checkpoint_times[checkpoint.checkpoint_id] = split_time
        else:
            self.best_checkpoint_times[checkpoint.checkpoint_id] = min(
                self.best_checkpoint_times[checkpoint.checkpoint_id],
                split_time
            )
        
        self.last_checkpoint_time = current_time
        self.total_checkpoints_passed += 1
        
        # Callback utilisateur
        if self.on_checkpoint_passed:
            self.on_checkpoint_passed(checkpoint.checkpoint_id, split_time, total_time)
        
        # Passe au checkpoint suivant
        self.current_checkpoint_index += 1
        
        # Vérifie si c'est la fin d'un tour
        if self.current_checkpoint_index >= len(self.checkpoints):
            self._on_lap_completed()
    
    def _on_lap_completed(self):
        """Appelé quand un tour complet est terminé"""
        lap_time = time.time() - self.start_time
        self.lap_count += 1
        self.lap_times.append(lap_time)
        
        # Met à jour le meilleur temps
        if lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
        
        # Callback utilisateur
        if self.on_lap_completed:
            self.on_lap_completed(lap_time, self.lap_count)
        
        # Réinitialise pour le prochain tour
        self.current_checkpoint_index = 0
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        
        for cp in self.checkpoints:
            cp.reset()
    
    def get_progress_percentage(self):
        """Retourne la progression en pourcentage (0-100)"""
        if len(self.checkpoints) == 0:
            return 0.0
        return (self.current_checkpoint_index / len(self.checkpoints)) * 100
    
    def get_fitness_score(self, time_weight=1.0, progress_weight=2.0):
        """
        Calcule un score de fitness pour l'algorithme génétique
        
        Args:
            time_weight: Poids du temps (plus bas = meilleur)
            progress_weight: Poids de la progression (plus haut = meilleur)
        
        Returns:
            float: Score de fitness (plus haut = meilleur)
        """
        if not self.start_time:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        progress = self.total_checkpoints_passed
        
        # Score basé sur progression / temps
        # Plus de checkpoints en moins de temps = meilleur score
        if elapsed_time > 0:
            fitness = (progress * progress_weight) - (elapsed_time * time_weight)
        else:
            fitness = progress * progress_weight
        
        return max(0, fitness)
    
    def get_detailed_fitness(self):
        """
        Retourne des métriques détaillées pour l'analyse
        
        Returns:
            dict: Dictionnaire avec toutes les métriques utiles
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_checkpoints_passed': self.total_checkpoints_passed,
            'current_checkpoint_index': self.current_checkpoint_index,
            'progress_percentage': self.get_progress_percentage(),
            'elapsed_time': elapsed_time,
            'lap_count': self.lap_count,
            'checkpoint_times': self.checkpoint_times.copy(),
            'lap_times': self.lap_times.copy(),
            'best_lap_time': self.best_lap_time if self.best_lap_time != float('inf') else None,
            'average_split_time': sum(ct['split_time'] for ct in self.checkpoint_times) / len(self.checkpoint_times) if self.checkpoint_times else 0,
            'fitness_score': self.get_fitness_score()
        }
    
    def reset(self):
        """Réinitialise complètement le gestionnaire"""
        self.current_checkpoint_index = 0
        self.lap_count = 0
        self.total_checkpoints_passed = 0
        self.checkpoint_times = []
        self.start_time = None
        self.last_checkpoint_time = None
        
        for cp in self.checkpoints:
            cp.reset()
    
    def set_checkpoints_visible(self, visible):
        """Active/désactive la visibilité des checkpoints (pour debug)"""
        for cp in self.checkpoints:
            cp.visible = visible
    
    def export_checkpoint_positions(self, filename):
        """Exporte les positions des checkpoints pour les sauvegarder"""
        data = {
            'track_name': self.track_name,
            'checkpoints': [
                {
                    'position': list(cp.position),
                    'rotation': list(cp.rotation),
                    'scale': list(cp.scale)
                }
                for cp in self.checkpoints
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Checkpoints exportés vers {filename}")


# Fonction utilitaire pour créer des checkpoints facilement
def create_checkpoint_manager_for_track(car, track_name, auto_generate=False):
    """
    Crée et configure un gestionnaire de checkpoints pour un circuit
    
    Args:
        car: Instance de Car
        track_name: Nom du circuit
        auto_generate: Si True, tente de charger depuis metadata ou génère automatiquement
    
    Returns:
        CheckpointManager: Gestionnaire configuré
    """
    manager = CheckpointManager(car, track_name)
    
    if auto_generate:
        try:
            manager.load_checkpoints_from_metadata()
        except:
            print(f"Impossible de charger les checkpoints pour {track_name}")
    
    return manager
