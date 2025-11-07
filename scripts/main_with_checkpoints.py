"""
Main avec système de checkpoints intégré
Affiche les checkpoints, les temps intermédiaires et permet le contrôle à distance
"""

from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask, request, jsonify
from threading import Thread
from ursina import *
import sys
import os

# Ajouter le répertoire courant au path pour importer checkpoint_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rallyrobopilot.checkpoint_system import CheckpointManager
except ImportError:
    print("ERREUR: checkpoint_system.py doit être dans le même dossier que ce fichier!")
    sys.exit(1)


class CheckpointDisplay(Entity):
    """Affichage en temps réel des informations de checkpoints"""
    
    def __init__(self, checkpoint_manager):
        super().__init__(parent=camera.ui, eternal=True)
        self.checkpoint_manager = checkpoint_manager
        
        # Checkpoint actuel
        self.current_checkpoint_text = Text(
            parent=self,
            text="Checkpoint: 0/0",
            position=(0.50, 0.35),
            scale=1.3,
            origin=(0, 0),
            color=color.yellow
        )
        
        # Temps écoulé
        self.time_text = Text(
            parent=self,
            text="Temps: 0.0s",
            position=(0.50, 0.30),
            scale=1.3,
            origin=(0, 0),
            color=color.white
        )
        
        # Dernier split time
        self.split_time_text = Text(
            parent=self,
            text="Dernier split: -",
            position=(0.50, 0.25),
            scale=1.2,
            origin=(0, 0),
            color=color.lime
        )
        
        # Tours complétés
        self.lap_text = Text(
            parent=self,
            text="Tours: 0",
            position=(0.50, 0.20),
            scale=1.3,
            origin=(0, 0),
            color=color.orange
        )
        
        # Meilleur temps
        self.best_lap_text = Text(
            parent=self,
            text="Meilleur tour: -",
            position=(0.50, 0.15),
            scale=1.2,
            origin=(0, 0),
            color=color.gold
        )
        
        # Message temporaire
        self.temp_message = Text(
            parent=self,
            text="",
            position=(0, 0.45),
            scale=2.0,
            origin=(0, 0),
            color=color.green,
            visible=False
        )
        
        self.last_split_time = 0
        
    def update(self):
        """Met à jour l'affichage"""
        if not self.checkpoint_manager.start_time:
            return
        
        metrics = self.checkpoint_manager.get_detailed_fitness()
        
        # Checkpoint actuel
        current = metrics['current_checkpoint_index']
        total = len(self.checkpoint_manager.checkpoints)
        self.current_checkpoint_text.text = f"Checkpoint: {current}/{total}"
        
        # Temps écoulé
        elapsed = metrics['elapsed_time']
        self.time_text.text = f"Temps: {elapsed:.2f}s"
        
        # Dernier split time
        if metrics['checkpoint_times']:
            last_split = metrics['checkpoint_times'][-1]['split_time']
            if last_split != self.last_split_time:
                self.split_time_text.text = f"Dernier split: {last_split:.2f}s"
                self.split_time_text.color = color.lime
                self.last_split_time = last_split
        else:
            self.split_time_text.text = "Dernier split: -"
        
        # Tours
        lap_count = metrics['lap_count']
        self.lap_text.text = f"Tours: {lap_count}"
        
        # Meilleur temps de tour
        if metrics['best_lap_time']:
            self.best_lap_text.text = f"Meilleur tour: {metrics['best_lap_time']:.2f}s"
        else:
            self.best_lap_text.text = "Meilleur tour: -"
    
    def show_message(self, message, duration=2, message_color=color.green):
        """Affiche un message temporaire"""
        self.temp_message.text = message
        self.temp_message.color = message_color
        self.temp_message.visible = True
        invoke(self.hide_message, delay=duration)
    
    def hide_message(self):
        """Cache le message temporaire"""
        self.temp_message.visible = False


class MainGameController(Entity):
    """Contrôleur principal du jeu avec checkpoints"""
    
    def __init__(self, car, track_name, remote_controller):
        super().__init__()
        self.car = car
        self.track_name = track_name
        self.remote_controller = remote_controller
        
        # Créer le gestionnaire de checkpoints
        self.checkpoint_manager = CheckpointManager(car, track_name)
        
        # Charger les checkpoints
        self._load_checkpoints()
        
        # Créer l'affichage
        #self.display = CheckpointDisplay(self.checkpoint_manager)
        
        # Configurer les callbacks
        self.checkpoint_manager.on_checkpoint_passed = self._on_checkpoint_passed
        self.checkpoint_manager.on_lap_completed = self._on_lap_completed
        
        # État
        self.checkpoints_visible = False
        self.timer_started = False
        
        print("\n" + "="*60)
        print("SYSTÈME DE CHECKPOINTS ACTIVÉ")
        print("="*60)
        print(f"Circuit: {track_name}")
        print(f"Checkpoints chargés: {len(self.checkpoint_manager.checkpoints)}")
        print("\nCommandes:")
        print("  [V] - Toggle visibilité des checkpoints")
        print("  [R] - Reset voiture et checkpoints")
        print("  [T] - Démarrer/Redémarrer le chronomètre")
        print("="*60 + "\n")
    
    def _load_checkpoints(self):
        """Charge les checkpoints pour le circuit"""
        import json
        from pathlib import Path
        
        # Essayer de charger depuis un fichier JSON
        checkpoint_file = f"checkpoints_{self.track_name}.json"
        
        if os.path.exists(checkpoint_file):
            print(f"Chargement des checkpoints depuis {checkpoint_file}")
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)
                    self.checkpoint_manager.create_checkpoints_manually(data["checkpoints"])
                print(f"✓ {len(self.checkpoint_manager.checkpoints)} checkpoints chargés")
                return
            except Exception as e:
                print(f"Erreur lors du chargement: {e}")
        
        # Essayer de charger depuis les metadata du circuit
        try:
            self.checkpoint_manager.load_checkpoints_from_metadata()
            if len(self.checkpoint_manager.checkpoints) > 0:
                print(f"✓ {len(self.checkpoint_manager.checkpoints)} checkpoints chargés depuis metadata")
                return
        except Exception as e:
            print(f"Pas de checkpoints dans metadata: {e}")
        
        # Rendre les checkpoints visibles
        self.checkpoint_manager.set_checkpoints_visible(False)
    
    def input(self, key):
        """Gestion des entrées clavier"""
        
        # Toggle visibilité des checkpoints
        if key == 'v':
            self.checkpoints_visible = not self.checkpoints_visible
            self.checkpoint_manager.set_checkpoints_visible(self.checkpoints_visible)
            status = "VISIBLE" if self.checkpoints_visible else "CACHÉ"
            #self.display.show_message(f"Checkpoints: {status}", duration=1.5)
            print(f"Checkpoints: {status}")
        
        # Reset
        if key == 'r':
            self.car.reset_car()
            self.checkpoint_manager.reset()
            self.timer_started = False
            #self.display.show_message("RESET", duration=1.5, message_color=color.orange)
            print("Reset effectué")
        
        # Démarrer le chronomètre
        if key == 't':
            self.checkpoint_manager.start_timing()
            self.timer_started = True
            #self.display.show_message("CHRONO DÉMARRÉ", duration=1.5, message_color=color.cyan)
            print("Chronomètre démarré")
    
    def update(self):
        """Mise à jour du contrôleur"""
        # Démarrer automatiquement le chrono au premier mouvement
        if not self.timer_started and (held_keys['w'] or held_keys['s'] or held_keys['a'] or held_keys['d']):
            self.checkpoint_manager.start_timing()
            self.timer_started = True
            #self.display.show_message("GO!", duration=1.5, message_color=color.lime)
    
    def _on_checkpoint_passed(self, checkpoint_id, split_time, total_time):
        """Callback quand un checkpoint est franchi"""
        print(f"✓ Checkpoint {checkpoint_id + 1} franchi | Split: {split_time:.2f}s | Total: {total_time:.2f}s")
        
        # Message visuel
        '''self.display.show_message(
            f"CHECKPOINT {checkpoint_id + 1}",
            duration=1.0,
            message_color=color.lime
        )'''
        
        # Changement de couleur temporaire du checkpoint
        cp = self.checkpoint_manager.checkpoints[checkpoint_id]
        original_color = cp.color
        cp.color = color.rgba(255, 255, 0, 200)  # Jaune vif
        
        def reset_color():
            cp.color = original_color
        
        invoke(reset_color, delay=0.5)
    
    def _on_lap_completed(self, lap_time, lap_number):
        """Callback quand un tour est complété"""
        print(f"\n{'='*60}")
        print(f"🏁 TOUR {lap_number} TERMINÉ EN {lap_time:.2f}s! 🏁")
        print(f"{'='*60}\n")
        
        # Message visuel spectaculaire
        '''self.display.show_message(
            f"TOUR COMPLET! {lap_time:.2f}s",
            duration=3.0,
            message_color=color.gold
        )'''
        
        # Mettre à jour le meilleur temps dans l'affichage
        metrics = self.checkpoint_manager.get_detailed_fitness()
        if metrics['best_lap_time'] == lap_time:
            print(f"⭐ NOUVEAU MEILLEUR TEMPS! ⭐")
            #self.display.best_lap_text.color = color.gold
            
            # Reset de la couleur après un délai
            '''def reset_best_color():
                self.display.best_lap_text.color = color.white
            invoke(reset_best_color, delay=3)'''


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Setup Flask
    flask_app = Flask(__name__)
    flask_thread = Thread(target=flask_app.run, kwargs={'host': "0.0.0.0", 'port': 5000})
    print("Flask server running on port 5000")
    flask_thread.start()
    
    # Nom du circuit à utiliser
    TRACK_NAME = "SimpleTrack"
    # TRACK_NAME = "VisualTrack/track_circuit2_metadata.json"  # Décommentez pour utiliser un autre circuit
    
    # Préparer le jeu
    app, car = prepare_game_app(TRACK_NAME)
    
    # Remote controller
    remote_controller = RemoteController(car=car, connection_port=7654, flask_app=flask_app)
    
    # Contrôleur principal avec checkpoints
    game_controller = MainGameController(car, TRACK_NAME, remote_controller)
    
    # Lancer le jeu
    app.run()
