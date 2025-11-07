"""
Outil de placement interactif de checkpoints
Permet de placer visuellement les checkpoints sur un circuit en conduisant
"""

from rallyrobopilot import prepare_game_app
from rallyrobopilot.checkpoint_system import CheckpointManager, Checkpoint
from ursina import *
import json


class CheckpointPlacementTool(Entity):
    """Outil pour placer des checkpoints interactivement"""
    
    def __init__(self, car, track_name):
        super().__init__()
        self.car = car
        self.track_name = track_name
        self.checkpoint_manager = CheckpointManager(car, track_name)
        
        self.placement_mode = False
        self.preview_checkpoint = None
        
        # Configuration par défaut des checkpoints
        self.checkpoint_width = 8
        self.checkpoint_height = 10
        self.checkpoint_depth = 1
        
        # Créer le checkpoint de prévisualisation
        self._create_preview_checkpoint()
        
        # Interface
        self._create_ui()
    
    def _create_preview_checkpoint(self):
        """Crée un checkpoint de prévisualisation"""
        self.preview_checkpoint = Entity(
            model='cube',
            color=color.rgba(255, 255, 0, 100),  # Jaune transparent
            scale=(self.checkpoint_width, self.checkpoint_height, self.checkpoint_depth),
            visible=False
        )
    
    def _create_ui(self):
        """Crée l'interface utilisateur"""
        self.ui_panel = Entity(parent=camera.ui, eternal=True)
        
        # Instructions
        self.instructions = Text(
            parent=self.ui_panel,
            text="[P] Mode placement ON/OFF\n[ESPACE] Placer checkpoint\n[Z] Supprimer dernier\n[S] Sauvegarder\n[V] Visibilité ON/OFF",
            position=(-0.85, 0.45),
            scale=1.2,
            origin=(0, 0)
        )
        
        # Compteur
        self.counter_text = Text(
            parent=self.ui_panel,
            text="Checkpoints: 0",
            position=(-0.85, 0.35),
            scale=1.5,
            origin=(0, 0)
        )
        
        # État
        self.status_text = Text(
            parent=self.ui_panel,
            text="Mode: Navigation",
            position=(-0.85, 0.30),
            scale=1.5,
            origin=(0, 0),
            color=color.white
        )
    
    def update(self):
        """Mise à jour du placement"""
        if self.placement_mode and self.preview_checkpoint:
            # Positionne le checkpoint devant la voiture
            offset_distance = 5  # Distance devant la voiture
            self.preview_checkpoint.position = (
                self.car.world_position + 
                self.car.forward * offset_distance
            )
            self.preview_checkpoint.rotation_y = self.car.rotation_y
            
    def input(self, key):
        """Gestion des entrées clavier"""
        
        # Toggle placement mode
        if key == 'p':
            self.placement_mode = not self.placement_mode
            self.preview_checkpoint.visible = self.placement_mode
            
            if self.placement_mode:
                self.status_text.text = "Mode: PLACEMENT ACTIF"
                self.status_text.color = color.green
            else:
                self.status_text.text = "Mode: Navigation"
                self.status_text.color = color.white
        
        # Placer un checkpoint
        if key == 'space' and self.placement_mode:
            self._place_checkpoint()
        
        # Supprimer le dernier checkpoint
        if key == 'z' and len(self.checkpoint_manager.checkpoints) > 0:
            self._remove_last_checkpoint()
        
        # Sauvegarder
        if key == 'm':
            self._save_checkpoints()
        
        # Toggle visibilité
        if key == 'v':
            visible = not self.checkpoint_manager.checkpoints[0].visible if self.checkpoint_manager.checkpoints else True
            self.checkpoint_manager.set_checkpoints_visible(visible)
        
        # Ajuster la taille (en mode placement)
        if self.placement_mode:
            if key == 'y' or key == '=':
                self.checkpoint_width += 1
                self.preview_checkpoint.scale_x = self.checkpoint_width
            elif key == 'x':
                self.checkpoint_width = max(3, self.checkpoint_width - 1)
                self.preview_checkpoint.scale_x = self.checkpoint_width
    
    def _place_checkpoint(self):
        """Place un checkpoint à la position actuelle"""
        checkpoint = Checkpoint(
            position=self.preview_checkpoint.world_position,
            rotation=(0, self.car.rotation_y, 0),
            scale=(self.checkpoint_width, self.checkpoint_height, self.checkpoint_depth),
            checkpoint_id=len(self.checkpoint_manager.checkpoints)
        )
        checkpoint.visible = True
        checkpoint.color = color.rgba(0, 255, 0, 100)  # Vert transparent
        
        self.checkpoint_manager.checkpoints.append(checkpoint)
        self._update_counter()
        
        # Feedback visuel
        self.status_text.text = f"Checkpoint #{checkpoint.checkpoint_id} placé!"
        invoke(self._reset_status_text, delay=1)
    
    def _remove_last_checkpoint(self):
        """Supprime le dernier checkpoint placé"""
        last_checkpoint = self.checkpoint_manager.checkpoints.pop()
        destroy(last_checkpoint)
        self._update_counter()
        
        self.status_text.text = "Dernier checkpoint supprimé"
        invoke(self._reset_status_text, delay=1)
    
    def _update_counter(self):
        """Met à jour le compteur de checkpoints"""
        self.counter_text.text = f"Checkpoints: {len(self.checkpoint_manager.checkpoints)}"
    
    def _reset_status_text(self):
        """Réinitialise le texte de statut"""
        if self.placement_mode:
            self.status_text.text = "Mode: PLACEMENT ACTIF"
        else:
            self.status_text.text = "Mode: Navigation"
    
    def _save_checkpoints(self):
        """Sauvegarde les checkpoints dans un fichier JSON"""
        if len(self.checkpoint_manager.checkpoints) == 0:
            print("Aucun checkpoint à sauvegarder!")
            return
        
        filename = f"checkpoints_{self.track_name}.json"
        self.checkpoint_manager.export_checkpoint_positions(filename)
        
        self.status_text.text = f"Sauvegardé: {filename}"
        self.status_text.color = color.cyan
        invoke(self._reset_status_text, delay=2)
        
        # Affiche aussi un résumé
        print(f"\n{'='*50}")
        print(f"CHECKPOINTS SAUVEGARDÉS POUR {self.track_name}")
        print(f"{'='*50}")
        print(f"Nombre total: {len(self.checkpoint_manager.checkpoints)}")
        print(f"Fichier: {filename}")
        print(f"{'='*50}\n")


def main():
    """Lance l'outil de placement de checkpoints"""
    # Prépare le jeu
    app, car = prepare_game_app("SimpleTrack")
    
    # Crée l'outil de placement
    placement_tool = CheckpointPlacementTool(car, "SimpleTrack")
    
    # Instructions de démarrage
    print("\n" + "="*60)
    print("OUTIL DE PLACEMENT DE CHECKPOINTS")
    print("="*60)
    print("\nCONTRÔLES:")
    print("  [P]       - Activer/Désactiver le mode placement")
    print("  [ESPACE]  - Placer un checkpoint")
    print("  [Z]       - Supprimer le dernier checkpoint")
    print("  [M]       - Sauvegarder les checkpoints")
    print("  [V]       - Toggle visibilité des checkpoints")
    print("  [Y/X]     - Ajuster la largeur (en mode placement)")
    print("\nCONSEILS:")
    print("  • Placez un checkpoint tous les 2-3 secondes de conduite")
    print("  • Assurez-vous que tous les checkpoints couvrent toute la piste")
    print("  • Le dernier checkpoint doit être proche de la ligne d'arrivée")
    print("="*60 + "\n")
    
    app.run()


if __name__ == "__main__":
    main()
