"""
Script principal pour créer un modèle "professeur" optimal
Combine apprentissage par imitation et algorithme génétique
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

from scripts.teacher_model import TeacherNetwork, ImitationLearningInitializer
from scripts.genetic_algorithm import GeneticAlgorithm
from scripts.simulation_evaluator import SimulationEvaluator, BatchEvaluator


class HybridTrainingPipeline:
    """
    Pipeline complet pour créer un modèle professeur optimal
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.track_name = config['track_name']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Composants principaux
        self.base_network = None
        self.genetic_algorithm = None
        self.evaluator = None
        
        print(f"\n{'='*70}")
        print(f"PIPELINE D'ENTRAÎNEMENT HYBRIDE")
        print(f"{'='*70}")
        print(f"Circuit: {self.track_name}")
        print(f"Répertoire de sortie: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def phase1_imitation_learning(self):
        """
        Phase 1: Initialisation par apprentissage par imitation
        Utilise les données de conduite humaine
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: APPRENTISSAGE PAR IMITATION")
        print(f"{'#'*70}\n")
        
        human_data_file = self.config['human_data_file']
        
        if not Path(human_data_file).exists():
            print(f"⚠ ATTENTION: Fichier de données non trouvé: {human_data_file}")
            print(f"  → Utilisation d'une initialisation aléatoire à la place")
            
            self.base_network = TeacherNetwork()
            return self.base_network
        
        print(f"Fichier de données: {human_data_file}")
        
        # Créer le réseau
        self.base_network = TeacherNetwork(
            input_size=17,
            hidden1_size=self.config.get('hidden1_size', 32),
            hidden2_size=self.config.get('hidden2_size', 16),
            output_size=4
        )
        
        # Initialiser avec les données humaines
        print("\nEntraînement par imitation...")
        initializer = ImitationLearningInitializer(self.base_network)
        
        try:
            self.base_network = initializer.initialize_from_data(
                human_data_file,
                epochs=self.config.get('imitation_epochs', 50),
                learning_rate=self.config.get('imitation_lr', 0.01)
            )
            
            # Sauvegarder le modèle initial
            model_path = self.output_dir / f"model_phase1_imitation_{self.track_name}.pkl"
            self.base_network.save(str(model_path))
            
            print(f"\n✓ Phase 1 terminée!")
            print(f"  Modèle sauvegardé: {model_path}")
            
        except Exception as e:
            print(f"\n✗ Erreur lors de l'apprentissage par imitation: {e}")
            print(f"  → Utilisation d'une initialisation aléatoire")
            self.base_network = TeacherNetwork()
        
        return self.base_network
    
    def phase2_genetic_evolution(self):
        """
        Phase 2: Évolution génétique
        Optimise le modèle initial via algorithme génétique
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: ÉVOLUTION GÉNÉTIQUE")
        print(f"{'#'*70}\n")
        
        if self.base_network is None:
            print("✗ Erreur: Aucun modèle de base disponible")
            return None
        
        # Configurer l'algorithme génétique
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=self.config.get('population_size', 50),
            elite_size=self.config.get('elite_size', 5),
            mutation_rate=self.config.get('mutation_rate', 0.1),
            mutation_strength=self.config.get('mutation_strength', 0.3),
            crossover_rate=self.config.get('crossover_rate', 0.7)
        )
        
        # Initialiser la population
        print("Initialisation de la population...")
        self.genetic_algorithm.initialize_population_from_model(
            self.base_network,
            noise_level=self.config.get('init_noise_level', 0.2)
        )
        
        # Configurer l'évaluateur
        checkpoint_file = self.config.get('checkpoint_file', 
                                         f"checkpoints_{self.track_name}.json")
        
        self.evaluator = SimulationEvaluator(
            game_host=self.config.get('game_host', '127.0.0.1'),
            game_port=self.config.get('game_port', 5000),
            max_simulation_time=self.config.get('simulation_time', 60),
            checkpoint_data_file=checkpoint_file
        )
        
        batch_evaluator = BatchEvaluator(self.evaluator)
        
        # Boucle évolutionnaire
        n_generations = self.config.get('n_generations', 20)
        
        print(f"\nÉvolution sur {n_generations} générations...")
        print(f"{'='*70}\n")
        
        best_overall_fitness = -float('inf')
        best_overall_network = None
        
        try:
            for generation in range(n_generations):
                print(f"\n{'='*70}")
                print(f"GÉNÉRATION {generation + 1}/{n_generations}")
                print(f"{'='*70}")
                
                # Évaluer la population
                fitness_scores = batch_evaluator.evaluate_population(
                    self.genetic_algorithm.population,
                    verbose_interval=self.config.get('verbose_interval', 5)
                )
                
                # Mettre à jour les scores dans l'algorithme génétique
                self.genetic_algorithm.fitness_scores = fitness_scores
                self.genetic_algorithm.generation = generation
                
                # Statistiques
                best_network, best_fitness = self.genetic_algorithm.get_best_individual()
                
                if best_fitness > best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_overall_network = best_network.copy()
                    
                    # Sauvegarder le meilleur modèle
                    best_model_path = self.output_dir / f"model_best_{self.track_name}.pkl"
                    best_overall_network.save(str(best_model_path))
                    print(f"\n🏆 NOUVEAU MEILLEUR MODÈLE! Fitness: {best_fitness:.2f}")
                    print(f"   Sauvegardé: {best_model_path}")
                
                # Sauvegarder un checkpoint
                if (generation + 1) % self.config.get('checkpoint_interval', 5) == 0:
                    checkpoint_path = self.output_dir / f"ga_checkpoint_gen{generation+1}_{self.track_name}.pkl"
                    self.genetic_algorithm.save_checkpoint(str(checkpoint_path))
                
                # Créer la prochaine génération
                if generation < n_generations - 1:
                    self.genetic_algorithm.evolve()
                    
                    # Adaptation dynamique des paramètres
                    if self.config.get('adaptive_params', True):
                        self._adapt_parameters(generation, n_generations)
        
        except KeyboardInterrupt:
            print("\n\n⚠ Entraînement interrompu par l'utilisateur")
            print("Sauvegarde du meilleur modèle trouvé jusqu'ici...")
            
            if best_overall_network is not None:
                interrupt_path = self.output_dir / f"model_interrupted_{self.track_name}.pkl"
                best_overall_network.save(str(interrupt_path))
        
        # Génération des graphiques
        print("\nGénération des graphiques de progression...")
        plot_path = self.output_dir / f"evolution_progress_{self.track_name}.png"
        self.genetic_algorithm.plot_progress(str(plot_path))
        
        print(f"\n✓ Phase 2 terminée!")
        print(f"  Meilleure fitness atteinte: {best_overall_fitness:.2f}")
        
        return best_overall_network
    
    def _adapt_parameters(self, generation, total_generations):
        """
        Adapte dynamiquement les paramètres de l'algorithme génétique
        
        Args:
            generation: Numéro de génération actuelle
            total_generations: Nombre total de générations
        """
        progress = generation / total_generations
        
        # Réduire le taux de mutation au fil du temps (exploitation vs exploration)
        initial_mutation_rate = self.config.get('mutation_rate', 0.1)
        final_mutation_rate = initial_mutation_rate * 0.3
        self.genetic_algorithm.mutation_rate = (
            initial_mutation_rate - 
            (initial_mutation_rate - final_mutation_rate) * progress
        )
        
        # Réduire la force de mutation
        initial_mutation_strength = self.config.get('mutation_strength', 0.3)
        final_mutation_strength = initial_mutation_strength * 0.5
        self.genetic_algorithm.mutation_strength = (
            initial_mutation_strength -
            (initial_mutation_strength - final_mutation_strength) * progress
        )
    
    def phase3_final_evaluation(self, best_network):
        """
        Phase 3: Évaluation finale et génération de dataset
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: ÉVALUATION FINALE")
        print(f"{'#'*70}\n")
        
        if best_network is None:
            print("✗ Aucun réseau à évaluer")
            return
        
        print("Évaluation détaillée du meilleur modèle...")
        
        # Faire plusieurs évaluations pour obtenir une moyenne
        n_evaluations = self.config.get('final_evaluations', 5)
        fitness_scores = []
        
        for i in range(n_evaluations):
            print(f"\nÉvaluation {i+1}/{n_evaluations}...")
            fitness = self.evaluator.evaluate_network(best_network, verbose=True)
            fitness_scores.append(fitness)
            time.sleep(2)  # Pause entre les évaluations
        
        avg_fitness = np.mean(fitness_scores)
        std_fitness = np.std(fitness_scores)
        
        print(f"\n{'='*70}")
        print(f"RÉSULTATS FINAUX")
        print(f"{'='*70}")
        print(f"Fitness moyenne: {avg_fitness:.2f} ± {std_fitness:.2f}")
        print(f"Meilleure: {max(fitness_scores):.2f}")
        print(f"Pire: {min(fitness_scores):.2f}")
        print(f"{'='*70}\n")
        
        # Sauvegarder le modèle final
        final_model_path = self.output_dir / f"teacher_model_final_{self.track_name}.pkl"
        best_network.save(str(final_model_path))
        
        print(f"✓ Modèle professeur final sauvegardé: {final_model_path}")
        
        return avg_fitness
    
    def run_full_pipeline(self):
        """Exécute le pipeline complet d'entraînement"""
        start_time = time.time()
        
        try:
            # Phase 1: Imitation Learning
            self.phase1_imitation_learning()
            
            # Phase 2: Genetic Evolution
            best_network = self.phase2_genetic_evolution()
            
            # Phase 3: Final Evaluation
            if best_network is not None:
                self.phase3_final_evaluation(best_network)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"PIPELINE TERMINÉ")
            print(f"{'='*70}")
            print(f"Temps total: {elapsed_time/60:.1f} minutes")
            print(f"Modèles sauvegardés dans: {self.output_dir}")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n✗ Erreur critique: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Entraînement hybride (Imitation + Génétique) pour modèle professeur"
    )
    
    parser.add_argument(
        '--track', 
        type=str, 
        default='SimpleTrack',
        help='Nom du circuit (default: SimpleTrack)'
    )
    
    parser.add_argument(
        '--human-data',
        type=str,
        default='record_0.npz',
        help='Fichier de données de conduite humaine'
    )
    
    parser.add_argument(
        '--population',
        type=int,
        default=30,
        help='Taille de la population (default: 30)'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=20,
        help='Nombre de générations (default: 20)'
    )
    
    parser.add_argument(
        '--simulation-time',
        type=int,
        default=60,
        help='Temps de simulation par individu en secondes (default: 60)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models',
        help='Répertoire de sortie pour les modèles'
    )
    
    parser.add_argument(
        '--skip-imitation',
        action='store_true',
        help='Sauter la phase d\'apprentissage par imitation'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'track_name': args.track,
        'human_data_file': args.human_data,
        'population_size': args.population,
        'n_generations': args.generations,
        'simulation_time': args.simulation_time,
        'output_dir': args.output_dir,
        'elite_size': max(2, args.population // 10),
        'mutation_rate': 0.15,
        'mutation_strength': 0.3,
        'crossover_rate': 0.7,
        'init_noise_level': 0.2,
        'imitation_epochs': 50 if not args.skip_imitation else 0,
        'imitation_lr': 0.01,
        'checkpoint_interval': 5,
        'verbose_interval': 5,
        'final_evaluations': 3,
        'adaptive_params': True,
        'game_host': '127.0.0.1',
        'game_port': 5000,
        'checkpoint_file': f'checkpoints_{args.track}.json'
    }
    
    # Vérification préalable
    print("\n" + "="*70)
    print("VÉRIFICATIONS PRÉALABLES")
    print("="*70)
    
    print(f"\n1. Circuit: {config['track_name']}")
    
    checkpoint_path = Path(config['checkpoint_file'])
    if checkpoint_path.exists():
        print(f"   ✓ Fichier de checkpoints trouvé: {checkpoint_path}")
    else:
        print(f"   ⚠ Fichier de checkpoints non trouvé: {checkpoint_path}")
        print(f"   Le système utilisera uniquement la distance parcourue")
    
    print(f"\n2. Données humaines: {config['human_data_file']}")
    data_path = Path(config['human_data_file'])
    if data_path.exists():
        print(f"   ✓ Fichier de données trouvé")
    else:
        print(f"   ⚠ Fichier non trouvé - initialisation aléatoire sera utilisée")
    
    print(f"\n3. Connexion au jeu:")
    print(f"   Host: {config['game_host']}")
    print(f"   Port: {config['game_port']}")
    print(f"   ⚠ IMPORTANT: Le jeu doit être lancé avec main_with_checkpoints.py")
    
    print("\n" + "="*70)
    
    response = input("\nLe jeu est-il lancé et prêt? (y/n): ")
    if response.lower() != 'y':
        print("Veuillez lancer le jeu avec: python main_with_checkpoints.py")
        print("Puis relancez ce script.")
        sys.exit(0)
    
    # Lancer le pipeline
    pipeline = HybridTrainingPipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
