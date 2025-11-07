"""
Algorithme Génétique pour optimiser le modèle professeur
"""

import numpy as np
import pickle
from pathlib import Path
import time
from scripts.teacher_model import TeacherNetwork


class GeneticAlgorithm:
    """
    Algorithme génétique pour faire évoluer un réseau de neurones
    """
    
    def __init__(self, 
                 population_size=50,
                 elite_size=5,
                 mutation_rate=0.1,
                 mutation_strength=0.3,
                 crossover_rate=0.7):
        """
        Args:
            population_size: Taille de la population
            elite_size: Nombre d'individus élites préservés
            mutation_rate: Probabilité de mutation pour chaque gène
            mutation_strength: Amplitude des mutations
            crossover_rate: Probabilité de crossover
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # Statistiques
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
    def initialize_population_random(self, network_template):
        """
        Initialise la population avec des réseaux aléatoires
        
        Args:
            network_template: Template de TeacherNetwork pour la structure
        """
        print(f"Initialisation de la population (aléatoire)...")
        self.population = []
        
        for i in range(self.population_size):
            network = TeacherNetwork(
                network_template.input_size,
                network_template.hidden1_size,
                network_template.hidden2_size,
                network_template.output_size
            )
            self.population.append(network)
        
        self.fitness_scores = [0.0] * self.population_size
        print(f"✓ Population de {self.population_size} individus créée")
    
    def initialize_population_from_model(self, base_network, noise_level=0.2):
        """
        Initialise la population autour d'un modèle pré-entraîné
        Permet de démarrer l'algorithme génétique avec une bonne base
        
        Args:
            base_network: Réseau de base (déjà entraîné par imitation)
            noise_level: Niveau de bruit à ajouter (0-1)
        """
        print(f"Initialisation de la population depuis modèle pré-entraîné...")
        self.population = []
        
        base_genome = base_network.get_genome()
        
        # Premier individu: copie exacte du modèle de base
        self.population.append(base_network.copy())
        
        # Autres individus: variations autour du modèle de base
        for i in range(1, self.population_size):
            network = base_network.copy()
            genome = base_genome.copy()
            
            # Ajouter du bruit gaussien
            noise = np.random.randn(len(genome)) * noise_level
            genome += noise
            
            network.set_genome(genome)
            self.population.append(network)
        
        self.fitness_scores = [0.0] * self.population_size
        print(f"✓ Population de {self.population_size} individus créée autour du modèle de base")
    
    def evaluate_population(self, fitness_function, *args, **kwargs):
        """
        Évalue la fitness de toute la population
        
        Args:
            fitness_function: Fonction qui prend un réseau et retourne un score
            *args, **kwargs: Arguments passés à fitness_function
        
        Returns:
            list: Scores de fitness
        """
        print(f"\nÉvaluation de la génération {self.generation}...")
        
        self.fitness_scores = []
        
        for i, network in enumerate(self.population):
            fitness = fitness_function(network, *args, **kwargs)
            self.fitness_scores.append(fitness)
            
            if (i + 1) % 10 == 0:
                print(f"  Évalué {i+1}/{self.population_size} individus...")
        
        # Statistiques
        best_fitness = np.max(self.fitness_scores)
        avg_fitness = np.mean(self.fitness_scores)
        diversity = self._calculate_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        print(f"\n✓ Génération {self.generation} évaluée:")
        print(f"    Meilleur: {best_fitness:.2f}")
        print(f"    Moyenne: {avg_fitness:.2f}")
        print(f"    Pire: {np.min(self.fitness_scores):.2f}")
        print(f"    Diversité: {diversity:.4f}")
        
        return self.fitness_scores
    
    def _calculate_diversity(self):
        """Calcule la diversité génétique de la population"""
        if len(self.population) < 2:
            return 0.0
        
        # Échantillonner quelques paires pour éviter O(n²)
        n_samples = min(100, len(self.population) * (len(self.population) - 1) // 2)
        distances = []
        
        for _ in range(n_samples):
            i, j = np.random.choice(len(self.population), 2, replace=False)
            genome_i = self.population[i].get_genome()
            genome_j = self.population[j].get_genome()
            distance = np.mean(np.abs(genome_i - genome_j))
            distances.append(distance)
        
        return np.mean(distances)
    
    def selection_tournament(self, tournament_size=3):
        """
        Sélection par tournoi
        
        Args:
            tournament_size: Nombre d'individus par tournoi
        
        Returns:
            TeacherNetwork: Individu sélectionné
        """
        # Choisir aléatoirement tournament_size individus
        indices = np.random.choice(
            len(self.population), 
            tournament_size, 
            replace=False
        )
        
        # Retourner le meilleur du tournoi
        best_idx = indices[np.argmax([self.fitness_scores[i] for i in indices])]
        return self.population[best_idx]
    
    def crossover_uniform(self, parent1, parent2):
        """
        Crossover uniforme entre deux parents
        
        Args:
            parent1, parent2: TeacherNetwork parents
        
        Returns:
            TeacherNetwork: Enfant
        """
        genome1 = parent1.get_genome()
        genome2 = parent2.get_genome()
        
        # Masque aléatoire pour choisir les gènes
        mask = np.random.random(len(genome1)) < 0.5
        
        child_genome = np.where(mask, genome1, genome2)
        
        child = parent1.copy()
        child.set_genome(child_genome)
        
        return child
    
    def crossover_blend(self, parent1, parent2, alpha=0.5):
        """
        Crossover par mélange (blend crossover)
        
        Args:
            parent1, parent2: TeacherNetwork parents
            alpha: Coefficient de mélange (0-1)
        
        Returns:
            TeacherNetwork: Enfant
        """
        genome1 = parent1.get_genome()
        genome2 = parent2.get_genome()
        
        # Mélange aléatoire
        blend_factor = np.random.random(len(genome1)) * (1 + 2*alpha) - alpha
        child_genome = genome1 + blend_factor * (genome2 - genome1)
        
        child = parent1.copy()
        child.set_genome(child_genome)
        
        return child
    
    def mutate(self, network):
        """
        Applique des mutations à un réseau
        
        Args:
            network: TeacherNetwork à muter
        
        Returns:
            TeacherNetwork: Réseau muté (modifié in-place)
        """
        genome = network.get_genome()
        
        # Masque de mutation
        mutation_mask = np.random.random(len(genome)) < self.mutation_rate
        
        # Générer des mutations gaussiennes
        mutations = np.random.randn(len(genome)) * self.mutation_strength
        
        # Appliquer les mutations
        genome[mutation_mask] += mutations[mutation_mask]
        
        network.set_genome(genome)
        return network
    
    def evolve(self):
        """
        Crée la prochaine génération
        
        Returns:
            list: Nouvelle population
        """
        print(f"\nCréation de la génération {self.generation + 1}...")
        
        # Trier la population par fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]  # Décroissant
        
        new_population = []
        
        # 1. Élitisme: garder les meilleurs
        for i in range(self.elite_size):
            elite = self.population[sorted_indices[i]].copy()
            new_population.append(elite)
        
        print(f"  ✓ {self.elite_size} élites préservés")
        
        # 2. Générer le reste par crossover et mutation
        while len(new_population) < self.population_size:
            # Sélection des parents
            parent1 = self.selection_tournament()
            parent2 = self.selection_tournament()
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child = self.crossover_blend(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        print(f"  ✓ {len(self.population)} nouveaux individus créés")
        
        return self.population
    
    def get_best_individual(self):
        """Retourne le meilleur individu de la population actuelle"""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]
    
    def save_checkpoint(self, filename):
        """Sauvegarde l'état complet de l'algorithme"""
        checkpoint = {
            'generation': self.generation,
            'population': [net.get_genome() for net in self.population],
            'fitness_scores': self.fitness_scores,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'network_architecture': {
                'input_size': self.population[0].input_size,
                'hidden1_size': self.population[0].hidden1_size,
                'hidden2_size': self.population[0].hidden2_size,
                'output_size': self.population[0].output_size
            },
            'ga_params': {
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': self.mutation_rate,
                'mutation_strength': self.mutation_strength,
                'crossover_rate': self.crossover_rate
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"\n✓ Checkpoint sauvegardé: {filename}")
    
    def load_checkpoint(self, filename):
        """Charge un checkpoint précédent"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.fitness_scores = checkpoint['fitness_scores']
        self.best_fitness_history = checkpoint['best_fitness_history']
        self.avg_fitness_history = checkpoint['avg_fitness_history']
        self.diversity_history = checkpoint['diversity_history']
        
        # Restaurer les paramètres
        arch = checkpoint['network_architecture']
        params = checkpoint['ga_params']
        
        self.population_size = params['population_size']
        self.elite_size = params['elite_size']
        self.mutation_rate = params['mutation_rate']
        self.mutation_strength = params['mutation_strength']
        self.crossover_rate = params['crossover_rate']
        
        # Reconstruire la population
        self.population = []
        for genome in checkpoint['population']:
            network = TeacherNetwork(
                arch['input_size'],
                arch['hidden1_size'],
                arch['hidden2_size'],
                arch['output_size']
            )
            network.set_genome(genome)
            self.population.append(network)
        
        print(f"\n✓ Checkpoint chargé: {filename}")
        print(f"  Génération: {self.generation}")
        print(f"  Meilleur fitness: {max(self.best_fitness_history):.2f}")
    
    def plot_progress(self, save_path=None):
        """
        Génère des graphiques de progression
        
        Args:
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Fitness
            generations = range(len(self.best_fitness_history))
            axes[0].plot(generations, self.best_fitness_history, 'g-', label='Meilleur', linewidth=2)
            axes[0].plot(generations, self.avg_fitness_history, 'b--', label='Moyenne', linewidth=1.5)
            axes[0].set_xlabel('Génération')
            axes[0].set_ylabel('Fitness')
            axes[0].set_title('Évolution de la Fitness')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Diversité
            axes[1].plot(generations, self.diversity_history, 'r-', linewidth=2)
            axes[1].set_xlabel('Génération')
            axes[1].set_ylabel('Diversité')
            axes[1].set_title('Diversité Génétique')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✓ Graphique sauvegardé: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠ matplotlib non disponible pour les graphiques")


if __name__ == "__main__":
    # Test de l'algorithme génétique
    print("Test du GeneticAlgorithm")
    print("=" * 60)
    
    # Créer un template de réseau
    template = TeacherNetwork()
    
    # Initialiser l'algorithme génétique
    ga = GeneticAlgorithm(population_size=20, elite_size=2)
    ga.initialize_population_random(template)
    
    # Fonction de fitness simple (pour le test)
    def dummy_fitness(network):
        # Fitness aléatoire pour le test
        return np.random.random() * 100
    
    # Simuler quelques générations
    for gen in range(3):
        ga.evaluate_population(dummy_fitness)
        best, best_fitness = ga.get_best_individual()
        print(f"Meilleur individu de la gen {gen}: fitness = {best_fitness:.2f}")
        
        if gen < 2:
            ga.evolve()
    
    print("\n" + "=" * 60)
    print("Test terminé! ✓")
