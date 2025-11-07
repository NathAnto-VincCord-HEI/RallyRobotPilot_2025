"""
EXEMPLE MINIMAL - Comprendre le système pas à pas
Ce fichier démontre les concepts clés de manière simplifiée
"""

import numpy as np
from teacher_model import TeacherNetwork
from genetic_algorithm import GeneticAlgorithm


def exemple_1_creer_reseau():
    """
    EXEMPLE 1: Créer et utiliser un réseau de neurones simple
    """
    print("\n" + "="*70)
    print("EXEMPLE 1: CRÉATION ET UTILISATION D'UN RÉSEAU")
    print("="*70)
    
    # Créer un réseau
    network = TeacherNetwork(
        input_size=17,      # 15 raycasts + vitesse + angle
        hidden1_size=32,    # Première couche cachée
        hidden2_size=16,    # Deuxième couche cachée
        output_size=4       # Forward, Backward, Left, Right
    )
    
    print(f"✓ Réseau créé avec {network.get_genome_size()} paramètres")
    
    # Simuler des données de capteurs
    raycasts = [100.0, 95.0, 80.0, 60.0, 40.0, 30.0, 25.0, 30.0,
                40.0, 60.0, 80.0, 95.0, 100.0, 100.0, 100.0]  # 15 valeurs
    vitesse = 25.0  # km/h
    angle = 45.0    # degrés
    
    print(f"\nDonnées de test:")
    print(f"  Raycasts avant: {raycasts[:7]}")
    print(f"  Vitesse: {vitesse}")
    print(f"  Angle: {angle}°")
    
    # Prédire les actions
    actions = network.predict(raycasts, vitesse, angle, threshold=0.5)
    forward, backward, left, right = actions
    
    print(f"\nPrédiction du réseau:")
    print(f"  Forward: {'OUI' if forward else 'NON'}")
    print(f"  Backward: {'OUI' if backward else 'NON'}")
    print(f"  Left: {'OUI' if left else 'NON'}")
    print(f"  Right: {'OUI' if right else 'NON'}")
    
    return network


def exemple_2_entrainer_imitation(network):
    """
    EXEMPLE 2: Entraîner par imitation (si données disponibles)
    """
    print("\n" + "="*70)
    print("EXEMPLE 2: APPRENTISSAGE PAR IMITATION")
    print("="*70)
    
    from pathlib import Path
    
    data_file = "record_0.npz"
    
    if not Path(data_file).exists():
        print(f"⚠ Fichier {data_file} non trouvé")
        print("  → Cet exemple nécessite des données de conduite humaine")
        print("  → Utilisez data_collector.py pour en créer")
        return network
    
    from teacher_model import ImitationLearningInitializer
    
    print(f"Chargement des données depuis {data_file}...")
    
    initializer = ImitationLearningInitializer(network)
    
    # Charger les données
    inputs, targets = initializer.load_human_data(data_file)
    
    print(f"\nDonnées chargées:")
    print(f"  Nombre d'exemples: {len(inputs)}")
    print(f"  Dimensions input: {inputs.shape}")
    print(f"  Dimensions target: {targets.shape}")
    
    # Entraîner (juste quelques époques pour la démo)
    print(f"\nEntraînement sur 10 époques...")
    initializer.train_supervised(inputs, targets, epochs=10, learning_rate=0.01)
    
    print(f"\n✓ Réseau entraîné par imitation!")
    
    return network


def exemple_3_algorithme_genetique():
    """
    EXEMPLE 3: Comprendre l'algorithme génétique
    """
    print("\n" + "="*70)
    print("EXEMPLE 3: ALGORITHME GÉNÉTIQUE (SIMPLIFIÉ)")
    print("="*70)
    
    # Créer un template
    template = TeacherNetwork()
    
    # Algorithme génétique avec petite population
    ga = GeneticAlgorithm(
        population_size=10,   # Petite population pour la démo
        elite_size=2,         # Garder les 2 meilleurs
        mutation_rate=0.15,   # 15% des gènes mutent
        mutation_strength=0.3 # Force de mutation
    )
    
    print(f"Configuration:")
    print(f"  Population: {ga.population_size}")
    print(f"  Élites: {ga.elite_size}")
    print(f"  Taux mutation: {ga.mutation_rate}")
    
    # Initialiser la population
    ga.initialize_population_random(template)
    print(f"\n✓ Population initialisée: {len(ga.population)} individus")
    
    # Fonction de fitness simple (pour la démo)
    def fitness_demo(network):
        """
        Fitness factice qui favorise des valeurs de poids moyennes
        Dans la réalité, on évaluerait dans le simulateur
        """
        genome = network.get_genome()
        # Favoriser les génomes avec des valeurs proches de 0
        fitness = 100 - np.mean(np.abs(genome))
        return max(0, fitness)
    
    print(f"\nSimulation de 5 générations...")
    
    for gen in range(5):
        # Évaluer
        ga.evaluate_population(fitness_demo)
        best, best_fitness = ga.get_best_individual()
        
        print(f"\nGénération {gen}:")
        print(f"  Meilleur: {best_fitness:.2f}")
        print(f"  Moyenne: {np.mean(ga.fitness_scores):.2f}")
        print(f"  Diversité: {ga.diversity_history[-1]:.4f}")
        
        # Évoluer (sauf dernière génération)
        if gen < 4:
            ga.evolve()
    
    print(f"\n✓ Évolution terminée!")
    print(f"  Amélioration: {ga.best_fitness_history[-1] - ga.best_fitness_history[0]:.2f}")
    
    return ga.get_best_individual()[0]


def exemple_4_operations_genetiques():
    """
    EXEMPLE 4: Opérations génétiques en détail
    """
    print("\n" + "="*70)
    print("EXEMPLE 4: OPÉRATIONS GÉNÉTIQUES")
    print("="*70)
    
    from genetic_algorithm import GeneticAlgorithm
    
    # Créer deux parents
    parent1 = TeacherNetwork()
    parent2 = TeacherNetwork()
    
    print("Création de deux parents...")
    genome1 = parent1.get_genome()
    genome2 = parent2.get_genome()
    
    print(f"  Parent 1: moyenne des poids = {np.mean(genome1):.4f}")
    print(f"  Parent 2: moyenne des poids = {np.mean(genome2):.4f}")
    
    # Crossover
    ga = GeneticAlgorithm()
    child = ga.crossover_blend(parent1, parent2, alpha=0.5)
    genome_child = child.get_genome()
    
    print(f"\n✓ Crossover (mélange):")
    print(f"  Enfant: moyenne des poids = {np.mean(genome_child):.4f}")
    print(f"  (Devrait être entre les deux parents)")
    
    # Mutation
    child_before = child.get_genome().copy()
    ga.mutation_rate = 0.2
    ga.mutation_strength = 0.5
    ga.mutate(child)
    child_after = child.get_genome()
    
    diff = np.sum(np.abs(child_after - child_before) > 0.01)
    total = len(child_before)
    
    print(f"\n✓ Mutation:")
    print(f"  Gènes mutés: {diff}/{total} ({diff/total*100:.1f}%)")
    print(f"  Changement moyen: {np.mean(np.abs(child_after - child_before)):.4f}")


def exemple_5_sauvegarder_charger():
    """
    EXEMPLE 5: Sauvegarder et charger un modèle
    """
    print("\n" + "="*70)
    print("EXEMPLE 5: SAUVEGARDE ET CHARGEMENT")
    print("="*70)
    
    # Créer un réseau
    network = TeacherNetwork()
    
    # Faire une prédiction avant sauvegarde
    test_raycasts = [50.0] * 15
    test_speed = 30.0
    test_angle = 90.0
    
    actions_before = network.predict(test_raycasts, test_speed, test_angle)
    print(f"Prédiction avant sauvegarde: {actions_before}")
    
    # Sauvegarder
    filename = "test_model_example.pkl"
    network.save(filename)
    print(f"\n✓ Modèle sauvegardé: {filename}")
    
    # Charger
    loaded_network = TeacherNetwork.load(filename)
    print(f"✓ Modèle chargé depuis: {filename}")
    
    # Vérifier que la prédiction est identique
    actions_after = loaded_network.predict(test_raycasts, test_speed, test_angle)
    print(f"Prédiction après chargement: {actions_after}")
    
    if actions_before == actions_after:
        print(f"\n✓ Les prédictions sont identiques!")
    else:
        print(f"\n✗ Erreur: les prédictions diffèrent!")
    
    # Nettoyer
    from pathlib import Path
    Path(filename).unlink()
    print(f"\n✓ Fichier temporaire supprimé")


def main():
    """
    Fonction principale qui exécute tous les exemples
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " EXEMPLES PÉDAGOGIQUES - MODÈLE PROFESSEUR ".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print("\nCes exemples démontrent les concepts clés du système:")
    print("  1. Création et utilisation d'un réseau de neurones")
    print("  2. Apprentissage par imitation")
    print("  3. Algorithme génétique")
    print("  4. Opérations génétiques (crossover, mutation)")
    print("  5. Sauvegarde et chargement")
    
    input("\nAppuyez sur Entrée pour commencer...")
    
    # Exemple 1
    network = exemple_1_creer_reseau()
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Exemple 2
    network = exemple_2_entrainer_imitation(network)
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Exemple 3
    best_network = exemple_3_algorithme_genetique()
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Exemple 4
    exemple_4_operations_genetiques()
    input("\nAppuyez sur Entrée pour continuer...")
    
    # Exemple 5
    exemple_5_sauvegarder_charger()
    
    # Résumé
    print("\n" + "#"*70)
    print("# RÉSUMÉ ".center(70) + "#")
    print("#"*70)
    
    print("""
    ✓ Vous avez vu comment:
      1. Créer un réseau de neurones pour le contrôle
      2. L'entraîner par imitation des données humaines
      3. L'optimiser via algorithme génétique
      4. Combiner crossover et mutation
      5. Sauvegarder et réutiliser les modèles
    
    Pour l'entraînement complet:
      → python train_teacher_model.py --help
    
    Pour plus de détails:
      → Consultez GUIDE_COMPLET.md
      → Consultez README.md
    """)
    
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
