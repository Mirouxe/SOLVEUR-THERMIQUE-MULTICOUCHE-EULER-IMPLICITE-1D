#!/usr/bin/env python3
"""
Script principal de validation du modèle thermique
==================================================

Ce script exécute l'ensemble des tests de validation et génère
un rapport de synthèse.

Tests effectués:
1. Validation contre solutions analytiques
2. Convergence spatiale (raffinement de maillage)
3. Convergence temporelle (raffinement du pas de temps)
4. Validation des conditions limites
5. Propriétés dépendantes de la température

Usage:
    python run_all_tests.py [--quick] [--verbose]
    
Options:
    --quick: Exécute uniquement les tests rapides
    --verbose: Affiche plus de détails
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour la génération de figures
import matplotlib.pyplot as plt


def print_header():
    """Affiche l'en-tête du rapport."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "VALIDATION DU MODÈLE THERMIQUE 1D" + " "*15 + "║")
    print("║" + " "*20 + "Conduction transitoire multicouche" + " "*13 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    print("Date:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__)
    print()


def run_test_module(module_name, test_function_name='run_all_tests'):
    """
    Exécute un module de test et capture les résultats.
    
    Args:
        module_name: Nom du module (sans .py)
        test_function_name: Nom de la fonction principale de test
        
    Returns:
        Tuple (passed, duration, error_message)
    """
    print(f"\n{'─'*70}")
    print(f"Exécution de {module_name}...")
    print(f"{'─'*70}")
    
    start_time = time.time()
    
    try:
        # Import dynamique du module
        module = __import__(module_name)
        test_func = getattr(module, test_function_name)
        
        # Exécution des tests
        passed = test_func()
        
        duration = time.time() - start_time
        return passed, duration, None
        
    except Exception as e:
        duration = time.time() - start_time
        import traceback
        error_msg = traceback.format_exc()
        print(f"\n✗ ERREUR lors de l'exécution de {module_name}:")
        print(error_msg)
        return False, duration, str(e)


def generate_summary_report(results):
    """
    Génère un rapport de synthèse des tests.
    
    Args:
        results: Dict {nom_test: (passed, duration, error)}
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*25 + "RAPPORT DE SYNTHÈSE" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r[0])
    total_duration = sum(r[1] for r in results.values())
    
    # Tableau des résultats
    print("┌" + "─"*35 + "┬" + "─"*10 + "┬" + "─"*12 + "┐")
    print("│" + " Module de test".ljust(35) + "│" + " Résultat ".center(10) + "│" + " Durée [s] ".center(12) + "│")
    print("├" + "─"*35 + "┼" + "─"*10 + "┼" + "─"*12 + "┤")
    
    for name, (passed, duration, error) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        status_color = status
        print(f"│ {name[:33].ljust(33)} │ {status_color.center(8)} │ {duration:10.2f} │")
    
    print("├" + "─"*35 + "┼" + "─"*10 + "┼" + "─"*12 + "┤")
    print(f"│ {'TOTAL'.ljust(33)} │ {f'{passed_tests}/{total_tests}'.center(8)} │ {total_duration:10.2f} │")
    print("└" + "─"*35 + "┴" + "─"*10 + "┴" + "─"*12 + "┘")
    
    # Résumé final
    print()
    if passed_tests == total_tests:
        print("╔" + "═"*68 + "╗")
        print("║" + " "*15 + "✓ TOUS LES TESTS SONT RÉUSSIS ✓" + " "*16 + "║")
        print("╚" + "═"*68 + "╝")
    else:
        print("╔" + "═"*68 + "╗")
        print("║" + " "*15 + f"✗ {total_tests - passed_tests} TEST(S) ÉCHOUÉ(S) ✗".ljust(38) + "║")
        print("╚" + "═"*68 + "╝")
        
        # Détails des échecs
        print("\nDétails des échecs:")
        for name, (passed, duration, error) in results.items():
            if not passed:
                print(f"  • {name}: {error if error else 'Voir logs ci-dessus'}")
    
    # Liste des figures générées
    print("\nFigures générées:")
    import os
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    for f in sorted(png_files):
        print(f"  • {f}")
    
    return passed_tests == total_tests


def main():
    """Fonction principale."""
    print_header()
    
    # Vérification des dépendances
    print("Vérification des dépendances...")
    try:
        from scipy import special
        print("  ✓ scipy disponible")
    except ImportError:
        print("  ✗ scipy non disponible (certains tests analytiques seront limités)")
    
    # Liste des modules de test
    test_modules = [
        ('test_analytic_solution', 'Validation solutions analytiques'),
        ('test_mesh_convergence', 'Convergence spatiale'),
        ('test_time_convergence', 'Convergence temporelle'),
        ('test_boundary_conditions', 'Conditions limites'),
        ('test_temperature_dependent', 'Propriétés f(T)'),
    ]
    
    # Mode rapide si demandé
    if '--quick' in sys.argv:
        print("\nMode rapide: exécution des tests essentiels uniquement")
        test_modules = test_modules[:3]
    
    # Exécution des tests
    results = {}
    
    for module_name, description in test_modules:
        passed, duration, error = run_test_module(module_name)
        results[description] = (passed, duration, error)
    
    # Génération du rapport
    all_passed = generate_summary_report(results)
    
    # Code de retour
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
