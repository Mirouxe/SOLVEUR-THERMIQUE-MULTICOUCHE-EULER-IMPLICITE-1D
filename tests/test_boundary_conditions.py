"""
Tests de validation des conditions limites
==========================================

Ce script vérifie que les conditions limites sont correctement implémentées:
1. Condition de Dirichlet (température imposée)
2. Condition de Neumann (flux imposé)
3. Condition de Robin (convection)
4. Condition adiabatique (flux nul)

Pour chaque condition:
- Vérification directe de la valeur ou du gradient au bord
- Comparaison avec le comportement physique attendu
- Bilan de flux aux interfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, create_constant_material, create_single_layer, compute_error_norms
from analytical_solutions import thermal_diffusivity


def test_dirichlet_bc():
    """
    Test de la condition de Dirichlet (température imposée).
    
    Vérifie que T(x=0) = T_imposed exactement à chaque instant.
    """
    print("\n" + "="*70)
    print("TEST CONDITION DE DIRICHLET (température imposée)")
    print("="*70)
    
    # Paramètres
    L = 0.1
    k, rho, cp = 50.0, 7800.0, 500.0
    
    T_init = 300.0
    T_imposed = 500.0  # Température imposée à gauche
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=51)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_imposed}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 50.0
    dt = 0.5
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=10)
    
    # Vérifier T(x=0) à chaque instant sauvegardé (sauf t=0 où c'est la CI)
    T_at_boundary = result['T'][1:, 0]  # Exclure t=0
    errors = np.abs(T_at_boundary - T_imposed)
    
    print(f"\n  Température imposée: T_left = {T_imposed} K")
    print(f"  Erreur max sur T(x=0) (t>0): {np.max(errors):.2e} K")
    print(f"  Erreur moyenne: {np.mean(errors):.2e} K")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//3, 2*len(result['time'])//3, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].axhline(y=T_imposed, color='r', linestyle='--', label=f'T_imposé={T_imposed}K')
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Température au bord (tous les temps pour le tracé)
    T_at_boundary_all = result['T'][:, 0]
    axes[1].plot(result['time'], T_at_boundary_all, 'b-', linewidth=2, label='T(x=0)')
    axes[1].axhline(y=T_imposed, color='r', linestyle='--', label='Valeur imposée')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('T(x=0) [K]')
    axes[1].set_title('Vérification condition de Dirichlet')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test Condition de Dirichlet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_bc_dirichlet.png', dpi=150)
    plt.show()
    
    # Critère: erreur < 1e-10 (précision machine) pour t > 0
    passed = np.max(errors) < 1e-10
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_neumann_bc():
    """
    Test de la condition de Neumann (flux imposé).
    
    Vérifie que le flux entrant q = -k*dT/dx correspond au flux imposé.
    """
    print("\n" + "="*70)
    print("TEST CONDITION DE NEUMANN (flux imposé)")
    print("="*70)
    
    # Paramètres
    L = 0.1
    k, rho, cp = 50.0, 7800.0, 500.0
    
    T_init = 300.0
    q_imposed = 5000.0  # W/m² - flux entrant
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=51)
    
    bc_left = lambda t: {'type': 'flux', 'q': q_imposed}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 50.0
    dt = 0.5
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=10)
    
    # Calculer le flux numérique au bord gauche: q = -k * dT/dx
    # Approximation par différences finies avant
    dx = solver.dx
    q_numerical = []
    
    for T in result['T']:
        dTdx = (T[1] - T[0]) / dx
        q_num = -k * dTdx  # Flux sortant (négatif = entrant)
        q_numerical.append(-q_num)  # Convertir en flux entrant
    
    q_numerical = np.array(q_numerical)
    
    # Erreur relative sur le flux
    errors_rel = np.abs(q_numerical - q_imposed) / q_imposed * 100
    
    print(f"\n  Flux imposé: q = {q_imposed} W/m²")
    print(f"  Flux numérique (moyenne): {np.mean(q_numerical[1:]):.2f} W/m²")
    print(f"  Erreur relative max: {np.max(errors_rel[1:]):.2f} %")
    
    # Vérification énergétique: l'énergie totale doit augmenter de q*A*t
    E_init = np.sum(rho * cp * result['T'][0, :] * dx)
    E_final = np.sum(rho * cp * result['T'][-1, :] * dx)
    dE_actual = E_final - E_init
    dE_expected = q_imposed * t_end  # Pour une surface unitaire
    
    print(f"\n  Bilan énergétique:")
    print(f"    Énergie entrée attendue: {dE_expected:.2f} J/m²")
    print(f"    Variation d'énergie: {dE_actual:.2f} J/m²")
    print(f"    Erreur: {abs(dE_actual - dE_expected)/dE_expected*100:.2f} %")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//3, 2*len(result['time'])//3, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température (chauffage par flux)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Flux au bord
    axes[1].plot(result['time'], q_numerical, 'b-', linewidth=2, label='Flux numérique')
    axes[1].axhline(y=q_imposed, color='r', linestyle='--', label=f'Flux imposé ({q_imposed} W/m²)')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Flux entrant [W/m²]')
    axes[1].set_title('Vérification condition de Neumann')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test Condition de Neumann (flux imposé)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_bc_neumann.png', dpi=150)
    plt.show()
    
    # Critère: erreur sur le bilan énergétique < 5%
    energy_error = abs(dE_actual - dE_expected) / dE_expected * 100
    passed = energy_error < 5
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: erreur bilan = {energy_error:.2f}%")
    
    return passed


def test_convection_bc():
    """
    Test de la condition de Robin (convection).
    
    Vérifie que: -k*dT/dx = h*(T_surface - T_inf)
    """
    print("\n" + "="*70)
    print("TEST CONDITION DE ROBIN (convection)")
    print("="*70)
    
    # Paramètres
    L = 0.02
    k, rho, cp = 50.0, 7800.0, 500.0
    
    T_init = 300.0
    h = 1000.0  # W/(m²·K) - coefficient de convection élevé
    T_inf = 500.0  # Température du fluide
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=41)
    
    bc_left = lambda t: {'type': 'convection', 'h': h, 'T_inf': T_inf}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 20.0
    dt = 0.1
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=20)
    
    # Vérifier la relation de Robin à chaque instant
    dx = solver.dx
    robin_errors = []
    
    for T in result['T'][1:]:  # Ignorer t=0
        T_surf = T[0]
        dTdx = (T[1] - T[0]) / dx
        
        # Flux conductif sortant: q_cond = -k * dT/dx (positif si T décroît vers l'intérieur)
        q_cond = -k * dTdx
        
        # Flux convectif entrant: q_conv = h * (T_inf - T_surf)
        q_conv = h * (T_inf - T_surf)
        
        # À l'équilibre du bord: q_cond = q_conv
        robin_errors.append(abs(q_cond - q_conv) / (abs(q_conv) + 1e-10))
    
    robin_errors = np.array(robin_errors)
    
    print(f"\n  Paramètres: h = {h} W/(m²·K), T_inf = {T_inf} K")
    print(f"  Erreur relative max sur la relation de Robin: {np.max(robin_errors)*100:.2f}%")
    print(f"  Erreur relative moyenne: {np.mean(robin_errors)*100:.2f}%")
    
    # Vérification de la convergence vers T_inf
    T_final_mean = np.mean(result['T'][-1, :])
    print(f"\n  Température finale moyenne: {T_final_mean:.2f} K")
    print(f"  Écart à T_inf: {abs(T_final_mean - T_inf):.2f} K")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//3, 2*len(result['time'])//3, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].axhline(y=T_inf, color='r', linestyle='--', label=f'T_inf={T_inf}K')
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température (chauffage par convection)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Température de surface vs temps
    T_surface = result['T'][:, 0]
    axes[1].plot(result['time'], T_surface, 'b-', linewidth=2, label='T_surface')
    axes[1].axhline(y=T_inf, color='r', linestyle='--', label=f'T_inf={T_inf}K')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Température de surface [K]')
    axes[1].set_title('Évolution de la température de surface')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test Condition de Robin (convection)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_bc_convection.png', dpi=150)
    plt.show()
    
    # Critère: erreur sur la relation de Robin < 10%
    passed = np.mean(robin_errors) < 0.10
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_adiabatic_bc():
    """
    Test de la condition adiabatique (flux nul).
    
    Vérifie que dT/dx = 0 au bord adiabatique.
    """
    print("\n" + "="*70)
    print("TEST CONDITION ADIABATIQUE (flux nul)")
    print("="*70)
    
    # Paramètres
    L = 0.1
    k, rho, cp = 50.0, 7800.0, 500.0
    
    T_init = 300.0
    T_left = 500.0  # Chauffage à gauche
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=51)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    # Simuler jusqu'au régime permanent (Fo >> 1)
    # τ = L²/α ≈ 780 s, simuler jusqu'à 3τ
    t_end = 2000.0
    dt = 5.0
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=40)
    
    # Calculer le gradient au bord droit
    dx = solver.dx
    gradients_right = []
    fluxes_right = []
    
    for T in result['T']:
        dTdx = (T[-1] - T[-2]) / dx
        gradients_right.append(dTdx)
        fluxes_right.append(-k * dTdx)
    
    gradients_right = np.array(gradients_right)
    fluxes_right = np.array(fluxes_right)
    
    print(f"\n  Gradient max au bord droit: {np.max(np.abs(gradients_right)):.4f} K/m")
    print(f"  Flux max au bord droit: {np.max(np.abs(fluxes_right)):.4f} W/m²")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//3, 2*len(result['time'])//3, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Flux au bord droit
    axes[1].plot(result['time'], fluxes_right, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', label='Flux nul (attendu)')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Flux au bord droit [W/m²]')
    axes[1].set_title('Vérification condition adiabatique')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test Condition Adiabatique', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_bc_adiabatic.png', dpi=150)
    plt.show()
    
    # Critère: en régime établi (Dirichlet-Adiabatique), T doit être uniforme = T_left
    # Donc le gradient et le flux doivent tendre vers zéro
    T_final = result['T'][-1, :]
    T_uniform = np.max(T_final) - np.min(T_final) < 1.0  # Moins de 1K de variation
    flux_final_small = np.abs(fluxes_right[-1]) < 100  # Flux < 100 W/m²
    
    passed = T_uniform and flux_final_small
    print(f"\n  Température finale uniforme: {'✓' if T_uniform else '✗'} (variation = {np.max(T_final) - np.min(T_final):.2f} K)")
    print(f"  Flux final faible: {'✓' if flux_final_small else '✗'} ({np.abs(fluxes_right[-1]):.1f} W/m²)")
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_variable_bc():
    """
    Test des conditions limites variables dans le temps.
    
    Vérifie que le solveur suit correctement une condition limite
    qui varie avec le temps (rampe de température).
    """
    print("\n" + "="*70)
    print("TEST CONDITIONS LIMITES VARIABLES DANS LE TEMPS")
    print("="*70)
    
    # Paramètres
    L = 0.05
    k, rho, cp = 50.0, 7800.0, 500.0
    
    T_init = 300.0
    
    # Rampe de température: T(t) = T_init + rate * t
    rate = 10.0  # K/s
    T_max = 600.0
    
    def T_ramp(t):
        return min(T_init + rate * t, T_max)
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=41)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_ramp(t)}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 50.0
    dt = 0.2
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=5)
    
    # Vérifier que T(x=0) suit la rampe
    T_imposed = np.array([T_ramp(t) for t in result['time']])
    T_boundary = result['T'][:, 0]
    
    errors = np.abs(T_boundary - T_imposed)
    
    print(f"\n  Rampe de température: {rate} K/s jusqu'à {T_max} K")
    print(f"  Erreur max sur T(x=0): {np.max(errors):.4f} K")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//4, len(result['time'])//2, 3*len(result['time'])//4, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température (rampe)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Température au bord vs rampe imposée
    axes[1].plot(result['time'], T_boundary, 'b-', linewidth=2, label='T(x=0) numérique')
    axes[1].plot(result['time'], T_imposed, 'r--', linewidth=2, label='Rampe imposée')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Température [K]')
    axes[1].set_title('Suivi de la rampe de température')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test Conditions Limites Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_bc_variable.png', dpi=150)
    plt.show()
    
    # Critère: erreur < 0.1 K
    passed = np.max(errors) < 0.1
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def run_all_tests():
    """Exécute tous les tests de conditions limites."""
    print("\n" + "#"*70)
    print("# TESTS DE VALIDATION DES CONDITIONS LIMITES")
    print("#"*70)
    
    results = {}
    
    results['dirichlet'] = test_dirichlet_bc()
    results['neumann'] = test_neumann_bc()
    results['convection'] = test_convection_bc()
    results['adiabatic'] = test_adiabatic_bc()
    results['variable'] = test_variable_bc()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS DE CONDITIONS LIMITES")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ RÉUSSI" if passed else "✗ ÉCHOUÉ"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "-"*70)
    if all_passed:
        print("TOUS LES TESTS SONT RÉUSSIS ✓")
    else:
        print("CERTAINS TESTS ONT ÉCHOUÉ ✗")
    print("-"*70)
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
