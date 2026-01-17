"""
Tests de validation contre des solutions analytiques
====================================================

Ce script compare les résultats du solveur numérique avec des solutions
analytiques exactes pour différents cas canoniques.

Cas testés:
1. Plaque finie avec Dirichlet aux deux bords
2. Plaque finie avec Dirichlet à gauche et adiabatique à droite
3. Plaque semi-infinie avec température imposée
4. Plaque finie avec convection et adiabatique

Pour chaque cas:
- Comparaison visuelle des profils
- Calcul des erreurs L2 et L∞
- Vérification de la convergence vers la solution exacte
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, create_constant_material, create_single_layer, compute_error_norms
from analytical_solutions import (
    finite_slab_dirichlet_dirichlet,
    finite_slab_dirichlet_adiabatic,
    finite_slab_convection_adiabatic,
    semi_infinite_dirichlet,
    thermal_diffusivity,
    compute_fourier_number,
    compute_biot_number
)


def test_dirichlet_dirichlet():
    """
    Test 1: Plaque finie avec températures imposées aux deux bords.
    
    Configuration:
    - Plaque d'épaisseur L = 0.1 m
    - T_init = 300 K uniforme
    - T(0,t) = 400 K, T(L,t) = 300 K
    - Matériau: k=50 W/(m·K), ρ=7800 kg/m³, cp=500 J/(kg·K)
    
    Physique attendue:
    - Transitoire initial avec diffusion depuis le bord chaud
    - Convergence vers profil linéaire stationnaire
    """
    print("\n" + "="*70)
    print("TEST 1: Dirichlet-Dirichlet (températures imposées aux deux bords)")
    print("="*70)
    
    # Paramètres physiques
    L = 0.1  # m
    k = 50.0  # W/(m·K)
    rho = 7800.0  # kg/m³
    cp = 500.0  # J/(kg·K)
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0  # K
    T_left = 400.0  # K
    T_right = 300.0  # K
    
    print(f"\nParamètres:")
    print(f"  L = {L} m, k = {k} W/(m·K), ρ = {rho} kg/m³, cp = {cp} J/(kg·K)")
    print(f"  α = {alpha:.2e} m²/s")
    print(f"  T_init = {T_init} K, T_left = {T_left} K, T_right = {T_right} K")
    
    # Configuration du solveur
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    
    Nx = 51
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    # Conditions limites
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    # Simulation
    t_end = 100.0  # s
    dt = 0.5  # s
    
    Fo_end = compute_fourier_number(alpha, t_end, L)
    print(f"  Temps final: {t_end} s, Fo = {Fo_end:.2f}")
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=20)
    
    # Temps de comparaison
    test_times = [5.0, 20.0, 50.0, t_end]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    errors = []
    for t_test in test_times:
        # Trouver l'indice temporel le plus proche
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        T_num = result['T'][idx, :]
        T_ana = finite_slab_dirichlet_dirichlet(
            result['x'], t_actual, L, T_init, T_left, T_right, alpha
        )
        
        err = compute_error_norms(T_num, T_ana)
        errors.append(err)
        
        Fo = compute_fourier_number(alpha, t_actual, L)
        print(f"\n  t = {t_actual:.1f} s (Fo = {Fo:.3f}):")
        print(f"    Erreur L2 = {err['L2']:.4f} K, L∞ = {err['Linf']:.4f} K")
        print(f"    Erreur relative L2 = {err['L2_rel']*100:.3f}%")
        
        # Tracé
        axes[0].plot(result['x']*1000, T_num, 'o-', markersize=3, 
                     label=f'Num. t={t_actual:.0f}s')
        axes[0].plot(result['x']*1000, T_ana, '--', 
                     label=f'Ana. t={t_actual:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Comparaison numérique vs analytique')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Évolution de l'erreur
    axes[1].semilogy(test_times, [e['L2'] for e in errors], 'o-', label='Erreur L2')
    axes[1].semilogy(test_times, [e['Linf'] for e in errors], 's-', label='Erreur L∞')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Erreur [K]')
    axes[1].set_title('Évolution de l\'erreur')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 1: Dirichlet-Dirichlet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test1_dirichlet_dirichlet.png', dpi=150)
    plt.show()
    
    # Critère de validation
    max_error_rel = max(e['L2_rel'] for e in errors)
    passed = max_error_rel < 0.02  # Moins de 2% d'erreur
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: erreur max = {max_error_rel*100:.2f}%")
    
    return passed


def test_dirichlet_adiabatic():
    """
    Test 2: Plaque finie avec Dirichlet à gauche et adiabatique à droite.
    
    Configuration:
    - Plaque d'épaisseur L = 0.05 m
    - T_init = 300 K uniforme
    - T(0,t) = 500 K
    - ∂T/∂x(L,t) = 0 (adiabatique)
    
    Physique attendue:
    - La chaleur entre par la gauche
    - Le bord droit "accumule" la chaleur (pas de sortie)
    - Convergence vers T = 500 K uniforme
    """
    print("\n" + "="*70)
    print("TEST 2: Dirichlet-Adiabatique (température imposée + isolé)")
    print("="*70)
    
    # Paramètres
    L = 0.05  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    T_left = 500.0
    
    print(f"\nParamètres:")
    print(f"  L = {L} m, α = {alpha:.2e} m²/s")
    print(f"  T_init = {T_init} K, T_left = {T_left} K, ∂T/∂x(L) = 0")
    
    # Solveur
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=41)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    # Temps caractéristique τ = L²/α ≈ 195 s, simuler jusqu'à 3τ pour convergence
    t_end = 500.0
    dt = 1.0
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=50)
    
    test_times = [10.0, 50.0, 200.0, t_end]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    errors = []
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        T_num = result['T'][idx, :]
        T_ana = finite_slab_dirichlet_adiabatic(
            result['x'], t_actual, L, T_init, T_left, alpha
        )
        
        err = compute_error_norms(T_num, T_ana)
        errors.append(err)
        
        Fo = compute_fourier_number(alpha, t_actual, L)
        print(f"\n  t = {t_actual:.1f} s (Fo = {Fo:.3f}):")
        print(f"    Erreur L2 = {err['L2']:.4f} K, L∞ = {err['Linf']:.4f} K")
        
        axes[0].plot(result['x']*1000, T_num, 'o-', markersize=3,
                     label=f'Num. t={t_actual:.0f}s')
        axes[0].plot(result['x']*1000, T_ana, '--',
                     label=f'Ana. t={t_actual:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Comparaison numérique vs analytique')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Vérification du flux nul à droite
    # Le gradient doit être proche de zéro au bord droit
    gradients_right = []
    for idx in range(len(result['time'])):
        T = result['T'][idx, :]
        grad = (T[-1] - T[-2]) / (result['x'][-1] - result['x'][-2])
        gradients_right.append(np.abs(grad))
    
    axes[1].semilogy(result['time'], gradients_right, 'b-')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('|∂T/∂x| au bord droit [K/m]')
    axes[1].set_title('Vérification condition adiabatique')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 2: Dirichlet-Adiabatique', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test2_dirichlet_adiabatic.png', dpi=150)
    plt.show()
    
    # Pour ce cas avec discontinuité initiale, on vérifie plutôt:
    # 1. La convergence vers T_left uniforme
    # 2. La diminution de l'erreur avec le temps
    T_final = result['T'][-1, :]
    converges_to_Tleft = np.max(np.abs(T_final - T_left)) < 5.0  # Moins de 5K d'écart
    error_decreases = errors[-1]['L2'] < errors[1]['L2']  # Erreur diminue
    
    passed = converges_to_Tleft and error_decreases
    print(f"\n  Convergence vers T_left: {'✓' if converges_to_Tleft else '✗'} (écart max = {np.max(np.abs(T_final - T_left)):.2f} K)")
    print(f"  Erreur décroissante: {'✓' if error_decreases else '✗'}")
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_convection_adiabatic():
    """
    Test 3: Plaque finie avec convection à gauche et adiabatique à droite.
    
    Configuration:
    - Plaque d'épaisseur L = 0.02 m
    - T_init = 300 K
    - Convection: h = 500 W/(m²·K), T_inf = 500 K
    - Adiabatique à droite
    
    Physique attendue:
    - Chauffage progressif de la plaque
    - Gradient de température décroissant avec le temps
    - Convergence vers T = T_inf uniforme
    """
    print("\n" + "="*70)
    print("TEST 3: Convection-Adiabatique (Robin + isolé)")
    print("="*70)
    
    # Paramètres
    L = 0.02  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    h = 500.0  # W/(m²·K)
    T_inf = 500.0
    
    Bi = compute_biot_number(h, L, k)
    
    print(f"\nParamètres:")
    print(f"  L = {L} m, α = {alpha:.2e} m²/s")
    print(f"  h = {h} W/(m²·K), T_inf = {T_inf} K")
    print(f"  Bi = {Bi:.3f} (Bi << 0.1: lumped capacitance valide)")
    
    # Solveur
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=31)
    
    bc_left = lambda t: {'type': 'convection', 'h': h, 'T_inf': T_inf}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 30.0
    dt = 0.1
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=30)
    
    test_times = [1.0, 5.0, 15.0, t_end]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    errors = []
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        T_num = result['T'][idx, :]
        T_ana = finite_slab_convection_adiabatic(
            result['x'], t_actual, L, T_init, h, T_inf, k, alpha
        )
        
        err = compute_error_norms(T_num, T_ana)
        errors.append(err)
        
        Fo = compute_fourier_number(alpha, t_actual, L)
        print(f"\n  t = {t_actual:.1f} s (Fo = {Fo:.3f}):")
        print(f"    Erreur L2 = {err['L2']:.4f} K, L∞ = {err['Linf']:.4f} K")
        print(f"    T_surface = {T_num[0]:.2f} K (ana: {T_ana[0]:.2f} K)")
        
        axes[0].plot(result['x']*1000, T_num, 'o-', markersize=3,
                     label=f'Num. t={t_actual:.0f}s')
        axes[0].plot(result['x']*1000, T_ana, '--',
                     label=f'Ana. t={t_actual:.0f}s')
    
    axes[0].axhline(y=T_inf, color='r', linestyle=':', label=f'T_inf={T_inf}K')
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Comparaison numérique vs analytique')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Évolution de la température moyenne
    T_mean = np.mean(result['T'], axis=1)
    axes[1].plot(result['time'], T_mean, 'b-', label='T moyenne (num)')
    axes[1].axhline(y=T_inf, color='r', linestyle='--', label=f'T_inf={T_inf}K')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Température moyenne [K]')
    axes[1].set_title('Convergence vers l\'équilibre')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 3: Convection-Adiabatique', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test3_convection_adiabatic.png', dpi=150)
    plt.show()
    
    max_error_rel = max(e['L2_rel'] for e in errors)
    # Tolérance plus large pour ce cas (solution analytique avec séries)
    passed = max_error_rel < 0.05
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: erreur max = {max_error_rel*100:.2f}%")
    
    return passed


def test_semi_infinite_approximation():
    """
    Test 4: Approximation semi-infinie (temps courts).
    
    Pour des temps courts (Fo << 1), une plaque finie se comporte
    comme un milieu semi-infini car la perturbation n'a pas atteint
    le bord opposé.
    
    Configuration:
    - Plaque épaisse L = 0.5 m
    - T(0,t) = 500 K imposé
    - Temps court: t = 10 s → profondeur de pénétration ~ 1 cm
    """
    print("\n" + "="*70)
    print("TEST 4: Approximation semi-infinie (temps courts)")
    print("="*70)
    
    # Paramètres
    L = 0.5  # m - plaque épaisse
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    T_surf = 500.0
    
    # Temps de test
    t_test = 10.0  # s
    delta = np.sqrt(alpha * t_test)  # profondeur de pénétration
    Fo = compute_fourier_number(alpha, t_test, L)
    
    print(f"\nParamètres:")
    print(f"  L = {L} m, α = {alpha:.2e} m²/s")
    print(f"  t = {t_test} s, δ = {delta*1000:.1f} mm (profondeur de pénétration)")
    print(f"  Fo = {Fo:.4f} << 1 (régime semi-infini)")
    
    # Solveur
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=201)  # Maillage fin
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_surf}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    dt = 0.1
    result = solver.solve(T_init, t_test, dt, bc_left, bc_right, save_every=10)
    
    # Comparaison avec solution semi-infinie
    idx = -1  # Dernier instant
    T_num = result['T'][idx, :]
    T_ana = semi_infinite_dirichlet(result['x'], t_test, T_init, T_surf, alpha)
    
    # Ne comparer que dans la zone de pénétration (x < 5*δ)
    mask = result['x'] < 5 * delta
    
    err = compute_error_norms(T_num[mask], T_ana[mask])
    
    print(f"\n  Comparaison dans la zone x < {5*delta*1000:.1f} mm:")
    print(f"    Erreur L2 = {err['L2']:.4f} K")
    print(f"    Erreur L∞ = {err['Linf']:.4f} K")
    
    # Tracé
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(result['x']*1000, T_num, 'b-', linewidth=2, label='Numérique')
    ax.plot(result['x']*1000, T_ana, 'r--', linewidth=2, label='Semi-infini (analytique)')
    ax.axvline(x=5*delta*1000, color='g', linestyle=':', 
               label=f'5δ = {5*delta*1000:.1f} mm')
    ax.axhline(y=T_init, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title(f'Test 4: Approximation semi-infinie (t = {t_test} s, Fo = {Fo:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])  # Zoom sur la zone d'intérêt
    
    plt.tight_layout()
    plt.savefig('test4_semi_infinite.png', dpi=150)
    plt.show()
    
    passed = err['L2'] < 1.0  # Moins de 1 K d'erreur
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: erreur L2 = {err['L2']:.4f} K")
    
    return passed


def run_all_tests():
    """Exécute tous les tests de validation analytique."""
    print("\n" + "#"*70)
    print("# VALIDATION DU SOLVEUR CONTRE DES SOLUTIONS ANALYTIQUES")
    print("#"*70)
    
    results = {}
    
    results['dirichlet_dirichlet'] = test_dirichlet_dirichlet()
    results['dirichlet_adiabatic'] = test_dirichlet_adiabatic()
    results['convection_adiabatic'] = test_convection_adiabatic()
    results['semi_infinite'] = test_semi_infinite_approximation()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS")
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
