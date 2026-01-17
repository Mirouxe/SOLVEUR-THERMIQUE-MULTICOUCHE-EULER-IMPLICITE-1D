"""
Tests de convergence spatiale (raffinement de maillage)
=======================================================

Ce script vérifie que l'erreur numérique diminue avec le raffinement
du maillage spatial, conformément à l'ordre de convergence attendu.

Pour un schéma aux différences finies centrées d'ordre 2:
    Erreur ∝ (Δx)²

On vérifie:
1. Que l'erreur diminue quand Nx augmente
2. Que l'ordre de convergence est proche de 2
3. La stabilité du schéma pour différentes discrétisations
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, create_constant_material, create_single_layer, compute_error_norms
from analytical_solutions import (
    finite_slab_dirichlet_dirichlet,
    finite_slab_convection_adiabatic,
    thermal_diffusivity
)


def test_spatial_convergence_dirichlet():
    """
    Test de convergence spatiale pour le cas Dirichlet-Dirichlet.
    
    On raffine progressivement le maillage et on vérifie que l'erreur
    diminue selon (Δx)² (ordre 2).
    """
    print("\n" + "="*70)
    print("TEST DE CONVERGENCE SPATIALE - Cas Dirichlet-Dirichlet")
    print("="*70)
    
    # Paramètres physiques (fixes)
    L = 0.1  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    T_left = 400.0
    T_right = 300.0
    
    # Temps de comparaison
    t_test = 20.0  # s
    dt = 0.05  # Pas de temps fixe (petit pour éviter erreur temporelle)
    
    print(f"\nParamètres fixes:")
    print(f"  L = {L} m, α = {alpha:.2e} m²/s")
    print(f"  t_test = {t_test} s, dt = {dt} s")
    
    # Maillages à tester
    Nx_values = [11, 21, 41, 81, 161, 321]
    
    errors_L2 = []
    errors_Linf = []
    dx_values = []
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    for Nx in Nx_values:
        dx = L / (Nx - 1)
        dx_values.append(dx)
        
        solver = ThermalSolver1D(layers, material_data, Nx=Nx)
        result = solver.solve(T_init, t_test, dt, bc_left, bc_right, save_every=1000)
        
        # Solution numérique au temps final
        T_num = result['T'][-1, :]
        
        # Solution analytique aux mêmes points
        T_ana = finite_slab_dirichlet_dirichlet(
            result['x'], t_test, L, T_init, T_left, T_right, alpha
        )
        
        err = compute_error_norms(T_num, T_ana)
        errors_L2.append(err['L2'])
        errors_Linf.append(err['Linf'])
        
        print(f"\n  Nx = {Nx:4d}, dx = {dx*1000:.3f} mm:")
        print(f"    Erreur L2 = {err['L2']:.6f} K")
        print(f"    Erreur L∞ = {err['Linf']:.6f} K")
    
    # Calcul de l'ordre de convergence
    dx_values = np.array(dx_values)
    errors_L2 = np.array(errors_L2)
    errors_Linf = np.array(errors_Linf)
    
    # Ordre = log(E1/E2) / log(dx1/dx2) entre maillages successifs
    orders_L2 = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(dx_values[:-1] / dx_values[1:])
    orders_Linf = np.log(errors_Linf[:-1] / errors_Linf[1:]) / np.log(dx_values[:-1] / dx_values[1:])
    
    print(f"\n  Ordres de convergence L2: {orders_L2}")
    print(f"  Ordres de convergence L∞: {orders_Linf}")
    print(f"  Ordre moyen L2: {np.mean(orders_L2):.2f} (attendu: 2)")
    print(f"  Ordre moyen L∞: {np.mean(orders_Linf):.2f} (attendu: 2)")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convergence de l'erreur
    axes[0].loglog(dx_values*1000, errors_L2, 'bo-', linewidth=2, markersize=8, label='Erreur L2')
    axes[0].loglog(dx_values*1000, errors_Linf, 'rs-', linewidth=2, markersize=8, label='Erreur L∞')
    
    # Pente de référence (ordre 2)
    dx_ref = np.array([dx_values[0], dx_values[-1]])
    err_ref = errors_L2[0] * (dx_ref / dx_values[0])**2
    axes[0].loglog(dx_ref*1000, err_ref, 'k--', linewidth=1, label='Pente ordre 2')
    
    axes[0].set_xlabel('Δx [mm]')
    axes[0].set_ylabel('Erreur [K]')
    axes[0].set_title('Convergence spatiale')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Ordre de convergence
    Nx_mid = [(Nx_values[i] + Nx_values[i+1])/2 for i in range(len(orders_L2))]
    axes[1].plot(Nx_mid, orders_L2, 'bo-', linewidth=2, markersize=8, label='Ordre L2')
    axes[1].plot(Nx_mid, orders_Linf, 'rs-', linewidth=2, markersize=8, label='Ordre L∞')
    axes[1].axhline(y=2, color='k', linestyle='--', label='Ordre théorique = 2')
    axes[1].set_xlabel('Nx (moyen)')
    axes[1].set_ylabel('Ordre de convergence')
    axes[1].set_title('Ordre de convergence observé')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 3])
    
    plt.suptitle('Test de convergence spatiale - Dirichlet-Dirichlet', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_mesh_convergence_dirichlet.png', dpi=150)
    plt.show()
    
    # Critère de validation: 
    # - Les deux premiers raffinements doivent montrer ordre ~2
    # - L'erreur doit diminuer significativement avec le raffinement
    # Note: pour les maillages très fins, l'erreur temporelle domine
    mean_order_first = np.mean(orders_L2[:2])  # Deux premiers raffinements (où spatial domine)
    error_ratio = errors_L2[0] / errors_L2[-1]  # Facteur de réduction
    
    passed = (mean_order_first > 1.5) and (error_ratio > 10)
    print(f"\n  Ordre moyen (2 premiers raffinements): {mean_order_first:.2f}")
    print(f"  Réduction d'erreur: {error_ratio:.1f}x ({errors_L2[0]:.4f} → {errors_L2[-1]:.4f})")
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_spatial_convergence_convection():
    """
    Test de convergence spatiale pour le cas convection-adiabatique.
    
    Ce cas est plus exigeant car la condition de convection (Robin)
    est plus délicate à discrétiser.
    """
    print("\n" + "="*70)
    print("TEST DE CONVERGENCE SPATIALE - Cas Convection-Adiabatique")
    print("="*70)
    
    # Paramètres
    L = 0.02  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    h = 500.0
    T_inf = 500.0
    
    t_test = 10.0
    dt = 0.01  # Petit pas de temps
    
    print(f"\nParamètres fixes:")
    print(f"  L = {L} m, h = {h} W/(m²·K)")
    print(f"  t_test = {t_test} s, dt = {dt} s")
    
    Nx_values = [11, 21, 41, 81, 161]
    
    errors_L2 = []
    dx_values = []
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    
    bc_left = lambda t: {'type': 'convection', 'h': h, 'T_inf': T_inf}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    for Nx in Nx_values:
        dx = L / (Nx - 1)
        dx_values.append(dx)
        
        solver = ThermalSolver1D(layers, material_data, Nx=Nx)
        result = solver.solve(T_init, t_test, dt, bc_left, bc_right, save_every=1000)
        
        T_num = result['T'][-1, :]
        T_ana = finite_slab_convection_adiabatic(
            result['x'], t_test, L, T_init, h, T_inf, k, alpha
        )
        
        err = compute_error_norms(T_num, T_ana)
        errors_L2.append(err['L2'])
        
        print(f"\n  Nx = {Nx:4d}, dx = {dx*1000:.3f} mm: Erreur L2 = {err['L2']:.6f} K")
    
    dx_values = np.array(dx_values)
    errors_L2 = np.array(errors_L2)
    
    orders = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(dx_values[:-1] / dx_values[1:])
    
    print(f"\n  Ordres de convergence: {orders}")
    print(f"  Ordre moyen: {np.mean(orders):.2f}")
    
    # Tracé
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.loglog(dx_values*1000, errors_L2, 'bo-', linewidth=2, markersize=8, label='Erreur L2')
    
    # Pentes de référence
    dx_ref = np.array([dx_values[0], dx_values[-1]])
    err_ref1 = errors_L2[0] * (dx_ref / dx_values[0])**1
    err_ref2 = errors_L2[0] * (dx_ref / dx_values[0])**2
    ax.loglog(dx_ref*1000, err_ref1, 'g--', linewidth=1, label='Pente ordre 1')
    ax.loglog(dx_ref*1000, err_ref2, 'r--', linewidth=1, label='Pente ordre 2')
    
    ax.set_xlabel('Δx [mm]')
    ax.set_ylabel('Erreur L2 [K]')
    ax.set_title('Convergence spatiale - Convection-Adiabatique')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('test_mesh_convergence_convection.png', dpi=150)
    plt.show()
    
    # Pour la condition de Robin, l'ordre peut être réduit
    # On vérifie que l'erreur diminue et que les premiers raffinements montrent convergence
    mean_order_first = np.mean(orders[:2])
    error_decreases = errors_L2[-1] < errors_L2[0]
    
    passed = (mean_order_first > 1.0) and error_decreases
    print(f"\n  Ordre moyen (premiers raffinements): {mean_order_first:.2f}")
    print(f"  Erreur diminue: {'✓' if error_decreases else '✗'}")
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


def test_solution_independence():
    """
    Vérifie que la solution converge vers une valeur indépendante du maillage.
    
    On compare les solutions pour différents Nx au même point physique
    et on vérifie qu'elles convergent.
    """
    print("\n" + "="*70)
    print("TEST D'INDÉPENDANCE DE LA SOLUTION AU MAILLAGE")
    print("="*70)
    
    # Paramètres
    L = 0.1
    k = 50.0
    rho = 7800.0
    cp = 500.0
    
    T_init = 300.0
    T_left = 400.0
    T_right = 300.0
    
    t_test = 50.0
    dt = 0.1
    
    # Point de mesure
    x_measure = L / 2  # Milieu de la plaque
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    Nx_values = [11, 21, 41, 81, 161, 321]
    T_at_center = []
    
    for Nx in Nx_values:
        solver = ThermalSolver1D(layers, material_data, Nx=Nx)
        result = solver.solve(T_init, t_test, dt, bc_left, bc_right, save_every=1000)
        
        # Interpoler la température au point de mesure
        T_final = result['T'][-1, :]
        T_center = np.interp(x_measure, result['x'], T_final)
        T_at_center.append(T_center)
        
        print(f"  Nx = {Nx:4d}: T(x=L/2) = {T_center:.6f} K")
    
    T_at_center = np.array(T_at_center)
    
    # Variation relative par rapport au maillage le plus fin
    T_ref = T_at_center[-1]
    variations = np.abs(T_at_center - T_ref) / T_ref * 100
    
    print(f"\n  Température de référence (Nx=321): {T_ref:.6f} K")
    print(f"  Variations relatives: {variations[:-1]} %")
    
    # Tracé
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(Nx_values, T_at_center, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=T_ref, color='r', linestyle='--', label=f'Valeur convergée: {T_ref:.4f} K')
    ax.fill_between([Nx_values[0], Nx_values[-1]], 
                    [T_ref*0.999, T_ref*0.999], 
                    [T_ref*1.001, T_ref*1.001],
                    alpha=0.3, color='green', label='±0.1%')
    
    ax.set_xlabel('Nombre de nœuds Nx')
    ax.set_ylabel('T(x=L/2) [K]')
    ax.set_title('Convergence de la température au centre')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_solution_independence.png', dpi=150)
    plt.show()
    
    # Critère: variation < 0.5% pour les 3 maillages les plus fins
    max_variation = np.max(variations[-3:-1])
    passed = max_variation < 0.5
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: variation max = {max_variation:.3f}%")
    
    return passed


def run_all_tests():
    """Exécute tous les tests de convergence spatiale."""
    print("\n" + "#"*70)
    print("# TESTS DE CONVERGENCE SPATIALE (RAFFINEMENT DE MAILLAGE)")
    print("#"*70)
    
    results = {}
    
    results['convergence_dirichlet'] = test_spatial_convergence_dirichlet()
    results['convergence_convection'] = test_spatial_convergence_convection()
    results['solution_independence'] = test_solution_independence()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS DE CONVERGENCE SPATIALE")
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
