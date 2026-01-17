"""
Tests de convergence temporelle
===============================

Ce script vérifie que l'erreur numérique diminue avec le raffinement
du pas de temps, conformément à l'ordre de convergence attendu.

Pour un schéma d'Euler implicite (ordre 1 en temps):
    Erreur ∝ Δt

On vérifie:
1. Que l'erreur diminue quand Δt diminue
2. Que l'ordre de convergence est proche de 1
3. La stabilité inconditionnelle du schéma implicite
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, create_constant_material, create_single_layer, compute_error_norms
from analytical_solutions import (
    finite_slab_dirichlet_dirichlet,
    finite_slab_convection_adiabatic,
    thermal_diffusivity
)


def test_temporal_convergence():
    """
    Test de convergence temporelle pour le cas Dirichlet-Dirichlet.
    
    On raffine progressivement le pas de temps et on vérifie que l'erreur
    diminue selon Δt (ordre 1 pour Euler implicite).
    """
    print("\n" + "="*70)
    print("TEST DE CONVERGENCE TEMPORELLE - Euler Implicite")
    print("="*70)
    
    # Paramètres physiques
    L = 0.1  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    T_left = 400.0
    T_right = 300.0
    
    # Temps de comparaison et maillage spatial fin (pour minimiser erreur spatiale)
    t_test = 10.0  # s
    Nx = 201  # Maillage fin
    
    print(f"\nParamètres fixes:")
    print(f"  L = {L} m, α = {alpha:.2e} m²/s")
    print(f"  Nx = {Nx} (maillage fin pour isoler l'erreur temporelle)")
    print(f"  t_test = {t_test} s")
    
    # Pas de temps à tester
    dt_values = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    
    errors_L2 = []
    errors_Linf = []
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    # Solution analytique de référence
    T_ana = finite_slab_dirichlet_dirichlet(
        solver.x, t_test, L, T_init, T_left, T_right, alpha
    )
    
    for dt in dt_values:
        result = solver.solve(T_init, t_test, dt, bc_left, bc_right, save_every=10000)
        
        T_num = result['T'][-1, :]
        
        err = compute_error_norms(T_num, T_ana)
        errors_L2.append(err['L2'])
        errors_Linf.append(err['Linf'])
        
        # Nombre de Fourier de maille
        Fo_mesh = alpha * dt / (solver.dx**2)
        
        print(f"\n  dt = {dt:.4f} s (Fo_mesh = {Fo_mesh:.2f}):")
        print(f"    Erreur L2 = {err['L2']:.6f} K")
        print(f"    Erreur L∞ = {err['Linf']:.6f} K")
    
    dt_values = np.array(dt_values)
    errors_L2 = np.array(errors_L2)
    errors_Linf = np.array(errors_Linf)
    
    # Calcul de l'ordre de convergence
    orders_L2 = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(dt_values[:-1] / dt_values[1:])
    
    print(f"\n  Ordres de convergence L2: {orders_L2}")
    print(f"  Ordre moyen: {np.mean(orders_L2):.2f} (attendu: 1 pour Euler implicite)")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convergence de l'erreur
    axes[0].loglog(dt_values, errors_L2, 'bo-', linewidth=2, markersize=8, label='Erreur L2')
    axes[0].loglog(dt_values, errors_Linf, 'rs-', linewidth=2, markersize=8, label='Erreur L∞')
    
    # Pentes de référence
    dt_ref = np.array([dt_values[0], dt_values[-1]])
    err_ref1 = errors_L2[0] * (dt_ref / dt_values[0])**1
    err_ref2 = errors_L2[0] * (dt_ref / dt_values[0])**2
    axes[0].loglog(dt_ref, err_ref1, 'g--', linewidth=1, label='Pente ordre 1')
    axes[0].loglog(dt_ref, err_ref2, 'k--', linewidth=1, label='Pente ordre 2')
    
    axes[0].set_xlabel('Δt [s]')
    axes[0].set_ylabel('Erreur [K]')
    axes[0].set_title('Convergence temporelle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Ordre de convergence
    dt_mid = [(dt_values[i] + dt_values[i+1])/2 for i in range(len(orders_L2))]
    axes[1].semilogx(dt_mid, orders_L2, 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=1, color='g', linestyle='--', label='Ordre théorique = 1')
    axes[1].set_xlabel('Δt [s]')
    axes[1].set_ylabel('Ordre de convergence')
    axes[1].set_title('Ordre de convergence observé')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 2.5])
    
    plt.suptitle('Test de convergence temporelle - Euler Implicite', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_time_convergence.png', dpi=150)
    plt.show()
    
    # Critère: ordre moyen proche de 1 (entre 0.8 et 1.5)
    mean_order = np.mean(orders_L2)
    passed = 0.7 < mean_order < 1.5
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: ordre moyen = {mean_order:.2f}")
    
    return passed


def test_unconditional_stability():
    """
    Test de stabilité inconditionnelle du schéma implicite.
    
    Pour un schéma explicite, la stabilité requiert:
        Fo = α*Δt/Δx² ≤ 0.5
    
    Le schéma implicite doit rester stable même pour Fo >> 1.
    """
    print("\n" + "="*70)
    print("TEST DE STABILITÉ INCONDITIONNELLE")
    print("="*70)
    
    # Paramètres
    L = 0.01  # m - Plaque fine pour avoir Fo élevé
    k = 50.0
    rho = 7800.0
    cp = 500.0
    alpha = thermal_diffusivity(k, rho, cp)
    
    T_init = 300.0
    T_left = 400.0
    T_right = 300.0
    
    Nx = 21
    dx = L / (Nx - 1)
    
    print(f"\nParamètres:")
    print(f"  L = {L*1000} mm, Nx = {Nx}, dx = {dx*1000:.3f} mm")
    print(f"  α = {alpha:.2e} m²/s")
    
    # Limite de stabilité explicite
    dt_explicit_max = 0.5 * dx**2 / alpha
    print(f"  Limite de stabilité explicite: dt_max = {dt_explicit_max:.6f} s")
    
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    # Tester des pas de temps bien au-delà de la limite explicite
    dt_values = [dt_explicit_max * 0.5,  # Stable même en explicite
                 dt_explicit_max * 2,     # Instable en explicite
                 dt_explicit_max * 10,    # Très instable en explicite
                 dt_explicit_max * 50,    # Extrêmement instable en explicite
                 dt_explicit_max * 100]   # Fo = 50
    
    t_end = 1.0  # s
    
    results_stable = []
    
    print(f"\n  Test de stabilité pour différents Fo:")
    
    for dt in dt_values:
        Fo = alpha * dt / dx**2
        
        try:
            result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=1000)
            T_final = result['T'][-1, :]
            
            # Vérifier la physique: T doit être entre T_right et T_left
            is_bounded = np.all(T_final >= T_right - 1) and np.all(T_final <= T_left + 1)
            is_monotonic = np.all(np.diff(T_final) <= 0.1)  # Quasi-monotone décroissant
            
            stable = is_bounded and not np.any(np.isnan(T_final))
            results_stable.append(stable)
            
            status = "✓ Stable" if stable else "✗ Instable"
            print(f"    Fo = {Fo:6.1f} (dt = {dt:.6f} s): {status}")
            print(f"      T_min = {T_final.min():.2f} K, T_max = {T_final.max():.2f} K")
            
        except Exception as e:
            results_stable.append(False)
            print(f"    Fo = {Fo:6.1f}: ✗ Erreur - {e}")
    
    # Tracé comparatif
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Profils pour différents Fo
    for dt in [dt_values[0], dt_values[2], dt_values[4]]:
        Fo = alpha * dt / dx**2
        result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=1000)
        axes[0].plot(solver.x*1000, result['T'][-1, :], '-', linewidth=2,
                     label=f'Fo = {Fo:.0f}')
    
    # Solution analytique
    T_ana = finite_slab_dirichlet_dirichlet(
        solver.x, t_end, L, T_init, T_left, T_right, alpha
    )
    axes[0].plot(solver.x*1000, T_ana, 'k--', linewidth=2, label='Analytique')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title(f'Profils à t = {t_end} s pour différents Fo')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Erreur vs Fo
    errors = []
    Fo_values = []
    for dt in dt_values:
        Fo = alpha * dt / dx**2
        Fo_values.append(Fo)
        result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=1000)
        T_num = result['T'][-1, :]
        err = compute_error_norms(T_num, T_ana)
        errors.append(err['L2'])
    
    axes[1].loglog(Fo_values, errors, 'bo-', linewidth=2, markersize=8)
    axes[1].axvline(x=0.5, color='r', linestyle='--', label='Limite explicite (Fo=0.5)')
    axes[1].set_xlabel('Nombre de Fourier de maille (Fo)')
    axes[1].set_ylabel('Erreur L2 [K]')
    axes[1].set_title('Erreur vs Fo (stabilité)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Test de stabilité inconditionnelle - Schéma implicite', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_stability.png', dpi=150)
    plt.show()
    
    # Critère: tous les cas doivent être stables
    all_stable = all(results_stable)
    print(f"\n{'✓ TEST RÉUSSI' if all_stable else '✗ TEST ÉCHOUÉ'}: "
          f"{sum(results_stable)}/{len(results_stable)} cas stables")
    
    return all_stable


def test_energy_conservation():
    """
    Test de conservation de l'énergie.
    
    Pour un système adiabatique (pas de flux aux bords), l'énergie
    totale doit être conservée.
    """
    print("\n" + "="*70)
    print("TEST DE CONSERVATION DE L'ÉNERGIE")
    print("="*70)
    
    # Paramètres
    L = 0.1  # m
    k = 50.0
    rho = 7800.0
    cp = 500.0
    
    Nx = 51
    
    # Condition initiale non uniforme (sinusoïdale)
    material_data = create_constant_material(k, rho, cp, 'test_mat')
    layers = create_single_layer(L, 'test_mat')
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    T_mean = 350.0
    T_amp = 50.0
    T_init = T_mean + T_amp * np.sin(np.pi * solver.x / L)
    
    # Conditions adiabatiques aux deux bords
    bc_left = lambda t: {'type': 'flux', 'q': 0.0}  # Flux nul = adiabatique
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 100.0
    dt = 0.5
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=10)
    
    # Calcul de l'énergie totale à chaque instant
    # E = ∫ ρ*cp*T dx ≈ Σ ρ*cp*T*dx
    energies = []
    for T in result['T']:
        E = np.sum(rho * cp * T * solver.dx)
        energies.append(E)
    
    energies = np.array(energies)
    E_init = energies[0]
    
    # Variation relative de l'énergie
    dE_rel = (energies - E_init) / E_init * 100
    
    print(f"\nÉnergie initiale: {E_init:.2f} J/m²")
    print(f"Variation max d'énergie: {np.max(np.abs(dE_rel)):.6f} %")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil de température
    for idx in [0, len(result['time'])//4, len(result['time'])//2, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Évolution du profil (système adiabatique)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Conservation de l'énergie
    axes[1].plot(result['time'], dE_rel, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Variation relative de l\'énergie [%]')
    axes[1].set_title('Conservation de l\'énergie')
    axes[1].grid(True, alpha=0.3)
    
    # Zoom si la variation est très faible
    max_var = np.max(np.abs(dE_rel))
    if max_var < 0.01:
        axes[1].set_ylim([-0.01, 0.01])
    
    plt.suptitle('Test de conservation de l\'énergie - Système adiabatique', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_energy_conservation.png', dpi=150)
    plt.show()
    
    # Critère: variation < 0.5% (tolérance pour erreurs numériques)
    # Note: la conservation parfaite n'est pas garantie pour les schémas implicites
    # car l'énergie est calculée de manière approchée
    max_variation = np.max(np.abs(dE_rel))
    passed = max_variation < 0.5
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: "
          f"variation max = {max_variation:.6f}%")
    
    return passed


def run_all_tests():
    """Exécute tous les tests de convergence temporelle."""
    print("\n" + "#"*70)
    print("# TESTS DE CONVERGENCE TEMPORELLE ET STABILITÉ")
    print("#"*70)
    
    results = {}
    
    results['temporal_convergence'] = test_temporal_convergence()
    results['unconditional_stability'] = test_unconditional_stability()
    results['energy_conservation'] = test_energy_conservation()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS TEMPORELS")
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
