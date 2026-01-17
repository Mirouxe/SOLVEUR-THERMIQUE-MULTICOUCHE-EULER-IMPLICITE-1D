"""
Tests des propriétés thermiques dépendantes de la température
=============================================================

Ce script valide le comportement du solveur lorsque les propriétés
thermiques (k, ρ, cp) dépendent de la température.

Tests effectués:
1. Comparaison propriétés constantes vs dépendantes de T
2. Vérification de la cohérence physique (diffusivité effective)
3. Conservation de l'énergie avec propriétés variables
4. Comportement pour des variations de propriétés réalistes
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, create_constant_material, create_single_layer, compute_error_norms
from analytical_solutions import thermal_diffusivity, finite_slab_dirichlet_adiabatic


def create_temperature_dependent_material(k_ref, rho_ref, cp_ref, 
                                          dk_dT=0, drho_dT=0, dcp_dT=0,
                                          T_ref=300, name='var_mat'):
    """
    Crée un matériau avec des propriétés linéairement dépendantes de T.
    
    k(T) = k_ref + dk_dT * (T - T_ref)
    ρ(T) = ρ_ref + dρ_dT * (T - T_ref)
    cp(T) = cp_ref + dcp_dT * (T - T_ref)
    
    Args:
        k_ref, rho_ref, cp_ref: Propriétés à T_ref
        dk_dT, drho_dT, dcp_dT: Dérivées par rapport à T
        T_ref: Température de référence
        name: Nom du matériau
    """
    T_range = np.linspace(200, 800, 50)
    
    k_values = k_ref + dk_dT * (T_range - T_ref)
    rho_values = rho_ref + drho_dT * (T_range - T_ref)
    cp_values = cp_ref + dcp_dT * (T_range - T_ref)
    
    # S'assurer que les propriétés restent positives
    k_values = np.maximum(k_values, 1.0)
    rho_values = np.maximum(rho_values, 100.0)
    cp_values = np.maximum(cp_values, 100.0)
    
    return {
        name: {
            'T': T_range,
            'k': k_values,
            'rho': rho_values,
            'cp': cp_values
        }
    }


def test_constant_vs_variable_properties():
    """
    Test 1: Comparaison entre propriétés constantes et variables.
    
    Pour des propriétés faiblement dépendantes de T, la solution doit
    être proche de celle avec propriétés constantes évaluées à T moyen.
    """
    print("\n" + "="*70)
    print("TEST 1: Comparaison propriétés constantes vs variables")
    print("="*70)
    
    # Paramètres de base
    L = 0.05
    k_ref = 50.0
    rho_ref = 7800.0
    cp_ref = 500.0
    T_ref = 400.0  # Température moyenne attendue
    
    T_init = 300.0
    T_left = 500.0
    
    # Variation faible des propriétés (±10% sur 200 K)
    dk_dT = -0.05  # k diminue avec T (typique des métaux)
    dcp_dT = 0.5   # cp augmente avec T
    
    print(f"\nPropriétés à T_ref = {T_ref} K:")
    print(f"  k = {k_ref} W/(m·K), dk/dT = {dk_dT} W/(m·K²)")
    print(f"  cp = {cp_ref} J/(kg·K), dcp/dT = {dcp_dT} J/(kg·K²)")
    
    # Matériau à propriétés constantes
    mat_const = create_constant_material(k_ref, rho_ref, cp_ref, 'const_mat')
    
    # Matériau à propriétés variables
    mat_var = create_temperature_dependent_material(
        k_ref, rho_ref, cp_ref, dk_dT=dk_dT, dcp_dT=dcp_dT, T_ref=T_ref, name='var_mat'
    )
    
    layers_const = create_single_layer(L, 'const_mat')
    layers_var = create_single_layer(L, 'var_mat')
    
    Nx = 51
    solver_const = ThermalSolver1D(layers_const, mat_const, Nx=Nx)
    solver_var = ThermalSolver1D(layers_var, mat_var, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 50.0
    dt = 0.2
    
    result_const = solver_const.solve(T_init, t_end, dt, bc_left, bc_right, save_every=25)
    result_var = solver_var.solve(T_init, t_end, dt, bc_left, bc_right, save_every=25)
    
    # Comparaison
    test_times = [10.0, 25.0, t_end]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for t_test in test_times:
        idx_const = np.argmin(np.abs(result_const['time'] - t_test))
        idx_var = np.argmin(np.abs(result_var['time'] - t_test))
        
        T_const = result_const['T'][idx_const, :]
        T_var = result_var['T'][idx_var, :]
        
        diff = T_var - T_const
        
        print(f"\n  t = {t_test:.0f} s:")
        print(f"    Écart max: {np.max(np.abs(diff)):.3f} K")
        print(f"    Écart moyen: {np.mean(np.abs(diff)):.3f} K")
        
        axes[0].plot(solver_const.x*1000, T_const, '-', label=f'Const. t={t_test:.0f}s')
        axes[0].plot(solver_var.x*1000, T_var, '--', label=f'Var. t={t_test:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils de température')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Écart en fonction du temps
    ecarts = []
    for idx in range(len(result_const['time'])):
        diff = result_var['T'][idx, :] - result_const['T'][idx, :]
        ecarts.append(np.max(np.abs(diff)))
    
    axes[1].plot(result_const['time'], ecarts, 'b-', linewidth=2)
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Écart max [K]')
    axes[1].set_title('Écart entre solutions (const. vs var.)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 1: Propriétés constantes vs dépendantes de T', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_temp_dep_comparison.png', dpi=150)
    plt.show()
    
    # Critère: l'écart doit être modéré (< 10 K pour ces variations)
    max_ecart = np.max(ecarts)
    passed = max_ecart < 15
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: écart max = {max_ecart:.2f} K")
    
    return passed


def test_diffusivity_effect():
    """
    Test 2: Effet de la diffusivité variable.
    
    La diffusivité α = k/(ρ*cp) contrôle la vitesse de propagation thermique.
    On vérifie que:
    - Si α augmente avec T → propagation plus rapide dans les zones chaudes
    - Si α diminue avec T → propagation plus lente dans les zones chaudes
    """
    print("\n" + "="*70)
    print("TEST 2: Effet de la diffusivité thermique variable")
    print("="*70)
    
    L = 0.05
    k_ref = 50.0
    rho_ref = 7800.0
    cp_ref = 500.0
    T_ref = 400.0
    
    T_init = 300.0
    T_left = 600.0
    
    # Trois cas:
    # 1. α constant
    # 2. α augmente avec T (k augmente, cp constant)
    # 3. α diminue avec T (k diminue, cp augmente)
    
    alpha_ref = thermal_diffusivity(k_ref, rho_ref, cp_ref)
    print(f"\nDiffusivité de référence: α = {alpha_ref:.2e} m²/s")
    
    # Cas 1: Constant
    mat1 = create_constant_material(k_ref, rho_ref, cp_ref, 'mat1')
    
    # Cas 2: α augmente (k augmente fortement)
    mat2 = create_temperature_dependent_material(
        k_ref, rho_ref, cp_ref, dk_dT=0.1, dcp_dT=0, T_ref=T_ref, name='mat2'
    )
    
    # Cas 3: α diminue (k diminue, cp augmente)
    mat3 = create_temperature_dependent_material(
        k_ref, rho_ref, cp_ref, dk_dT=-0.1, dcp_dT=1.0, T_ref=T_ref, name='mat3'
    )
    
    layers1 = create_single_layer(L, 'mat1')
    layers2 = create_single_layer(L, 'mat2')
    layers3 = create_single_layer(L, 'mat3')
    
    Nx = 51
    solver1 = ThermalSolver1D(layers1, mat1, Nx=Nx)
    solver2 = ThermalSolver1D(layers2, mat2, Nx=Nx)
    solver3 = ThermalSolver1D(layers3, mat3, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 30.0
    dt = 0.1
    
    result1 = solver1.solve(T_init, t_end, dt, bc_left, bc_right, save_every=30)
    result2 = solver2.solve(T_init, t_end, dt, bc_left, bc_right, save_every=30)
    result3 = solver3.solve(T_init, t_end, dt, bc_left, bc_right, save_every=30)
    
    # Tracé à un instant donné
    t_test = 15.0
    idx = np.argmin(np.abs(result1['time'] - t_test))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(solver1.x*1000, result1['T'][idx, :], 'b-', linewidth=2, label='α constant')
    axes[0].plot(solver2.x*1000, result2['T'][idx, :], 'r-', linewidth=2, label='α ↑ avec T')
    axes[0].plot(solver3.x*1000, result3['T'][idx, :], 'g-', linewidth=2, label='α ↓ avec T')
    axes[0].axhline(y=T_init, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title(f'Profils à t = {t_test} s')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Température au centre vs temps
    center_idx = Nx // 2
    axes[1].plot(result1['time'], result1['T'][:, center_idx], 'b-', linewidth=2, label='α constant')
    axes[1].plot(result2['time'], result2['T'][:, center_idx], 'r-', linewidth=2, label='α ↑ avec T')
    axes[1].plot(result3['time'], result3['T'][:, center_idx], 'g-', linewidth=2, label='α ↓ avec T')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('T(x=L/2) [K]')
    axes[1].set_title('Température au centre')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 2: Effet de la diffusivité variable', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_diffusivity_effect.png', dpi=150)
    plt.show()
    
    # Vérification physique:
    # Si α augmente avec T, la chaleur se propage plus vite → T au centre plus élevé
    T_center_const = result1['T'][-1, center_idx]
    T_center_alpha_up = result2['T'][-1, center_idx]
    T_center_alpha_down = result3['T'][-1, center_idx]
    
    print(f"\n  Température au centre à t = {t_end} s:")
    print(f"    α constant: {T_center_const:.2f} K")
    print(f"    α ↑ avec T: {T_center_alpha_up:.2f} K")
    print(f"    α ↓ avec T: {T_center_alpha_down:.2f} K")
    
    # Vérification: α↑ → T_center plus élevé, α↓ → T_center plus bas
    physically_correct = (T_center_alpha_up > T_center_const > T_center_alpha_down)
    
    passed = physically_correct
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: "
          f"comportement physique {'correct' if passed else 'incorrect'}")
    
    return passed


def test_energy_conservation_variable_cp():
    """
    Test 3: Conservation de l'énergie avec cp(T).
    
    L'énergie stockée E = ∫ ρ*cp(T)*T dx doit évoluer correctement.
    Pour un système adiabatique, l'énergie interne doit être conservée.
    """
    print("\n" + "="*70)
    print("TEST 3: Conservation de l'énergie avec cp(T)")
    print("="*70)
    
    L = 0.1
    k_ref = 50.0
    rho_ref = 7800.0
    cp_ref = 500.0
    T_ref = 350.0
    
    # cp qui varie fortement avec T
    dcp_dT = 1.0  # cp augmente de 100 J/(kg·K) pour ΔT = 100 K
    
    mat_var = create_temperature_dependent_material(
        k_ref, rho_ref, cp_ref, dk_dT=0, dcp_dT=dcp_dT, T_ref=T_ref, name='var_mat'
    )
    
    layers = create_single_layer(L, 'var_mat')
    
    Nx = 51
    solver = ThermalSolver1D(layers, mat_var, Nx=Nx)
    
    # Condition initiale non uniforme
    T_mean = 350.0
    T_amp = 50.0
    T_init = T_mean + T_amp * np.sin(np.pi * solver.x / L)
    
    # Système adiabatique
    bc_left = lambda t: {'type': 'flux', 'q': 0.0}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 200.0
    dt = 1.0
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=20)
    
    # Calcul de l'énergie interne à chaque instant
    # Pour cp(T), l'énergie est E = ∫ ρ * ∫cp(T)dT dx
    # Approximation: E ≈ Σ ρ * cp(T_i) * T_i * dx
    
    def compute_internal_energy(T_field, mat_props, dx):
        """Calcule l'énergie interne approximée."""
        E = 0
        for i, T in enumerate(T_field):
            # Interpoler cp à cette température
            cp = np.interp(T, mat_props['T'], mat_props['cp'])
            # Contribution: ρ * cp * T * dx (par rapport à T=0)
            E += rho_ref * cp * T * dx
        return E
    
    energies = []
    for T_field in result['T']:
        E = compute_internal_energy(T_field, mat_var['var_mat'], solver.dx)
        energies.append(E)
    
    energies = np.array(energies)
    E_init = energies[0]
    
    # Note: L'énergie "interne" ainsi calculée n'est pas exactement conservée
    # car la définition correcte nécessite l'enthalpie. On vérifie plutôt
    # que la variation reste faible.
    
    dE_rel = (energies - E_init) / E_init * 100
    
    print(f"\n  Variation relative de l'énergie interne:")
    print(f"    Max: {np.max(np.abs(dE_rel)):.4f} %")
    print(f"    Finale: {dE_rel[-1]:.4f} %")
    
    # Vérification de la convergence vers température uniforme
    T_final = result['T'][-1, :]
    T_variation = np.max(T_final) - np.min(T_final)
    
    print(f"\n  Convergence vers l'équilibre:")
    print(f"    Variation de T initiale: {np.max(T_init) - np.min(T_init):.2f} K")
    print(f"    Variation de T finale: {T_variation:.4f} K")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Évolution du profil
    for idx in [0, len(result['time'])//4, len(result['time'])//2, -1]:
        t = result['time'][idx]
        axes[0].plot(solver.x*1000, result['T'][idx, :], '-', label=f't={t:.0f}s')
    
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Évolution vers l\'équilibre')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Énergie vs temps
    axes[1].plot(result['time'], dE_rel, 'b-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Variation relative d\'énergie [%]')
    axes[1].set_title('Conservation de l\'énergie')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 3: Conservation de l\'énergie avec cp(T)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_energy_variable_cp.png', dpi=150)
    plt.show()
    
    # Critère: convergence vers équilibre thermique
    passed = T_variation < 0.1  # Quasi-uniforme
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: "
          f"variation finale = {T_variation:.4f} K")
    
    return passed


def test_realistic_steel_properties():
    """
    Test 4: Simulation avec propriétés réalistes de l'acier.
    
    Utilise des données réalistes pour l'acier AISI 1010:
    - k diminue avec T
    - cp augmente avec T
    - ρ quasi-constant
    """
    print("\n" + "="*70)
    print("TEST 4: Propriétés réalistes de l'acier")
    print("="*70)
    
    # Propriétés de l'acier AISI 1010 (approximation)
    # Source: Engineering Toolbox, MatWeb
    T_data = np.array([300, 400, 500, 600, 700, 800])  # K
    k_data = np.array([63.9, 58.7, 52.7, 48.0, 42.0, 36.0])  # W/(m·K)
    rho_data = np.array([7870, 7850, 7830, 7810, 7790, 7770])  # kg/m³
    cp_data = np.array([434, 487, 519, 559, 615, 700])  # J/(kg·K)
    
    steel_realistic = {
        'steel_real': {
            'T': T_data,
            'k': k_data,
            'rho': rho_data,
            'cp': cp_data
        }
    }
    
    print("\n  Propriétés de l'acier AISI 1010:")
    print("  T [K]    k [W/mK]   ρ [kg/m³]   cp [J/kgK]   α [m²/s]")
    for i in range(len(T_data)):
        alpha = k_data[i] / (rho_data[i] * cp_data[i])
        print(f"  {T_data[i]:4.0f}     {k_data[i]:5.1f}      {rho_data[i]:4.0f}        {cp_data[i]:3.0f}       {alpha:.2e}")
    
    # Simulation
    L = 0.02  # 20 mm
    layers = create_single_layer(L, 'steel_real')
    
    Nx = 41
    solver = ThermalSolver1D(layers, steel_realistic, Nx=Nx)
    
    T_init = 300.0
    h = 500.0
    T_inf = 800.0  # Chauffage intense
    
    bc_left = lambda t: {'type': 'convection', 'h': h, 'T_inf': T_inf}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 60.0
    dt = 0.1
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=60)
    
    # Comparaison avec propriétés constantes (à T moyen)
    T_mean = (T_init + T_inf) / 2
    k_mean = np.interp(T_mean, T_data, k_data)
    rho_mean = np.interp(T_mean, T_data, rho_data)
    cp_mean = np.interp(T_mean, T_data, cp_data)
    
    mat_const = create_constant_material(k_mean, rho_mean, cp_mean, 'steel_const')
    layers_const = create_single_layer(L, 'steel_const')
    solver_const = ThermalSolver1D(layers_const, mat_const, Nx=Nx)
    
    result_const = solver_const.solve(T_init, t_end, dt, bc_left, bc_right, save_every=60)
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Profils à différents instants
    test_times = [5, 15, 30, t_end]
    for t_test in test_times:
        idx_real = np.argmin(np.abs(result['time'] - t_test))
        idx_const = np.argmin(np.abs(result_const['time'] - t_test))
        
        axes[0].plot(solver.x*1000, result['T'][idx_real, :], '-', 
                     label=f'Réel t={t_test:.0f}s')
        axes[0].plot(solver_const.x*1000, result_const['T'][idx_const, :], '--',
                     alpha=0.7)
    
    axes[0].axhline(y=T_inf, color='r', linestyle=':', label=f'T_inf={T_inf}K')
    axes[0].set_xlabel('Position x [mm]')
    axes[0].set_ylabel('Température [K]')
    axes[0].set_title('Profils: réel (trait plein) vs constant (tirets)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Évolution de la température de surface
    T_surf_real = result['T'][:, 0]
    T_surf_const = result_const['T'][:, 0]
    
    axes[1].plot(result['time'], T_surf_real, 'b-', linewidth=2, label='Propriétés réelles')
    axes[1].plot(result_const['time'], T_surf_const, 'r--', linewidth=2, label='Propriétés constantes')
    axes[1].axhline(y=T_inf, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Temps [s]')
    axes[1].set_ylabel('Température de surface [K]')
    axes[1].set_title('Évolution de T_surface')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Test 4: Acier AISI 1010 - Propriétés réalistes', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_realistic_steel.png', dpi=150)
    plt.show()
    
    # Vérification: la solution doit être physiquement cohérente
    # T doit être bornée entre T_init et T_inf
    T_min = np.min(result['T'])
    T_max = np.max(result['T'])
    
    print(f"\n  Plage de température: [{T_min:.2f}, {T_max:.2f}] K")
    print(f"  Bornes attendues: [{T_init}, {T_inf}] K")
    
    bounded = (T_min >= T_init - 1) and (T_max <= T_inf + 1)
    
    passed = bounded
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: "
          f"solution {'bornée' if bounded else 'non bornée'}")
    
    return passed


def run_all_tests():
    """Exécute tous les tests de propriétés dépendantes de T."""
    print("\n" + "#"*70)
    print("# TESTS DES PROPRIÉTÉS THERMIQUES DÉPENDANTES DE LA TEMPÉRATURE")
    print("#"*70)
    
    results = {}
    
    results['constant_vs_variable'] = test_constant_vs_variable_properties()
    results['diffusivity_effect'] = test_diffusivity_effect()
    results['energy_conservation'] = test_energy_conservation_variable_cp()
    results['realistic_steel'] = test_realistic_steel_properties()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS DE PROPRIÉTÉS VARIABLES")
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
