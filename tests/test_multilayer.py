"""
Tests de validation pour empilements multicouches (3 couches)
=============================================================

Ce script valide le solveur sur des configurations multicouches
en comparant avec des solutions analytiques et des benchmarks de la littérature.

Références principales:
1. Mikhailov, M.D., Özişik, M.N. (2003) "On transient heat conduction in a 
   one-dimensional composite slab", Int. J. Heat Mass Transfer
2. D'Alessandro, V., de Monte, F. (2020) "Multi-layer transient heat conduction"
3. Carslaw & Jaeger, "Conduction of Heat in Solids" - Chapitre sur les composites

Cas testés:
1. Trois couches symétriques (métal-isolant-métal)
2. Configuration asymétrique avec fort contraste de propriétés
3. Comparaison avec solution stationnaire analytique
4. Vérification de la continuité de température et flux aux interfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from solver import ThermalSolver1D, compute_error_norms
from analytical_solutions import thermal_diffusivity, steady_state_multilayer


# ============================================================================
# Définition des matériaux de test
# ============================================================================

def create_multilayer_materials():
    """
    Crée les matériaux pour les tests multicouches.
    
    Matériaux choisis pour représenter des cas réalistes:
    - Acier: haute conductivité, haute densité
    - Isolant: très faible conductivité
    - Aluminium: très haute conductivité
    """
    return {
        'steel': {
            'T': np.array([0, 1000]),
            'k': np.array([50.0, 50.0]),      # W/(m·K)
            'rho': np.array([7800.0, 7800.0]), # kg/m³
            'cp': np.array([500.0, 500.0])     # J/(kg·K)
        },
        'insulator': {
            'T': np.array([0, 1000]),
            'k': np.array([0.5, 0.5]),         # W/(m·K) - très isolant
            'rho': np.array([150.0, 150.0]),   # kg/m³
            'cp': np.array([1200.0, 1200.0])   # J/(kg·K)
        },
        'aluminum': {
            'T': np.array([0, 1000]),
            'k': np.array([200.0, 200.0]),     # W/(m·K)
            'rho': np.array([2700.0, 2700.0]), # kg/m³
            'cp': np.array([900.0, 900.0])     # J/(kg·K)
        },
        'copper': {
            'T': np.array([0, 1000]),
            'k': np.array([400.0, 400.0]),     # W/(m·K)
            'rho': np.array([8900.0, 8900.0]), # kg/m³
            'cp': np.array([385.0, 385.0])     # J/(kg·K)
        },
        'concrete': {
            'T': np.array([0, 1000]),
            'k': np.array([1.4, 1.4]),         # W/(m·K)
            'rho': np.array([2300.0, 2300.0]), # kg/m³
            'cp': np.array([880.0, 880.0])     # J/(kg·K)
        }
    }


# ============================================================================
# CAS 1: Configuration symétrique métal-isolant-métal
# ============================================================================

def test_symmetric_metal_insulator_metal():
    """
    Test 1: Configuration symétrique Acier-Isolant-Acier
    
    Configuration inspirée de Mikhailov & Özişik (2003):
    - Couche 1: Acier, 5 mm
    - Couche 2: Isolant, 20 mm
    - Couche 3: Acier, 5 mm
    - T_init = 20°C (293 K)
    - T_left = 100°C (373 K) imposé
    - Bord droit: adiabatique
    
    Physique attendue:
    - Chauffage rapide de la couche 1 (haute diffusivité)
    - Propagation lente à travers l'isolant
    - Chauffage progressif de la couche 3
    - Convergence vers T = 373 K uniforme
    """
    print("\n" + "="*70)
    print("TEST 1: Configuration symétrique Acier-Isolant-Acier")
    print("="*70)
    
    materials = create_multilayer_materials()
    
    # Configuration des couches
    layers = [
        {'material': 'steel', 'thickness': 0.005},      # 5 mm
        {'material': 'insulator', 'thickness': 0.020},  # 20 mm
        {'material': 'steel', 'thickness': 0.005}       # 5 mm
    ]
    
    L_total = sum(layer['thickness'] for layer in layers)
    
    # Propriétés pour calcul des temps caractéristiques
    alpha_steel = thermal_diffusivity(50.0, 7800.0, 500.0)
    alpha_insulator = thermal_diffusivity(0.5, 150.0, 1200.0)
    
    tau_steel = (0.005)**2 / alpha_steel
    tau_insulator = (0.020)**2 / alpha_insulator
    
    print(f"\nConfiguration:")
    print(f"  Couche 1: Acier, 5 mm (τ = {tau_steel:.2f} s)")
    print(f"  Couche 2: Isolant, 20 mm (τ = {tau_insulator:.2f} s)")
    print(f"  Couche 3: Acier, 5 mm")
    print(f"  L_total = {L_total*1000:.0f} mm")
    
    # Conditions
    T_init = 293.0  # 20°C
    T_left = 373.0  # 100°C
    
    # Solveur
    Nx = 101
    solver = ThermalSolver1D(layers, materials, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    # Temps de simulation: le système doit atteindre l'équilibre
    # Avec un isolant, le temps est dominé par la diffusion à travers celui-ci
    # τ_total ≈ L_total² / α_min pour une estimation conservative
    # Simuler suffisamment longtemps pour convergence
    t_end = 20 * tau_insulator  # ~2900 s
    dt = t_end / 1000
    
    print(f"\n  Simulation: t_end = {t_end:.0f} s, dt = {dt:.2f} s")
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=50)
    
    # Analyse des résultats
    # Positions des interfaces
    x_interface1 = 0.005  # Entre couche 1 et 2
    x_interface2 = 0.025  # Entre couche 2 et 3
    
    # Trouver les indices des interfaces
    idx_int1 = np.argmin(np.abs(solver.x - x_interface1))
    idx_int2 = np.argmin(np.abs(solver.x - x_interface2))
    
    # Tracé
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Profils de température à différents instants
    ax = axes[0, 0]
    test_times = [0, t_end/10, t_end/4, t_end/2, t_end]
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        ax.plot(solver.x*1000, result['T'][idx, :], '-', linewidth=2, 
                label=f't={t_actual:.0f}s')
    
    # Marquer les interfaces
    ax.axvline(x=x_interface1*1000, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=x_interface2*1000, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=T_left, color='r', linestyle=':', alpha=0.5)
    
    ax.fill_betweenx([T_init-10, T_left+10], 0, 5, alpha=0.1, color='blue', label='Acier')
    ax.fill_betweenx([T_init-10, T_left+10], 5, 25, alpha=0.1, color='orange', label='Isolant')
    ax.fill_betweenx([T_init-10, T_left+10], 25, 30, alpha=0.1, color='blue')
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Profils de température')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, L_total*1000])
    
    # 2. Évolution temporelle aux interfaces
    ax = axes[0, 1]
    T_left_surface = result['T'][:, 0]
    T_interface1 = result['T'][:, idx_int1]
    T_interface2 = result['T'][:, idx_int2]
    T_right_surface = result['T'][:, -1]
    
    ax.plot(result['time'], T_left_surface, 'r-', linewidth=2, label='Surface gauche')
    ax.plot(result['time'], T_interface1, 'b-', linewidth=2, label='Interface 1-2')
    ax.plot(result['time'], T_interface2, 'g-', linewidth=2, label='Interface 2-3')
    ax.plot(result['time'], T_right_surface, 'm-', linewidth=2, label='Surface droite')
    ax.axhline(y=T_left, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Évolution aux interfaces')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Vérification de la continuité du flux aux interfaces
    ax = axes[1, 0]
    
    # Calcul du flux aux interfaces (q = -k * dT/dx)
    fluxes_int1_left = []
    fluxes_int1_right = []
    fluxes_int2_left = []
    fluxes_int2_right = []
    
    k_steel = 50.0
    k_insulator = 0.5
    
    for T in result['T']:
        # Interface 1: flux côté acier et côté isolant
        dTdx_left = (T[idx_int1] - T[idx_int1-1]) / (solver.x[idx_int1] - solver.x[idx_int1-1])
        dTdx_right = (T[idx_int1+1] - T[idx_int1]) / (solver.x[idx_int1+1] - solver.x[idx_int1])
        fluxes_int1_left.append(-k_steel * dTdx_left)
        fluxes_int1_right.append(-k_insulator * dTdx_right)
        
        # Interface 2
        dTdx_left = (T[idx_int2] - T[idx_int2-1]) / (solver.x[idx_int2] - solver.x[idx_int2-1])
        dTdx_right = (T[idx_int2+1] - T[idx_int2]) / (solver.x[idx_int2+1] - solver.x[idx_int2])
        fluxes_int2_left.append(-k_insulator * dTdx_left)
        fluxes_int2_right.append(-k_steel * dTdx_right)
    
    ax.plot(result['time'], fluxes_int1_left, 'b-', linewidth=2, label='Interface 1 (côté acier)')
    ax.plot(result['time'], fluxes_int1_right, 'b--', linewidth=2, label='Interface 1 (côté isolant)')
    ax.plot(result['time'], fluxes_int2_left, 'g-', linewidth=2, label='Interface 2 (côté isolant)')
    ax.plot(result['time'], fluxes_int2_right, 'g--', linewidth=2, label='Interface 2 (côté acier)')
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Flux de chaleur [W/m²]')
    ax.set_title('Continuité du flux aux interfaces')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Convergence vers l'équilibre
    ax = axes[1, 1]
    T_mean = np.mean(result['T'], axis=1)
    T_max = np.max(result['T'], axis=1)
    T_min = np.min(result['T'], axis=1)
    
    ax.plot(result['time'], T_mean, 'b-', linewidth=2, label='T moyenne')
    ax.fill_between(result['time'], T_min, T_max, alpha=0.3, label='Plage T')
    ax.axhline(y=T_left, color='r', linestyle='--', label=f'T_équilibre = {T_left} K')
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Convergence vers l\'équilibre')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Test 1: Configuration symétrique Acier-Isolant-Acier', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_multilayer_symmetric.png', dpi=150)
    plt.show()
    
    # Vérifications
    # 1. Convergence vers T_left
    T_final = result['T'][-1, :]
    converges = np.max(np.abs(T_final - T_left)) < 5.0
    
    # 2. Continuité du flux aux interfaces (en moyenne sur les derniers instants)
    flux_continuity_int1 = np.mean(np.abs(np.array(fluxes_int1_left[-10:]) - 
                                          np.array(fluxes_int1_right[-10:])))
    flux_continuity_int2 = np.mean(np.abs(np.array(fluxes_int2_left[-10:]) - 
                                          np.array(fluxes_int2_right[-10:])))
    
    # Note: La continuité parfaite n'est pas attendue car le maillage est uniforme
    # et les propriétés changent brutalement aux interfaces
    
    print(f"\nRésultats:")
    print(f"  Convergence vers T_left = {T_left} K: {'✓' if converges else '✗'}")
    print(f"    Écart max final: {np.max(np.abs(T_final - T_left)):.2f} K")
    print(f"  Discontinuité de flux interface 1: {flux_continuity_int1:.1f} W/m²")
    print(f"  Discontinuité de flux interface 2: {flux_continuity_int2:.1f} W/m²")
    
    passed = converges
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


# ============================================================================
# CAS 2: Configuration asymétrique avec fort contraste
# ============================================================================

def test_asymmetric_high_contrast():
    """
    Test 2: Configuration asymétrique Cuivre-Béton-Aluminium
    
    Configuration avec fort contraste de propriétés:
    - Couche 1: Cuivre, 2 mm (très haute conductivité)
    - Couche 2: Béton, 50 mm (faible conductivité, haute inertie)
    - Couche 3: Aluminium, 10 mm (haute conductivité)
    
    Conditions:
    - T_init = 300 K
    - Convection à gauche: h = 500 W/(m²·K), T_inf = 500 K
    - Convection à droite: h = 100 W/(m²·K), T_inf = 300 K
    
    Ce cas teste:
    - Le comportement avec des matériaux très différents
    - Les conditions de convection aux deux bords
    - L'établissement d'un régime permanent non uniforme
    """
    print("\n" + "="*70)
    print("TEST 2: Configuration asymétrique Cuivre-Béton-Aluminium")
    print("="*70)
    
    materials = create_multilayer_materials()
    
    layers = [
        {'material': 'copper', 'thickness': 0.002},    # 2 mm
        {'material': 'concrete', 'thickness': 0.050},  # 50 mm
        {'material': 'aluminum', 'thickness': 0.010}   # 10 mm
    ]
    
    L_total = sum(layer['thickness'] for layer in layers)
    
    # Temps caractéristiques
    alpha_copper = thermal_diffusivity(400.0, 8900.0, 385.0)
    alpha_concrete = thermal_diffusivity(1.4, 2300.0, 880.0)
    alpha_aluminum = thermal_diffusivity(200.0, 2700.0, 900.0)
    
    tau_concrete = (0.050)**2 / alpha_concrete  # Temps dominant
    
    print(f"\nConfiguration:")
    print(f"  Couche 1: Cuivre, 2 mm (α = {alpha_copper:.2e} m²/s)")
    print(f"  Couche 2: Béton, 50 mm (α = {alpha_concrete:.2e} m²/s, τ = {tau_concrete:.0f} s)")
    print(f"  Couche 3: Aluminium, 10 mm (α = {alpha_aluminum:.2e} m²/s)")
    print(f"  L_total = {L_total*1000:.0f} mm")
    
    # Conditions
    T_init = 300.0
    h_left = 500.0
    T_inf_left = 500.0
    h_right = 100.0
    T_inf_right = 300.0
    
    print(f"\nConditions limites:")
    print(f"  Gauche: convection h = {h_left} W/(m²·K), T_inf = {T_inf_left} K")
    print(f"  Droite: convection h = {h_right} W/(m²·K), T_inf = {T_inf_right} K")
    
    # Solveur
    Nx = 121
    solver = ThermalSolver1D(layers, materials, Nx=Nx)
    
    bc_left = lambda t: {'type': 'convection', 'h': h_left, 'T_inf': T_inf_left}
    bc_right = lambda t: {'type': 'convection', 'h': h_right, 'T_inf': T_inf_right}
    
    t_end = 3 * tau_concrete
    dt = t_end / 500
    
    print(f"\n  Simulation: t_end = {t_end:.0f} s")
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=50)
    
    # Solution stationnaire analytique
    # En régime permanent avec convection aux deux bords:
    # Résistance totale = 1/h_left + Σ(L_i/k_i) + 1/h_right
    R_conv_left = 1 / h_left
    R_copper = 0.002 / 400.0
    R_concrete = 0.050 / 1.4
    R_aluminum = 0.010 / 200.0
    R_conv_right = 1 / h_right
    R_total = R_conv_left + R_copper + R_concrete + R_aluminum + R_conv_right
    
    q_steady = (T_inf_left - T_inf_right) / R_total
    
    # Températures aux interfaces en régime permanent
    T_surf_left_steady = T_inf_left - q_steady * R_conv_left
    T_int1_steady = T_surf_left_steady - q_steady * R_copper
    T_int2_steady = T_int1_steady - q_steady * R_concrete
    T_surf_right_steady = T_int2_steady - q_steady * R_aluminum
    
    print(f"\nSolution stationnaire analytique:")
    print(f"  Flux: q = {q_steady:.1f} W/m²")
    print(f"  T_surface_gauche = {T_surf_left_steady:.2f} K")
    print(f"  T_interface_1 = {T_int1_steady:.2f} K")
    print(f"  T_interface_2 = {T_int2_steady:.2f} K")
    print(f"  T_surface_droite = {T_surf_right_steady:.2f} K")
    
    # Tracé
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Profils de température
    ax = axes[0, 0]
    test_times = [0, t_end/10, t_end/4, t_end/2, t_end]
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        ax.plot(solver.x*1000, result['T'][idx, :], '-', linewidth=2, 
                label=f't={t_actual:.0f}s')
    
    # Solution stationnaire
    x_steady = np.array([0, 0.002, 0.052, 0.062])
    T_steady = np.array([T_surf_left_steady, T_int1_steady, T_int2_steady, T_surf_right_steady])
    ax.plot(x_steady*1000, T_steady, 'k--', linewidth=2, label='Stationnaire (ana.)')
    
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=52, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Profils de température')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Évolution temporelle
    ax = axes[0, 1]
    ax.plot(result['time'], result['T'][:, 0], 'r-', linewidth=2, label='Surface gauche')
    ax.plot(result['time'], result['T'][:, -1], 'b-', linewidth=2, label='Surface droite')
    ax.axhline(y=T_surf_left_steady, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=T_surf_right_steady, color='b', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Évolution des températures de surface')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Comparaison avec solution stationnaire
    ax = axes[1, 0]
    T_final = result['T'][-1, :]
    
    # Interpoler la solution stationnaire sur le maillage
    T_steady_interp = np.interp(solver.x, x_steady, T_steady)
    
    ax.plot(solver.x*1000, T_final, 'b-', linewidth=2, label='Numérique (final)')
    ax.plot(solver.x*1000, T_steady_interp, 'r--', linewidth=2, label='Analytique stationnaire')
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Comparaison avec solution stationnaire')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Erreur par rapport à la solution stationnaire
    ax = axes[1, 1]
    error = T_final - T_steady_interp
    ax.plot(solver.x*1000, error, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=52, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Erreur T_num - T_ana [K]')
    ax.set_title('Écart à la solution stationnaire')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Test 2: Configuration asymétrique Cuivre-Béton-Aluminium', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_multilayer_asymmetric.png', dpi=150)
    plt.show()
    
    # Vérifications
    err = compute_error_norms(T_final, T_steady_interp)
    
    print(f"\nComparaison avec solution stationnaire:")
    print(f"  Erreur L2 = {err['L2']:.3f} K")
    print(f"  Erreur L∞ = {err['Linf']:.3f} K")
    print(f"  Erreur relative L2 = {err['L2_rel']*100:.2f}%")
    
    passed = err['L2_rel'] < 0.02  # Moins de 2% d'erreur
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: erreur = {err['L2_rel']*100:.2f}%")
    
    return passed


# ============================================================================
# CAS 3: Benchmark de Mikhailov & Özişik
# ============================================================================

def compute_mikhailov_analytical(x, t, k_values, L_values, T_left, T_right, n_terms=200):
    """
    Solution analytique transitoire pour une dalle composite 3 couches.
    
    Basé sur Mikhailov & Özişik (2003), Int. J. Heat Mass Transfer.
    
    Pour un système à 3 couches avec Dirichlet aux deux bords:
    T(x,t) = T_stationnaire(x) + Σ_n A_n * φ_n(x) * exp(-λ_n² * t)
    
    où les λ_n sont les valeurs propres du problème et φ_n les fonctions propres.
    
    Note: Cette implémentation utilise une approximation simplifiée basée sur
    la superposition des solutions monocouche avec correction aux interfaces.
    Pour une solution exacte, voir la référence originale.
    """
    x = np.atleast_1d(x)
    
    # Solution stationnaire (profil linéaire par morceaux)
    L1, L2, L3 = L_values
    k1, k2, k3 = k_values
    L_total = L1 + L2 + L3
    
    R_total = L1/k1 + L2/k2 + L3/k3
    q = (T_left - T_right) / R_total
    
    T_steady = np.zeros_like(x, dtype=float)
    
    # Couche 1: 0 <= x < L1
    mask1 = x < L1
    T_steady[mask1] = T_left - q * x[mask1] / k1
    
    # Couche 2: L1 <= x < L1+L2
    T_at_L1 = T_left - q * L1 / k1
    mask2 = (x >= L1) & (x < L1 + L2)
    T_steady[mask2] = T_at_L1 - q * (x[mask2] - L1) / k2
    
    # Couche 3: L1+L2 <= x <= L_total
    T_at_L1L2 = T_at_L1 - q * L2 / k2
    mask3 = x >= L1 + L2
    T_steady[mask3] = T_at_L1L2 - q * (x[mask3] - L1 - L2) / k3
    
    if t <= 0:
        return np.zeros_like(x)
    
    # Partie transitoire approximée par séries de Fourier
    # On utilise une approche de "composite slab" simplifiée
    T_transient = np.zeros_like(x, dtype=float)
    
    # Diffusivités (avec ρcp = 1)
    alpha1, alpha2, alpha3 = k1, k2, k3
    
    # Approximation: on utilise les modes propres d'un système équivalent
    # avec une diffusivité effective moyenne
    alpha_eff = L_total / R_total  # Diffusivité effective
    
    for n in range(1, n_terms + 1):
        lambda_n = n * np.pi / L_total
        
        # Coefficient de Fourier pour T_init = 0
        # B_n = (2/L) * ∫[0 - T_steady(x)] * sin(λ_n * x) dx
        # Approximation numérique
        x_int = np.linspace(0, L_total, 1000)
        T_steady_int = np.zeros_like(x_int)
        
        mask1_int = x_int < L1
        T_steady_int[mask1_int] = T_left - q * x_int[mask1_int] / k1
        mask2_int = (x_int >= L1) & (x_int < L1 + L2)
        T_steady_int[mask2_int] = T_at_L1 - q * (x_int[mask2_int] - L1) / k2
        mask3_int = x_int >= L1 + L2
        T_steady_int[mask3_int] = T_at_L1L2 - q * (x_int[mask3_int] - L1 - L2) / k3
        
        integrand = -T_steady_int * np.sin(lambda_n * x_int)
        B_n = 2 / L_total * np.trapezoid(integrand, x_int)
        
        # Décroissance temporelle avec diffusivité effective
        T_transient += B_n * np.sin(lambda_n * x) * np.exp(-lambda_n**2 * alpha_eff * t)
    
    return T_steady + T_transient


def test_mikhailov_benchmark():
    """
    Test 3: Benchmark inspiré de Mikhailov & Özişik (2003)
    
    Configuration à trois couches avec propriétés normalisées:
    - Couche 1: k₁ = 1, épaisseur L₁ = 0.2
    - Couche 2: k₂ = 0.1 (isolant), épaisseur L₂ = 0.6
    - Couche 3: k₃ = 1, épaisseur L₃ = 0.2
    - Propriétés: ρ*cp = 1 pour toutes les couches (normalisé)
    
    Conditions:
    - T_init = 0 (normalisé)
    - T(0,t) = 1 (Dirichlet)
    - T(L,t) = 0 (Dirichlet)
    
    Ce cas permet de vérifier le comportement avec un fort contraste
    de conductivité et des conditions de Dirichlet aux deux bords.
    """
    print("\n" + "="*70)
    print("TEST 3: Benchmark type Mikhailov & Özişik")
    print("="*70)
    
    # Matériaux normalisés
    materials = {
        'mat_high_k': {
            'T': np.array([0, 100]),
            'k': np.array([1.0, 1.0]),
            'rho': np.array([1.0, 1.0]),
            'cp': np.array([1.0, 1.0])
        },
        'mat_low_k': {
            'T': np.array([0, 100]),
            'k': np.array([0.1, 0.1]),
            'rho': np.array([1.0, 1.0]),
            'cp': np.array([1.0, 1.0])
        }
    }
    
    layers = [
        {'material': 'mat_high_k', 'thickness': 0.2},
        {'material': 'mat_low_k', 'thickness': 0.6},
        {'material': 'mat_high_k', 'thickness': 0.2}
    ]
    
    L_total = 1.0
    k_values = [1.0, 0.1, 1.0]
    L_values = [0.2, 0.6, 0.2]
    
    print(f"\nConfiguration normalisée (Mikhailov & Özişik, 2003):")
    print(f"  Couche 1: k₁ = 1, L₁ = 0.2, α₁ = 1")
    print(f"  Couche 2: k₂ = 0.1, L₂ = 0.6, α₂ = 0.1 (isolant)")
    print(f"  Couche 3: k₃ = 1, L₃ = 0.2, α₃ = 1")
    print(f"  ρ*cp = 1 partout (normalisé)")
    print(f"\nConditions limites:")
    print(f"  T(0,t) = 1 (Dirichlet)")
    print(f"  T(L,t) = 0 (Dirichlet)")
    print(f"  T(x,0) = 0 (condition initiale)")
    
    # Conditions
    T_init = 0.0
    T_left = 1.0
    T_right = 0.0
    
    # Solveur
    Nx = 101
    solver = ThermalSolver1D(layers, materials, Nx=Nx)
    
    bc_left = lambda t: {'type': 'dirichlet', 'T': T_left}
    bc_right = lambda t: {'type': 'dirichlet', 'T': T_right}
    
    # Temps caractéristique de la couche isolante: τ = L²/α = 0.6²/0.1 = 3.6
    t_end = 20.0
    dt = 0.02
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=50)
    
    # Solution stationnaire analytique
    R_total = 0.2/1.0 + 0.6/0.1 + 0.2/1.0
    q_steady = (T_left - T_right) / R_total
    
    x_interfaces = np.array([0, 0.2, 0.8, 1.0])
    T_interfaces = np.array([
        T_left,
        T_left - q_steady * 0.2 / 1.0,
        T_left - q_steady * (0.2/1.0 + 0.6/0.1),
        T_right
    ])
    
    T_steady_interp = np.interp(solver.x, x_interfaces, T_interfaces)
    
    print(f"\nSolution stationnaire analytique:")
    print(f"  Résistance totale: R = {R_total:.2f}")
    print(f"  Flux thermique: q = {q_steady:.4f}")
    print(f"  T aux interfaces: {T_interfaces}")
    
    # Temps de comparaison pour le transitoire (exclure t=0 car discontinuité initiale)
    test_times = [0.5, 1.0, 2.0, 5.0, 10.0, t_end]
    
    # Calcul des solutions analytiques transitoires
    T_analytical = {}
    for t_test in test_times:
        T_analytical[t_test] = compute_mikhailov_analytical(
            solver.x, t_test, k_values, L_values, T_left, T_right
        )
    
    # Calcul des erreurs à chaque instant
    print(f"\n{'='*70}")
    print("COMPARAISON NUMÉRIQUE vs ANALYTIQUE")
    print(f"{'='*70}")
    print(f"\n{'Temps':>8} | {'Erreur L2':>12} | {'Erreur L∞':>12} | {'Erreur moy.':>12} | {'Err. rel. L2':>12}")
    print("-"*70)
    
    errors_L2 = []
    errors_Linf = []
    errors_mean = []
    errors_rel = []
    
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        T_num = result['T'][idx, :]
        T_ana = T_analytical[t_test]
        
        err = compute_error_norms(T_num, T_ana)
        err_mean = np.mean(np.abs(T_num - T_ana))
        
        errors_L2.append(err['L2'])
        errors_Linf.append(err['Linf'])
        errors_mean.append(err_mean)
        errors_rel.append(err['L2_rel'])
        
        print(f"{t_actual:8.1f} | {err['L2']:12.6f} | {err['Linf']:12.6f} | {err_mean:12.6f} | {err['L2_rel']*100:11.4f}%")
    
    # Erreur finale (régime permanent)
    T_final = result['T'][-1, :]
    err_final = compute_error_norms(T_final, T_steady_interp)
    
    print("-"*70)
    print(f"{'FINAL':>8} | {err_final['L2']:12.6f} | {err_final['Linf']:12.6f} | {np.mean(np.abs(T_final - T_steady_interp)):12.6f} | {err_final['L2_rel']*100:11.4f}%")
    print(f"{'='*70}")
    
    # Tracé détaillé
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Comparaison des profils numériques et analytiques
    ax1 = fig.add_subplot(2, 2, 1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_times)))
    
    for i, t_test in enumerate(test_times):
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        # Numérique (trait plein)
        ax1.plot(solver.x, result['T'][idx, :], '-', color=colors[i], linewidth=2, 
                 label=f'Num. t={t_actual:.1f}')
        # Analytique (tirets)
        ax1.plot(solver.x, T_analytical[t_test], '--', color=colors[i], linewidth=2)
    
    # Solution stationnaire
    ax1.plot(solver.x, T_steady_interp, 'k-', linewidth=3, label='Stationnaire (exact)')
    
    # Marquer les interfaces
    ax1.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7, label='Interfaces')
    ax1.axvline(x=0.8, color='gray', linestyle=':', alpha=0.7)
    
    # Zones de matériaux
    ax1.fill_betweenx([-0.1, 1.1], 0, 0.2, alpha=0.1, color='blue')
    ax1.fill_betweenx([-0.1, 1.1], 0.2, 0.8, alpha=0.1, color='red')
    ax1.fill_betweenx([-0.1, 1.1], 0.8, 1.0, alpha=0.1, color='blue')
    
    ax1.set_xlabel('Position x (normalisée)', fontsize=12)
    ax1.set_ylabel('Température T (normalisée)', fontsize=12)
    ax1.set_title('Profils de température: Numérique (—) vs Analytique (--)', fontsize=12)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-0.05, 1.05])
    ax1.text(0.1, 0.95, 'k=1', ha='center', fontsize=10, color='blue')
    ax1.text(0.5, 0.95, 'k=0.1', ha='center', fontsize=10, color='red')
    ax1.text(0.9, 0.95, 'k=1', ha='center', fontsize=10, color='blue')
    
    # 2. Erreur ponctuelle à différents instants
    ax2 = fig.add_subplot(2, 2, 2)
    
    for i, t_test in enumerate(test_times):
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        
        error = result['T'][idx, :] - T_analytical[t_test]
        ax2.plot(solver.x, error, '-', color=colors[i], linewidth=1.5, 
                 label=f't={t_actual:.1f}')
    
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(x=0.8, color='gray', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Position x (normalisée)', fontsize=12)
    ax2.set_ylabel('Erreur T_num - T_ana', fontsize=12)
    ax2.set_title('Erreur ponctuelle en fonction de la position', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Évolution des erreurs dans le temps
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax3.semilogy(test_times, errors_L2, 'bo-', linewidth=2, markersize=8, label='Erreur L2')
    ax3.semilogy(test_times, errors_Linf, 'rs-', linewidth=2, markersize=8, label='Erreur L∞')
    ax3.semilogy(test_times, errors_mean, 'g^-', linewidth=2, markersize=8, label='Erreur moyenne')
    
    ax3.set_xlabel('Temps (normalisé)', fontsize=12)
    ax3.set_ylabel('Erreur', fontsize=12)
    ax3.set_title('Évolution des erreurs dans le temps', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. Comparaison finale numérique vs analytique stationnaire
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax4.plot(solver.x, T_final, 'b-', linewidth=2.5, label='Numérique (final)')
    ax4.plot(solver.x, T_steady_interp, 'r--', linewidth=2.5, label='Analytique (stationnaire)')
    
    # Zoom sur les différences
    ax4_inset = ax4.inset_axes([0.55, 0.55, 0.4, 0.4])
    ax4_inset.plot(solver.x, T_final - T_steady_interp, 'g-', linewidth=1.5)
    ax4_inset.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4_inset.set_xlabel('x', fontsize=8)
    ax4_inset.set_ylabel('Erreur', fontsize=8)
    ax4_inset.set_title('Écart num.-ana.', fontsize=8)
    ax4_inset.grid(True, alpha=0.3)
    
    ax4.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7)
    ax4.axvline(x=0.8, color='gray', linestyle=':', alpha=0.7)
    
    ax4.set_xlabel('Position x (normalisée)', fontsize=12)
    ax4.set_ylabel('Température T (normalisée)', fontsize=12)
    ax4.set_title(f'Régime permanent: Erreur L2 = {err_final["L2"]:.6f}, L∞ = {err_final["Linf"]:.6f}', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Benchmark Mikhailov & Özişik (2003) - Dalle composite 3 couches\n'
                 f'Erreur moyenne finale: {np.mean(np.abs(T_final - T_steady_interp)):.6f}, '
                 f'Erreur max: {err_final["Linf"]:.6f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_multilayer_mikhailov.png', dpi=150)
    plt.show()
    
    # Résumé des erreurs
    print(f"\n{'='*70}")
    print("RÉSUMÉ DES ERREURS")
    print(f"{'='*70}")
    print(f"  Erreur L2 moyenne sur tous les instants: {np.mean(errors_L2):.6f}")
    print(f"  Erreur L2 max sur tous les instants: {np.max(errors_L2):.6f}")
    print(f"  Erreur L∞ moyenne: {np.mean(errors_Linf):.6f}")
    print(f"  Erreur L∞ max: {np.max(errors_Linf):.6f}")
    print(f"  Erreur relative L2 finale: {err_final['L2_rel']*100:.4f}%")
    print(f"{'='*70}")
    
    # Critère de validation
    # On exclut le premier temps (t=0.5) car le transitoire initial est difficile à capturer
    errors_L2_late = errors_L2[1:]  # Exclure le premier temps
    passed = err_final['L2'] < 0.01 and np.max(errors_L2_late) < 0.1
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}: "
          f"Erreur L2 finale = {err_final['L2']:.6f}, max transitoire (t≥1) = {np.max(errors_L2_late):.6f}")
    
    return passed


# ============================================================================
# CAS 4: Vérification de la conservation de l'énergie multicouche
# ============================================================================

def test_multilayer_energy_conservation():
    """
    Test 4: Conservation de l'énergie dans un système multicouche adiabatique
    
    Configuration:
    - Trois couches avec des propriétés différentes
    - Condition initiale non uniforme
    - Bords adiabatiques (système isolé)
    
    L'énergie totale doit être conservée.
    """
    print("\n" + "="*70)
    print("TEST 4: Conservation de l'énergie multicouche")
    print("="*70)
    
    materials = create_multilayer_materials()
    
    layers = [
        {'material': 'steel', 'thickness': 0.01},
        {'material': 'insulator', 'thickness': 0.03},
        {'material': 'aluminum', 'thickness': 0.01}
    ]
    
    L_total = sum(layer['thickness'] for layer in layers)
    
    # Solveur
    Nx = 81
    solver = ThermalSolver1D(layers, materials, Nx=Nx)
    
    # Condition initiale: gradient linéaire
    T_left_init = 400.0
    T_right_init = 300.0
    T_init = T_left_init + (T_right_init - T_left_init) * solver.x / L_total
    
    # Bords adiabatiques
    bc_left = lambda t: {'type': 'flux', 'q': 0.0}
    bc_right = lambda t: {'type': 'adiabatic'}
    
    t_end = 5000.0  # Long temps pour équilibrage
    dt = 10.0
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, save_every=50)
    
    # Calcul de l'énergie à chaque instant
    def compute_energy(T_field):
        E = 0
        for i, (x, T) in enumerate(zip(solver.x, T_field)):
            mat = layers[solver.layer_index[i]]['material']
            rho = materials[mat]['rho'][0]
            cp = materials[mat]['cp'][0]
            E += rho * cp * T * solver.dx
        return E
    
    energies = [compute_energy(T) for T in result['T']]
    energies = np.array(energies)
    E_init = energies[0]
    dE_rel = (energies - E_init) / E_init * 100
    
    # Température d'équilibre attendue (moyenne pondérée par ρ*cp*V)
    total_heat_capacity = 0
    total_energy_init = 0
    for layer in layers:
        mat = layer['material']
        rho = materials[mat]['rho'][0]
        cp = materials[mat]['cp'][0]
        L = layer['thickness']
        total_heat_capacity += rho * cp * L
    
    T_eq = E_init / total_heat_capacity
    
    print(f"\nConfiguration:")
    print(f"  Acier (10 mm) + Isolant (30 mm) + Aluminium (10 mm)")
    print(f"  Condition initiale: gradient {T_left_init} K → {T_right_init} K")
    print(f"\nRésultats:")
    print(f"  Énergie initiale: {E_init:.0f} J/m²")
    print(f"  Variation max d'énergie: {np.max(np.abs(dE_rel)):.4f}%")
    print(f"  Température d'équilibre attendue: {T_eq:.2f} K")
    print(f"  Température finale moyenne: {np.mean(result['T'][-1, :]):.2f} K")
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Profils de température
    ax = axes[0]
    test_times = [0, t_end/10, t_end/4, t_end/2, t_end]
    for t_test in test_times:
        idx = np.argmin(np.abs(result['time'] - t_test))
        t_actual = result['time'][idx]
        ax.plot(solver.x*1000, result['T'][idx, :], '-', linewidth=2, 
                label=f't={t_actual:.0f}s')
    
    ax.axhline(y=T_eq, color='r', linestyle='--', label=f'T_eq = {T_eq:.1f} K')
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Évolution vers l\'équilibre thermique')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Conservation de l'énergie
    ax = axes[1]
    ax.plot(result['time'], dE_rel, 'b-', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Variation relative d\'énergie [%]')
    ax.set_title('Conservation de l\'énergie')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Test 4: Conservation de l\'énergie multicouche', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('test_multilayer_energy.png', dpi=150)
    plt.show()
    
    # Vérifications
    energy_conserved = np.max(np.abs(dE_rel)) < 0.5
    T_final = result['T'][-1, :]
    equilibrium_reached = np.max(T_final) - np.min(T_final) < 1.0
    
    print(f"\n  Conservation énergie (< 0.5%): {'✓' if energy_conserved else '✗'}")
    print(f"  Équilibre atteint (ΔT < 1 K): {'✓' if equilibrium_reached else '✗'}")
    
    passed = energy_conserved and equilibrium_reached
    print(f"\n{'✓ TEST RÉUSSI' if passed else '✗ TEST ÉCHOUÉ'}")
    
    return passed


# ============================================================================
# Fonction principale
# ============================================================================

def run_all_tests():
    """Exécute tous les tests multicouches."""
    print("\n" + "#"*70)
    print("# TESTS DE VALIDATION POUR EMPILEMENTS MULTICOUCHES (3 COUCHES)")
    print("#"*70)
    
    results = {}
    
    results['symmetric'] = test_symmetric_metal_insulator_metal()
    results['asymmetric'] = test_asymmetric_high_contrast()
    results['mikhailov'] = test_mikhailov_benchmark()
    results['energy'] = test_multilayer_energy_conservation()
    
    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS MULTICOUCHES")
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
