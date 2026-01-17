"""
Exemple de simulation thermique multicouche
===========================================

Ce script montre comment utiliser le solveur et la bibliothèque de matériaux
pour simuler la conduction thermique transitoire dans un empilement multicouche.

Cas simulé:
- Empilement: Acier inoxydable 304 (5 mm) + Isolant céramique (20 mm) + Aluminium 6061 (10 mm)
- Condition gauche: Convection variable (données du fichier flux.csv)
- Condition droite: Adiabatique
- Température initiale: 300 K
"""

import numpy as np
import matplotlib.pyplot as plt

# Import du solveur et de la bibliothèque de matériaux
from solver import ThermalSolver1D
from material_library import get_materials, MaterialLibrary


def main():
    """Simulation principale."""
    
    print("="*70)
    print("SIMULATION THERMIQUE MULTICOUCHE")
    print("="*70)
    
    # ========================================================================
    # 1. DÉFINITION DES MATÉRIAUX
    # ========================================================================
    
    print("\n1. Chargement des matériaux depuis la bibliothèque...")
    
    # Charger les matériaux depuis la bibliothèque
    material_data = get_materials('steel_304', 'ceramic_fiber', 'aluminum_6061')
    
    # Afficher les propriétés à température ambiante
    print("\n   Propriétés à 300 K:")
    for name in ['steel_304', 'ceramic_fiber', 'aluminum_6061']:
        k, rho, cp = MaterialLibrary.get_properties(name, 300)
        alpha = k / (rho * cp)
        print(f"   - {name}: k={k:.2f} W/(m·K), ρ={rho:.0f} kg/m³, "
              f"cp={cp:.0f} J/(kg·K), α={alpha:.2e} m²/s")
    
    # ========================================================================
    # 2. DÉFINITION DE LA GÉOMÉTRIE
    # ========================================================================
    
    print("\n2. Définition de la géométrie...")
    
    layers = [
        {'material': 'steel_304', 'thickness': 0.005},      # 5 mm
        {'material': 'ceramic_fiber', 'thickness': 0.020},  # 20 mm
        {'material': 'aluminum_6061', 'thickness': 0.010}   # 10 mm
    ]
    
    L_total = sum(layer['thickness'] for layer in layers)
    
    print(f"\n   Configuration:")
    for i, layer in enumerate(layers):
        print(f"   - Couche {i+1}: {layer['material']}, {layer['thickness']*1000:.0f} mm")
    print(f"   - Épaisseur totale: {L_total*1000:.0f} mm")
    
    # ========================================================================
    # 3. CRÉATION DU SOLVEUR
    # ========================================================================
    
    print("\n3. Création du solveur...")
    
    Nx = 101  # Nombre de nœuds
    solver = ThermalSolver1D(layers, material_data, Nx=Nx)
    
    print(f"   - Nombre de nœuds: {Nx}")
    print(f"   - Pas spatial: {solver.dx*1000:.3f} mm")
    
    # ========================================================================
    # 4. LECTURE DES CONDITIONS LIMITES
    # ========================================================================
    
    print("\n4. Lecture des conditions limites (flux.csv)...")
    
    data = np.loadtxt('flux.csv', delimiter=',', skiprows=1)
    t_flux = data[:, 0]
    h_data = data[:, 1]
    T_inf_data = data[:, 2]
    
    print(f"   - Temps: {t_flux[0]:.0f} s → {t_flux[-1]:.0f} s")
    print(f"   - h: {h_data.min():.0f} → {h_data.max():.0f} W/(m²·K)")
    print(f"   - T_inf: {T_inf_data.min():.0f} → {T_inf_data.max():.0f} K")
    
    # Fonctions d'interpolation pour les CL
    def bc_left(t):
        h = np.interp(t, t_flux, h_data)
        T_inf = np.interp(t, t_flux, T_inf_data)
        return {'type': 'convection', 'h': h, 'T_inf': T_inf}
    
    def bc_right(t):
        return {'type': 'adiabatic'}
    
    # ========================================================================
    # 5. SIMULATION
    # ========================================================================
    
    print("\n5. Lancement de la simulation...")
    
    T_init = 300.0  # Température initiale [K]
    t_end = t_flux[-1]
    dt = 0.1  # Pas de temps [s]
    
    print(f"   - T_init: {T_init} K")
    print(f"   - Durée: {t_end} s")
    print(f"   - Pas de temps: {dt} s")
    print(f"   - Nombre de pas: {int(t_end/dt)}")
    
    result = solver.solve(T_init, t_end, dt, bc_left, bc_right, 
                          save_every=10, verbose=True)
    
    print(f"\n   Simulation terminée!")
    print(f"   - {len(result['time'])} instants sauvegardés")
    print(f"   - T_min = {result['T'].min():.2f} K")
    print(f"   - T_max = {result['T'].max():.2f} K")
    
    # ========================================================================
    # 6. VISUALISATION
    # ========================================================================
    
    print("\n6. Génération des graphiques...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Graphique 1: Profils de température ---
    ax = axes[0, 0]
    
    n_profiles = 6
    indices = np.linspace(0, len(result['time'])-1, n_profiles, dtype=int)
    colors = plt.cm.hot(np.linspace(0.2, 0.8, n_profiles))
    
    for idx, color in zip(indices, colors):
        t = result['time'][idx]
        T = result['T'][idx, :]
        ax.plot(solver.x*1000, T, '-', color=color, linewidth=2, label=f't={t:.0f}s')
    
    # Marquer les interfaces
    cum_thick = np.cumsum([layer['thickness'] for layer in layers])
    for x_int in cum_thick[:-1]:
        ax.axvline(x=x_int*1000, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Profils de température')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annotations des couches
    ax.text(2.5, ax.get_ylim()[1]*0.95, 'Acier\n304', ha='center', fontsize=8)
    ax.text(15, ax.get_ylim()[1]*0.95, 'Isolant\ncéramique', ha='center', fontsize=8)
    ax.text(30, ax.get_ylim()[1]*0.95, 'Alu\n6061', ha='center', fontsize=8)
    
    # --- Graphique 2: Évolution temporelle ---
    ax = axes[0, 1]
    
    # Trouver les indices des positions clés
    x_positions = [0, 0.005, 0.015, 0.025, 0.035]  # Bord, interfaces, milieux
    labels = ['Surface gauche', 'Interface 1', 'Milieu isolant', 'Interface 2', 'Surface droite']
    
    for x_pos, label in zip(x_positions, labels):
        idx = np.argmin(np.abs(solver.x - x_pos))
        ax.plot(result['time'], result['T'][:, idx], '-', linewidth=2, label=label)
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('Température [K]')
    ax.set_title('Évolution temporelle')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 3: Conditions limites ---
    ax = axes[1, 0]
    
    ax2 = ax.twinx()
    
    l1 = ax.plot(t_flux, h_data, 'b-', linewidth=2, label='h')
    l2 = ax2.plot(t_flux, T_inf_data, 'r-', linewidth=2, label='T_inf')
    
    ax.set_xlabel('Temps [s]')
    ax.set_ylabel('h [W/(m²·K)]', color='b')
    ax2.set_ylabel('T_inf [K]', color='r')
    ax.set_title('Conditions limites (convection)')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    ax.grid(True, alpha=0.3)
    
    # --- Graphique 4: Carte de température ---
    ax = axes[1, 1]
    
    # Créer une grille temps-espace
    T_grid = result['T']
    extent = [solver.x[0]*1000, solver.x[-1]*1000, result['time'][0], result['time'][-1]]
    
    im = ax.imshow(T_grid, aspect='auto', origin='lower', extent=extent,
                   cmap='hot', interpolation='bilinear')
    
    # Marquer les interfaces
    for x_int in cum_thick[:-1]:
        ax.axvline(x=x_int*1000, color='white', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Position x [mm]')
    ax.set_ylabel('Temps [s]')
    ax.set_title('Carte spatio-temporelle de température')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Température [K]')
    
    plt.suptitle('Simulation thermique multicouche: Acier 304 + Isolant + Aluminium 6061',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('simulation_multicouche.png', dpi=150)
    plt.show()
    
    print(f"\n   Figure sauvegardée: simulation_multicouche.png")
    
    # ========================================================================
    # 7. RÉSUMÉ
    # ========================================================================
    
    print("\n" + "="*70)
    print("RÉSUMÉ DE LA SIMULATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Empilement: {' + '.join([l['material'] for l in layers])}")
    print(f"  - Épaisseur totale: {L_total*1000:.0f} mm")
    print(f"  - Maillage: {Nx} nœuds, dx = {solver.dx*1000:.3f} mm")
    print(f"\nSimulation:")
    print(f"  - Durée: {t_end} s")
    print(f"  - Pas de temps: {dt} s")
    print(f"\nRésultats:")
    print(f"  - T initiale: {T_init} K")
    print(f"  - T finale min: {result['T'][-1].min():.2f} K")
    print(f"  - T finale max: {result['T'][-1].max():.2f} K")
    print(f"  - T finale moyenne: {result['T'][-1].mean():.2f} K")
    print("="*70)


if __name__ == '__main__':
    main()
