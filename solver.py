"""
Solveur thermique 1D transitoire multicouche - Module réutilisable
==================================================================

Ce module contient le solveur de conduction thermique 1D implicite,
séparé de la définition des cas tests pour faciliter la validation.

Équation résolue:
    ρ(T) * cp(T) * ∂T/∂t = ∂/∂x [k(T) * ∂T/∂x]

Schéma numérique:
    - Temporel: Euler implicite (inconditionnellement stable)
    - Spatial: Différences finies centrées

Conditions limites supportées:
    - Gauche: Convection (Robin) ou flux imposé (Neumann) ou isotherme (Dirichlet)
    - Droite: Adiabatique (Neumann homogène) ou isotherme (Dirichlet)
"""

import numpy as np


class ThermalSolver1D:
    """
    Solveur de conduction thermique 1D transitoire multicouche.
    
    Attributs:
        layers: Liste de dictionnaires définissant les couches
        material_data: Dictionnaire des propriétés des matériaux
        Nx: Nombre de nœuds spatiaux
        dx: Pas spatial [m]
        x: Coordonnées des nœuds [m]
        L_total: Longueur totale du domaine [m]
    """
    
    def __init__(self, layers, material_data, Nx=101):
        """
        Initialise le solveur.
        
        Args:
            layers: Liste de dict {'material': str, 'thickness': float}
            material_data: Dict des propriétés thermiques par matériau
            Nx: Nombre de nœuds spatiaux
        """
        self.layers = layers
        self.material_data = material_data
        self.Nx = Nx
        
        # Calcul du maillage
        self.L_total = sum(layer['thickness'] for layer in layers)
        self.dx = self.L_total / (Nx - 1)
        self.x = np.linspace(0, self.L_total, Nx)
        
        # Déterminer l'indice de couche pour chaque nœud
        cum_thick = np.cumsum([layer['thickness'] for layer in layers])
        self.layer_index = np.searchsorted(cum_thick, self.x)
        # Corriger l'indice pour le dernier point (peut dépasser)
        self.layer_index = np.clip(self.layer_index, 0, len(layers) - 1)
    
    def interp_property(self, mat_props, T):
        """
        Interpole les propriétés thermiques (k, rho, cp) à la température T.
        
        Args:
            mat_props: Dict contenant 'T', 'k', 'rho', 'cp'
            T: Température d'interpolation [K] (scalaire)
            
        Returns:
            Tuple (k, rho, cp) interpolés (scalaires)
        """
        T_tab = mat_props['T']
        k_tab = mat_props['k']
        rho_tab = mat_props['rho']
        cp_tab = mat_props['cp']
        # Convertir T en scalaire si nécessaire
        T_val = float(T)
        k = float(np.interp(T_val, T_tab, k_tab))
        rho = float(np.interp(T_val, T_tab, rho_tab))
        cp = float(np.interp(T_val, T_tab, cp_tab))
        return k, rho, cp
    
    def get_properties_at_nodes(self, T):
        """
        Calcule les propriétés thermiques à chaque nœud.
        
        Args:
            T: Champ de température actuel [K]
            
        Returns:
            Tuple (k_nodes, rho_nodes, cp_nodes)
        """
        k_nodes = np.zeros(self.Nx)
        rho_nodes = np.zeros(self.Nx)
        cp_nodes = np.zeros(self.Nx)
        
        for i in range(self.Nx):
            mat = self.layers[self.layer_index[i]]['material']
            k_nodes[i], rho_nodes[i], cp_nodes[i] = self.interp_property(
                self.material_data[mat], T[i]
            )
        
        return k_nodes, rho_nodes, cp_nodes
    
    def build_system(self, T, dt, bc_left, bc_right):
        """
        Construit le système linéaire A*T_new = b pour le schéma implicite.
        
        Args:
            T: Champ de température actuel [K]
            dt: Pas de temps [s]
            bc_left: Dict définissant la CL gauche
                     {'type': 'convection', 'h': float, 'T_inf': float}
                     {'type': 'flux', 'q': float}  (q positif = entrant)
                     {'type': 'dirichlet', 'T': float}
            bc_right: Dict définissant la CL droite
                      {'type': 'adiabatic'}
                      {'type': 'dirichlet', 'T': float}
                      
        Returns:
            Tuple (A, b) du système linéaire
        """
        Nx = self.Nx
        dx = self.dx
        
        A = np.zeros((Nx, Nx))
        b = np.zeros(Nx)
        
        # Propriétés aux nœuds
        k_nodes, rho_nodes, cp_nodes = self.get_properties_at_nodes(T)
        
        # === Condition limite gauche (i=0) ===
        if bc_left['type'] == 'dirichlet':
            A[0, 0] = 1.0
            b[0] = bc_left['T']
        elif bc_left['type'] == 'convection':
            h = bc_left['h']
            T_inf = bc_left['T_inf']
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            # Bilan thermique sur demi-maille de bord avec convection
            # ρcp * (dx/2) * (T_new - T_old)/dt = h*(T_inf - T_new) + k*(T_1 - T_0)/dx
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx + h)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * h * T_inf
        elif bc_left['type'] == 'flux':
            q = bc_left['q']  # flux entrant positif
            k0p = 0.5 * (k_nodes[0] + k_nodes[1])
            rho0 = rho_nodes[0]
            cp0 = cp_nodes[0]
            # Bilan sur demi-maille: ρcp*(dx/2)*(T_new-T_old)/dt = q + k*(T_1-T_0)/dx
            coef = dt / (rho0 * cp0 * dx / 2)
            A[0, 0] = 1 + coef * (k0p / dx)
            A[0, 1] = -coef * (k0p / dx)
            b[0] = T[0] + coef * q
        else:
            raise ValueError(f"Type de CL gauche inconnu: {bc_left['type']}")
        
        # === Nœuds intérieurs (1 ≤ i ≤ Nx-2) ===
        for i in range(1, Nx - 1):
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])  # conductivité moyenne gauche
            kip = 0.5 * (k_nodes[i] + k_nodes[i+1])  # conductivité moyenne droite
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            
            coef = dt / (rhoi * cpi * dx**2)
            A[i, i-1] = -coef * kim
            A[i, i] = 1 + coef * (kim + kip)
            A[i, i+1] = -coef * kip
            b[i] = T[i]
        
        # === Condition limite droite (i=Nx-1) ===
        i = Nx - 1
        if bc_right['type'] == 'dirichlet':
            A[i, i] = 1.0
            b[i] = bc_right['T']
        elif bc_right['type'] == 'adiabatic':
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            # Bilan sur demi-maille de bord avec flux nul
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx)
            b[i] = T[i]
        elif bc_right['type'] == 'convection':
            h = bc_right['h']
            T_inf = bc_right['T_inf']
            kim = 0.5 * (k_nodes[i] + k_nodes[i-1])
            rhoi = rho_nodes[i]
            cpi = cp_nodes[i]
            coef = dt / (rhoi * cpi * dx / 2)
            A[i, i-1] = -coef * (kim / dx)
            A[i, i] = 1 + coef * (kim / dx + h)
            b[i] = T[i] + coef * h * T_inf
        else:
            raise ValueError(f"Type de CL droite inconnu: {bc_right['type']}")
        
        return A, b
    
    def solve_step(self, T, dt, bc_left, bc_right):
        """
        Résout un pas de temps.
        
        Args:
            T: Champ de température actuel [K]
            dt: Pas de temps [s]
            bc_left: Condition limite gauche
            bc_right: Condition limite droite
            
        Returns:
            Nouveau champ de température [K]
        """
        A, b = self.build_system(T, dt, bc_left, bc_right)
        T_new = np.linalg.solve(A, b)
        return T_new
    
    def solve(self, T_init, t_end, dt, bc_left_func, bc_right_func, 
              save_every=1, verbose=False):
        """
        Résout le problème transitoire complet.
        
        Args:
            T_init: Condition initiale (scalaire ou array)
            t_end: Temps final [s]
            dt: Pas de temps [s]
            bc_left_func: Fonction(t) retournant le dict de CL gauche
            bc_right_func: Fonction(t) retournant le dict de CL droite
            save_every: Sauvegarder tous les N pas de temps
            verbose: Afficher la progression
            
        Returns:
            Dict contenant:
                'time': Array des temps sauvegardés
                'T': Array 2D (nt_saved, Nx) des températures
                'x': Coordonnées spatiales
        """
        # Initialisation
        if np.isscalar(T_init):
            T = np.ones(self.Nx) * T_init
        else:
            T = np.array(T_init).copy()
        
        nt = int(t_end / dt) + 1
        time_steps = np.linspace(0, t_end, nt)
        
        # Stockage
        save_indices = list(range(0, nt, save_every))
        if (nt - 1) not in save_indices:
            save_indices.append(nt - 1)
        
        T_hist = []
        time_hist = []
        
        for n, t in enumerate(time_steps):
            if n in save_indices:
                T_hist.append(T.copy())
                time_hist.append(t)
            
            if n < nt - 1:  # Pas de résolution après le dernier pas
                bc_left = bc_left_func(t + dt)
                bc_right = bc_right_func(t + dt)
                T = self.solve_step(T, dt, bc_left, bc_right)
            
            if verbose and n % 100 == 0:
                print(f"  t = {t:.2f} s, T_min = {T.min():.2f} K, T_max = {T.max():.2f} K")
        
        return {
            'time': np.array(time_hist),
            'T': np.array(T_hist),
            'x': self.x.copy()
        }


def create_constant_material(k, rho, cp, name='constant'):
    """
    Crée un matériau avec des propriétés constantes.
    
    Args:
        k: Conductivité thermique [W/(m·K)]
        rho: Masse volumique [kg/m³]
        cp: Capacité thermique massique [J/(kg·K)]
        name: Nom du matériau
        
    Returns:
        Dict compatible avec le solveur
    """
    # Table triviale (2 points suffisent pour interpolation constante)
    return {
        name: {
            'T': np.array([0, 1000]),
            'k': np.array([k, k]),
            'rho': np.array([rho, rho]),
            'cp': np.array([cp, cp])
        }
    }


def create_single_layer(L, material_name):
    """
    Crée une configuration monocouche.
    
    Args:
        L: Épaisseur [m]
        material_name: Nom du matériau
        
    Returns:
        Liste de couches (1 élément)
    """
    return [{'material': material_name, 'thickness': L}]


# ============================================================================
# Fonctions utilitaires pour l'analyse des résultats
# ============================================================================

def compute_error_norms(T_num, T_ref):
    """
    Calcule les normes d'erreur L2 et Linf.
    
    Args:
        T_num: Solution numérique
        T_ref: Solution de référence
        
    Returns:
        Dict avec 'L2', 'Linf', 'L2_rel', 'Linf_rel'
    """
    err = T_num - T_ref
    L2 = np.sqrt(np.mean(err**2))
    Linf = np.max(np.abs(err))
    
    # Erreurs relatives (éviter division par zéro)
    T_range = np.max(T_ref) - np.min(T_ref)
    if T_range > 1e-10:
        L2_rel = L2 / T_range
        Linf_rel = Linf / T_range
    else:
        L2_rel = L2 / (np.mean(np.abs(T_ref)) + 1e-10)
        Linf_rel = Linf / (np.mean(np.abs(T_ref)) + 1e-10)
    
    return {
        'L2': L2,
        'Linf': Linf,
        'L2_rel': L2_rel,
        'Linf_rel': Linf_rel
    }


def compute_heat_flux(T, x, k):
    """
    Calcule le flux de chaleur q = -k * dT/dx.
    
    Args:
        T: Champ de température
        x: Coordonnées spatiales
        k: Conductivité (scalaire ou array)
        
    Returns:
        Flux aux interfaces (Nx-1 valeurs)
    """
    dTdx = np.diff(T) / np.diff(x)
    if np.isscalar(k):
        k_interf = k
    else:
        k_interf = 0.5 * (k[:-1] + k[1:])
    return -k_interf * dTdx


def compute_energy_balance(T, T_prev, rho, cp, dx, dt, q_left, q_right):
    """
    Vérifie le bilan d'énergie sur un pas de temps.
    
    Args:
        T: Température au temps n+1
        T_prev: Température au temps n
        rho, cp: Propriétés (arrays)
        dx: Pas spatial
        dt: Pas de temps
        q_left: Flux entrant à gauche
        q_right: Flux entrant à droite
        
    Returns:
        Dict avec 'dE_stored', 'Q_in', 'residual'
    """
    # Énergie stockée
    dE = np.sum(rho * cp * (T - T_prev) * dx)
    
    # Énergie entrante
    Q_in = (q_left + q_right) * dt
    
    return {
        'dE_stored': dE,
        'Q_in': Q_in,
        'residual': dE - Q_in,
        'residual_rel': (dE - Q_in) / (np.abs(Q_in) + 1e-10)
    }
