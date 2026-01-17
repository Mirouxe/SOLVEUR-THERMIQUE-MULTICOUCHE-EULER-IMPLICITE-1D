"""
Solutions analytiques de référence pour la validation
=====================================================

Ce module contient des solutions analytiques exactes de l'équation de la chaleur 1D
pour différentes configurations de conditions initiales et limites.

Références:
- Carslaw & Jaeger, "Conduction of Heat in Solids", Oxford (1959)
- Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"
- Özışık, "Heat Conduction", Wiley (1993)

Équation de base:
    ∂T/∂t = α * ∂²T/∂x²
    
où α = k/(ρ*cp) est la diffusivité thermique [m²/s]
"""

import numpy as np
from scipy import special


# ============================================================================
# CAS 1: Plaque semi-infinie avec température de surface imposée
# ============================================================================

def semi_infinite_dirichlet(x, t, T_init, T_surf, alpha):
    """
    Plaque semi-infinie (x ≥ 0), initialement à T_init,
    avec température de surface T_surf imposée à x=0 pour t>0.
    
    Solution exacte (Carslaw & Jaeger, §2.4):
        T(x,t) = T_surf + (T_init - T_surf) * erf(x / (2*sqrt(α*t)))
    
    Args:
        x: Position(s) [m]
        t: Temps [s] (doit être > 0)
        T_init: Température initiale [K]
        T_surf: Température de surface [K]
        alpha: Diffusivité thermique [m²/s]
        
    Returns:
        Température T(x,t) [K]
    """
    if t <= 0:
        return np.ones_like(x) * T_init
    
    eta = x / (2 * np.sqrt(alpha * t))
    return T_surf + (T_init - T_surf) * special.erf(eta)


def semi_infinite_dirichlet_flux(x, t, T_init, T_surf, alpha, k):
    """
    Flux de chaleur pour le cas semi-infini avec Dirichlet.
    
    q(x,t) = k * (T_surf - T_init) / sqrt(π*α*t) * exp(-x²/(4*α*t))
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        T_init, T_surf: Températures [K]
        alpha: Diffusivité [m²/s]
        k: Conductivité [W/(m·K)]
        
    Returns:
        Flux de chaleur [W/m²]
    """
    if t <= 0:
        return np.zeros_like(x)
    
    return k * (T_surf - T_init) / np.sqrt(np.pi * alpha * t) * np.exp(-x**2 / (4 * alpha * t))


# ============================================================================
# CAS 2: Plaque semi-infinie avec flux de surface imposé
# ============================================================================

def semi_infinite_neumann(x, t, T_init, q0, alpha, k):
    """
    Plaque semi-infinie avec flux constant q0 imposé à x=0.
    
    Solution exacte (Carslaw & Jaeger, §2.9):
        T(x,t) = T_init + (2*q0/k) * sqrt(α*t/π) * exp(-x²/(4*α*t))
                        - (q0*x/k) * erfc(x/(2*sqrt(α*t)))
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        T_init: Température initiale [K]
        q0: Flux de surface imposé [W/m²] (positif = entrant)
        alpha: Diffusivité thermique [m²/s]
        k: Conductivité thermique [W/(m·K)]
        
    Returns:
        Température T(x,t) [K]
    """
    if t <= 0:
        return np.ones_like(x) * T_init
    
    x = np.atleast_1d(x)
    sqrt_at = np.sqrt(alpha * t)
    eta = x / (2 * sqrt_at)
    
    term1 = 2 * q0 / k * sqrt_at / np.sqrt(np.pi) * np.exp(-eta**2)
    term2 = q0 * x / k * special.erfc(eta)
    
    return T_init + term1 - term2


# ============================================================================
# CAS 3: Plaque finie avec conditions de Dirichlet aux deux bords
# ============================================================================

def finite_slab_dirichlet_dirichlet(x, t, L, T_init, T_left, T_right, alpha, n_terms=100):
    """
    Plaque finie 0 ≤ x ≤ L, initialement à T_init,
    avec T(0,t) = T_left et T(L,t) = T_right pour t > 0.
    
    Solution par séries de Fourier (Carslaw & Jaeger, §3.3):
        T(x,t) = T_left + (T_right - T_left)*x/L
                 + Σ_n B_n * sin(n*π*x/L) * exp(-n²*π²*α*t/L²)
    
    où B_n = (2/L) * ∫[T_init - T_stationnaire] * sin(n*π*x/L) dx
    
    Pour T_init uniforme:
        B_n = 2/(n*π) * [(T_init - T_left) - (-1)^n * (T_init - T_right)]
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        L: Épaisseur de la plaque [m]
        T_init: Température initiale uniforme [K]
        T_left: Température à x=0 [K]
        T_right: Température à x=L [K]
        alpha: Diffusivité thermique [m²/s]
        n_terms: Nombre de termes de la série
        
    Returns:
        Température T(x,t) [K]
    """
    x = np.atleast_1d(x)
    
    # Solution stationnaire
    T_stat = T_left + (T_right - T_left) * x / L
    
    if t <= 0:
        return np.ones_like(x) * T_init
    
    # Partie transitoire
    T_trans = np.zeros_like(x, dtype=float)
    
    for n in range(1, n_terms + 1):
        # Coefficient de Fourier pour T_init uniforme
        if n % 2 == 1:  # n impair
            Bn = 2 / (n * np.pi) * ((T_init - T_left) + (T_init - T_right))
        else:  # n pair
            Bn = 2 / (n * np.pi) * ((T_init - T_left) - (T_init - T_right))
        
        lambda_n = n * np.pi / L
        T_trans += Bn * np.sin(lambda_n * x) * np.exp(-lambda_n**2 * alpha * t)
    
    return T_stat + T_trans


# ============================================================================
# CAS 4: Plaque finie isolée à droite (adiabatique) avec Dirichlet à gauche
# ============================================================================

def finite_slab_dirichlet_adiabatic(x, t, L, T_init, T_left, alpha, n_terms=200):
    """
    Plaque finie 0 ≤ x ≤ L, initialement à T_init,
    avec T(0,t) = T_left et ∂T/∂x(L,t) = 0 (adiabatique).
    
    Solution par séries de Fourier (Carslaw & Jaeger, §3.4):
    La solution utilise les fonctions propres cos((2n-1)*π*x/(2L))
    qui satisfont les conditions aux limites.
    
    T(x,t) = T_left + Σ_n A_n * cos(λ_n*x) * exp(-λ_n²*α*t)
    
    où λ_n = (2n-1)*π/(2L) et A_n = 4*(T_init-T_left)/((2n-1)*π) * sin((2n-1)*π/2)
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        L: Épaisseur [m]
        T_init: Température initiale [K]
        T_left: Température imposée à gauche [K]
        alpha: Diffusivité thermique [m²/s]
        n_terms: Nombre de termes
        
    Returns:
        Température T(x,t) [K]
    """
    x = np.atleast_1d(x)
    
    if t <= 0:
        return np.ones_like(x) * T_init
    
    # Solution stationnaire = T_left (uniforme car adiabatique à droite)
    T = np.ones_like(x, dtype=float) * T_left
    
    # Partie transitoire avec séries de Fourier
    for n in range(1, n_terms + 1):
        lambda_n = (2*n - 1) * np.pi / (2 * L)
        # Coefficient de Fourier pour T_init uniforme
        # A_n = (2/L) * ∫_0^L (T_init - T_left) * cos(λ_n*x) dx
        # = 2*(T_init - T_left) * sin(λ_n*L) / (λ_n*L)
        # = 2*(T_init - T_left) * sin((2n-1)*π/2) / ((2n-1)*π/2)
        # = 4*(T_init - T_left) * (-1)^(n+1) / ((2n-1)*π)
        An = 4 * (T_init - T_left) / ((2*n - 1) * np.pi) * ((-1)**(n + 1))
        T += An * np.cos(lambda_n * x) * np.exp(-lambda_n**2 * alpha * t)
    
    return T


# ============================================================================
# CAS 5: Plaque finie avec convection à gauche et adiabatique à droite
# ============================================================================

def finite_slab_convection_adiabatic(x, t, L, T_init, h, T_inf, k, alpha, n_terms=50):
    """
    Plaque finie 0 ≤ x ≤ L, initialement à T_init,
    avec convection à gauche: -k*∂T/∂x(0,t) = h*(T(0,t) - T_inf)
    et adiabatique à droite: ∂T/∂x(L,t) = 0.
    
    Solution par séries de Fourier (Özışık, Heat Conduction):
        T(x,t) = T_inf + Σ_n C_n * cos(λ_n*(L-x)) * exp(-λ_n²*α*t)
    
    où les λ_n sont solutions de: λ_n*L*tan(λ_n*L) = Bi = h*L/k
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        L: Épaisseur [m]
        T_init: Température initiale [K]
        h: Coefficient de convection [W/(m²·K)]
        T_inf: Température du fluide [K]
        k: Conductivité [W/(m·K)]
        alpha: Diffusivité [m²/s]
        n_terms: Nombre de termes
        
    Returns:
        Température T(x,t) [K]
    """
    x = np.atleast_1d(x)
    
    if t <= 0:
        return np.ones_like(x) * T_init
    
    Bi = h * L / k  # Nombre de Biot
    
    # Trouver les valeurs propres λ_n*L telles que λ*L*tan(λ*L) = Bi
    eigenvalues = _find_eigenvalues_convection_adiabatic(Bi, n_terms)
    
    T = np.ones_like(x, dtype=float) * T_inf
    
    for zeta_n in eigenvalues:  # zeta_n = λ_n * L
        lambda_n = zeta_n / L
        
        # Coefficient de Fourier
        # C_n = 4*sin(zeta_n) * (T_init - T_inf) / (2*zeta_n + sin(2*zeta_n))
        C_n = 4 * np.sin(zeta_n) * (T_init - T_inf) / (2 * zeta_n + np.sin(2 * zeta_n))
        
        T += C_n * np.cos(lambda_n * (L - x)) * np.exp(-lambda_n**2 * alpha * t)
    
    return T


def _find_eigenvalues_convection_adiabatic(Bi, n_terms):
    """
    Trouve les n_terms premières racines de: ζ*tan(ζ) = Bi
    
    Les racines sont dans les intervalles ((n-1)*π, (n-0.5)*π) pour n=1,2,3...
    """
    from scipy.optimize import brentq
    
    eigenvalues = []
    
    def equation(zeta):
        return zeta * np.tan(zeta) - Bi
    
    for n in range(1, n_terms + 1):
        # Intervalle de recherche (éviter les singularités de tan)
        a = (n - 1) * np.pi + 1e-10
        b = (n - 0.5) * np.pi - 1e-10
        
        try:
            root = brentq(equation, a, b)
            eigenvalues.append(root)
        except ValueError:
            # Pas de racine dans cet intervalle (peut arriver pour Bi très petit)
            continue
    
    return np.array(eigenvalues)


# ============================================================================
# CAS 6: Solution de Lumped Capacitance (Biot << 1)
# ============================================================================

def lumped_capacitance(t, T_init, h, T_inf, rho, cp, V, A):
    """
    Modèle à capacité globale (valide si Bi = h*Lc/k << 0.1).
    
    Solution exacte:
        T(t) = T_inf + (T_init - T_inf) * exp(-t/τ)
    
    où τ = ρ*cp*V / (h*A) est la constante de temps.
    
    Args:
        t: Temps [s]
        T_init: Température initiale [K]
        h: Coefficient de convection [W/(m²·K)]
        T_inf: Température du fluide [K]
        rho: Masse volumique [kg/m³]
        cp: Capacité thermique [J/(kg·K)]
        V: Volume [m³]
        A: Surface d'échange [m²]
        
    Returns:
        Température T(t) [K]
    """
    tau = rho * cp * V / (h * A)
    return T_inf + (T_init - T_inf) * np.exp(-t / tau)


# ============================================================================
# CAS 7: Plaque avec flux sinusoïdal (régime périodique établi)
# ============================================================================

def periodic_flux_steady_state(x, t, L, q0, omega, k, alpha):
    """
    Plaque semi-infinie soumise à un flux sinusoïdal q(t) = q0*sin(ωt) à x=0.
    
    Solution en régime établi (partie oscillante):
        T(x,t) = (q0/k) * sqrt(α/ω) * exp(-x*sqrt(ω/(2α))) 
                 * sin(ωt - x*sqrt(ω/(2α)) - π/4)
    
    Args:
        x: Position(s) [m]
        t: Temps [s]
        L: Non utilisé (semi-infini)
        q0: Amplitude du flux [W/m²]
        omega: Pulsation [rad/s]
        k: Conductivité [W/(m·K)]
        alpha: Diffusivité [m²/s]
        
    Returns:
        Température oscillante [K]
    """
    x = np.atleast_1d(x)
    
    delta = np.sqrt(2 * alpha / omega)  # Profondeur de pénétration
    
    amplitude = q0 / k * delta / np.sqrt(2)
    phase = omega * t - x / delta - np.pi / 4
    
    return amplitude * np.exp(-x / delta) * np.sin(phase)


# ============================================================================
# CAS 8: Solution stationnaire multicouche
# ============================================================================

def steady_state_multilayer(x, layers, k_values, T_left, T_right):
    """
    Profil de température stationnaire dans un milieu multicouche.
    
    En régime permanent, le flux est constant et la température est
    linéaire par morceaux dans chaque couche.
    
    Args:
        x: Position(s) [m]
        layers: Liste de dict avec 'thickness'
        k_values: Conductivité de chaque couche [W/(m·K)]
        T_left: Température à gauche [K]
        T_right: Température à droite [K]
        
    Returns:
        Température T(x) [K]
    """
    x = np.atleast_1d(x)
    
    # Résistances thermiques
    R_total = sum(layer['thickness'] / k for layer, k in zip(layers, k_values))
    
    # Flux (constant dans tout le domaine)
    q = (T_left - T_right) / R_total
    
    # Température par morceaux
    T = np.zeros_like(x, dtype=float)
    
    cum_thick = np.cumsum([0] + [layer['thickness'] for layer in layers])
    T_interface = T_left
    
    for i, (layer, k) in enumerate(zip(layers, k_values)):
        x_start = cum_thick[i]
        x_end = cum_thick[i + 1]
        mask = (x >= x_start) & (x <= x_end)
        
        # Température linéaire dans la couche
        T[mask] = T_interface - q * (x[mask] - x_start) / k
        
        # Mise à jour pour la couche suivante
        T_interface = T_interface - q * layer['thickness'] / k
    
    return T


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def compute_fourier_number(alpha, t, L):
    """
    Calcule le nombre de Fourier: Fo = α*t/L²
    
    Fo >> 1 indique que le régime permanent est atteint.
    """
    return alpha * t / L**2


def compute_biot_number(h, L, k):
    """
    Calcule le nombre de Biot: Bi = h*L/k
    
    Bi << 0.1: modèle à capacité globale valide
    Bi >> 1: résistance de convection négligeable
    """
    return h * L / k


def thermal_diffusivity(k, rho, cp):
    """
    Calcule la diffusivité thermique: α = k/(ρ*cp) [m²/s]
    """
    return k / (rho * cp)


def thermal_time_constant(L, alpha):
    """
    Temps caractéristique de diffusion: τ = L²/α [s]
    """
    return L**2 / alpha


def penetration_depth(alpha, t):
    """
    Profondeur de pénétration thermique: δ ≈ sqrt(α*t) [m]
    
    Distance à laquelle la perturbation thermique s'est propagée.
    """
    return np.sqrt(alpha * t)
