import numpy as np
import matplotlib.pyplot as plt

# Définitions de matériels avec leurs tables de propriétés thermiques (exemple fictif)
# Chaque table contient des valeurs de température (T_tab) 
# et les valeurs correspondantes de k, rho, cp (à remplacer par des données réelles).
material_data = {
    'steel': {
        'T':   np.array([300, 400, 500, 600]), 
        'k':   np.array([50,  45,  40,  35 ]),      # W/(m·K)
        'rho': np.array([7800,7800,7800,7800]),    # kg/m^3
        'cp':  np.array([500,  550,  600,  650])   # J/(kg·K)
    },
    'aluminum': {
        'T':   np.array([300, 400, 500, 600]),
        'k':   np.array([200, 190, 180, 170]),
        'rho': np.array([2700,2700,2700,2700]),
        'cp':  np.array([900,  920,  940,  960])
    },
    'titanium': {
        'T':   np.array([300, 400, 500, 600]),
        'k':   np.array([7,    6.5,  6,    5.5]),
        'rho': np.array([4500,4500,4500,4500]),
        'cp':  np.array([520,  540,  560,  580])
    }
}

# Fonction d'interpolation des propriétés k, rho, cp à partir de la table d'un matériau
def interp_property(mat_props, T):
    """
    Retourne (k, rho, cp) pour la température (ou les températures) T 
    en interpolant linéairement les tables du matériau.
    """
    T_tab = mat_props['T']
    k_tab = mat_props['k']
    rho_tab = mat_props['rho']
    cp_tab = mat_props['cp']
    T = np.array(T, ndmin=1)  # assurer un array numpy
    k   = np.interp(T, T_tab, k_tab)
    rho = np.interp(T, T_tab, rho_tab)
    cp  = np.interp(T, T_tab, cp_tab)
    return k, rho, cp

# Définition des couches (chaque couche a un matériau et une épaisseur)
layers = [
    {'material': 'steel',    'thickness': 0.005},  # acier, 5 mm
    {'material': 'aluminum', 'thickness': 0.010},  # aluminium, 10 mm
    {'material': 'titanium','thickness': 0.003}    # titane, 3 mm
]
L_total = sum(layer['thickness'] for layer in layers)  # longueur totale du domaine
Nx = 101               # nombre de nœuds (modifiable)
dx = L_total / (Nx-1)  # pas spatial uniforme
x = np.linspace(0, L_total, Nx)

# Déterminer pour chaque nœud son indice de couche (multi-matériaux)
cum_thick = np.cumsum([layer['thickness'] for layer in layers])
layer_index = np.searchsorted(cum_thick, x)  # 0 pour 1ère couche, 1 pour 2ème, etc.

# Condition initiale : température uniforme
T_init = 300.0  # [K]
T = np.ones(Nx) * T_init

# Lecture du fichier CSV des données de flux convectif
# (colonnes : temps [s], coefficient h [W/m^2K], température du fluide [K])
data = np.loadtxt('flux.csv', delimiter=',', skiprows=1)
t_flux    = data[:,0]
h_data    = data[:,1]
T_inf_data = data[:,2]

# Paramètres temporels
t0    = t_flux[0]
t_end = t_flux[-1]
dt    = 0.1                  # pas de temps [s]
nt    = int((t_end - t0)/dt) + 1
time  = np.linspace(t0, t_end, nt)

# Stockage de l'historique pour visualisation
T_hist = np.zeros((nt, Nx))

for n in range(nt):
    t = time[n]
    # Interpoler h(t) et T_inf(t) depuis les données CSV
    h  = np.interp(t, t_flux, h_data)
    T_inf = np.interp(t, t_flux, T_inf_data)
    
    # Construire la matrice A et le vecteur b du système linéaire implicite
    A = np.zeros((Nx, Nx))
    b = np.zeros(Nx)
    
    # Interpoler les propriétés thermiques à chaque nœud selon la température actuelle T[i]
    k_nodes   = np.zeros(Nx)
    rho_nodes = np.zeros(Nx)
    cp_nodes  = np.zeros(Nx)
    for i in range(Nx):
        mat = layers[layer_index[i]]['material']
        k_nodes[i], rho_nodes[i], cp_nodes[i] = interp_property(material_data[mat], T[i])
    
    # *** Bord gauche (i=0) : flux convectif h*(T_inf - T0) ***
    # Moyenne de conductivité entre nœud 0 et 1
    k0p = 0.5*(k_nodes[0] + k_nodes[1])
    rho0 = rho_nodes[0]
    cp0  = cp_nodes[0]
    # Équation implicite modifiée pour i=0
    A[0,0] = 1 + (dt/(rho0*cp0))*(k0p/dx + h)
    A[0,1] = - (dt/(rho0*cp0)) * (k0p/dx)
    # Terme source dû à la température du fluide externe
    b[0] = T[0] + (dt/(rho0*cp0)) * h * T_inf
    
    # *** Nœuds intérieurs (1 ≤ i ≤ Nx-2) ***
    for i in range(1, Nx-1):
        kim = 0.5*(k_nodes[i]   + k_nodes[i-1])  # conductivité moyenne gauche
        kip = 0.5*(k_nodes[i]   + k_nodes[i+1])  # conductivité moyenne droite
        rhoi = rho_nodes[i]
        cpi  = cp_nodes[i]
        A[i,i-1] = - (dt/(rhoi*cpi)) * (kim/dx**2)
        A[i,i]   = 1 + (dt/(rhoi*cpi)) * ((kim+kip)/dx**2)
        A[i,i+1] = - (dt/(rhoi*cpi)) * (kip/dx**2)
        b[i] = T[i]
    
    # *** Bord droit (i=Nx-1) : adiabatique (∂T/∂x=0) ***
    # On impose T[N-1] = T[N-2] en pratique
    i = Nx-1
    kim = 0.5*(k_nodes[i] + k_nodes[i-1])
    rhoi = rho_nodes[i]
    cpi  = cp_nodes[i]
    A[i,i-1] = - (dt/(rhoi*cpi)) * (kim/dx**2)
    A[i,i]   = 1 + (dt/(rhoi*cpi)) * (kim/dx**2)
    b[i] = T[i]  # pas de terme de source additionnel pour adiabatique
    
    # Résoudre le système linéaire pour obtenir T à l'instant suivant
    T_new = np.linalg.solve(A, b)
    T = T_new.copy()
    T_hist[n,:] = T

# Visualisation de l’évolution de la température dans le domaine
plt.figure(figsize=(6,4))
for idx in [0, nt//3, 2*nt//3, nt-1]:
    plt.plot(x, T_hist[idx,:], label=f't={time[idx]:.1f} s')
plt.xlabel("Position $x$ [m]")
plt.ylabel("Température $T$ [K]")
plt.legend()
plt.title("Évolution temporelle du profil de température")
plt.show()
