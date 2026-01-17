"""
Bibliothèque de matériaux pour le modèle thermique
==================================================

Ce module contient une base de données de propriétés thermiques pour
différents matériaux, avec dépendance en température lorsque disponible.

Sources des données:
- Engineering Toolbox (https://www.engineeringtoolbox.com)
- MatWeb Material Property Data
- NIST Chemistry WebBook
- ASM Handbook Vol. 1 & 2
- Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"

Propriétés stockées:
- k: Conductivité thermique [W/(m·K)]
- rho: Masse volumique [kg/m³]
- cp: Capacité thermique massique [J/(kg·K)]

Usage:
    from material_library import MaterialLibrary, get_material
    
    # Obtenir un matériau
    steel = get_material('steel_1010')
    
    # Lister tous les matériaux
    MaterialLibrary.list_materials()
    
    # Obtenir les propriétés à une température donnée
    k, rho, cp = MaterialLibrary.get_properties('aluminum_6061', T=400)
"""

import numpy as np


class MaterialLibrary:
    """
    Bibliothèque centralisée des propriétés thermiques des matériaux.
    """
    
    # ========================================================================
    # MÉTAUX
    # ========================================================================
    
    METALS = {
        # ------------------------------------------------------------------
        # ACIERS
        # ------------------------------------------------------------------
        'steel_1010': {
            'name': 'Acier AISI 1010 (bas carbone)',
            'category': 'metal',
            'source': 'ASM Handbook',
            'T': np.array([300, 400, 500, 600, 700, 800, 900, 1000]),
            'k': np.array([63.9, 58.7, 52.7, 48.0, 42.0, 36.0, 31.0, 27.0]),
            'rho': np.array([7870, 7850, 7830, 7810, 7790, 7770, 7750, 7730]),
            'cp': np.array([434, 487, 519, 559, 615, 700, 750, 800])
        },
        'steel_304': {
            'name': 'Acier inoxydable AISI 304',
            'category': 'metal',
            'source': 'MatWeb',
            'T': np.array([300, 400, 500, 600, 700, 800, 900, 1000]),
            'k': np.array([14.9, 16.6, 18.3, 19.8, 21.3, 22.8, 24.2, 25.6]),
            'rho': np.array([7900, 7880, 7860, 7840, 7820, 7800, 7780, 7760]),
            'cp': np.array([477, 515, 540, 565, 590, 615, 640, 665])
        },
        'steel_316': {
            'name': 'Acier inoxydable AISI 316',
            'category': 'metal',
            'source': 'MatWeb',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([13.4, 15.2, 16.8, 18.3, 19.8, 21.2]),
            'rho': np.array([7990, 7970, 7950, 7930, 7910, 7890]),
            'cp': np.array([500, 530, 550, 570, 590, 610])
        },
        'steel_tool': {
            'name': 'Acier à outils (H13)',
            'category': 'metal',
            'source': 'ASM Handbook',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([24.3, 25.6, 26.8, 27.9, 28.8, 29.5]),
            'rho': np.array([7760, 7740, 7720, 7700, 7680, 7660]),
            'cp': np.array([460, 490, 520, 550, 580, 610])
        },
        
        # ------------------------------------------------------------------
        # ALUMINIUM ET ALLIAGES
        # ------------------------------------------------------------------
        'aluminum_pure': {
            'name': 'Aluminium pur (99.9%)',
            'category': 'metal',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600, 700, 800, 900]),
            'k': np.array([237, 240, 236, 231, 225, 218, 210]),
            'rho': np.array([2702, 2680, 2657, 2634, 2611, 2588, 2565]),
            'cp': np.array([903, 949, 996, 1040, 1080, 1120, 1160])
        },
        'aluminum_6061': {
            'name': 'Aluminium 6061-T6',
            'category': 'metal',
            'source': 'MatWeb',
            'T': np.array([300, 350, 400, 450, 500, 550, 600]),
            'k': np.array([167, 172, 177, 182, 186, 190, 193]),
            'rho': np.array([2700, 2690, 2680, 2670, 2660, 2650, 2640]),
            'cp': np.array([896, 920, 945, 970, 995, 1020, 1045])
        },
        'aluminum_7075': {
            'name': 'Aluminium 7075-T6',
            'category': 'metal',
            'source': 'MatWeb',
            'T': np.array([300, 400, 500, 600]),
            'k': np.array([130, 145, 155, 165]),
            'rho': np.array([2810, 2790, 2770, 2750]),
            'cp': np.array([960, 1000, 1040, 1080])
        },
        
        # ------------------------------------------------------------------
        # CUIVRE ET ALLIAGES
        # ------------------------------------------------------------------
        'copper_pure': {
            'name': 'Cuivre pur (OFHC)',
            'category': 'metal',
            'source': 'NIST',
            'T': np.array([300, 400, 500, 600, 700, 800, 900, 1000]),
            'k': np.array([401, 393, 386, 379, 372, 365, 358, 351]),
            'rho': np.array([8933, 8890, 8847, 8804, 8761, 8718, 8675, 8632]),
            'cp': np.array([385, 397, 408, 417, 425, 433, 440, 447])
        },
        'brass': {
            'name': 'Laiton (70Cu-30Zn)',
            'category': 'metal',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600]),
            'k': np.array([109, 128, 144, 155]),
            'rho': np.array([8530, 8500, 8470, 8440]),
            'cp': np.array([380, 395, 410, 425])
        },
        'bronze': {
            'name': 'Bronze phosphoreux',
            'category': 'metal',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600]),
            'k': np.array([50, 55, 60, 65]),
            'rho': np.array([8800, 8770, 8740, 8710]),
            'cp': np.array([380, 395, 410, 425])
        },
        
        # ------------------------------------------------------------------
        # TITANE ET ALLIAGES
        # ------------------------------------------------------------------
        'titanium_pure': {
            'name': 'Titane pur',
            'category': 'metal',
            'source': 'ASM Handbook',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([21.9, 20.4, 19.4, 18.8, 18.4, 18.2]),
            'rho': np.array([4506, 4490, 4474, 4458, 4442, 4426]),
            'cp': np.array([522, 540, 555, 570, 585, 600])
        },
        'titanium_6al4v': {
            'name': 'Titane Ti-6Al-4V',
            'category': 'metal',
            'source': 'MatWeb',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([6.7, 7.4, 8.7, 10.3, 12.0, 14.0]),
            'rho': np.array([4430, 4415, 4400, 4385, 4370, 4355]),
            'cp': np.array([526, 560, 590, 620, 650, 680])
        },
        
        # ------------------------------------------------------------------
        # NICKEL ET SUPERALLIAGES
        # ------------------------------------------------------------------
        'nickel_pure': {
            'name': 'Nickel pur',
            'category': 'metal',
            'source': 'ASM Handbook',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([90.7, 80.2, 72.1, 65.6, 60.5, 56.5]),
            'rho': np.array([8908, 8880, 8852, 8824, 8796, 8768]),
            'cp': np.array([444, 473, 499, 523, 545, 565])
        },
        'inconel_718': {
            'name': 'Inconel 718',
            'category': 'metal',
            'source': 'Special Metals',
            'T': np.array([300, 400, 500, 600, 700, 800, 900, 1000]),
            'k': np.array([11.4, 13.4, 15.5, 17.5, 19.6, 21.6, 23.7, 25.7]),
            'rho': np.array([8190, 8170, 8150, 8130, 8110, 8090, 8070, 8050]),
            'cp': np.array([435, 460, 485, 510, 535, 560, 585, 610])
        },
        
        # ------------------------------------------------------------------
        # AUTRES MÉTAUX
        # ------------------------------------------------------------------
        'tungsten': {
            'name': 'Tungstène',
            'category': 'metal',
            'source': 'NIST',
            'T': np.array([300, 500, 700, 1000, 1500, 2000]),
            'k': np.array([174, 159, 143, 125, 108, 95]),
            'rho': np.array([19300, 19250, 19200, 19100, 18950, 18800]),
            'cp': np.array([132, 137, 142, 148, 156, 163])
        },
        'gold': {
            'name': 'Or',
            'category': 'metal',
            'source': 'NIST',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([317, 311, 304, 298, 292, 287]),
            'rho': np.array([19300, 19230, 19160, 19090, 19020, 18950]),
            'cp': np.array([129, 131, 133, 135, 137, 139])
        },
        'silver': {
            'name': 'Argent',
            'category': 'metal',
            'source': 'NIST',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([429, 425, 419, 412, 405, 397]),
            'rho': np.array([10500, 10450, 10400, 10350, 10300, 10250]),
            'cp': np.array([235, 239, 243, 247, 251, 255])
        },
        'lead': {
            'name': 'Plomb',
            'category': 'metal',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600]),
            'k': np.array([35.3, 34.0, 32.8, 31.5]),
            'rho': np.array([11340, 11240, 11140, 11040]),
            'cp': np.array([129, 133, 137, 141])
        },
        'zinc': {
            'name': 'Zinc',
            'category': 'metal',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 350, 400, 450, 500]),
            'k': np.array([116, 113, 110, 107, 104]),
            'rho': np.array([7140, 7100, 7060, 7020, 6980]),
            'cp': np.array([388, 394, 400, 406, 412])
        },
    }
    
    # ========================================================================
    # ISOLANTS ET CÉRAMIQUES
    # ========================================================================
    
    INSULATORS = {
        'glass_wool': {
            'name': 'Laine de verre',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 350, 400, 450, 500]),
            'k': np.array([0.038, 0.045, 0.052, 0.060, 0.070]),
            'rho': np.array([24, 24, 24, 24, 24]),
            'cp': np.array([840, 840, 840, 840, 840])
        },
        'rock_wool': {
            'name': 'Laine de roche',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600, 700]),
            'k': np.array([0.037, 0.050, 0.065, 0.085, 0.110]),
            'rho': np.array([100, 100, 100, 100, 100]),
            'cp': np.array([840, 860, 880, 900, 920])
        },
        'polyurethane_foam': {
            'name': 'Mousse polyuréthane',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([250, 300, 350, 400]),
            'k': np.array([0.022, 0.026, 0.030, 0.035]),
            'rho': np.array([32, 32, 32, 32]),
            'cp': np.array([1400, 1400, 1400, 1400])
        },
        'polystyrene_expanded': {
            'name': 'Polystyrène expansé (EPS)',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([250, 300, 350]),
            'k': np.array([0.033, 0.038, 0.044]),
            'rho': np.array([25, 25, 25]),
            'cp': np.array([1300, 1300, 1300])
        },
        'calcium_silicate': {
            'name': 'Silicate de calcium',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600, 700, 800]),
            'k': np.array([0.055, 0.065, 0.075, 0.090, 0.105, 0.125]),
            'rho': np.array([240, 240, 240, 240, 240, 240]),
            'cp': np.array([840, 880, 920, 960, 1000, 1040])
        },
        'ceramic_fiber': {
            'name': 'Fibre céramique',
            'category': 'insulator',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 500, 700, 900, 1100, 1300]),
            'k': np.array([0.06, 0.10, 0.15, 0.22, 0.30, 0.40]),
            'rho': np.array([128, 128, 128, 128, 128, 128]),
            'cp': np.array([1130, 1130, 1130, 1130, 1130, 1130])
        },
        'alumina': {
            'name': 'Alumine (Al2O3)',
            'category': 'ceramic',
            'source': 'MatWeb',
            'T': np.array([300, 400, 500, 600, 800, 1000, 1200]),
            'k': np.array([35, 26, 20, 16, 11, 8, 6]),
            'rho': np.array([3970, 3960, 3950, 3940, 3920, 3900, 3880]),
            'cp': np.array([765, 880, 960, 1020, 1100, 1160, 1200])
        },
        'zirconia': {
            'name': 'Zircone (ZrO2)',
            'category': 'ceramic',
            'source': 'MatWeb',
            'T': np.array([300, 500, 700, 900, 1100, 1300]),
            'k': np.array([2.0, 2.0, 2.0, 2.0, 2.1, 2.2]),
            'rho': np.array([5680, 5660, 5640, 5620, 5600, 5580]),
            'cp': np.array([450, 520, 560, 590, 610, 630])
        },
        'silicon_carbide': {
            'name': 'Carbure de silicium (SiC)',
            'category': 'ceramic',
            'source': 'MatWeb',
            'T': np.array([300, 500, 700, 900, 1100, 1300]),
            'k': np.array([120, 80, 55, 42, 33, 28]),
            'rho': np.array([3210, 3200, 3190, 3180, 3170, 3160]),
            'cp': np.array([750, 950, 1080, 1160, 1220, 1260])
        },
    }
    
    # ========================================================================
    # MATÉRIAUX DE CONSTRUCTION
    # ========================================================================
    
    CONSTRUCTION = {
        'concrete': {
            'name': 'Béton',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 350, 400, 450, 500]),
            'k': np.array([1.4, 1.4, 1.4, 1.4, 1.4]),
            'rho': np.array([2300, 2300, 2300, 2300, 2300]),
            'cp': np.array([880, 900, 920, 940, 960])
        },
        'brick': {
            'name': 'Brique',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600]),
            'k': np.array([0.72, 0.75, 0.78, 0.82]),
            'rho': np.array([1920, 1900, 1880, 1860]),
            'cp': np.array([835, 860, 885, 910])
        },
        'firebite': {
            'name': 'Brique réfractaire',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 500, 700, 900, 1100, 1300]),
            'k': np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
            'rho': np.array([2100, 2080, 2060, 2040, 2020, 2000]),
            'cp': np.array([880, 960, 1020, 1070, 1110, 1140])
        },
        'glass': {
            'name': 'Verre (soda-lime)',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600, 700]),
            'k': np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
            'rho': np.array([2500, 2490, 2480, 2470, 2460]),
            'cp': np.array([840, 880, 920, 960, 1000])
        },
        'granite': {
            'name': 'Granite',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500]),
            'k': np.array([2.8, 2.6, 2.4]),
            'rho': np.array([2750, 2740, 2730]),
            'cp': np.array([790, 850, 910])
        },
        'marble': {
            'name': 'Marbre',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500]),
            'k': np.array([2.8, 2.6, 2.4]),
            'rho': np.array([2700, 2690, 2680]),
            'cp': np.array([880, 920, 960])
        },
        'wood_oak': {
            'name': 'Bois de chêne',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([280, 300, 320, 340, 360]),
            'k': np.array([0.16, 0.17, 0.18, 0.19, 0.20]),
            'rho': np.array([750, 750, 750, 750, 750]),
            'cp': np.array([2000, 2100, 2200, 2300, 2400])
        },
        'plaster': {
            'name': 'Plâtre',
            'category': 'construction',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 350, 400]),
            'k': np.array([0.48, 0.50, 0.52]),
            'rho': np.array([1680, 1670, 1660]),
            'cp': np.array([840, 860, 880])
        },
    }
    
    # ========================================================================
    # POLYMÈRES
    # ========================================================================
    
    POLYMERS = {
        'ptfe': {
            'name': 'PTFE (Téflon)',
            'category': 'polymer',
            'source': 'MatWeb',
            'T': np.array([250, 300, 350, 400, 450, 500]),
            'k': np.array([0.25, 0.25, 0.25, 0.26, 0.27, 0.28]),
            'rho': np.array([2200, 2190, 2180, 2170, 2160, 2150]),
            'cp': np.array([1000, 1050, 1100, 1150, 1200, 1250])
        },
        'nylon': {
            'name': 'Nylon 6/6',
            'category': 'polymer',
            'source': 'MatWeb',
            'T': np.array([250, 300, 350, 400]),
            'k': np.array([0.24, 0.25, 0.26, 0.27]),
            'rho': np.array([1140, 1130, 1120, 1110]),
            'cp': np.array([1600, 1700, 1800, 1900])
        },
        'peek': {
            'name': 'PEEK',
            'category': 'polymer',
            'source': 'MatWeb',
            'T': np.array([300, 350, 400, 450, 500]),
            'k': np.array([0.25, 0.26, 0.27, 0.28, 0.29]),
            'rho': np.array([1320, 1310, 1300, 1290, 1280]),
            'cp': np.array([1340, 1450, 1560, 1670, 1780])
        },
        'epoxy': {
            'name': 'Résine époxy',
            'category': 'polymer',
            'source': 'MatWeb',
            'T': np.array([300, 350, 400]),
            'k': np.array([0.20, 0.21, 0.22]),
            'rho': np.array([1200, 1190, 1180]),
            'cp': np.array([1100, 1200, 1300])
        },
        'silicone_rubber': {
            'name': 'Caoutchouc silicone',
            'category': 'polymer',
            'source': 'MatWeb',
            'T': np.array([250, 300, 350, 400, 450, 500]),
            'k': np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.30]),
            'rho': np.array([1100, 1090, 1080, 1070, 1060, 1050]),
            'cp': np.array([1100, 1150, 1200, 1250, 1300, 1350])
        },
    }
    
    # ========================================================================
    # FLUIDES (pour référence)
    # ========================================================================
    
    FLUIDS = {
        'air': {
            'name': 'Air (1 atm)',
            'category': 'gas',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 400, 500, 600, 700, 800, 1000]),
            'k': np.array([0.0263, 0.0338, 0.0407, 0.0469, 0.0524, 0.0573, 0.0667]),
            'rho': np.array([1.177, 0.883, 0.706, 0.588, 0.504, 0.441, 0.353]),
            'cp': np.array([1007, 1014, 1030, 1051, 1075, 1099, 1141])
        },
        'water': {
            'name': 'Eau liquide',
            'category': 'liquid',
            'source': 'NIST',
            'T': np.array([280, 300, 320, 340, 360, 373]),
            'k': np.array([0.582, 0.613, 0.640, 0.660, 0.674, 0.679]),
            'rho': np.array([1000, 996, 989, 979, 967, 958]),
            'cp': np.array([4198, 4182, 4180, 4188, 4203, 4217])
        },
        'oil_engine': {
            'name': 'Huile moteur',
            'category': 'liquid',
            'source': 'Engineering Toolbox',
            'T': np.array([300, 320, 340, 360, 380, 400]),
            'k': np.array([0.145, 0.143, 0.141, 0.139, 0.137, 0.135]),
            'rho': np.array([884, 871, 858, 845, 832, 819]),
            'cp': np.array([1909, 1993, 2077, 2161, 2245, 2329])
        },
    }
    
    # Dictionnaire combiné de tous les matériaux
    ALL_MATERIALS = {}
    ALL_MATERIALS.update(METALS)
    ALL_MATERIALS.update(INSULATORS)
    ALL_MATERIALS.update(CONSTRUCTION)
    ALL_MATERIALS.update(POLYMERS)
    ALL_MATERIALS.update(FLUIDS)
    
    @classmethod
    def list_materials(cls, category=None):
        """
        Liste tous les matériaux disponibles.
        
        Args:
            category: Filtrer par catégorie ('metal', 'insulator', 'construction', 
                      'polymer', 'gas', 'liquid')
        """
        print("\n" + "="*80)
        print("BIBLIOTHÈQUE DE MATÉRIAUX")
        print("="*80)
        
        categories = {
            'metal': cls.METALS,
            'insulator': cls.INSULATORS,
            'ceramic': cls.INSULATORS,  # Inclus dans insulators
            'construction': cls.CONSTRUCTION,
            'polymer': cls.POLYMERS,
            'gas': cls.FLUIDS,
            'liquid': cls.FLUIDS
        }
        
        if category:
            materials = {k: v for k, v in cls.ALL_MATERIALS.items() 
                        if v.get('category') == category}
            print(f"\nCatégorie: {category}")
        else:
            materials = cls.ALL_MATERIALS
            
        current_cat = None
        for key, mat in sorted(materials.items(), key=lambda x: x[1].get('category', '')):
            cat = mat.get('category', 'unknown')
            if cat != current_cat:
                current_cat = cat
                print(f"\n--- {cat.upper()} ---")
            
            T_range = f"[{mat['T'].min():.0f}-{mat['T'].max():.0f}] K"
            k_range = f"k=[{mat['k'].min():.2f}-{mat['k'].max():.2f}]"
            print(f"  {key:25s} | {mat['name']:30s} | {T_range:15s} | {k_range}")
        
        print("\n" + "="*80)
        print(f"Total: {len(materials)} matériaux")
        print("="*80)
    
    @classmethod
    def get_material(cls, name):
        """
        Retourne les propriétés d'un matériau.
        
        Args:
            name: Nom du matériau (clé du dictionnaire)
            
        Returns:
            Dict avec les propriétés du matériau au format du solveur
        """
        if name not in cls.ALL_MATERIALS:
            available = list(cls.ALL_MATERIALS.keys())
            raise ValueError(f"Matériau '{name}' non trouvé. Disponibles: {available}")
        
        mat = cls.ALL_MATERIALS[name]
        return {
            name: {
                'T': mat['T'].copy(),
                'k': mat['k'].copy(),
                'rho': mat['rho'].copy(),
                'cp': mat['cp'].copy()
            }
        }
    
    @classmethod
    def get_properties(cls, name, T):
        """
        Retourne les propriétés interpolées à une température donnée.
        
        Args:
            name: Nom du matériau
            T: Température [K]
            
        Returns:
            Tuple (k, rho, cp)
        """
        if name not in cls.ALL_MATERIALS:
            raise ValueError(f"Matériau '{name}' non trouvé.")
        
        mat = cls.ALL_MATERIALS[name]
        k = np.interp(T, mat['T'], mat['k'])
        rho = np.interp(T, mat['T'], mat['rho'])
        cp = np.interp(T, mat['T'], mat['cp'])
        
        return float(k), float(rho), float(cp)
    
    @classmethod
    def get_material_info(cls, name):
        """
        Affiche les informations détaillées d'un matériau.
        """
        if name not in cls.ALL_MATERIALS:
            raise ValueError(f"Matériau '{name}' non trouvé.")
        
        mat = cls.ALL_MATERIALS[name]
        
        print(f"\n{'='*60}")
        print(f"Matériau: {mat['name']}")
        print(f"{'='*60}")
        print(f"Catégorie: {mat.get('category', 'N/A')}")
        print(f"Source: {mat.get('source', 'N/A')}")
        print(f"\nPropriétés thermiques:")
        print(f"{'T [K]':>10} | {'k [W/mK]':>12} | {'ρ [kg/m³]':>12} | {'cp [J/kgK]':>12} | {'α [m²/s]':>12}")
        print("-"*65)
        
        for i in range(len(mat['T'])):
            T = mat['T'][i]
            k = mat['k'][i]
            rho = mat['rho'][i]
            cp = mat['cp'][i]
            alpha = k / (rho * cp)
            print(f"{T:10.0f} | {k:12.3f} | {rho:12.1f} | {cp:12.1f} | {alpha:12.2e}")
        
        print(f"{'='*60}")


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def get_material(name):
    """
    Raccourci pour obtenir un matériau.
    
    Args:
        name: Nom du matériau
        
    Returns:
        Dict au format du solveur
    """
    return MaterialLibrary.get_material(name)


def get_materials(*names):
    """
    Obtenir plusieurs matériaux combinés dans un seul dictionnaire.
    
    Args:
        *names: Noms des matériaux
        
    Returns:
        Dict combiné au format du solveur
    """
    result = {}
    for name in names:
        result.update(get_material(name))
    return result


def create_custom_material(name, T, k, rho, cp, category='custom', source='User'):
    """
    Crée un matériau personnalisé.
    
    Args:
        name: Nom du matériau
        T: Array de températures [K]
        k: Array de conductivités [W/(m·K)]
        rho: Array de masses volumiques [kg/m³]
        cp: Array de capacités thermiques [J/(kg·K)]
        category: Catégorie du matériau
        source: Source des données
        
    Returns:
        Dict au format du solveur
    """
    return {
        name: {
            'T': np.array(T),
            'k': np.array(k),
            'rho': np.array(rho),
            'cp': np.array(cp),
            'category': category,
            'source': source
        }
    }


def create_constant_material(name, k, rho, cp):
    """
    Crée un matériau avec des propriétés constantes.
    
    Args:
        name: Nom du matériau
        k: Conductivité [W/(m·K)]
        rho: Masse volumique [kg/m³]
        cp: Capacité thermique [J/(kg·K)]
        
    Returns:
        Dict au format du solveur
    """
    return {
        name: {
            'T': np.array([0, 2000]),
            'k': np.array([k, k]),
            'rho': np.array([rho, rho]),
            'cp': np.array([cp, cp])
        }
    }


# ============================================================================
# Script de démonstration
# ============================================================================

if __name__ == '__main__':
    # Lister tous les matériaux
    MaterialLibrary.list_materials()
    
    # Informations détaillées sur un matériau
    print("\n")
    MaterialLibrary.get_material_info('steel_304')
    
    # Exemple d'utilisation
    print("\n\nExemple d'utilisation:")
    print("-"*40)
    
    # Obtenir un matériau
    steel = get_material('steel_1010')
    print(f"Matériau 'steel_1010' obtenu: {list(steel.keys())}")
    
    # Obtenir les propriétés à 500 K
    k, rho, cp = MaterialLibrary.get_properties('steel_1010', 500)
    print(f"Propriétés à 500 K: k={k:.1f} W/(m·K), ρ={rho:.0f} kg/m³, cp={cp:.0f} J/(kg·K)")
    
    # Combiner plusieurs matériaux
    materials = get_materials('steel_1010', 'aluminum_6061', 'glass_wool')
    print(f"Matériaux combinés: {list(materials.keys())}")
