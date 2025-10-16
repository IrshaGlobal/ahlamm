"""
Generate synthetic blade cutting performance dataset using Taylor's Tool Life Equation
and material constants from ASM Handbook Vol. 16.

Master's Thesis Project - Mechanical Engineering
Physics-informed dataset for cutting blade optimization.

Outputs: data/blade_dataset.csv (8,000 samples, thesis-compliant)
"""

import numpy as np
import pandas as pd

# Reproducibility seeds
np.random.seed(42)

# Dataset size - thesis appropriate
N_SAMPLES = 8000

# Material-specific Taylor constants (C in min, n dimensionless)
# Source: ASM Handbook Vol. 16 (verified values)
TAYLOR_CONSTANTS = {
    ('Steel', 'HSS'):      {'C': 80,   'n': 0.25},
    ('Steel', 'Carbide'):  {'C': 300,  'n': 0.22},
    ('Aluminum', 'HSS'):   {'C': 400,  'n': 0.18},
    ('Aluminum', 'Carbide'): {'C': 900, 'n': 0.15},
    ('Titanium', 'HSS'):   {'C': 60,   'n': 0.30},
    ('Titanium', 'Carbide'): {'C': 180, 'n': 0.27},
}

# Friction coefficients (dry/lubricated)
FRICTION_BASE = {
    'Steel': 0.60,
    'Aluminum': 0.30,
    'Titanium': 0.65,
}
FRICTION_LUBE_FACTOR = 0.6

# Thesis-approved materials and ranges
MATERIALS_TO_CUT = ['Steel', 'Aluminum', 'Titanium']
BLADE_MATERIALS = ['HSS', 'Carbide']

# Thesis-approved input ranges
CUTTING_ANGLE_DEG_RANGE = (15, 45)
BLADE_THICKNESS_MM_RANGE = (0.5, 5.0)
CUTTING_SPEED_M_PER_MIN_RANGE = (20, 200)
APPLIED_FORCE_N_RANGE = (100, 2000)
OPERATING_TEMPERATURE_C_RANGE = (20, 800)
LUBRICATION_OPTIONS = [True, False]

def sample_inputs(n):
    """Randomly sample input features according to thesis specifications."""
    data = {
        'material_to_cut': np.random.choice(MATERIALS_TO_CUT, n),
        'blade_material': np.random.choice(BLADE_MATERIALS, n),
        'cutting_angle_deg': np.random.uniform(*CUTTING_ANGLE_DEG_RANGE, n),
        'blade_thickness_mm': np.random.uniform(*BLADE_THICKNESS_MM_RANGE, n),
        'cutting_speed_m_per_min': np.random.uniform(*CUTTING_SPEED_M_PER_MIN_RANGE, n),
        'applied_force_N': np.random.uniform(*APPLIED_FORCE_N_RANGE, n),
        'operating_temperature_C': np.random.uniform(*OPERATING_TEMPERATURE_C_RANGE, n),
        'lubrication': np.random.choice(LUBRICATION_OPTIONS, n),
    }
    
    # Auto-calculate friction coefficient from material and lubrication
    friction = []
    for mat, lube in zip(data['material_to_cut'], data['lubrication']):
        base = FRICTION_BASE[mat]
        fc = base * FRICTION_LUBE_FACTOR if lube else base
        # Add small random variation (±3%)
        fc += np.random.normal(0, 0.03)
        friction.append(round(max(0.1, min(fc, 1.2)), 3))
    data['friction_coefficient'] = friction
    return pd.DataFrame(data)

def compute_outputs(df):
    """Compute 3 outputs using Taylor's equation and physics-based rules."""
    blade_lifespan_hrs = []
    wear_pct = []
    efficiency_pct = []

    for i, row in df.iterrows():
        key = (row['material_to_cut'], row['blade_material'])
        
        if key not in TAYLOR_CONSTANTS:
            # Fallback to Steel+Carbide for missing combinations
            key = ('Steel', 'Carbide')
        
        C = TAYLOR_CONSTANTS[key]['C']
        n = TAYLOR_CONSTANTS[key]['n']
        V = row['cutting_speed_m_per_min']

        # Taylor's Tool Life Equation: T (min) = C * V^(-n)
        T_min = C * (V ** (-n))
        T_min *= np.random.uniform(0.90, 1.10)  # ±10% noise
        T_hrs = T_min / 60

        # Wear estimation: physics-based factors
        wear = (
            0.3 * (V / 200) +  # Speed factor
            0.2 * (row['applied_force_N'] / 2000) +  # Force factor
            0.2 * (row['operating_temperature_C'] / 800) +  # Temperature factor
            0.2 * (row['friction_coefficient'] / 1.2) -  # Friction factor
            (0.15 if row['lubrication'] else 0)  # Lubrication benefit
        )
        wear = np.clip(wear * 100 + np.random.uniform(-3, 3), 5, 95)

        # Cutting efficiency: optimal conditions around 30° angle, moderate speeds
        angle_factor = 1 - (abs(row['cutting_angle_deg'] - 30) / 30) ** 2
        thickness_factor = 1 - (abs(row['blade_thickness_mm'] - 2.5) / 4) ** 2
        speed_factor = np.exp(-((row['cutting_speed_m_per_min'] - 100) ** 2) / 5000)
        force_factor = 1 / (1 + (row['applied_force_N'] / 1000) * 0.3)
        temp_factor = 1 - (row['operating_temperature_C'] / 800) * 0.4
        material_bonus = 0.15 if row['blade_material'] == 'Carbide' else 0.0
        
        eff = (
            0.25 * angle_factor +
            0.15 * thickness_factor +
            0.20 * speed_factor +
            0.15 * force_factor +
            0.10 * temp_factor +
            0.10 * (1 - row['friction_coefficient'] / 1.2) +
            0.05 * material_bonus
        )
        eff = np.clip(eff * 100 + np.random.uniform(-2, 2), 35, 98)

        blade_lifespan_hrs.append(round(T_hrs, 2))
        wear_pct.append(round(wear, 1))
        efficiency_pct.append(round(eff, 1))

    df['blade_lifespan_hrs'] = blade_lifespan_hrs
    df['wear_estimation_pct'] = wear_pct
    df['cutting_efficiency_pct'] = efficiency_pct
    return df

def main():
    """Generate thesis-compliant synthetic dataset."""
    print(f"🔧 Generating synthetic dataset with {N_SAMPLES} samples...")
    print(f"📊 Materials: {MATERIALS_TO_CUT}")
    print(f"🔩 Blade types: {BLADE_MATERIALS}")
    
    df_inputs = sample_inputs(N_SAMPLES)
    df_full = compute_outputs(df_inputs)
    df_full.to_csv('data/blade_dataset.csv', index=False)
    
    print(f"\n✅ Dataset saved: {len(df_full)} rows, {len(df_full.columns)} columns")
    print(f"   Columns: {list(df_full.columns)}")
    print(f"📈 Sample distribution:")
    print(f"   Materials: {df_full['material_to_cut'].value_counts().to_dict()}")
    print(f"   Blade types: {df_full['blade_material'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
