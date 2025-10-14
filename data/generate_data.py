"""
Generate synthetic blade cutting performance dataset using Taylor's Tool Life Equation
and material constants from ASM Handbook Vol. 16.

Outputs: data/blade_dataset.csv (8,000+ rows)
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Material-specific Taylor constants (C in min, n dimensionless)
# Source: ASM Handbook Vol. 16 (approximate values)
TAYLOR_CONSTANTS = {
    ('Steel', 'HSS'):      {'C': 80,   'n': 0.25},
    ('Steel', 'Carbide'):  {'C': 300,  'n': 0.22},
    ('Aluminum', 'HSS'):   {'C': 400,  'n': 0.18},
    ('Aluminum', 'Carbide'): {'C': 900, 'n': 0.15},
    ('Titanium', 'HSS'):   {'C': 60,   'n': 0.30},
    ('Titanium', 'Carbide'): {'C': 180, 'n': 0.27},
}

# Friction coefficients (approximate, dry/lubricated)
FRICTION_BASE = {
    'Steel':     0.7,
    'Aluminum':  0.5,
    'Titanium':  0.8,
}
FRICTION_LUBE_FACTOR = 0.6  # Lubrication reduces friction by ~40%

# Input ranges
N_SAMPLES = 20000

MATERIALS_TO_CUT = ['Steel', 'Aluminum', 'Titanium']
BLADE_MATERIALS = ['HSS', 'Carbide']
CUTTING_ANGLE_DEG_RANGE = (5, 25)
BLADE_THICKNESS_MM_RANGE = (2, 10)
CUTTING_SPEED_M_PER_MIN_RANGE = (20, 200)
APPLIED_FORCE_N_RANGE = (100, 2000)
OPERATING_TEMPERATURE_C_RANGE = (20, 600)
LUBRICATION_OPTIONS = [True, False]

def sample_inputs(n):
    """Randomly sample input features."""
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
    # Friction coefficient derived from material and lubrication
    friction = []
    for mat, lube in zip(data['material_to_cut'], data['lubrication']):
        base = FRICTION_BASE[mat]
        fc = base * FRICTION_LUBE_FACTOR if lube else base
        # Add small random variation
        fc += np.random.normal(0, 0.03)
        friction.append(round(max(0.1, min(fc, 1.2)), 3))
    data['friction_coefficient'] = friction
    return pd.DataFrame(data)

def compute_outputs(df):
    """Compute outputs using Taylor's equation and rule-based logic."""
    blade_lifespan_hrs = []
    wear_pct = []
    efficiency_pct = []
    perf_score = []

    for i, row in df.iterrows():
        key = (row['material_to_cut'], row['blade_material'])
        C = TAYLOR_CONSTANTS[key]['C']
        n = TAYLOR_CONSTANTS[key]['n']
        V = row['cutting_speed_m_per_min']

        # Taylor's equation: T (min) = C * V^(-n)
        T_min = C * (V ** (-n))
        # Add ±15% noise
        T_min *= np.random.uniform(0.85, 1.15)
        T_hrs = T_min / 60

        # Wear estimation: increases with speed, force, temp, friction; decreases with lubrication
        wear = (
            0.3 * (V / CUTTING_SPEED_M_PER_MIN_RANGE[1]) +
            0.2 * (row['applied_force_N'] / APPLIED_FORCE_N_RANGE[1]) +
            0.2 * (row['operating_temperature_C'] / OPERATING_TEMPERATURE_C_RANGE[1]) +
            0.2 * (row['friction_coefficient'] / 1.2) -
            (0.15 if row['lubrication'] else 0)
        )
        wear = np.clip(wear * 100 + np.random.uniform(-5, 5), 5, 95)

        # Cutting efficiency: physics-based with multiple factors
        # Optimal angle around 15°, optimal thickness around 6mm
        angle_factor = 1 - (abs(row['cutting_angle_deg'] - 15) / 20) ** 2
        thickness_factor = 1 - (abs(row['blade_thickness_mm'] - 6) / 8) ** 2
        
        # Speed efficiency (Goldilocks zone: 80-120 m/min is optimal)
        speed_normalized = row['cutting_speed_m_per_min'] / 100.0
        speed_factor = np.exp(-((speed_normalized - 1.0) ** 2) / 0.5)
        
        # Force efficiency (lower force relative to speed is better)
        force_normalized = row['applied_force_N'] / 1000.0
        force_factor = 1 / (1 + force_normalized * speed_normalized * 0.3)
        
        # Temperature penalty (high temp reduces efficiency)
        temp_factor = 1 - (row['operating_temperature_C'] / 600.0) * 0.4
        
        # Material-specific efficiency
        if row['blade_material'] == 'Carbide':
            material_bonus = 0.15
        else:
            material_bonus = 0.0
        
        # Add force-temperature interaction (high force & temp penalizes efficiency)
        force_temp_interaction = 1 - ((row['applied_force_N'] / 2000.0) * (row['operating_temperature_C'] / 600.0)) * 0.25
        eff = (
            0.23 * angle_factor +
            0.13 * thickness_factor +
            0.18 * speed_factor +
            0.13 * force_factor +
            0.09 * temp_factor +
            0.09 * (1 - row['friction_coefficient'] / 1.2) +
            0.05 * material_bonus +
            0.10 * force_temp_interaction
        )
        
        # Convert to percentage and add realistic noise
        eff = np.clip(eff * 100 + np.random.uniform(-4, 4), 35, 98)

        # Performance score: weighted sum of outputs
        score = (
            0.35 * (T_hrs / (max(T_hrs, 20))) * 100 +
            0.25 * (100 - wear) +
            0.25 * eff +
            0.15 * (1 if row['lubrication'] else 0) * 100
        )
        score = np.clip(score + np.random.uniform(-2, 2), 10, 100)

        blade_lifespan_hrs.append(round(T_hrs, 2))
        wear_pct.append(round(wear, 1))
        efficiency_pct.append(round(eff, 1))
        perf_score.append(round(score, 1))

    df['blade_lifespan_hrs'] = blade_lifespan_hrs
    df['wear_estimation_pct'] = wear_pct
    df['cutting_efficiency_pct'] = efficiency_pct
    df['performance_score'] = perf_score
    return df

def main():
    df_inputs = sample_inputs(N_SAMPLES)
    df_full = compute_outputs(df_inputs)
    df_full.to_csv('data/blade_dataset.csv', index=False)
    print(f"Saved synthetic dataset with {len(df_full)} rows to data/blade_dataset.csv")

if __name__ == "__main__":
    main()
