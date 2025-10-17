"""
Enhanced data generation for blade optimization system.
Generates physics-based synthetic data for thesis project.

Mechanical Engineering - Post-Graduation Project
Physics-informed data generation using Taylor's Tool Life Equation.

Scope:
- 6 Materials to Cut
- 7 Blade Materials
- 4 Blade Types
- 168 Total Combinations
- ~50,000 Total Samples

Outputs: 3 predictions only
1. Blade Lifespan (hrs)
2. Wear Estimation (%)
3. Cutting Efficiency (%)

Performance Score calculated post-prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# MATERIAL DEFINITIONS (CORRECTED PER USER REQUEST)
# ============================================================================

MATERIALS_TO_CUT = [
    "Steel",
    "Stainless Steel",
    "Aluminum",
    "Cast Iron",
    "Brass",
    "Titanium"
]

BLADE_MATERIALS = [
    "HSS",
    "Carbide",
    "Coated Carbide (TiN)",
    "Coated Carbide (TiAlN)",
    "Ceramic",
    "CBN",
    "PCD"
]

BLADE_TYPES = [
    "Straight Blade",
    "Circular Blade",
    "Insert/Replaceable Tip Blade",
    "Toothed Blade"
]

# ============================================================================
# MATERIAL PROPERTIES (Engineering Reference Data)
# ============================================================================

# Material hardness ranges (HRC)
MATERIAL_HARDNESS = {
    "Steel": (20, 40),
    "Stainless Steel": (25, 45),
    "Aluminum": (10, 30),
    "Cast Iron": (20, 60),
    "Brass": (10, 20),
    "Titanium": (30, 42)
}

# Blade material performance factors (relative to HSS = 1.0)
BLADE_PERFORMANCE = {
    "HSS": {"hardness": 850, "max_temp": 600, "cost_factor": 1.0, "wear_resistance": 1.0},
    "Carbide": {"hardness": 1750, "max_temp": 1000, "cost_factor": 3.0, "wear_resistance": 3.0},
    "Coated Carbide (TiN)": {"hardness": 2250, "max_temp": 1000, "cost_factor": 4.0, "wear_resistance": 4.5},
    "Coated Carbide (TiAlN)": {"hardness": 2750, "max_temp": 1100, "cost_factor": 5.0, "wear_resistance": 5.5},
    "Ceramic": {"hardness": 2250, "max_temp": 1200, "cost_factor": 6.0, "wear_resistance": 4.0},
    "CBN": {"hardness": 4500, "max_temp": 1400, "cost_factor": 20.0, "wear_resistance": 12.0},
    "PCD": {"hardness": 9000, "max_temp": 700, "cost_factor": 50.0, "wear_resistance": 15.0}
}

# Blade type characteristics
BLADE_TYPE_PROPERTIES = {
    "Straight Blade": {"speed_factor": 1.0, "wear_factor": 1.0, "efficiency_factor": 1.0},
    "Circular Blade": {"speed_factor": 1.3, "wear_factor": 0.9, "efficiency_factor": 1.2},
    "Insert/Replaceable Tip Blade": {"speed_factor": 1.1, "wear_factor": 0.7, "efficiency_factor": 1.1},
    "Toothed Blade": {"speed_factor": 0.9, "wear_factor": 1.2, "efficiency_factor": 0.95}
}

# Cutting speed recommendations (m/min) by material
CUTTING_SPEED_RANGES = {
    "Steel": (80, 150),
    "Stainless Steel": (50, 100),
    "Aluminum": (150, 300),
    "Cast Iron": (80, 180),
    "Brass": (100, 250),
    "Titanium": (30, 80)
}

# ============================================================================
# PHYSICS-BASED CALCULATIONS (Taylor's Tool Life Equation)
# ============================================================================

def calculate_tool_life_taylor(cutting_speed, material, blade_material, temp, lubrication):
    """
    Simplified tool life calculation based on cutting mechanics.
    Returns realistic tool life in hours (typically 0.5 - 10 hrs).
    """
    # Base tool life hours (at optimal conditions)
    base_life = {
        "Steel": 3.0,
        "Stainless Steel": 2.0,
        "Aluminum": 5.0,
        "Cast Iron": 4.0,
        "Brass": 6.0,
        "Titanium": 1.5
    }
    
    # Get base life for material
    life = base_life[material]
    
    # Blade material multiplier (better blades last longer)
    life *= BLADE_PERFORMANCE[blade_material]["wear_resistance"] ** 0.5
    
    # Speed effect (higher speed = shorter life)
    optimal_speed = np.mean(CUTTING_SPEED_RANGES[material])
    speed_ratio = cutting_speed / optimal_speed
    life *= (1 / speed_ratio) ** 0.3  # Inverse relationship
    
    # Temperature effect (higher temp = shorter life)
    if temp > 400:
        temp_penalty = (800 - temp) / 400
        temp_penalty = max(0.3, min(1.0, temp_penalty))
        life *= temp_penalty
    
    # Lubrication bonus (20-40% improvement)
    if lubrication:
        life *= np.random.uniform(1.2, 1.4)
    
    # Add realistic variation
    life *= np.random.uniform(0.8, 1.2)
    
    # Clamp to realistic range (0.2 to 15 hours)
    life = max(0.2, min(15.0, life))
    
    return life

def calculate_wear_estimation(lifespan, blade_material, operating_temp, friction):
    """
    Estimate wear percentage based on operating conditions.
    Wear increases with temperature, friction, and decreases with blade quality.
    """
    # Base wear (inversely proportional to lifespan)
    base_wear = 100 / (lifespan + 1)  # Higher lifespan = lower wear
    
    # Temperature effect (normalized to 0-1 scale)
    temp_effect = operating_temp / 800  # Max temp 800°C
    
    # Friction effect
    friction_effect = friction / 0.8  # Max friction ~0.8
    
    # Blade material resistance
    resistance = BLADE_PERFORMANCE[blade_material]["wear_resistance"]
    
    # Calculate wear
    wear = base_wear * (1 + temp_effect * 0.5 + friction_effect * 0.3) / resistance
    
    # Add noise and clamp to [0, 100]
    wear = wear + np.random.normal(0, 5)
    wear = max(0, min(100, wear))
    
    return wear

def calculate_cutting_efficiency(material, blade_material, blade_type,
                                 cutting_angle, speed, force, temp, friction, lubrication):
    """
    Improved cutting efficiency model using smooth optima and synergy terms.
    - Bell-shaped response around optimal cutting speed and angle
    - Moderate force optimum
    - Lubrication bonus and friction penalty
    - Blade-material hardness synergy and blade-type efficiency factor
    Output clamped to [20, 100] (%).
    """
    # Angle kernel (optimum around 20° with sigma ~10°)
    angle_sigma = 10.0
    angle_kernel = np.exp(-((cutting_angle - 20.0) ** 2) / (2 * angle_sigma ** 2))  # 0..1

    # Speed kernel (optimum at material's nominal speed, sigma = 35% of optimum)
    optimal_speed = np.mean(CUTTING_SPEED_RANGES[material])
    speed_ratio = speed / (optimal_speed + 1e-6)
    speed_sigma = 0.35
    speed_kernel = np.exp(-((speed_ratio - 1.0) ** 2) / (2 * speed_sigma ** 2))  # 0..1

    # Force kernel (optimum ~900 N, sigma ~500 N)
    force_sigma = 500.0
    force_kernel = np.exp(-((force - 900.0) ** 2) / (2 * force_sigma ** 2))

    # Temperature penalty (soft penalty past 450°C, harsh past 650°C)
    if temp <= 450:
        temp_factor = 1.0
    elif temp <= 650:
        temp_factor = 1.0 - (temp - 450.0) / 600.0  # down to ~0.67
    else:
        temp_factor = max(0.35, 1.0 - (temp - 650.0) / 300.0)  # clamp at 0.35

    # Lubrication bonus and friction penalty
    lub_bonus = 1.10 if lubrication else 1.0
    friction_penalty = 1.0 - min(0.35, max(0.0, (friction - 0.25) * 0.6))

    # Blade-material synergy via hardness ratio
    mat_hard = np.mean(MATERIAL_HARDNESS[material])
    blade_hard = BLADE_PERFORMANCE[blade_material]["hardness"]
    hardness_ratio = blade_hard / (mat_hard + 1e-6)
    # Smooth cap of synergy in [0.8, 1.15]
    synergy = 0.8 + 0.35 * (np.tanh((hardness_ratio - 1.0) * 0.6) + 1) / 2.0

    # Blade type factor
    type_factor = BLADE_TYPE_PROPERTIES[blade_type]["efficiency_factor"]

    # Combine multiplicatively with weighted geometric mean
    base = (angle_kernel ** 0.25) * (speed_kernel ** 0.35) * (force_kernel ** 0.15)
    modifiers = temp_factor * lub_bonus * friction_penalty * synergy * type_factor
    efficiency = 100.0 * np.clip(base * modifiers, 0.2, 1.0)

    # Small noise
    efficiency = efficiency + np.random.normal(0, 3.0)
    efficiency = float(np.clip(efficiency, 20.0, 100.0))
    return efficiency

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_sample(material, blade_material, blade_type):
    """Generate a single realistic sample for given combination."""
    
    # Generate input parameters with realistic ranges
    cutting_angle = np.random.uniform(15, 45)  # degrees
    blade_thickness = np.random.uniform(0.5, 5.0)  # mm
    
    # Cutting speed based on material recommendation
    speed_range = CUTTING_SPEED_RANGES[material]
    cutting_speed = np.random.uniform(speed_range[0], speed_range[1])
    
    # Apply blade type speed factor
    cutting_speed *= BLADE_TYPE_PROPERTIES[blade_type]["speed_factor"]
    
    applied_force = np.random.uniform(100, 2000)  # Newtons
    operating_temp = np.random.uniform(50, 700)  # Celsius
    
    # Lubrication (80% of samples use lubrication)
    lubrication = np.random.random() < 0.8
    
    # Friction coefficient (depends on material and lubrication)
    base_friction = {
        "Steel": 0.60,
        "Stainless Steel": 0.65,
        "Aluminum": 0.30,
        "Cast Iron": 0.55,
        "Brass": 0.35,
        "Titanium": 0.65
    }
    friction = base_friction[material] * (0.6 if lubrication else 1.0)
    friction += np.random.normal(0, 0.05)
    friction = max(0.1, min(0.8, friction))
    
    # Calculate outputs using physics-based models
    blade_lifespan = calculate_tool_life_taylor(
        cutting_speed, material, blade_material, operating_temp, lubrication
    )
    
    wear_estimation = calculate_wear_estimation(
        blade_lifespan, blade_material, operating_temp, friction
    )
    
    cutting_efficiency = calculate_cutting_efficiency(
        material, blade_material, blade_type,
        cutting_angle, cutting_speed, applied_force, operating_temp, friction, lubrication
    )

    # Derived features (for EDA; training recomputes these internally)
    mat_hard_avg = np.mean(MATERIAL_HARDNESS[material])
    blade_hard = BLADE_PERFORMANCE[blade_material]["hardness"]
    optimal_speed = np.mean(CUTTING_SPEED_RANGES[material])
    derived = {
        "speed_ratio": cutting_speed / (optimal_speed + 1e-6),
        "thermal_load_ratio": operating_temp / (BLADE_PERFORMANCE[blade_material]["max_temp"] + 1e-6),
        "hardness_ratio": blade_hard / (mat_hard_avg + 1e-6),
        "force_speed_ratio": applied_force / (cutting_speed + 1e-6),
        "geometry_cos": np.cos(np.deg2rad(cutting_angle)),
        "specific_energy_proxy": applied_force * friction / (cutting_speed + 1e-6),
        "blade_type_efficiency_factor": BLADE_TYPE_PROPERTIES[blade_type]["efficiency_factor"],
        "blade_material_wear_resistance": BLADE_PERFORMANCE[blade_material]["wear_resistance"],
        "blade_hardness": blade_hard,
        "material_hardness_avg": mat_hard_avg,
    }
    
    base = {
        # Input features
        "material_to_cut": material,
        "blade_material": blade_material,
        "blade_type": blade_type,
        "cutting_angle_deg": cutting_angle,
        "blade_thickness_mm": blade_thickness,
        "cutting_speed_m_per_min": cutting_speed,
        "applied_force_N": applied_force,
        "operating_temperature_C": operating_temp,
        "friction_coefficient": friction,
        "lubrication": lubrication,
        
        # Output targets (3 only!)
        "blade_lifespan_hrs": blade_lifespan,
        "wear_estimation_pct": wear_estimation,
        "cutting_efficiency_pct": cutting_efficiency
    }
    # Merge base with derived for easier offline analysis
    base.update(derived)
    return base

def generate_expanded_dataset(total_samples=50000):
    """
    Generate expanded dataset with all material combinations.
    
    Returns:
        DataFrame with ~50,000 samples across 144 combinations
    """
    print("=" * 70)
    print("BLADE OPTIMIZER - EXPANDED DATASET GENERATION")
    print("=" * 70)
    print(f"\n📊 Configuration:")
    print(f"   Materials to Cut: {len(MATERIALS_TO_CUT)}")
    print(f"   Blade Materials: {len(BLADE_MATERIALS)}")
    print(f"   Blade Types: {len(BLADE_TYPES)}")
    print(f"   Total Combinations: {len(MATERIALS_TO_CUT) * len(BLADE_MATERIALS) * len(BLADE_TYPES)}")
    print(f"   Target Samples: {total_samples:,}")
    
    # Calculate samples per combination
    num_combinations = len(MATERIALS_TO_CUT) * len(BLADE_MATERIALS) * len(BLADE_TYPES)
    samples_per_combo = total_samples // num_combinations
    print(f"   Samples per Combination: ~{samples_per_combo}")
    
    print(f"\n🔧 Generating samples...")
    
    samples = []
    
    for material in MATERIALS_TO_CUT:
        for blade_material in BLADE_MATERIALS:
            for blade_type in BLADE_TYPES:
                # Generate samples for this combination
                for _ in range(samples_per_combo):
                    sample = generate_sample(material, blade_material, blade_type)
                    samples.append(sample)
                
                # Progress indicator
                combo_count = len(samples) // samples_per_combo
                if combo_count % 12 == 0:  # Update every 12 combinations
                    print(f"   Progress: {combo_count}/{num_combinations} combinations ({len(samples):,} samples)")
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    print(f"\n✅ Generation Complete!")
    print(f"   Total Samples: {len(df):,}")
    print(f"   Total Combinations: {df.groupby(['material_to_cut', 'blade_material', 'blade_type']).ngroups}")
    
    return df

def validate_dataset(df):
    """Validate dataset quality and physics compliance."""
    print(f"\n🔍 Validating Dataset Quality...")
    
    issues = []
    
    # Check 1: No missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        issues.append(f"❌ Found {missing} missing values")
    else:
        print(f"   ✅ No missing values")
    
    # Check 2: All combinations present
    expected_combos = len(MATERIALS_TO_CUT) * len(BLADE_MATERIALS) * len(BLADE_TYPES)
    actual_combos = df.groupby(['material_to_cut', 'blade_material', 'blade_type']).ngroups
    if actual_combos == expected_combos:
        print(f"   ✅ All {expected_combos} combinations present")
    else:
        issues.append(f"❌ Expected {expected_combos} combinations, found {actual_combos}")
    
    # Check 3: Parameter ranges
    checks = [
        ("cutting_angle_deg", 15, 45),
        ("blade_thickness_mm", 0.5, 5.0),
        ("cutting_speed_m_per_min", 20, 400),
        ("applied_force_N", 100, 2000),
        ("operating_temperature_C", 50, 700),
        ("friction_coefficient", 0.1, 0.8),
        ("blade_lifespan_hrs", 0.2, 15),
        ("wear_estimation_pct", 0, 100),
        ("cutting_efficiency_pct", 20, 100)
    ]
    
    for col, min_val, max_val in checks:
        if df[col].min() < min_val or df[col].max() > max_val:
            issues.append(f"❌ {col} out of range [{min_val}, {max_val}]: [{df[col].min():.2f}, {df[col].max():.2f}]")
        else:
            print(f"   ✅ {col} in valid range")
    
    # Check 4: Data balance
    combo_counts = df.groupby(['material_to_cut', 'blade_material', 'blade_type']).size()
    balance_ratio = combo_counts.min() / combo_counts.max()
    if balance_ratio > 0.9:
        print(f"   ✅ Data well-balanced (ratio: {balance_ratio:.2f})")
    else:
        issues.append(f"⚠️  Data imbalance detected (ratio: {balance_ratio:.2f})")
    
    # Check 5: Physics relationships
    # Higher wear should correlate with lower lifespan
    corr = df['blade_lifespan_hrs'].corr(df['wear_estimation_pct'])
    if corr < -0.3:
        print(f"   ✅ Physics: Lifespan vs Wear correlation = {corr:.3f} (negative as expected)")
    else:
        issues.append(f"⚠️  Physics: Unexpected lifespan-wear correlation: {corr:.3f}")
    
    if issues:
        print(f"\n⚠️  Validation Issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ All validation checks passed!")
    
    return len(issues) == 0

def save_dataset(df, filename="blade_dataset_expanded.csv"):
    """Save dataset to CSV file."""
    output_path = Path(__file__).parent / filename
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return output_path

def print_dataset_summary(df):
    """Print comprehensive dataset summary."""
    print(f"\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Basic Statistics:")
    print(f"   Total Samples: {len(df):,}")
    print(f"   Features: {len(df.columns) - 3} (inputs)")
    print(f"   Targets: 3 (outputs)")
    
    print(f"\n🔧 Materials to Cut ({len(df['material_to_cut'].unique())}):")
    for mat in sorted(df['material_to_cut'].unique()):
        count = len(df[df['material_to_cut'] == mat])
        print(f"   • {mat}: {count:,} samples")
    
    print(f"\n🔨 Blade Materials ({len(df['blade_material'].unique())}):")
    for blade in sorted(df['blade_material'].unique()):
        count = len(df[df['blade_material'] == blade])
        print(f"   • {blade}: {count:,} samples")
    
    print(f"\n⚙️  Blade Types ({len(df['blade_type'].unique())}):")
    for btype in sorted(df['blade_type'].unique()):
        count = len(df[df['blade_type'] == btype])
        print(f"   • {btype}: {count:,} samples")
    
    print(f"\n📈 Output Statistics:")
    print(f"   Blade Lifespan:")
    print(f"      Mean: {df['blade_lifespan_hrs'].mean():.2f} hrs")
    print(f"      Std:  {df['blade_lifespan_hrs'].std():.2f} hrs")
    print(f"      Range: [{df['blade_lifespan_hrs'].min():.2f}, {df['blade_lifespan_hrs'].max():.2f}]")
    
    print(f"   Wear Estimation:")
    print(f"      Mean: {df['wear_estimation_pct'].mean():.2f}%")
    print(f"      Std:  {df['wear_estimation_pct'].std():.2f}%")
    print(f"      Range: [{df['wear_estimation_pct'].min():.2f}, {df['wear_estimation_pct'].max():.2f}]")
    
    print(f"   Cutting Efficiency:")
    print(f"      Mean: {df['cutting_efficiency_pct'].mean():.2f}%")
    print(f"      Std:  {df['cutting_efficiency_pct'].std():.2f}%")
    print(f"      Range: [{df['cutting_efficiency_pct'].min():.2f}, {df['cutting_efficiency_pct'].max():.2f}]")
    
    print(f"\n" + "=" * 70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting dataset generation...\n")
    
    # Generate dataset
    df = generate_expanded_dataset(total_samples=50000)
    
    # Validate
    is_valid = validate_dataset(df)
    
    # Print summary
    print_dataset_summary(df)
    
    # Save
    if is_valid:
        output_path = save_dataset(df)
        print(f"\n✅ SUCCESS! Dataset ready for model training.")
    else:
        print(f"\n⚠️  Dataset has validation issues. Review before training.")
        # Save anyway for inspection
        output_path = save_dataset(df)
    
    print(f"\n🎓 Dataset generation complete for thesis project!")
    print(f"   Next step: Train model with this data")
    print("=" * 70)
