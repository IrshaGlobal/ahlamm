"""
Physics-based synthetic data generator for blade machining optimization.

This module generates synthetic data based on classical machining theory,
simulating the relationship between cutting parameters and machining outcomes.

Key physics principles:
- Taylor's tool life equation
- Merchant's cutting force model
- Surface roughness correlations
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Constants based on machining theory
TAYLOR_EXPONENT = 0.2  # Typical value for high-speed steel
MERCHANT_CONSTANT = 0.7  # Shear angle constant
ROUGHNESS_FACTOR = 0.8  # Surface roughness scaling


def calculate_tool_life(cutting_speed: float, feed_rate: float, depth_of_cut: float) -> float:
    """
    Calculate tool life using Taylor's tool life equation.
    
    T = C / (v^n * f^m * d^p)
    where T is tool life, v is cutting speed, f is feed rate, d is depth of cut
    """
    C = 500  # Taylor constant (minutes)
    n = 0.25  # Speed exponent
    m = 0.5   # Feed exponent
    p = 0.15  # Depth exponent
    
    tool_life = C / (cutting_speed**n * feed_rate**m * depth_of_cut**p)
    return max(1, tool_life)  # Minimum 1 minute


def calculate_cutting_force(cutting_speed: float, feed_rate: float, 
                            depth_of_cut: float, material_hardness: float) -> float:
    """
    Calculate cutting force based on Merchant's model.
    
    F = K * a * f * hardness_factor
    where K is specific cutting force constant
    """
    K = 2000  # Specific cutting force (N/mm²)
    area = feed_rate * depth_of_cut  # Chip cross-section
    hardness_factor = material_hardness / 200  # Normalized hardness
    
    force = K * area * hardness_factor * (1 + 0.5 / cutting_speed)
    return force


def calculate_surface_roughness(cutting_speed: float, feed_rate: float, 
                                tool_nose_radius: float) -> float:
    """
    Calculate theoretical surface roughness (Ra).
    
    Ra ≈ f² / (32 * r)
    where f is feed rate and r is tool nose radius
    """
    theoretical_ra = (feed_rate**2) / (32 * tool_nose_radius)
    
    # Add speed-dependent factor
    speed_factor = 1 - 0.3 * np.tanh((cutting_speed - 100) / 50)
    
    ra = theoretical_ra * speed_factor * ROUGHNESS_FACTOR
    return ra


def calculate_power_consumption(cutting_force: float, cutting_speed: float) -> float:
    """
    Calculate power consumption.
    
    P = F * v / 60000
    where P is in kW, F is in N, v is in m/min
    """
    power = (cutting_force * cutting_speed) / 60000
    return power


def calculate_material_removal_rate(cutting_speed: float, feed_rate: float, 
                                    depth_of_cut: float) -> float:
    """
    Calculate material removal rate (MRR).
    
    MRR = v * f * d
    """
    mrr = cutting_speed * feed_rate * depth_of_cut
    return mrr


def generate_sample(idx: int, noise_level: float = 0.05) -> dict:
    """
    Generate a single sample with realistic parameter ranges.
    """
    # Input parameters with realistic ranges
    cutting_speed = np.random.uniform(50, 300)  # m/min
    feed_rate = np.random.uniform(0.1, 0.5)     # mm/rev
    depth_of_cut = np.random.uniform(0.5, 3.0)  # mm
    tool_nose_radius = np.random.uniform(0.4, 1.2)  # mm
    material_hardness = np.random.uniform(150, 300)  # HB (Brinell hardness)
    
    # Calculate physics-based outputs
    tool_life = calculate_tool_life(cutting_speed, feed_rate, depth_of_cut)
    cutting_force = calculate_cutting_force(cutting_speed, feed_rate, 
                                           depth_of_cut, material_hardness)
    surface_roughness = calculate_surface_roughness(cutting_speed, feed_rate, 
                                                   tool_nose_radius)
    power = calculate_power_consumption(cutting_force, cutting_speed)
    mrr = calculate_material_removal_rate(cutting_speed, feed_rate, depth_of_cut)
    
    # Add realistic noise
    noise = lambda x: x * (1 + np.random.normal(0, noise_level))
    
    return {
        'sample_id': idx,
        'cutting_speed': round(cutting_speed, 2),
        'feed_rate': round(feed_rate, 3),
        'depth_of_cut': round(depth_of_cut, 2),
        'tool_nose_radius': round(tool_nose_radius, 2),
        'material_hardness': round(material_hardness, 1),
        'tool_life': round(noise(tool_life), 2),
        'cutting_force': round(noise(cutting_force), 2),
        'surface_roughness': round(noise(surface_roughness), 4),
        'power_consumption': round(noise(power), 3),
        'material_removal_rate': round(noise(mrr), 2)
    }


def generate_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate complete dataset with n_samples.
    """
    np.random.seed(seed)
    
    samples = [generate_sample(i) for i in range(n_samples)]
    df = pd.DataFrame(samples)
    
    return df


def main():
    """
    Main function to generate and save the dataset.
    """
    print("Generating blade machining dataset...")
    print("Based on classical machining theory:")
    print("- Taylor's tool life equation")
    print("- Merchant's cutting force model")
    print("- Surface roughness correlations")
    print()
    
    # Generate dataset
    df = generate_dataset(n_samples=8000)
    
    # Display statistics
    print(f"Generated {len(df)} samples")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nStatistical summary:")
    print(df.describe())
    
    # Save to CSV
    output_path = 'blade_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
