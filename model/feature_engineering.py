"""Feature engineering utilities for the Blade Optimizer project.

Defines a scikit-learn compatible transformer that adds derived features
needed to improve efficiency accuracy. This module must be importable at
inference time so that joblib can unpickle the preprocessor pipeline.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Keep these in sync with training and generator assumptions
DERIVED_FEATURES: List[str] = [
    "speed_ratio",
    "thermal_load_ratio",
    "hardness_ratio",
    "force_speed_ratio",
    "geometry_cos",
    "specific_energy_proxy",
    "blade_type_efficiency_factor",
    "blade_material_wear_resistance",
    "blade_hardness",
    "material_hardness_avg",
]


class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds domain-derived features to the input DataFrame.

    Expected base columns present in X:
    - material_to_cut, blade_material, blade_type, lubrication (categorical)
    - cutting_angle_deg, blade_thickness_mm, cutting_speed_m_per_min,
      applied_force_N, operating_temperature_C, friction_coefficient (numerical)
    """

    # Hardcoded reference tables (duplicated minimal constants to avoid import cycles)
    _MATERIAL_HARDNESS = {
        "Steel": (20, 40),
        "Stainless Steel": (25, 45),
        "Aluminum": (10, 30),
        "Cast Iron": (20, 60),
        "Brass": (10, 20),
        "Titanium": (30, 42),
    }

    _BLADE_PERFORMANCE = {
        "HSS": {"hardness": 850, "max_temp": 600, "wear_resistance": 1.0},
        "Carbide": {"hardness": 1750, "max_temp": 1000, "wear_resistance": 3.0},
        "Coated Carbide (TiN)": {"hardness": 2250, "max_temp": 1000, "wear_resistance": 4.5},
        "Coated Carbide (TiAlN)": {"hardness": 2750, "max_temp": 1100, "wear_resistance": 5.5},
        "Ceramic": {"hardness": 2250, "max_temp": 1200, "wear_resistance": 4.0},
        "CBN": {"hardness": 4500, "max_temp": 1400, "wear_resistance": 12.0},
        "PCD": {"hardness": 9000, "max_temp": 700, "wear_resistance": 15.0},
    }

    _BLADE_TYPE_PROPERTIES = {
        "Straight Blade": {"efficiency_factor": 1.0},
        "Circular Blade": {"efficiency_factor": 1.2},
        "Insert/Replaceable Tip Blade": {"efficiency_factor": 1.1},
        "Toothed Blade": {"efficiency_factor": 0.95},
    }

    _CUTTING_SPEED_RANGES = {
        "Steel": (80, 150),
        "Stainless Steel": (50, 100),
        "Aluminum": (150, 300),
        "Cast Iron": (80, 180),
        "Brass": (100, 250),
        "Titanium": (30, 80),
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Assume columns order consistent with training; convert to DataFrame
            df = pd.DataFrame(X)

        # Map optimal speed and hardness values row-wise
        mat = df["material_to_cut"].astype(str)
        blade_mat = df["blade_material"].astype(str)
        blade_type = df["blade_type"].astype(str)

        optimal_speed = mat.map(lambda m: np.mean(self._CUTTING_SPEED_RANGES.get(m, (100, 200))))
        mat_hard_avg = mat.map(lambda m: float(np.mean(self._MATERIAL_HARDNESS.get(m, (20, 40)))))
        blade_hard = blade_mat.map(lambda b: float(self._BLADE_PERFORMANCE.get(b, {"hardness": 1000})["hardness"]))
        blade_max_temp = blade_mat.map(lambda b: float(self._BLADE_PERFORMANCE.get(b, {"max_temp": 800})["max_temp"]))
        blade_wear_res = blade_mat.map(lambda b: float(self._BLADE_PERFORMANCE.get(b, {"wear_resistance": 1.0})["wear_resistance"]))
        type_eff = blade_type.map(lambda t: float(self._BLADE_TYPE_PROPERTIES.get(t, {"efficiency_factor": 1.0})["efficiency_factor"]))

        speed = df["cutting_speed_m_per_min"].astype(float)
        temp = df["operating_temperature_C"].astype(float)
        force = df["applied_force_N"].astype(float)
        angle = df["cutting_angle_deg"].astype(float)
        friction = df["friction_coefficient"].astype(float)

        # Derived features
        df["speed_ratio"] = speed / (optimal_speed + 1e-6)
        df["thermal_load_ratio"] = temp / (blade_max_temp + 1e-6)
        df["hardness_ratio"] = blade_hard / (mat_hard_avg + 1e-6)
        df["force_speed_ratio"] = force / (speed + 1e-6)
        df["geometry_cos"] = np.cos(np.deg2rad(angle))
        df["specific_energy_proxy"] = force * friction / (speed + 1e-6)
        df["blade_type_efficiency_factor"] = type_eff
        df["blade_material_wear_resistance"] = blade_wear_res
        df["blade_hardness"] = blade_hard
        df["material_hardness_avg"] = mat_hard_avg

        return df
