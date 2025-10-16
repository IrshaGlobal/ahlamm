#!/usr/bin/env python3
"""
Test script to verify the input shape mismatch issue is resolved.
Run this script to test the prediction workflow.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path

def test_prediction():
    """Test the prediction workflow exactly like the Streamlit app."""
    
    PROJECT_ROOT = Path.cwd()
    MODEL_PATH = PROJECT_ROOT / "model" / "blade_model.h5"
    PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"
    ENSEMBLE_SEEDS = [42, 1337, 2025]
    
    print("🔧 Testing Blade Performance Prediction...")
    
    # Load preprocessor
    print("📦 Loading preprocessor...")
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    print(f"   Expected input features: {list(preprocessor.feature_names_in_)}")
    
    # Load models
    ensemble_paths = [PROJECT_ROOT / "model" / f"blade_model_seed{seed}.h5" for seed in ENSEMBLE_SEEDS]
    ensemble_available = all(p.exists() for p in ensemble_paths)
    
    models = []
    if ensemble_available:
        print("🎯 Loading ensemble models...")
        for p in ensemble_paths:
            m = load_model(str(p), compile=False)
            m.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models.append(m)
            print(f"   ✓ {p.name}: input_shape = {m.input_shape}")
    else:
        print("🎯 Loading single model...")
        m = load_model(str(MODEL_PATH), compile=False)
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
        models = [m]
        print(f"   ✓ {MODEL_PATH.name}: input_shape = {m.input_shape}")
    
    # Test different input scenarios
    test_cases = [
        {
            "name": "Steel + HSS",
            "material_to_cut": "Steel",
            "blade_material": "HSS",
            "cutting_angle_deg": 15,
            "blade_thickness_mm": 6.0,
            "cutting_speed_m_per_min": 100,
            "applied_force_N": 800,
            "operating_temperature_C": 300,
            "friction_coefficient": 0.5,
            "lubrication": True
        },
        {
            "name": "Titanium + Carbide",
            "material_to_cut": "Titanium",
            "blade_material": "Carbide",
            "cutting_angle_deg": 12,
            "blade_thickness_mm": 4.5,
            "cutting_speed_m_per_min": 60,
            "applied_force_N": 1200,
            "operating_temperature_C": 400,
            "friction_coefficient": 0.65,
            "lubrication": True
        },
        {
            "name": "Aluminum + Coated_Carbide",
            "material_to_cut": "Aluminum",
            "blade_material": "Coated_Carbide",
            "cutting_angle_deg": 20,
            "blade_thickness_mm": 8.0,
            "cutting_speed_m_per_min": 150,
            "applied_force_N": 600,
            "operating_temperature_C": 200,
            "friction_coefficient": 0.3,
            "lubrication": False
        }
    ]
    
    print(f"\n🧪 Running {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        # Create input DataFrame
        input_data = pd.DataFrame([test_case])
        print(f"   Input shape: {input_data.shape}")
        print(f"   Input columns: {list(input_data.columns)}")
        
        try:
            # Preprocess
            X_processed = preprocessor.transform(input_data)
            print(f"   Processed shape: {X_processed.shape}")
            
            # Predict
            if len(models) > 1:
                # Ensemble prediction
                preds = [m.predict(X_processed, verbose=0)[0] for m in models]
                predictions = np.mean(preds, axis=0)
                print(f"   Ensemble prediction (avg of {len(models)} models): ✓")
            else:
                # Single model prediction
                predictions = models[0].predict(X_processed, verbose=0)[0]
                print(f"   Single model prediction: ✓")
            
            lifespan, wear, efficiency, performance = predictions
            print(f"   Results:")
            print(f"     - Blade Lifespan: {lifespan:.2f} hours")
            print(f"     - Wear Estimation: {wear:.1f}%")
            print(f"     - Cutting Efficiency: {efficiency:.1f}%")
            print(f"     - Performance Score: {performance:.1f}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            return False
    
    print(f"\n✅ ALL TESTS PASSED!")
    print(f"   The input shape mismatch issue has been resolved.")
    print(f"   Your Streamlit app should now work correctly.")
    return True

if __name__ == "__main__":
    success = test_prediction()
    exit(0 if success else 1)