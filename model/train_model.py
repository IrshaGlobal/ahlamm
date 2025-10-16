"""Train lightweight MLP model for blade performance prediction.

Master's Thesis Project - Mechanical Engineering
Physics-informed model for cutting blade optimization.

Predicts (3 outputs only):
  - blade_lifespan_hrs
  - wear_estimation_pct  
  - cutting_efficiency_pct

Architecture: 128 → 64 → 32 neurons (thesis-appropriate)
Performance score computed in application, not predicted.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Reproducibility seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "blade_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "blade_model.h5"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METRICS_PATH = MODEL_DIR / "metrics_report.txt"

METRICS_PATH = MODEL_DIR / "metrics_report.txt"

# Thesis-approved targets (3 outputs only)
TARGET_COLUMNS = [
    "blade_lifespan_hrs",
    "wear_estimation_pct",
    "cutting_efficiency_pct",
]

FEATURE_COLUMNS_CAT = ["material_to_cut", "blade_material"]
FEATURE_COLUMNS_BOOL = ["lubrication"]
FEATURE_COLUMNS_NUM = [
    "cutting_angle_deg",
    "blade_thickness_mm",
    "cutting_speed_m_per_min",
    "applied_force_N",
    "operating_temperature_C",
    "friction_coefficient",
]

def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate thesis-compliant dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run data/generate_data.py first.")
    df = pd.read_csv(path)
    missing_targets = [t for t in TARGET_COLUMNS if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Dataset missing target columns: {missing_targets}")
    return df

def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for input features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), FEATURE_COLUMNS_CAT),
            ("num", StandardScaler(), FEATURE_COLUMNS_NUM),
            ("bool", "passthrough", FEATURE_COLUMNS_BOOL),
        ]
    )

def build_model(input_dim: int) -> tf.keras.Model:
    """Build lightweight MLP architecture as per thesis specification."""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(len(TARGET_COLUMNS), name="outputs"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def evaluate(model: tf.keras.Model, X_test: np.ndarray, y_test: pd.DataFrame) -> str:
    """Evaluate model performance on test set."""
    y_pred = model.predict(X_test, verbose=0)
    lines = ["Thesis Model Evaluation (Test Set):"]
    for i, target in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        lines.append(f"  - {target}: MAE={mae:.4f}, R²={r2:.4f}")
    return "\n".join(lines)

def main():
    """Train thesis-compliant model with 3 outputs."""
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X = df[FEATURE_COLUMNS_CAT + FEATURE_COLUMNS_NUM + FEATURE_COLUMNS_BOOL]
    y = df[TARGET_COLUMNS]

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    print(f"Processed feature matrix shape: {X_processed.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    print("Building lightweight model (128→64→32)...")
    model = build_model(X_processed.shape[1])
    model.summary()

    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
        ],
    )

    print("Evaluating model...")
    metrics_report = evaluate(model, X_test, y_test)
    print(metrics_report)

    print("Saving artifacts...")
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_report + "\n")

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"✅ Preprocessor saved to {PREPROCESSOR_PATH}")
    print(f"✅ Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
