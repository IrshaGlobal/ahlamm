"""
Train an ensemble of models with different random seeds and average their predictions.
Saves individual models as blade_model_seed{seed}.h5 and writes ensemble metrics.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers, optimizers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "blade_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"

TARGET_COLUMNS = [
    "blade_lifespan_hrs",
    "wear_estimation_pct",
    "cutting_efficiency_pct",
    "performance_score",
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

SEEDS = [42, 1337, 2025]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run data/generate_data.py first.")
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), FEATURE_COLUMNS_CAT),
            ("num", StandardScaler(), FEATURE_COLUMNS_NUM),
            ("bool", "passthrough", FEATURE_COLUMNS_BOOL),
        ]
    )


def build_model(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.0005)),
        Dropout(0.3),
        Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        Dropout(0.3),
        Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.0005)),
        Dense(len(TARGET_COLUMNS), name="outputs"),
    ])
    optimizer = optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    lines = ["Per-target evaluation (Test Set, Ensemble):"]
    for i, target in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        lines.append(f"  - {target}: MAE={mae:.4f}, R2={r2:.4f}")
    return "\n".join(lines)


def main():
    # Load data
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS_CAT + FEATURE_COLUMNS_NUM + FEATURE_COLUMNS_BOOL]
    y = df[TARGET_COLUMNS].values

    # Preprocess
    pre = build_preprocessor()
    Xp = pre.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=42)

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(pre, MODEL_DIR / "preprocessor.pkl")

    preds = []
    for seed in SEEDS:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model = build_model(Xp.shape[1])
        model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=160,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=24, restore_best_weights=True)
            ],
        )
        # Save model per seed
        model.save(MODEL_DIR / f"blade_model_seed{seed}.h5")
        preds.append(model.predict(X_test, verbose=0))

    # Average predictions
    y_pred = np.mean(preds, axis=0)

    report = evaluate(y_test, y_pred)
    print(report)

    with open(MODEL_DIR / "metrics_report_ensemble.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
