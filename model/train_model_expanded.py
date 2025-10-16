"""
Enhanced model training for expanded blade optimization dataset.
Mechanical Engineering Thesis Project.

Trains multi-task neural network for 3 outputs:
1. Blade Lifespan (hrs)
2. Wear Estimation (%)
3. Cutting Efficiency (%)

Target: R² ≥ 0.95 for ALL outputs
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "blade_dataset_expanded.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "blade_model.h5"
PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"

print("=" * 80)
print("BLADE OPTIMIZER - ENHANCED MODEL TRAINING")
print("=" * 80)
print(f"\n📁 Data source: {DATA_PATH}")
print(f"🎯 Target: R² ≥ 0.95 for all outputs")
print(f"🧠 Architecture: Multi-task neural network")

# ============================================================================
# DATA LOADING
# ============================================================================

print(f"\n📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Total samples: {len(df):,}")
print(f"   Features: {df.shape[1] - 3} (10 inputs)")
print(f"   Targets: 3 (lifespan, wear, efficiency)")

# Define features and targets
INPUT_FEATURES = [
    "material_to_cut",
    "blade_material",
    "blade_type",
    "cutting_angle_deg",
    "blade_thickness_mm",
    "cutting_speed_m_per_min",
    "applied_force_N",
    "operating_temperature_C",
    "friction_coefficient",
    "lubrication"
]

TARGET_COLUMNS = [
    "blade_lifespan_hrs",
    "wear_estimation_pct",
    "cutting_efficiency_pct"
]

X = df[INPUT_FEATURES].copy()
y = df[TARGET_COLUMNS].copy()

print(f"\n✅ Data loaded successfully")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# ============================================================================
# PREPROCESSING
# ============================================================================

print(f"\n🔧 Creating preprocessing pipeline...")

# Identify categorical and numerical columns
categorical_features = ["material_to_cut", "blade_material", "blade_type", "lubrication"]
numerical_features = [col for col in INPUT_FEATURES if col not in categorical_features]

print(f"   Categorical features: {len(categorical_features)}")
print(f"   Numerical features: {len(numerical_features)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# Split data
print(f"\n✂️  Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
)

print(f"   Train: {len(X_train):,} samples (80%)")
print(f"   Val:   {len(X_val):,} samples (10%)")
print(f"   Test:  {len(X_test):,} samples (10%)")

# Fit and transform
print(f"\n🔄 Fitting preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"   Processed feature count: {X_train_processed.shape[1]}")

# Save preprocessor
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"   💾 Preprocessor saved to: {PREPROCESSOR_PATH}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print(f"\n🏗️  Building enhanced neural network...")

def create_model(input_dim, learning_rate=0.001):
    """
    Create enhanced multi-task neural network with improved architecture.
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Enhanced shared feature extraction layers (deeper, more neurons)
    x = layers.Dense(512, activation='relu', name='shared_1')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu', name='shared_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(128, activation='relu', name='shared_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name='shared_4')(x)
    x = layers.BatchNormalization()(x)
    
    # Enhanced task-specific heads (larger heads for better accuracy)
    
    # Head 1: Blade Lifespan (always positive, 0.2-15 hrs)
    lifespan_head = layers.Dense(64, activation='relu', name='lifespan_hidden_1')(x)
    lifespan_head = layers.BatchNormalization()(lifespan_head)
    lifespan_head = layers.Dropout(0.15)(lifespan_head)
    lifespan_head = layers.Dense(32, activation='relu', name='lifespan_hidden_2')(lifespan_head)
    lifespan_output = layers.Dense(1, activation='relu', name='blade_lifespan_hrs')(lifespan_head)
    
    # Head 2: Wear Estimation (0-100%)
    wear_head = layers.Dense(64, activation='relu', name='wear_hidden_1')(x)
    wear_head = layers.BatchNormalization()(wear_head)
    wear_head = layers.Dropout(0.15)(wear_head)
    wear_head = layers.Dense(32, activation='relu', name='wear_hidden_2')(wear_head)
    wear_sigmoid = layers.Dense(1, activation='sigmoid')(wear_head)
    wear_output = layers.Multiply(name='wear_estimation_pct')([wear_sigmoid, tf.constant([100.0])])  # Scale to 0-100
    
    # Head 3: Cutting Efficiency (20-100%)
    efficiency_head = layers.Dense(64, activation='relu', name='efficiency_hidden_1')(x)
    efficiency_head = layers.BatchNormalization()(efficiency_head)
    efficiency_head = layers.Dropout(0.15)(efficiency_head)
    efficiency_head = layers.Dense(32, activation='relu', name='efficiency_hidden_2')(efficiency_head)
    efficiency_sigmoid = layers.Dense(1, activation='sigmoid')(efficiency_head)
    efficiency_scaled = layers.Multiply()([efficiency_sigmoid, tf.constant([80.0])])  # Scale 0-1 to 0-80
    efficiency_output = layers.Add(name='cutting_efficiency_pct')([efficiency_scaled, tf.constant([20.0])])  # Add 20 to get 20-100
    
    # Create model
    model = Model(
        inputs=inputs,
        outputs=[lifespan_output, wear_output, efficiency_output],
        name='blade_optimizer_model'
    )
    
    # Compile with task-specific losses and adjusted weights
    # Give more weight to efficiency since it was underperforming
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'blade_lifespan_hrs': 'mse',
            'wear_estimation_pct': 'mse',
            'cutting_efficiency_pct': 'mse'
        },
        loss_weights={
            'blade_lifespan_hrs': 1.0,
            'wear_estimation_pct': 1.2,  # Increased weight
            'cutting_efficiency_pct': 1.5  # Increased weight (was worst performer)
        },
        metrics={
            'blade_lifespan_hrs': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
            'wear_estimation_pct': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
            'cutting_efficiency_pct': ['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        }
    )
    
    return model

# Create model
input_dim = X_train_processed.shape[1]
model = create_model(input_dim, learning_rate=0.001)

print(f"\n📋 Model Architecture:")
model.summary()

# ============================================================================
# TRAINING
# ============================================================================

print(f"\n🚀 Training model...")

# Enhanced callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=30,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive LR reduction
        patience=12,
        min_lr=1e-7,
        verbose=1
    )
]

# Prepare targets
y_train_dict = {
    'blade_lifespan_hrs': y_train['blade_lifespan_hrs'].values,
    'wear_estimation_pct': y_train['wear_estimation_pct'].values,
    'cutting_efficiency_pct': y_train['cutting_efficiency_pct'].values
}

y_val_dict = {
    'blade_lifespan_hrs': y_val['blade_lifespan_hrs'].values,
    'wear_estimation_pct': y_val['wear_estimation_pct'].values,
    'cutting_efficiency_pct': y_val['cutting_efficiency_pct'].values
}

# Train with more epochs and smaller batch size for better convergence
history = model.fit(
    X_train_processed,
    y_train_dict,
    validation_data=(X_val_processed, y_val_dict),
    epochs=250,  # Increased from 150
    batch_size=32,  # Reduced from 64 for better gradient updates
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training complete!")

# ============================================================================
# EVALUATION
# ============================================================================

print(f"\n📊 Evaluating model on test set...")

y_test_dict = {
    'blade_lifespan_hrs': y_test['blade_lifespan_hrs'].values,
    'wear_estimation_pct': y_test['wear_estimation_pct'].values,
    'cutting_efficiency_pct': y_test['cutting_efficiency_pct'].values
}

# Predictions
predictions = model.predict(X_test_processed, verbose=0)
y_pred_lifespan = predictions[0].flatten()
y_pred_wear = predictions[1].flatten()
y_pred_efficiency = predictions[2].flatten()

# Calculate R² scores
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred, name):
    """Calculate and print metrics for a target."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"\n   {name}:")
    print(f"      R² Score: {r2:.4f} {'✅' if r2 >= 0.95 else '⚠️'}")
    print(f"      MAE:      {mae:.4f}")
    print(f"      RMSE:     {rmse:.4f}")
    print(f"      MAPE:     {mape:.2f}%")
    
    return r2, mae, rmse, mape

print(f"\n🎯 Test Set Performance:")
print("=" * 60)

r2_lifespan, mae_lifespan, rmse_lifespan, mape_lifespan = calculate_metrics(
    y_test['blade_lifespan_hrs'].values, y_pred_lifespan, "Blade Lifespan (hrs)"
)

r2_wear, mae_wear, rmse_wear, mape_wear = calculate_metrics(
    y_test['wear_estimation_pct'].values, y_pred_wear, "Wear Estimation (%)"
)

r2_efficiency, mae_efficiency, rmse_efficiency, mape_efficiency = calculate_metrics(
    y_test['cutting_efficiency_pct'].values, y_pred_efficiency, "Cutting Efficiency (%)"
)

print("=" * 60)

# Overall assessment
avg_r2 = (r2_lifespan + r2_wear + r2_efficiency) / 3
print(f"\n📈 Overall Performance:")
print(f"   Average R²: {avg_r2:.4f}")

if avg_r2 >= 0.95:
    print(f"   ✅ EXCELLENT! Target achieved (R² ≥ 0.95)")
elif avg_r2 >= 0.90:
    print(f"   ✅ GOOD! Close to target")
else:
    print(f"   ⚠️  Below target, consider retraining with more epochs or data")

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\n💾 Saving model...")
model.save(MODEL_PATH)
print(f"   Model saved to: {MODEL_PATH}")

# Save metrics report
metrics_report = f"""
BLADE OPTIMIZER - MODEL PERFORMANCE REPORT
Generated: {pd.Timestamp.now()}
==========================================

DATASET INFORMATION:
- Total Samples: {len(df):,}
- Training Samples: {len(X_train):,}
- Validation Samples: {len(X_val):,}
- Test Samples: {len(X_test):,}
- Input Features: {X_train_processed.shape[1]}
- Output Targets: 3

MODEL ARCHITECTURE:
- Type: Multi-task Neural Network
- Shared Layers: 3 (256→128→64 neurons)
- Task-Specific Heads: 3 (32 neurons each)
- Total Parameters: {model.count_params():,}
- Activation: ReLU (hidden), Custom (outputs)
- Optimizer: Adam
- Learning Rate: 0.001

TEST SET PERFORMANCE:
====================

1. Blade Lifespan (hrs)
   - R² Score: {r2_lifespan:.4f} {'✅' if r2_lifespan >= 0.95 else '⚠️'}
   - MAE: {mae_lifespan:.4f} hrs
   - RMSE: {rmse_lifespan:.4f} hrs
   - MAPE: {mape_lifespan:.2f}%

2. Wear Estimation (%)
   - R² Score: {r2_wear:.4f} {'✅' if r2_wear >= 0.95 else '⚠️'}
   - MAE: {mae_wear:.4f}%
   - RMSE: {rmse_wear:.4f}%
   - MAPE: {mape_wear:.2f}%

3. Cutting Efficiency (%)
   - R² Score: {r2_efficiency:.4f} {'✅' if r2_efficiency >= 0.95 else '⚠️'}
   - MAE: {mae_efficiency:.4f}%
   - RMSE: {rmse_efficiency:.4f}%
   - MAPE: {mape_efficiency:.2f}%

OVERALL:
- Average R²: {avg_r2:.4f}
- Status: {'✅ TARGET ACHIEVED' if avg_r2 >= 0.95 else '⚠️ BELOW TARGET'}

THESIS-READY: {'YES ✅' if avg_r2 >= 0.95 else 'NEEDS IMPROVEMENT ⚠️'}
"""

metrics_path = PROJECT_ROOT / "model" / "metrics_report_expanded.txt"
with open(metrics_path, 'w') as f:
    f.write(metrics_report)

print(f"   📄 Metrics report saved to: {metrics_path}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n📈 Creating training history plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Training History', fontsize=16, fontweight='bold')

# Plot 1: Total Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Lifespan MAE
axes[0, 1].plot(history.history['blade_lifespan_hrs_mae'], label='Train MAE', linewidth=2)
axes[0, 1].plot(history.history['val_blade_lifespan_hrs_mae'], label='Val MAE', linewidth=2)
axes[0, 1].set_title('Blade Lifespan - MAE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (hrs)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Wear MAE
axes[1, 0].plot(history.history['wear_estimation_pct_mae'], label='Train MAE', linewidth=2)
axes[1, 0].plot(history.history['val_wear_estimation_pct_mae'], label='Val MAE', linewidth=2)
axes[1, 0].set_title('Wear Estimation - MAE')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE (%)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Efficiency MAE
axes[1, 1].plot(history.history['cutting_efficiency_pct_mae'], label='Train MAE', linewidth=2)
axes[1, 1].plot(history.history['val_cutting_efficiency_pct_mae'], label='Val MAE', linewidth=2)
axes[1, 1].set_title('Cutting Efficiency - MAE')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('MAE (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = PROJECT_ROOT / "model" / "training_history_expanded.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"   📊 Training plots saved to: {plot_path}")

print(f"\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎓 Ready for thesis application!")
print(f"   Next step: Update app.py with new materials and blade types")
print("=" * 80)
