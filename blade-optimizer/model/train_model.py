"""
MLP (Multi-Layer Perceptron) training script for blade machining optimization.

This script trains a neural network to predict machining outcomes based on
cutting parameters, enabling optimization of the machining process.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict


def load_data(filepath: str) -> pd.DataFrame:
    """Load the blade dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Prepare input features and output targets.
    
    Inputs: cutting_speed, feed_rate, depth_of_cut, tool_nose_radius, material_hardness
    Outputs: tool_life, cutting_force, surface_roughness, power_consumption, material_removal_rate
    """
    input_features = [
        'cutting_speed', 'feed_rate', 'depth_of_cut', 
        'tool_nose_radius', 'material_hardness'
    ]
    
    output_features = [
        'tool_life', 'cutting_force', 'surface_roughness', 
        'power_consumption', 'material_removal_rate'
    ]
    
    X = df[input_features].values
    y = df[output_features].values
    
    print(f"\nInput features ({X.shape[1]}): {input_features}")
    print(f"Output features ({y.shape[1]}): {output_features}")
    
    return X, y, input_features, output_features


def build_mlp_model(input_dim: int, output_dim: int) -> keras.Model:
    """
    Build a Multi-Layer Perceptron model.
    
    Architecture:
    - Input layer: input_dim neurons
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Hidden layer 3: 32 neurons, ReLU activation
    - Output layer: output_dim neurons, linear activation
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.Dense(output_dim, activation='linear', name='output')
    ], name='blade_mlp_model')
    
    return model


def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100) -> keras.callbacks.History:
    """Train the MLP model."""
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                   output_features: list) -> Dict[str, float]:
    """Evaluate model performance."""
    print("\nEvaluating model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each output
    metrics = {}
    print("\nPer-output metrics:")
    print("-" * 70)
    
    for i, feature in enumerate(output_features):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        metrics[f'{feature}_mae'] = mae
        metrics[f'{feature}_rmse'] = rmse
        metrics[f'{feature}_r2'] = r2
        
        print(f"{feature:30s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | R²: {r2:6.4f}")
    
    # Overall metrics
    overall_mae = mean_absolute_error(y_test, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics['overall_mae'] = overall_mae
    metrics['overall_rmse'] = overall_rmse
    
    print("-" * 70)
    print(f"{'Overall':30s} | MAE: {overall_mae:8.4f} | RMSE: {overall_rmse:8.4f}")
    
    return metrics


def plot_training_history(history: keras.callbacks.History, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to: {save_path}")
    
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Blade Machining Optimization - MLP Training")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    df = load_data('../data/blade_dataset.csv')
    
    # Prepare features
    X, y, input_features, output_features = prepare_features(df)
    
    # Split data
    print("\nSplitting data (70% train, 15% validation, 15% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    print("\nBuilding MLP model...")
    model = build_mlp_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    model.summary()
    
    # Train model
    history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=100)
    
    # Plot training history
    plot_training_history(history, save_path='training_history.png')
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, output_features)
    
    # Save model and preprocessor
    print("\nSaving model and preprocessor...")
    model.save('blade_model.keras')
    print("Model saved to: blade_model.keras")
    
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Preprocessor saved to: preprocessor.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to: model_metrics.csv")
    
    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
