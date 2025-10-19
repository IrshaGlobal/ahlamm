---
title: Blade Optimizer
emoji: 🔪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app/app.py
pinned: false
license: mit
python_version: 3.11
---

# Blade Cutting Optimizer

An AI-powered tool for optimizing industrial blade cutting operations using deep learning.

## Features

- **Multi-task Neural Network**: Predicts blade lifespan, wear rate, and cutting efficiency simultaneously
- **Physics-Informed Model**: Trained on synthetic data with realistic physics-based efficiency calculations
- **Derived Feature Engineering**: Automatically computes 10 derived features for improved predictions
- **Interactive Streamlit Interface**: Easy-to-use web interface for blade optimization

## Model Performance

- **Blade Lifespan R²**: 0.957
- **Wear Rate R²**: 0.904
- **Cutting Efficiency R²**: 0.970
- **Overall R²**: 0.943

## Input Parameters

- **Material**: Wood, Plastic, Mild Steel, Stainless Steel, Aluminum, Copper
- **Blade Material**: HSS, Carbide, Diamond, Ceramic, Tool Steel, Cobalt Steel, Tungsten Carbide
- **Blade Type**: Straight, Serrated, Circular, Band
- **Cutting Speed**: 10-500 m/min
- **Cutting Angle**: 15-75 degrees
- **Applied Force**: 50-500 N
- **Blade Temperature**: 20-200°C
- **Lubrication**: Yes/No
- **Blade Hardness**: 40-95 HRC
- **Friction Coefficient**: 0.1-0.8

## Technology Stack

- **TensorFlow 2.15.0**: Deep learning framework
- **scikit-learn 1.5.1**: Feature engineering and preprocessing
- **Streamlit 1.38.0**: Web interface
- **Plotly**: Interactive visualizations

## Usage

The app provides predictions for:
1. **Predicted Blade Lifespan** (hours)
2. **Predicted Wear Rate** (mm/hr)
3. **Predicted Cutting Efficiency** (%)

Simply adjust the input parameters using the sidebar controls and get instant predictions!
