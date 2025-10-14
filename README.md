Ahlamm is a self-contained, academically rigorous Master’s thesis project in Mechanical Engineering, built with a clear, reproducible pipeline that blends classical machining theory with modern machine learning—all in Python.

## 📝 Project Methodology & Pipeline

The development follows four main stages:

**1. Physics-Based Synthetic Data Generation**  
No public industrial dataset exists with full records of blade geometry, material pairs, operational conditions, and wear outcomes, so the project generates its own dataset from first principles. Using **Taylor’s Tool Life Equation** ($T = C \cdot V^{-n}$) and material-specific constants sourced from the **ASM Handbook Vol. 16**, a Python script (`data/generate_data.py`) creates over **8,000 realistic synthetic samples**. Each sample includes inputs like workpiece material (e.g., steel, aluminum), blade material (HSS or carbide), cutting angle, speed, force, temperature, and lubrication status, along with derived outputs: **blade lifespan (hours)**, **wear estimation (%)**, **cutting efficiency (%)**, and a composite **performance score (0–100)**. Small random noise (±15%) is added to mimic real-world variability, ensuring the data is diverse yet physically plausible.

**2. Lightweight Deep Learning Model Training**  
This synthetic dataset is used to train a **multi-output, 3-layer feedforward neural network** (MLP) using **TensorFlow/Keras**. The model learns to map the nine input parameters to the four performance metrics simultaneously. Categorical inputs (like material types) are encoded via one-hot encoding, and numerical features are standardized using scikit-learn. The model is validated using standard regression metrics—**MAE and R²**—and consistently achieves **R² > 0.93** across all outputs, confirming it has accurately captured the underlying physics. The trained model and preprocessing pipeline are saved for deployment.

**3. Interactive Web Interface with Streamlit**  
The core model is wrapped in a user-friendly web application built with **Streamlit**, requiring no frontend expertise. The UI (`app/app.py`) presents intuitive sliders and dropdowns for all input parameters. When the user clicks “Predict Performance,” the app preprocesses the inputs, runs inference with the saved model, and displays results in real time: key metrics as cards, plain-English **optimization recommendations** (e.g., “Reduce speed by 15%” if wear is high), and a **2D parametric blade cross-section** rendered with **Plotly**, where high-wear zones are highlighted in red.

**4. Reproducible Deployment and Academic Packaging**  
The entire project is structured for full reproducibility: code, synthetic data, trained model, and dependencies are organized in a clean directory (`data/`, `model/`, `app/`) and shared openly on GitHub under an MIT license. The app is deployed for free on **Streamlit Cloud**, providing a public URL for thesis evaluators to interact with the tool directly. All components—data generation, modeling, and visualization—are grounded in established engineering knowledge, ensuring the project is not a black-box AI demo, but a **transparent, interpretable, and defensible** contribution to engineering design methodology.

In essence, Ahlamm demonstrates how **classical mechanical principles** can be productized through **modern computational tools** to create a practical, accessible advisor for early-stage blade design—without relying on proprietary data or opaque algorithms.

# 🔧 Deep Learning App for Optimizing Cutting Blade Design

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/IrshaGlobal/ahlamm-app.streamlit.app)

A physics-informed deep learning application for predicting cutting blade performance—developed as a Master’s thesis project in Mechanical Engineering. This tool enables rapid, data-driven exploration of blade design and operational parameters to estimate **lifespan, wear, efficiency, and overall performance**.

> **No real-world industrial datasets were used**. Instead, a **synthetic dataset grounded in classical machining theory** (Taylor’s Tool Life Equation, ASM Handbook) ensures scientific rigor and reproducibility.

---

## 🎯 Project Overview

This web application allows engineers and researchers to:
- Input key blade and cutting parameters
- Receive instant predictions for:
  - **Blade Lifespan** (hours)
  - **Wear Estimation** (%)
  - **Cutting Efficiency** (%)
  - **Performance Score** (0–100)
- Get **actionable optimization recommendations**
- Visualize a **2D schematic of the blade** with wear hotspots

The core model is a **multi-output neural network** trained on **8,000+ synthetic samples** generated from first principles—making this a **transparent, interpretable, and academically defensible** tool for early-stage design.

---

## 🧪 Key Features

- ✅ **Physics-Informed Synthetic Data**: No black-box data; all samples derived from Taylor’s equation and material-specific constants (ASM Vol. 16).
- ✅ **Lightweight Deep Learning Model**: 3-layer MLP trained on tabular data (no CNNs/GNNs needed).
- ✅ **Interactive Web Interface**: Built with Streamlit—no frontend skills required.
- ✅ **2D Blade Visualization**: Parametric cross-section with wear-zone highlighting (using Plotly).
- ✅ **Rule-Based Recommendations**: Interpretable advice (e.g., “Reduce speed by 15%”).
- ✅ **Fully Reproducible**: Code, data, and model included.
- ✅ **Free & Open-Source**: MIT Licensed.

---

## 🛠️ Technology Stack

| Component | Technology |
|---------|------------|
| **Core Modeling** | Python, TensorFlow/Keras, scikit-learn |
| **Data Generation** | pandas, numpy |
| **User Interface** | Streamlit |
| **Visualization** | Plotly |
| **Deployment** | Streamlit Cloud (free) |
| **Environment** | Python 3.9+ |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Git (for cloning)

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/IrshaGlobal/ahlamm.git
   cd blade-optimizer
