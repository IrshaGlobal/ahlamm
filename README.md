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
