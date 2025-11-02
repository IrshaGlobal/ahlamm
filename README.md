# 🔧 Ahlamm - Blade Performance Predictor# 🔧 Ahlamm - Blade Performance PredictorAhlamm is a self-contained, academically rigorous Master’s thesis project in Mechanical Engineering, built with a clear, reproducible pipeline that blends classical machining theory with modern machine learning—all in Python.



**Master's Thesis Project - Mechanical Engineering**



Physics-informed deep learning for cutting blade optimization using ensemble neural networks and REST API.**Master's Thesis Project - Mechanical Engineering**  ## 📝 Project Methodology & Pipeline



[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)Physics-informed deep learning for cutting blade optimization using ensemble neural networks.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)The development follows four main stages:



---## 🎯 Overview



## 🎯 Overview**1. Physics-Based Synthetic Data Generation**  



Ahlamm predicts three critical blade performance metrics through a modern web application with REST API:This application predicts three critical blade performance metrics:No public industrial dataset exists with full records of blade geometry, material pairs, operational conditions, and wear outcomes, so the project generates its own dataset from first principles. Using **Taylor’s Tool Life Equation** ($T = C \cdot V^{-n}$) and material-specific constants sourced from the **ASM Handbook Vol. 16**, a Python script (`data/generate_data.py`) creates over **8,000 realistic synthetic samples**. Each sample includes inputs like workpiece material (e.g., steel, aluminum), blade material (HSS or carbide), cutting angle, speed, force, temperature, and lubrication status, along with derived outputs: **blade lifespan (hours)**, **wear estimation (%)**, **cutting efficiency (%)**, and a composite **performance score (0–100)**. Small random noise (±15%) is added to mimic real-world variability, ensuring the data is diverse yet physically plausible.



1. **Blade Lifespan** (hours of operation)1. **Blade Lifespan** (hours of operation)

2. **Wear Estimation** (percentage)  

3. **Cutting Efficiency** (percentage)2. **Wear Estimation** (percentage)**2. Lightweight Deep Learning Model Training**  



**Features:**3. **Cutting Efficiency** (percentage)This synthetic dataset is used to train a **multi-output, 3-layer feedforward neural network** (MLP) using **TensorFlow/Keras**. The model learns to map the nine input parameters to the four performance metrics simultaneously. Categorical inputs (like material types) are encoded via one-hot encoding, and numerical features are standardized using scikit-learn. The model is validated using standard regression metrics—**MAE and R²**—and consistently achieves **R² > 0.93** across all outputs, confirming it has accurately captured the underlying physics. The trained model and preprocessing pipeline are saved for deployment.

- ✅ **Modern Web UI**: Bootstrap 5.3 with responsive design and number inputs

- ✅ **REST API**: Full FastAPI backend with OpenAPI documentation

- ✅ **5-Model Ensemble**: Average predictions from 5 neural networks (R² > 0.94)

- ✅ **168 Material Combinations**: 6 workpiece × 7 blade materials × 4 blade typesUses a 5-model ensemble with composite weighted loss function optimized for wear prediction accuracy.**3. Interactive Web Interface with Streamlit**  

- ✅ **Physics-Informed**: Based on Taylor's Tool Life Equation & ASM Handbook

- ✅ **Production Ready**: Docker support, comprehensive documentationThe core model is wrapped in a user-friendly web application built with **Streamlit**, requiring no frontend expertise. The UI (`app/app.py`) presents intuitive sliders and dropdowns for all input parameters. When the user clicks “Predict Performance,” the app preprocesses the inputs, runs inference with the saved model, and displays results in real time: key metrics as cards, plain-English **optimization recommendations** (e.g., “Reduce speed by 15%” if wear is high), and a **2D parametric blade cross-section** rendered with **Plotly**, where high-wear zones are highlighted in red.



---## 🚀 Quick Start



## 🚀 Quick Start**4. Reproducible Deployment and Academic Packaging**  



### Prerequisites### PrerequisitesThe entire project is structured for full reproducibility: code, synthetic data, trained model, and dependencies are organized in a clean directory (`data/`, `model/`, `app/`) and shared openly on GitHub under an MIT license. The app is deployed for free on **Streamlit Cloud**, providing a public URL for thesis evaluators to interact with the tool directly. All components—data generation, modeling, and visualization—are grounded in established engineering knowledge, ensuring the project is not a black-box AI demo, but a **transparent, interpretable, and defensible** contribution to engineering design methodology.

- Python 3.11+

- pip- Python 3.11+



### Installation- pipIn essence, Ahlamm demonstrates how **classical mechanical principles** can be productized through **modern computational tools** to create a practical, accessible advisor for early-stage blade design—without relying on proprietary data or opaque algorithms.



```bash

# Clone repository

git clone https://github.com/IrshaGlobal/ahlamm.git### Installation# 🔧 Deep Learning App for Optimizing Cutting Blade Design

cd ahlamm



# Install dependencies

pip install -r requirements.txt```bash[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/IrshaGlobal/ahlamm-app.streamlit.app)

```

# Clone repository

### Run Application

git clone https://github.com/IrshaGlobal/ahlamm.gitA physics-informed deep learning application for predicting cutting blade performance—developed as a Master’s thesis project in Mechanical Engineering. This tool enables rapid, data-driven exploration of blade design and operational parameters to estimate **lifespan, wear, efficiency, and overall performance**.

```bash

# Start FastAPI servercd ahlamm

python server.py

```> **No real-world industrial datasets were used**. Instead, a **synthetic dataset grounded in classical machining theory** (Taylor’s Tool Life Equation, ASM Handbook) ensures scientific rigor and reproducibility.



Access at:# Install dependencies

- **Web UI**: http://localhost:8000

- **API Docs**: http://localhost:8000/api/docspip install -r requirements.txt---

- **API**: http://localhost:8000/api/predict

```

---

## 🎯 Project Overview

## 📁 Project Structure

### Run Application

```

ahlamm/This web application allows engineers and researchers to:

├── api/

│   └── main.py                      # FastAPI backend```bash- Input key blade and cutting parameters

├── frontend/

│   └── index.html                   # Modern web UIstreamlit run app/app.py- Receive instant predictions for:

├── server.py                        # Combined server

├── model/```  - **Blade Lifespan** (hours)

│   ├── blade_model_seed*.h5         # 5 ensemble models

│   ├── preprocessor.pkl             # Feature preprocessor  - **Wear Estimation** (%)

│   ├── feature_engineering.py       # Derived features

│   └── train_model_advanced.py      # Training scriptAccess at: `http://localhost:8501`  - **Cutting Efficiency** (%)

├── data/

│   ├── blade_dataset_expanded.csv   # Training data (10K samples)  - **Performance Score** (0–100)

│   └── generate_expanded_data.py    # Data generation

├── Dockerfile                       # Docker configuration## 📁 Project Structure- Get **actionable optimization recommendations**

├── requirements.txt                 # Python dependencies

├── DEPLOYMENT_FASTAPI.md           # Deployment guide- Visualize a **2D schematic of the blade** with wear hotspots

└── README.md                        # This file

``````



---ahlamm/The core model is a **multi-output neural network** trained on **8,000+ synthetic samples** generated from first principles—making this a **transparent, interpretable, and academically defensible** tool for early-stage design.



## 🎓 Methodology & Pipeline├── app/



### 1. Physics-Based Synthetic Data Generation│   └── app.py                      # Streamlit web interface---



Using **Taylor's Tool Life Equation** and **ASM Handbook** material constants:├── data/



```│   ├── blade_dataset_expanded.csv  # Training dataset (10,000 samples)## 🧪 Key Features

T = C · V^(-n)

```│   └── generate_expanded_data.py   # Data generation script



Generated 10,000 samples with:├── model/- ✅ **Physics-Informed Synthetic Data**: No black-box data; all samples derived from Taylor’s equation and material-specific constants (ASM Vol. 16).

- **Inputs**: Material pairs, geometry, operating conditions (10 features)

- **Outputs**: Lifespan, wear, efficiency│   ├── blade_model_seed42.h5       # Ensemble model 1- ✅ **Lightweight Deep Learning Model**: 3-layer MLP trained on tabular data (no CNNs/GNNs needed).

- **Variability**: ±15% noise for realism

│   ├── blade_model_seed1337.h5     # Ensemble model 2- ✅ **Interactive Web Interface**: Built with Streamlit—no frontend skills required.

### 2. Deep Learning Model Training

│   ├── blade_model_seed2025.h5     # Ensemble model 3- ✅ **2D Blade Visualization**: Parametric cross-section with wear-zone highlighting (using Plotly).

**Architecture:**

- Multi-output feedforward neural network│   ├── blade_model_seed7.h5        # Ensemble model 4- ✅ **Rule-Based Recommendations**: Interpretable advice (e.g., “Reduce speed by 15%”).

- 3 dense layers with BatchNormalization

- Composite weighted loss (1.0, 1.8, 1.0) optimized for wear│   ├── blade_model_seed101.h5      # Ensemble model 5- ✅ **Fully Reproducible**: Code, data, and model included.

- 5-model ensemble (seeds: 42, 1337, 2025, 7, 101)

│   ├── preprocessor.pkl            # Feature scaler- ✅ **Free & Open-Source**: MIT Licensed.

**Performance:**

- Lifespan R²: 0.958│   ├── train_model_advanced.py     # Training script

- Wear R²: 0.903

- Efficiency R²: 0.972│   └── feature_engineering.py      # Feature extraction---

- **Overall R²: 0.944**

├── requirements.txt                # Python dependencies

### 3. Feature Engineering

├── runtime.txt                     # Python version specification## 🛠️ Technology Stack

**10 Input Features:**

- material_to_cut, blade_material, blade_type└── README.md                       # This file

- cutting_angle_deg, blade_thickness_mm

- cutting_speed_m_per_min, applied_force_N```| Component | Technology |

- operating_temperature_C, friction_coefficient, lubrication

|---------|------------|

**10 Derived Features** (auto-calculated):

- speed_ratio, thermal_load_ratio, hardness_ratio## 🧠 Model Architecture| **Core Modeling** | Python, TensorFlow/Keras, scikit-learn |

- force_speed_ratio, geometry_cos, specific_energy_proxy

- blade_type_efficiency_factor, blade_material_wear_resistance| **Data Generation** | pandas, numpy |

- blade_hardness, material_hardness_avg

### Ensemble Configuration| **User Interface** | Streamlit |

**Total: 21 features** after preprocessing

- **5 models** with different random seeds (42, 1337, 2025, 7, 101)| **Visualization** | Plotly |

### 4. Web Application & REST API

- **Ensemble averaging** for robust predictions| **Deployment** | Streamlit Cloud (free) |

**Frontend:**

- Bootstrap 5.3 responsive design- **Best validation loss**: 23.719 (Seed 1337)| **Environment** | Python 3.9+ |

- Number inputs for precision

- Real-time friction coefficient calculation

- Plotly 2D blade visualization

- Material-specific optimization tips### Training Details---



**Backend:**- **Loss Function**: Composite weighted MSE + Huber loss

- FastAPI with async support

- Pydantic validation  - Lifespan: MSE (weight 1.0)## 🚀 Quick Start

- OpenAPI/Swagger documentation

- CORS enabled for integration  - Wear: Huber loss (weight 1.8) - robust to outliers

- Comprehensive error handling

  - Efficiency: MSE (weight 1.0)### Prerequisites

---

- **Architecture**: Deep neural network with BatchNormalization, L2 regularization- Python 3.9 or higher

## 🧪 API Usage

- **Training Data**: 10,000 samples with physics-informed constraints- Git (for cloning)

### Prediction Endpoint

- **Validation**: 80/20 train-test split

**POST** `/api/predict`

### Local Setup

```json

{### Model Performance1. Clone the repository:

  "workpiece_material": "Aluminum",

  "blade_material": "Carbide",| Seed | Best Val Loss | Rank |   ```bash

  "blade_type": "Circular Blade",

  "thickness": 2.5,|------|---------------|------|   git clone https://github.com/IrshaGlobal/ahlamm.git

  "cutting_angle": 30,

  "cutting_speed": 100,| 1337 | 23.719 | 🥇 |   cd blade-optimizer

  "applied_force": 800,

  "operating_temperature": 300,| 101  | 23.742 | 🥈 |

  "lubrication": true| 42   | 23.933 | 🥉 |

}| 7    | 24.006 | ✅ |

```| 2025 | 24.007 | ✅ |



**Response:**## 📊 Features

```json

{### Input Parameters

  "blade_lifespan": 12.38,1. **Blade Thickness** (0.1 - 5.0 mm)

  "wear_estimation": 2.91,2. **Cutting Angle** (10 - 60 degrees)

  "cutting_efficiency": 95.49,3. **Material Hardness** (100 - 700 HV)

  "performance_score": 97.48,4. **Cutting Speed** (50 - 500 m/min)

  "friction_coefficient": 0.18,5. **Feed Rate** (0.05 - 1.0 mm/rev)

  "optimization_tips": [6. **Workpiece Material** (Steel, Aluminum, Cast Iron, etc.)

    "✅ Optimal Operating Conditions",7. **Lubrication** (Yes/No)

    "• Current parameters are well-balanced",

    "🌟 Excellent performance expected!",### Output Predictions

    "**Aluminum Tips:** Higher speeds acceptable...",- **Blade Lifespan**: Hours of operation before replacement

    "**Circular Blade:** Continuous cutting improves efficiency..."- **Wear Estimation**: Percentage of blade degradation

  ]- **Cutting Efficiency**: Performance effectiveness percentage

}- **Performance Score**: Composite metric (0-100)

```

### Visualizations

---- 2D blade cross-section with wear zones

- Color-coded performance indicators

## 🐳 Docker Deployment- Rule-based optimization recommendations



```bash## 🔬 Technical Stack

# Build image

docker build -t ahlamm-app .- **Backend**: TensorFlow/Keras 2.19.1

- **Frontend**: Streamlit 1.38.0

# Run container- **Visualization**: Plotly 5.23.0

docker run -p 8000:8000 ahlamm-app- **Data Processing**: Pandas, NumPy, scikit-learn

```- **Model Format**: HDF5 (.h5)



---## 🎓 Research Context



## 🌐 Deployment OptionsThis project implements physics-informed deep learning for industrial blade optimization:

- Addresses real-world manufacturing constraints

See [DEPLOYMENT_FASTAPI.md](DEPLOYMENT_FASTAPI.md) for detailed guides:- Balances multiple performance objectives

- Provides actionable optimization recommendations

1. **Hugging Face Spaces** (Recommended, Free)- Supports thesis defense with interactive demonstrations

2. **Railway.app** ($5/month)

3. **Render.com** ($7/month)## 📝 Training Your Own Models

4. **Google Cloud Run** (Pay-per-use)

5. **Docker** (Self-hosted)To retrain with custom data:



---```bash

# Generate expanded dataset (optional)

## 📊 Model Capabilitiespython data/generate_expanded_data.py



**Supported Parameters:**# Train ensemble models

- ✅ 6 workpiece materials (Steel, Stainless, Aluminum, Cast Iron, Brass, Titanium)python model/train_model_advanced.py

- ✅ 7 blade materials (HSS, Carbide, Coated Carbide, Ceramic, CBN, PCD)```

- ✅ 4 blade types (Straight, Circular, Insert/Replaceable, Toothed)

- ✅ Geometric optimization (thickness, angle)Training configuration in `model/train_model_advanced.py`:

- ✅ Operating conditions (speed, force, temperature)- `ENSEMBLE_SEEDS = [42, 1337, 2025, 7, 101]`

- ✅ Thermal management- `LOSS_WEIGHTS = (1.0, 1.8, 1.0)` for (lifespan, wear, efficiency)

- ✅ Lubrication effects- `MAX_EPOCHS = 250` with early stopping (patience=30)



**Analysis Capabilities:**## 🌐 Deployment

- Material compatibility assessment

- Speed optimization for material### Streamlit Cloud

- Thermal load management```bash

- Hardness matching# Ensure runtime.txt and requirements.txt are present

- Wear prediction# Connect GitHub repo to Streamlit Cloud

- Efficiency optimization```

- Trade-off analysis

### Docker

---```dockerfile

FROM python:3.11-slim

## 📈 Training DataWORKDIR /app

COPY requirements.txt .

**Dataset Characteristics:**RUN pip install --no-cache-dir -r requirements.txt

- **Size**: 10,000 samplesCOPY . .

- **Combinations**: 168 material pairsEXPOSE 8501

- **Features**: 10 input + 10 derived = 21 totalCMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

- **Outputs**: 3 metrics```

- **Source**: Physics-based synthesis (Taylor's equation + ASM constants)

### Alternative Platforms

**Material Hardness Values (HV):**- **Hugging Face Spaces** (recommended for ML apps)

- Steel: 30, Stainless: 35, Aluminum: 20- **Railway.app** (easy deployment)

- Cast Iron: 40, Brass: 15, Titanium: 36- **Render.com** (free tier available)



**Blade Material Properties:**## 📄 License

- HSS: 850 HV, 600°C max, 1.0× wear resistance

- Carbide: 1750 HV, 1000°C max, 3.0× wear resistanceSee LICENSE file for details.

- CBN: 4500 HV, 1400°C max, 12.0× wear resistance

- PCD: 9000 HV, 700°C max, 15.0× wear resistance## 👨‍🎓 Author



---Master's Thesis Project - Mechanical Engineering  

Year: 2025

## 🔬 Academic Contribution

## 🙏 Acknowledgments

This project demonstrates:

- Thesis supervisor and committee

1. **Transparent AI**: No black-box models; all data derived from established engineering principles- Department of Mechanical Engineering

2. **Reproducibility**: Complete pipeline from data generation to deployment- Open-source ML community (TensorFlow, Streamlit, Plotly)

3. **Practical Tool**: Accessible web interface for engineers

4. **Modern Stack**: REST API enables integration with other systems---

5. **Full-Stack Skills**: Backend (Python/FastAPI) + Frontend (HTML/CSS/JS) + ML (TensorFlow)

**Status**: Production Ready ✅  

---**Last Updated**: November 2025  

**Version**: 2.0 (5-Model Ensemble)

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@mastersthesis{faci2025blade,
  title={Physics-Informed Deep Learning for Cutting Blade Optimization},
  author={Ahlam Faci},
  year={2025},
  type={Master's Thesis},
  department={Mechanical Engineering}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is an academic project. For questions or suggestions:

1. Open an issue on GitHub
2. Contact: [your-email@university.edu]

---

## 🔗 Links

- **GitHub**: https://github.com/IrshaGlobal/ahlamm
- **API Documentation**: http://localhost:8000/api/docs (when running)
- **Deployment Guide**: [DEPLOYMENT_FASTAPI.md](DEPLOYMENT_FASTAPI.md)

---

**Built with** ❤️ **for advancing mechanical engineering through modern AI/ML techniques**
