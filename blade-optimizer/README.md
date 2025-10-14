# Blade Machining Optimizer

A machine learning-based tool for optimizing blade machining parameters. This project combines classical machining theory with modern neural networks to predict machining outcomes and recommend optimal cutting parameters.

## 🎯 Project Overview

This project was developed as part of a Master's thesis in Mechanical Engineering, demonstrating how classical machining theory can be combined with modern machine learning to create practical engineering tools—without relying on proprietary or unavailable real-world data.

### Key Features

- **Physics-Based Data Generation**: Synthetic dataset generated using established machining equations (Taylor's tool life, Merchant's force model, etc.)
- **Deep Learning Model**: Multi-layer perceptron (MLP) neural network for predicting 5 machining outcomes
- **Interactive Web Application**: Streamlit-based interface for real-time predictions and optimization
- **Comprehensive Metrics**: Predicts tool life, cutting force, surface roughness, power consumption, and material removal rate

## 📁 Project Structure

```
blade-optimizer/
├── data/
│   ├── generate_data.py        # Physics-based synthetic data generator
│   └── blade_dataset.csv       # Generated dataset (8k+ samples)
├── model/
│   ├── train_model.py          # MLP training & validation script
│   ├── blade_model.h5          # Trained Keras model
│   └── preprocessor.pkl        # Saved sklearn preprocessor
├── app/
│   └── app.py                  # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/IrshaGlobal/ahlamm.git
cd ahlamm/blade-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Generate Dataset

Generate the physics-based synthetic dataset:

```bash
cd data
python generate_data.py
```

This will create `blade_dataset.csv` with 8,000+ samples based on classical machining theory.

#### Step 2: Train the Model

Train the neural network model:

```bash
cd ../model
python train_model.py
```

This will:
- Train an MLP model on the generated data
- Save the trained model as `blade_model.h5`
- Save the preprocessor as `preprocessor.pkl`
- Generate training visualizations and metrics

#### Step 3: Run the Web Application

Launch the interactive Streamlit application:

```bash
cd ../app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔬 Technical Details

### Input Parameters

The model takes 5 input parameters:

1. **Cutting Speed** (m/min): Linear velocity at the tool-workpiece interface
2. **Feed Rate** (mm/rev): Distance the tool advances per revolution
3. **Depth of Cut** (mm): Depth of material removed in a single pass
4. **Tool Nose Radius** (mm): Radius of the tool nose/tip
5. **Material Hardness** (HB): Brinell hardness of the workpiece material

### Output Predictions

The model predicts 5 machining outcomes:

1. **Tool Life** (min): Expected operational time before tool replacement
2. **Cutting Force** (N): Primary force acting on the cutting tool
3. **Surface Roughness** (μm): Average surface finish quality (Ra)
4. **Power Consumption** (kW): Estimated electrical power required
5. **Material Removal Rate** (mm³/min): Productivity metric

### Machine Learning Model

- **Architecture**: Multi-Layer Perceptron (MLP)
- **Layers**: 
  - Input: 5 neurons
  - Hidden 1: 128 neurons + ReLU + BatchNorm + Dropout(0.2)
  - Hidden 2: 64 neurons + ReLU + BatchNorm + Dropout(0.2)
  - Hidden 3: 32 neurons + ReLU
  - Output: 5 neurons (linear)
- **Training**: 
  - Loss: Mean Squared Error (MSE)
  - Optimizer: Adam (lr=0.001)
  - Early stopping with patience=15
  - Learning rate reduction on plateau
- **Data Split**: 70% training, 15% validation, 15% test

### Physics-Based Data Generation

The synthetic data is generated using established machining equations:

1. **Taylor's Tool Life Equation**: 
   ```
   T = C / (v^n × f^m × d^p)
   ```

2. **Merchant's Cutting Force Model**: 
   ```
   F = K × a × f × hardness_factor
   ```

3. **Surface Roughness**: 
   ```
   Ra ≈ f² / (32 × r)
   ```

4. **Power Consumption**: 
   ```
   P = F × v / 60000
   ```

5. **Material Removal Rate**: 
   ```
   MRR = v × f × d
   ```

## 📊 Model Performance

The trained model achieves strong performance across all output metrics:

- **R² Score**: > 0.95 for all outputs
- **Training Time**: ~2-5 minutes on CPU
- **Inference Time**: < 100ms per prediction

## 🎨 Web Application Features

- **Interactive Parameter Adjustment**: Sliders for all 5 input parameters
- **Real-time Predictions**: Instant results with visual gauges
- **Smart Recommendations**: Actionable insights based on predictions
- **Visual Metrics**: Plotly-based gauge charts and metrics display
- **Summary Tables**: Clear presentation of inputs and outputs

## 🛠️ Technology Stack

- **Machine Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Language**: Python 3.8+

## 📝 Use Cases

This tool is valuable for:

- **Manufacturing Engineers**: Optimize cutting parameters for specific requirements
- **Production Planning**: Predict tool life and power consumption
- **Education**: Demonstrate the relationship between cutting parameters and outcomes
- **Research**: Baseline for comparing with real experimental data
- **Cost Optimization**: Balance productivity (MRR) with tool life and surface quality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 👤 Author

**IrshaGlobal**

- GitHub: [@IrshaGlobal](https://github.com/IrshaGlobal)

## 🙏 Acknowledgments

- Classical machining theory references from standard mechanical engineering textbooks
- Inspired by the need for practical ML tools in manufacturing
- Built as part of a Master's thesis in Mechanical Engineering

## 📚 References

1. Taylor, F.W. (1907). "On the Art of Cutting Metals"
2. Merchant, M.E. (1945). "Mechanics of the Metal Cutting Process"
3. Modern machining process optimization literature

---

**Note**: This project uses synthetic data based on physics equations. For production use, consider validating against real experimental data from your specific machining setup.
