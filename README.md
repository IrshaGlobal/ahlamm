# ahlamm

This project was developed as a Master's thesis in Mechanical Engineering. It demonstrates how classical machining theory can be combined with modern machine learning to create practical engineering tools—without relying on proprietary or unavailable real-world data.

## 🔧 Blade Machining Optimizer

A complete machine learning-based tool for optimizing blade machining parameters. The project combines physics-based data generation with deep learning to predict and optimize machining outcomes.

### Features

- **Physics-Based Data Generation**: 8,000+ synthetic samples based on classical machining equations
- **Deep Learning Model**: Multi-layer perceptron (MLP) for predicting 5 machining outcomes
- **Interactive Web App**: Streamlit-based interface for real-time predictions
- **Comprehensive Metrics**: Tool life, cutting force, surface roughness, power consumption, and material removal rate

### Quick Start

```bash
# Navigate to the project directory
cd blade-optimizer

# Install dependencies
pip install -r requirements.txt

# Generate dataset
cd data && python generate_data.py

# Train model
cd ../model && python train_model.py

# Run web application
cd ../app && streamlit run app.py
```

### Project Structure

```
blade-optimizer/
├── data/               # Data generation and dataset
├── model/              # ML model training and artifacts
├── app/                # Streamlit web application
└── README.md           # Detailed documentation
```

For detailed documentation, see [blade-optimizer/README.md](blade-optimizer/README.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
