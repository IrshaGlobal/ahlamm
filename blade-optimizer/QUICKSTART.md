# Quick Start Guide

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/IrshaGlobal/ahlamm.git
cd ahlamm/blade-optimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Use Pre-trained Model (Recommended)

The model and data are already included in the repository, so you can directly run the application:

```bash
cd app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Option 2: Regenerate Everything from Scratch

If you want to regenerate the data and retrain the model:

**Step 1: Generate Data**
```bash
cd data
python generate_data.py
```
This creates `blade_dataset.csv` with 8,000+ samples based on physics equations.

**Step 2: Train Model**
```bash
cd ../model
python train_model.py
```
This trains the neural network and saves:
- `blade_model.keras` - trained model
- `preprocessor.pkl` - data preprocessor
- `model_metrics.csv` - performance metrics
- `training_history.png` - training visualization

**Step 3: Run Application**
```bash
cd ../app
streamlit run app.py
```

## Using the Application

1. **Adjust Parameters**: Use the sliders in the sidebar to set:
   - Cutting Speed (50-300 m/min)
   - Feed Rate (0.1-0.5 mm/rev)
   - Depth of Cut (0.5-3.0 mm)
   - Tool Nose Radius (0.4-1.2 mm)
   - Material Hardness (150-300 HB)

2. **Get Predictions**: Click the "🔮 Predict Outcomes" button

3. **Review Results**: The app will display:
   - Tool Life (minutes)
   - Cutting Force (Newtons)
   - Surface Roughness (micrometers)
   - Power Consumption (kilowatts)
   - Material Removal Rate (mm³/min)
   - Visual gauge charts
   - Smart recommendations

## Example Use Cases

### High Productivity Scenario
- Cutting Speed: 250 m/min
- Feed Rate: 0.4 mm/rev
- Depth of Cut: 2.5 mm
→ Results in high material removal rate but lower tool life

### High Quality Scenario
- Cutting Speed: 100 m/min
- Feed Rate: 0.15 mm/rev
- Depth of Cut: 1.0 mm
→ Results in excellent surface finish and longer tool life

### Balanced Scenario
- Cutting Speed: 150 m/min
- Feed Rate: 0.25 mm/rev
- Depth of Cut: 1.5 mm
→ Good balance between productivity and quality

## Troubleshooting

### Model not found error
If you see "Model not found" error:
1. Make sure you're in the correct directory
2. Check that `model/blade_model.keras` exists
3. Try regenerating the model with `python model/train_model.py`

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port already in use
If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

## Technical Details

- **Dataset**: 8,000 physics-based synthetic samples
- **Model**: MLP with 128→64→32 neurons
- **Training Time**: ~2-5 minutes on CPU
- **Inference Time**: <100ms per prediction
- **Model Size**: ~185KB

## Support

For issues or questions, please open an issue on GitHub.
