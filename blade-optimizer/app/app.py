"""
Streamlit Web Application for Blade Machining Optimization

This application provides an interactive interface to predict machining outcomes
and optimize cutting parameters using the trained machine learning model.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Blade Machining Optimizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    # Get the base path (blade-optimizer directory)
    app_dir = Path(__file__).resolve().parent
    base_path = app_dir.parent
    model_path = base_path / 'model' / 'blade_model.keras'
    preprocessor_path = base_path / 'model' / 'preprocessor.pkl'
    
    try:
        if not model_path.exists():
            return None, None, False
        model = keras.models.load_model(str(model_path))
        with open(str(preprocessor_path), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False


def predict_outcomes(model, scaler, cutting_speed, feed_rate, depth_of_cut, 
                     tool_nose_radius, material_hardness):
    """Make predictions using the trained model."""
    # Prepare input
    X = np.array([[cutting_speed, feed_rate, depth_of_cut, 
                   tool_nose_radius, material_hardness]])
    
    # Standardize
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled, verbose=0)
    
    return {
        'Tool Life (min)': float(predictions[0][0]),
        'Cutting Force (N)': float(predictions[0][1]),
        'Surface Roughness (μm)': float(predictions[0][2]),
        'Power Consumption (kW)': float(predictions[0][3]),
        'Material Removal Rate (mm³/min)': float(predictions[0][4])
    }


def create_gauge_chart(value, title, min_val, max_val, unit):
    """Create a gauge chart for a metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        number={'suffix': f" {unit}"},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.33 + min_val], 'color': "lightgreen"},
                {'range': [(max_val-min_val)*0.33 + min_val, (max_val-min_val)*0.67 + min_val], 'color': "yellow"},
                {'range': [(max_val-min_val)*0.67 + min_val, max_val], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    """Main application function."""
    
    # Header
    st.title("⚙️ Blade Machining Optimizer")
    st.markdown("""
    This application uses machine learning to predict machining outcomes and optimize cutting parameters.
    Based on classical machining theory combined with modern neural networks.
    """)
    
    # Load model
    model, scaler, loaded = load_model_and_preprocessor()
    
    if not loaded:
        st.warning("⚠️ Model not found. Please train the model first by running `python model/train_model.py`")
        st.info("""
        **To get started:**
        1. Generate data: `cd blade-optimizer/data && python generate_data.py`
        2. Train model: `cd blade-optimizer/model && python train_model.py`
        3. Restart this application
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar - Input Parameters
    st.sidebar.header("Cutting Parameters")
    st.sidebar.markdown("Adjust the parameters below to predict machining outcomes:")
    
    cutting_speed = st.sidebar.slider(
        "Cutting Speed (m/min)",
        min_value=50.0,
        max_value=300.0,
        value=150.0,
        step=5.0,
        help="Linear cutting velocity at the tool-workpiece interface"
    )
    
    feed_rate = st.sidebar.slider(
        "Feed Rate (mm/rev)",
        min_value=0.1,
        max_value=0.5,
        value=0.25,
        step=0.01,
        help="Distance the tool advances per revolution"
    )
    
    depth_of_cut = st.sidebar.slider(
        "Depth of Cut (mm)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Depth of material removed in single pass"
    )
    
    tool_nose_radius = st.sidebar.slider(
        "Tool Nose Radius (mm)",
        min_value=0.4,
        max_value=1.2,
        value=0.8,
        step=0.1,
        help="Radius of the tool nose/tip"
    )
    
    material_hardness = st.sidebar.slider(
        "Material Hardness (HB)",
        min_value=150.0,
        max_value=300.0,
        value=225.0,
        step=5.0,
        help="Brinell hardness of workpiece material"
    )
    
    # Predict button
    if st.sidebar.button("🔮 Predict Outcomes", type="primary"):
        with st.spinner("Calculating..."):
            predictions = predict_outcomes(
                model, scaler, cutting_speed, feed_rate, depth_of_cut,
                tool_nose_radius, material_hardness
            )
            
            # Store predictions in session state
            st.session_state['predictions'] = predictions
            st.session_state['inputs'] = {
                'Cutting Speed': cutting_speed,
                'Feed Rate': feed_rate,
                'Depth of Cut': depth_of_cut,
                'Tool Nose Radius': tool_nose_radius,
                'Material Hardness': material_hardness
            }
    
    # Display results
    if 'predictions' in st.session_state:
        st.header("📊 Predicted Outcomes")
        
        predictions = st.session_state['predictions']
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Tool Life",
                value=f"{predictions['Tool Life (min)']:.2f} min",
                help="Expected tool life before replacement"
            )
            st.metric(
                label="Surface Roughness",
                value=f"{predictions['Surface Roughness (μm)']:.4f} μm",
                help="Average surface roughness (Ra)"
            )
        
        with col2:
            st.metric(
                label="Cutting Force",
                value=f"{predictions['Cutting Force (N)']:.2f} N",
                help="Primary cutting force"
            )
            st.metric(
                label="Material Removal Rate",
                value=f"{predictions['Material Removal Rate (mm³/min)']:.2f} mm³/min",
                help="Volume of material removed per minute"
            )
        
        with col3:
            st.metric(
                label="Power Consumption",
                value=f"{predictions['Power Consumption (kW)']:.3f} kW",
                help="Estimated power consumption"
            )
        
        # Gauge charts
        st.header("📈 Visual Metrics")
        
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            fig1 = create_gauge_chart(
                predictions['Tool Life (min)'],
                "Tool Life",
                0, 500,
                "min"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with gauge_col2:
            fig2 = create_gauge_chart(
                predictions['Cutting Force (N)'],
                "Cutting Force",
                0, 3000,
                "N"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with gauge_col3:
            fig3 = create_gauge_chart(
                predictions['Power Consumption (kW)'],
                "Power",
                0, 15,
                "kW"
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Summary table
        st.header("📋 Summary")
        
        col_sum1, col_sum2 = st.columns(2)
        
        with col_sum1:
            st.subheader("Input Parameters")
            inputs_df = pd.DataFrame({
                'Parameter': list(st.session_state['inputs'].keys()),
                'Value': [f"{v:.2f}" for v in st.session_state['inputs'].values()]
            })
            st.dataframe(inputs_df, hide_index=True, use_container_width=True)
        
        with col_sum2:
            st.subheader("Predicted Outcomes")
            outputs_df = pd.DataFrame({
                'Metric': list(predictions.keys()),
                'Value': [f"{v:.4f}" for v in predictions.values()]
            })
            st.dataframe(outputs_df, hide_index=True, use_container_width=True)
        
        # Recommendations
        st.header("💡 Recommendations")
        
        recommendations = []
        
        if predictions['Tool Life (min)'] < 50:
            recommendations.append("⚠️ Tool life is low. Consider reducing cutting speed or feed rate.")
        elif predictions['Tool Life (min)'] > 200:
            recommendations.append("✅ Excellent tool life. Consider increasing MRR for better productivity.")
        
        if predictions['Surface Roughness (μm)'] > 3.0:
            recommendations.append("⚠️ Surface roughness is high. Consider increasing tool nose radius or reducing feed rate.")
        elif predictions['Surface Roughness (μm)'] < 1.0:
            recommendations.append("✅ Excellent surface finish achieved.")
        
        if predictions['Material Removal Rate (mm³/min)'] < 100:
            recommendations.append("⚠️ Low material removal rate. Consider increasing cutting parameters for better productivity.")
        elif predictions['Material Removal Rate (mm³/min)'] > 300:
            recommendations.append("✅ High material removal rate. Good productivity.")
        
        if predictions['Power Consumption (kW)'] > 10:
            recommendations.append("⚠️ High power consumption. Consider optimizing cutting parameters.")
        
        if not recommendations:
            recommendations.append("✅ All parameters are within optimal range.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    else:
        st.info("👈 Adjust the cutting parameters in the sidebar and click **Predict Outcomes** to see results.")
        
        # Display sample information
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### Features
        - **Real-time Prediction**: Get instant predictions for machining outcomes
        - **Interactive Visualization**: Visual gauges and metrics for easy interpretation
        - **Smart Recommendations**: Actionable insights based on predictions
        - **Physics-Based**: Built on classical machining theory
        
        ### Predicted Metrics
        1. **Tool Life**: Expected operational time before tool replacement
        2. **Cutting Force**: Primary force acting on the cutting tool
        3. **Surface Roughness**: Average surface finish quality (Ra)
        4. **Power Consumption**: Estimated electrical power required
        5. **Material Removal Rate**: Productivity metric (volume/time)
        
        ### Input Parameters
        - **Cutting Speed**: Linear velocity at cutting interface (m/min)
        - **Feed Rate**: Tool advance per revolution (mm/rev)
        - **Depth of Cut**: Material removal depth per pass (mm)
        - **Tool Nose Radius**: Tool tip radius (mm)
        - **Material Hardness**: Workpiece hardness (Brinell scale)
        
        ### Technology Stack
        - **Machine Learning**: TensorFlow/Keras MLP
        - **Frontend**: Streamlit
        - **Visualization**: Plotly
        - **Data Processing**: NumPy, Pandas, Scikit-learn
        """)
    
    # Footer
    st.divider()
    st.caption("🔧 Blade Machining Optimizer | Combining Classical Machining Theory with Modern Machine Learning")


if __name__ == "__main__":
    main()
