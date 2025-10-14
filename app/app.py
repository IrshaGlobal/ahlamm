"""
Streamlit web application for blade performance prediction.

Features:
- Interactive input controls for blade parameters
- Real-time prediction of 4 performance metrics
- 2D blade visualization with wear zones
- Rule-based optimization recommendations
- Physics-informed synthetic data approach

Usage: streamlit run app.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ML components
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import joblib
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Ahlamm - Blade Performance Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "blade_model.h5"
ENSEMBLE_SEEDS = [42, 1337, 2025]
PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"

@st.cache_resource
def load_model_artifacts():
    """Load trained model or ensemble and preprocessor with caching."""
    if not PREPROCESSOR_PATH.exists():
        st.error(f"❌ Preprocessor not found at {PREPROCESSOR_PATH}")
        st.stop()

    preprocessor = joblib.load(str(PREPROCESSOR_PATH))

    # If ensemble models exist, load them; otherwise load single model
    ensemble_paths = [PROJECT_ROOT / "model" / f"blade_model_seed{seed}.h5" for seed in ENSEMBLE_SEEDS]
    ensemble_available = all(p.exists() for p in ensemble_paths)

    models = []
    try:
        if ensemble_available:
            for p in ensemble_paths:
                m = load_model(str(p), compile=False)
                m.compile(optimizer='adam', loss='mse', metrics=['mae'])
                models.append(m)
        else:
            if not MODEL_PATH.exists():
                st.error(f"❌ Model not found at {MODEL_PATH}")
                st.error("Please run: `python model/train_model.py` or `python model/train_ensemble.py`")
                st.stop()
            m = load_model(str(MODEL_PATH), compile=False)
            m.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models = [m]
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.error("Try retraining the model with current Keras version")
        st.stop()

    return models, preprocessor

def estimate_friction_coefficient(material: str, lubrication: bool) -> float:
    """Estimate friction coefficient based on material and lubrication."""
    base_friction = {
        "Steel": 0.7,
        "Aluminum": 0.5, 
        "Titanium": 0.8
    }
    
    base = base_friction.get(material, 0.6)
    # Lubrication reduces friction by ~40%
    return base * (0.6 if lubrication else 1.0)

def create_blade_visualization(
    thickness: float,
    cutting_angle: float,
    wear_pct: float,
    efficiency_pct: float
) -> go.Figure:
    """Create 2D blade cross-section visualization."""
    
    # Blade dimensions (simplified rectangular profile with angled tip)
    length = 10.0  # mm
    angle_rad = np.radians(cutting_angle)
    tip_offset = thickness / np.tan(angle_rad) if np.tan(angle_rad) > 0 else thickness
    
    # Blade outline coordinates
    x_coords = [0, length - tip_offset, length, length, 0, 0]
    y_coords = [0, 0, thickness/2, thickness, thickness, 0]
    
    fig = go.Figure()
    
    # Main blade body
    blade_color = "lightblue" if wear_pct < 50 else "orange" if wear_pct < 75 else "red"
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        fill="toself",
        fillcolor=blade_color,
        line=dict(color="black", width=2),
        name="Blade",
        hovertemplate="Blade Profile<br>Thickness: %{y:.1f}mm<extra></extra>"
    ))
    
    # Cutting edge highlight
    fig.add_trace(go.Scatter(
        x=[length - tip_offset, length],
        y=[0, thickness/2],
        mode="lines",
        line=dict(color="darkblue", width=4),
        name="Cutting Edge",
        hovertemplate="Cutting Edge<br>Angle: %{customdata}°<extra></extra>",
        customdata=[cutting_angle]
    ))
    
    # Wear zone indicator
    if wear_pct > 60:
        wear_zone_x = [length - 1.5, length - 0.5, length - 0.5, length - 1.5]
        wear_zone_y = [0, 0, thickness/2, thickness/4]
        
        fig.add_trace(go.Scatter(
            x=wear_zone_x,
            y=wear_zone_y,
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red", width=2, dash="dash"),
            name="High Wear Zone",
            hovertemplate="Wear Zone<br>Estimated Wear: %{customdata:.1f}%<extra></extra>",
            customdata=[wear_pct]
        ))
    
    # Efficiency indicator (arrow showing cutting direction)
    arrow_color = "green" if efficiency_pct > 70 else "orange" if efficiency_pct > 50 else "red"
    
    fig.add_annotation(
        x=length + 1,
        y=thickness/2,
        ax=length - 2,
        ay=thickness/2,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor=arrow_color
    )
    
    fig.add_annotation(
        x=length + 1.5,
        y=thickness/2,
        text=f"Efficiency: {efficiency_pct:.1f}%",
        showarrow=False,
        font=dict(size=12, color=arrow_color)
    )
    
    # Layout settings
    fig.update_layout(
        title="2D Blade Cross-Section with Performance Indicators",
        xaxis_title="Length (mm)",
        yaxis_title="Thickness (mm)",
        showlegend=True,
        height=400,
        xaxis=dict(range=[-0.5, length + 3]),
        yaxis=dict(range=[-0.5, thickness + 1], scaleanchor="x", scaleratio=1),
        template="plotly_white"
    )
    
    return fig

def generate_recommendations(
    lifespan: float,
    wear_pct: float, 
    efficiency_pct: float,
    performance_score: float,
    inputs: Dict
) -> str:
    """Generate rule-based optimization recommendations."""
    
    recommendations = []
    
    # High wear recommendations
    if wear_pct > 70:
        recommendations.append("🔴 **High Wear Detected**")
        recommendations.append("• Reduce cutting speed by 15-20%")
        recommendations.append("• Enable lubrication if not already used")
        if inputs["blade_material"] == "HSS" and inputs["material_to_cut"] in ["Steel", "Titanium"]:
            recommendations.append("• Consider switching to carbide blade for harder materials")
    
    # Low efficiency recommendations  
    elif efficiency_pct < 50:
        recommendations.append("🟡 **Low Cutting Efficiency**")
        recommendations.append("• Optimize cutting angle closer to 15°")
        recommendations.append("• Reduce applied force if possible")
        recommendations.append("• Check blade sharpness and condition")
    
    # Short lifespan recommendations
    elif lifespan < 1.0:
        recommendations.append("🟠 **Short Blade Lifespan**")
        recommendations.append("• Reduce cutting speed significantly")
        recommendations.append("• Lower operating temperature") 
        recommendations.append("• Ensure adequate lubrication")
    
    # Optimal conditions
    else:
        recommendations.append("✅ **Optimal Operating Conditions**")
        recommendations.append("• Current parameters are well-balanced")
        recommendations.append("• Monitor for wear and adjust as needed")
        recommendations.append("• Consider slight speed increase if productivity needed")
    
    # Material-specific advice
    if inputs["material_to_cut"] == "Titanium":
        recommendations.append("\n**Titanium-Specific Tips:**")
        recommendations.append("• Use flood coolant to manage heat")
        recommendations.append("• Maintain sharp cutting edges")
    elif inputs["material_to_cut"] == "Aluminum":
        recommendations.append("\n**Aluminum-Specific Tips:**") 
        recommendations.append("• Higher speeds generally acceptable")
        recommendations.append("• Watch for built-up edge formation")
    
    return "\n".join(recommendations)

def main():
    """Main Streamlit application."""
    
    # Load model artifacts
    model, preprocessor = load_model_artifacts()
    
    # App header
    st.title("🔧 Ahlamm - Blade Performance Predictor")
    st.markdown("""
    **Physics-informed deep learning for cutting blade optimization** | Master's Thesis Project
    
    Predict blade performance using synthetic data generated from Taylor's Tool Life Equation and ASM Handbook constants.
    """)
    
    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Blade & Cutting Parameters")
        
        # Material selection
        material_to_cut = st.selectbox(
            "Material to Cut",
            ["Steel", "Aluminum", "Titanium"],
            help="Workpiece material being machined"
        )
        
        blade_material = st.selectbox(
            "Blade Material", 
            ["HSS", "Carbide"],
            help="HSS = High-Speed Steel"
        )
        
        # Geometric parameters
        st.subheader("Geometry")
        cutting_angle = st.number_input(
            "Cutting Angle (°)", 
            min_value=5, max_value=25, value=15, step=1,
            help="Blade cutting angle in degrees (optimal: ~15°)"
        )
        
        blade_thickness = st.number_input(
            "Blade Thickness (mm)",
            min_value=2.0, max_value=10.0, value=6.0, step=0.1, format="%.1f",
            help="Blade thickness in millimeters (optimal: ~6mm)"
        )
        
        # Cutting conditions  
        st.subheader("Cutting Conditions")
        cutting_speed = st.number_input(
            "Cutting Speed (m/min)",
            min_value=20, max_value=200, value=100, step=5,
            help="Cutting speed in meters per minute (optimal: 80-120)"
        )
        
        applied_force = st.number_input(
            "Applied Force (N)",
            min_value=100, max_value=2000, value=800, step=50,
            help="Applied cutting force in Newtons"
        )
        
        operating_temp = st.number_input(
            "Operating Temperature (°C)",
            min_value=20, max_value=600, value=300, step=10,
            help="Operating temperature in Celsius"
        )
        
        lubrication = st.checkbox(
            "Lubrication Enabled",
            value=True,
            help="Whether lubrication/coolant is used"
        )
        
        # Friction coefficient control
        st.subheader("Friction Coefficient")
        use_auto_friction = st.checkbox(
            "Auto-calculate friction",
            value=True,
            help="Automatically estimate friction based on material and lubrication"
        )
        
        if use_auto_friction:
            friction_coeff = estimate_friction_coefficient(material_to_cut, lubrication)
            st.info(f"📊 Estimated friction: **{friction_coeff:.3f}**")
        else:
            friction_coeff = st.number_input(
                "Friction Coefficient",
                min_value=0.1, max_value=1.2, value=0.5, step=0.01, format="%.3f",
                help="Manual friction coefficient (0.1-1.2)"
            )
            st.warning("⚠️ Using manual friction value")
    
    # Prediction section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔍 Predict Performance", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame([{
                "material_to_cut": material_to_cut,
                "blade_material": blade_material, 
                "cutting_angle_deg": cutting_angle,
                "blade_thickness_mm": blade_thickness,
                "cutting_speed_m_per_min": cutting_speed,
                "applied_force_N": applied_force,
                "operating_temperature_C": operating_temp,
                "friction_coefficient": friction_coeff,
                "lubrication": lubrication
            }])
            
            try:
                # Preprocess and predict
                X_processed = preprocessor.transform(input_data)
                # Support ensemble: average predictions if multiple models are loaded
                if isinstance(model, list):
                    preds = [m.predict(X_processed, verbose=0)[0] for m in model]
                    predictions = np.mean(preds, axis=0)
                else:
                    predictions = model.predict(X_processed, verbose=0)[0]
                
                lifespan, wear, efficiency, performance = predictions
                
                # Store results in session state
                st.session_state.predictions = {
                    "lifespan": float(lifespan),
                    "wear": float(wear), 
                    "efficiency": float(efficiency),
                    "performance": float(performance),
                    "inputs": input_data.iloc[0].to_dict()
                }
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.markdown("### Quick Parameter Guide")
        st.markdown("""
        - **Steel**: General purpose, moderate speeds
        - **Aluminum**: Higher speeds OK, watch for buildup  
        - **Titanium**: Lower speeds, excellent lubrication needed
        - **HSS**: Cost-effective, moderate performance
        - **Carbide**: Higher performance, harder materials
        """)
    
    # Display results if available
    if hasattr(st.session_state, 'predictions'):
        pred = st.session_state.predictions
        
        st.markdown("---")
        st.subheader("📊 Predicted Performance Metrics")
        
        # Show input summary
        with st.expander("📝 Input Parameters Used", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Material to Cut:** {pred['inputs']['material_to_cut']}")
                st.write(f"**Blade Material:** {pred['inputs']['blade_material']}")
                st.write(f"**Cutting Angle:** {pred['inputs']['cutting_angle_deg']:.1f}°")
            with col2:
                st.write(f"**Thickness:** {pred['inputs']['blade_thickness_mm']:.1f} mm")
                st.write(f"**Speed:** {pred['inputs']['cutting_speed_m_per_min']:.0f} m/min")
                st.write(f"**Force:** {pred['inputs']['applied_force_N']:.0f} N")
            with col3:
                st.write(f"**Temperature:** {pred['inputs']['operating_temperature_C']:.0f}°C")
                st.write(f"**Friction:** {pred['inputs']['friction_coefficient']:.3f}")
                st.write(f"**Lubrication:** {'✓ Yes' if pred['inputs']['lubrication'] else '✗ No'}")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Blade Lifespan",
                f"{pred['lifespan']:.2f} hrs",
                delta="Good" if pred['lifespan'] > 2 else "Low" if pred['lifespan'] < 1 else "OK"
            )
        
        with col2:
            wear_delta = "High" if pred['wear'] > 70 else "Moderate" if pred['wear'] > 40 else "Low"
            st.metric(
                "Wear Estimation", 
                f"{pred['wear']:.1f}%",
                delta=wear_delta,
                delta_color="inverse"
            )
        
        with col3:
            eff_delta = "Good" if pred['efficiency'] > 70 else "Moderate" if pred['efficiency'] > 50 else "Low"
            st.metric(
                "Cutting Efficiency",
                f"{pred['efficiency']:.1f}%", 
                delta=eff_delta
            )
        
        with col4:
            score_delta = "Excellent" if pred['performance'] > 80 else "Good" if pred['performance'] > 60 else "Poor"
            st.metric(
                "Performance Score",
                f"{pred['performance']:.0f}/100",
                delta=score_delta
            )
        
        # Visualization and recommendations
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📐 Blade Visualization")
            fig = create_blade_visualization(
                blade_thickness, cutting_angle, 
                pred['wear'], pred['efficiency']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💡 Optimization Recommendations")
            recommendations = generate_recommendations(
                pred['lifespan'], pred['wear'], pred['efficiency'],
                pred['performance'], pred['inputs']
            )
            st.markdown(recommendations)
    
    # Footer with model accuracy info
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style='color: #666;'>
        <p><strong>Academic Project Notice</strong>: This tool uses physics-informed synthetic data for educational purposes. 
        Not intended for production use without validation against real-world testing.</p>
        <p>🎓 Master's Thesis Project | Mechanical Engineering | Physics-Informed Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Model Performance (R²)**")
        st.markdown("""
        - Lifespan: **0.92** ✓
        - Wear: **0.96** ✓
        - Efficiency: **0.69** ⚠️
        - Score: **0.96** ✓
        
        *R² > 0.90 = Excellent*
        """)
    
    st.info("💡 **Tip**: Results are most accurate when parameters are within normal operating ranges (Speed: 80-120 m/min, Angle: 12-18°, Temperature: 200-400°C)")

if __name__ == "__main__":
    main()