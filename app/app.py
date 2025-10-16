"""
Streamlit web application for blade performance prediction.

Master's Thesis Project - Mechanical Engineering
Physics-informed deep learning for cutting blade optimization.

Features:
- Interactive input controls for blade parameters
- Real-time prediction of 3 performance metrics
- Post-prediction performance score calculation
- 2D blade visualization with wear zones
- Rule-based optimization recommendations

Usage: streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
PREPROCESSOR_PATH = PROJECT_ROOT / "model" / "preprocessor.pkl"

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessor with caching."""
    if not PREPROCESSOR_PATH.exists():
        st.error(f"❌ Preprocessor not found at {PREPROCESSOR_PATH}")
        st.stop()

    preprocessor = joblib.load(str(PREPROCESSOR_PATH))

    try:
        if not MODEL_PATH.exists():
            st.error(f"❌ Model not found at {MODEL_PATH}")
            st.error("Please run: `python model/train_model.py`")
            st.stop()
        
        # Enable unsafe deserialization for our trusted model with Lambda layers
        import keras
        keras.config.enable_unsafe_deserialization()
        
        model = load_model(str(MODEL_PATH), compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.error("Try retraining the model with current version")
        st.stop()

    return model, preprocessor

def estimate_friction_coefficient(material: str, lubrication: bool) -> float:
    """Auto-calculate friction coefficient based on material and lubrication."""
    base_friction = {
        "Steel": 0.60,
        "Stainless Steel": 0.65,
        "Aluminum": 0.30,
        "Cast Iron": 0.55,
        "Brass": 0.35,
        "Titanium": 0.65,
    }
    
    base = base_friction.get(material, 0.6)
    return base * (0.6 if lubrication else 1.0)

def compute_performance_score(lifespan: float, wear: float, efficiency: float) -> float:
    """Compute performance score post-prediction as per thesis specification."""
    performance_score = 0.4 * (100 - wear) + 0.3 * efficiency + 0.3 * min(lifespan / 10, 1) * 100
    return min(100, performance_score)

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
        recommendations.append("• Low cutting speeds essential due to poor thermal conductivity")
    elif inputs["material_to_cut"] == "Aluminum":
        recommendations.append("\n**Aluminum-Specific Tips:**") 
        recommendations.append("• Higher speeds generally acceptable")
        recommendations.append("• Watch for built-up edge formation")
        recommendations.append("• Use sharp tools with polished cutting edges")
    elif inputs["material_to_cut"] == "Stainless Steel":
        recommendations.append("\n**Stainless Steel-Specific Tips:**")
        recommendations.append("• Work hardens rapidly - use sharp tools")
        recommendations.append("• Lower cutting speeds, positive rake angles")
        recommendations.append("• Ensure adequate lubrication/cooling")
        recommendations.append("• Avoid tool rubbing to prevent work hardening")
    elif inputs["material_to_cut"] == "Cast Iron":
        recommendations.append("\n**Cast Iron-Specific Tips:**")
        recommendations.append("• Dry cutting often preferred (graphite acts as lubricant)")
        recommendations.append("• Higher cutting speeds can be used")
        recommendations.append("• Good chip evacuation important")
        recommendations.append("• Abrasive material - carbide tools recommended")
    elif inputs["material_to_cut"] == "Brass":
        recommendations.append("\n**Brass-Specific Tips:**")
        recommendations.append("• Easy to machine with high speeds")
        recommendations.append("• Prevent built-up edge with sharp tools")
        recommendations.append("• Excellent surface finish achievable")
        recommendations.append("• Minimal lubrication needed")
    elif inputs["material_to_cut"] == "Steel":
        recommendations.append("\n**Steel-Specific Tips:**")
        recommendations.append("• Balanced cutting parameters work well")
        recommendations.append("• Adjust speed based on hardness")
        recommendations.append("• Good general-purpose machining characteristics")
    
    # Blade type-specific advice
    blade_type = inputs.get("blade_type", "")
    if "Circular" in blade_type:
        recommendations.append("\n**Circular Blade Tips:**")
        recommendations.append("• Continuous cutting action improves efficiency")
        recommendations.append("• Monitor for uniform wear around circumference")
    elif "Insert" in blade_type or "Replaceable" in blade_type:
        recommendations.append("\n**Insert/Replaceable Tip Tips:**")
        recommendations.append("• Cost-effective - replace inserts when worn")
        recommendations.append("• Ensure proper insert clamping")
    elif "Toothed" in blade_type:
        recommendations.append("\n**Toothed Blade Tips:**")
        recommendations.append("• Multiple cutting edges distribute wear")
        recommendations.append("• Good for softer materials and high removal rates")
    elif "Straight" in blade_type:
        recommendations.append("\n**Straight Blade Tips:**")
        recommendations.append("• Simple geometry, easy to sharpen")
        recommendations.append("• Versatile for various cutting operations")
    
    return "\n".join(recommendations)

def main():
    """Main Streamlit application."""
    
    # Load model artifacts
    model, preprocessor = load_model_artifacts()
    
    # App header
    st.title("🔧 Ahlamm - Blade Performance Predictor")
    st.markdown("""
    **Physics-informed deep learning for cutting blade optimization** | Master's Thesis Project
    
    Predict blade performance across **168 material combinations** (6 workpiece materials × 7 blade materials × 4 blade types)  
    using synthetic data generated from Taylor's Tool Life Equation and ASM Handbook constants.
    """)
    
    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Blade & Cutting Parameters")
        
        # Material selection - expanded to 6 materials
        material_to_cut = st.selectbox(
            "Material to Cut",
            ["Steel", "Stainless Steel", "Aluminum", "Cast Iron", "Brass", "Titanium"],
            help="Workpiece material being machined"
        )
        
        # Blade material - expanded to 7 options
        blade_material = st.selectbox(
            "Blade Material", 
            ["HSS", "Carbide", "Coated Carbide (TiN)", "Coated Carbide (TiAlN)", "Ceramic", "CBN", "PCD"],
            help="HSS=High-Speed Steel, CBN=Cubic Boron Nitride, PCD=Polycrystalline Diamond"
        )
        
        # NEW: Blade type selection - 4 types
        blade_type = st.selectbox(
            "Blade Type",
            ["Straight Blade", "Circular Blade", "Insert/Replaceable Tip Blade", "Toothed Blade"],
            help="Physical blade configuration and design"
        )
        
        # Geometric parameters
        st.subheader("Geometry")
        cutting_angle = st.number_input(
            "Cutting Angle (°)", 
            min_value=15, max_value=45, value=30, step=1,
            help="Blade cutting angle in degrees (thesis range: 15-45°)"
        )
        
        blade_thickness = st.number_input(
            "Blade Thickness (mm)",
            min_value=0.5, max_value=5.0, value=2.5, step=0.1, format="%.1f",
            help="Blade thickness in millimeters (thesis range: 0.5-5.0mm)"
        )
        
        # Cutting conditions  
        st.subheader("Cutting Conditions")
        cutting_speed = st.number_input(
            "Cutting Speed (m/min)",
            min_value=20, max_value=200, value=100, step=5,
            help="Cutting speed in meters per minute"
        )
        
        applied_force = st.number_input(
            "Applied Force (N)",
            min_value=100, max_value=2000, value=800, step=50,
            help="Applied cutting force in Newtons"
        )
        
        operating_temp = st.number_input(
            "Operating Temperature (°C)",
            min_value=20, max_value=800, value=300, step=10,
            help="Operating temperature in Celsius (thesis range: 20-800°C)"
        )
        
        lubrication = st.checkbox(
            "Lubrication Enabled",
            value=True,
            help="Whether lubrication/coolant is used"
        )
        
        # Auto-calculate friction coefficient
        friction_coeff = estimate_friction_coefficient(material_to_cut, lubrication)
        st.info(f"📊 Auto-calculated friction: **{friction_coeff:.3f}**")
    
    # Prediction section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔍 Predict Performance", type="primary", use_container_width=True):
            # Prepare input data (now includes blade_type)
            input_data = pd.DataFrame([{
                "material_to_cut": material_to_cut,
                "blade_material": blade_material,
                "blade_type": blade_type,
                "cutting_angle_deg": cutting_angle,
                "blade_thickness_mm": blade_thickness,
                "cutting_speed_m_per_min": cutting_speed,
                "applied_force_N": applied_force,
                "operating_temperature_C": operating_temp,
                "friction_coefficient": friction_coeff,
                "lubrication": lubrication
            }])
            
            try:
                # Preprocess and predict (3 outputs only)
                X_processed = preprocessor.transform(input_data)
                predictions = model.predict(X_processed, verbose=0)
                
                # Model returns 3 separate arrays [lifespan_arr, wear_arr, efficiency_arr]
                lifespan = float(predictions[0][0])
                wear = float(predictions[1][0])
                efficiency = float(predictions[2][0])
                
                # Compute performance score post-prediction
                performance = compute_performance_score(lifespan, wear, efficiency)
                
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
        **Materials to Cut:**
        - **Steel**: General purpose, moderate speeds
        - **Stainless Steel**: Work hardening, use sharp tools, lower speeds
        - **Aluminum**: Higher speeds OK, watch for buildup  
        - **Cast Iron**: Abrasive, higher cutting speeds acceptable
        - **Brass**: Easy to machine, good surface finish
        - **Titanium**: Lower speeds, excellent lubrication needed
        
        **Blade Materials:**
        - **HSS**: Cost-effective, moderate performance
        - **Carbide**: Higher performance, harder materials
        - **Coated Carbide (TiN/TiAlN)**: Best wear resistance
        - **Ceramic**: High-speed machining, hard materials
        - **CBN**: Hardened steels, high-performance
        - **PCD**: Non-ferrous materials, long life
        
        **Blade Types:**
        - **Straight**: General purpose, simple geometry
        - **Circular**: Continuous cutting, high efficiency
        - **Insert/Replaceable**: Cost-effective, quick changes
        - **Toothed**: Multiple cutting edges, good for soft materials
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
                st.write(f"**Blade Type:** {pred['inputs'].get('blade_type', 'N/A')}")
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
        - Lifespan: **0.95** ✅
        - Wear: **0.90** ⚠️
        - Efficiency: **0.68** ⚠️
        - Overall: **0.84** ⚠️
        
        *Trained on 168 combinations*  
        *(6 materials × 7 blades × 4 types)*
        """)
    
    st.info("💡 **Tip**: Results are most accurate when parameters are within normal operating ranges (Speed: 80-120 m/min, Angle: 12-18°, Temperature: 200-400°C)")

if __name__ == "__main__":
    main()