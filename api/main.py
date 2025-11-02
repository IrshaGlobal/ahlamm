"""
FastAPI backend for Blade Performance Predictor.

Author: Ahlam Faci
Master's Thesis Project - Mechanical Engineering

Provides REST API endpoints for blade performance predictions using
a 5-model ensemble with composite weighted loss optimization.

Endpoints:
- GET /: API documentation and health check
- POST /predict: Blade performance prediction
- GET /health: Health check endpoint
- GET /models/info: Model ensemble information
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import joblib
from pathlib import Path
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Blade Performance Predictor API - Ahlam Faci",
    description="Physics-informed deep learning for cutting blade optimization | Master's Thesis Project",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessor
models = []
preprocessor = None
MODEL_DIR = Path(__file__).parent.parent / "model"

# Material friction coefficients
FRICTION_COEFFICIENTS = {
    "Steel": 0.60,
    "Stainless Steel": 0.65,
    "Aluminum": 0.30,
    "Cast Iron": 0.55,
    "Brass": 0.35,
    "Titanium": 0.65,
}


def generate_detailed_recommendations(
    lifespan: float,
    wear: float,
    efficiency: float,
    performance_score: float,
    material_to_cut: str,
    blade_material: str,
    blade_type: str
) -> List[str]:
    """Generate comprehensive optimization recommendations based on predictions."""
    
    recommendations = []
    
    # High wear recommendations
    if wear > 70:
        recommendations.append("🔴 **High Wear Detected**")
        recommendations.append("• Reduce cutting speed by 15-20%")
        recommendations.append("• Enable lubrication if not already used")
        if blade_material == "HSS" and material_to_cut in ["Steel", "Titanium"]:
            recommendations.append("• Consider switching to carbide blade for harder materials")
    
    # Low efficiency recommendations  
    elif efficiency < 50:
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
        recommendations.append("🌟 Excellent performance expected!")
    
    # Material-specific advice
    if material_to_cut == "Titanium":
        recommendations.append("**Titanium Tips:** Use flood coolant, maintain sharp edges, low speeds due to poor thermal conductivity")
    elif material_to_cut == "Aluminum":
        recommendations.append("**Aluminum Tips:** Higher speeds acceptable, watch for built-up edge, use sharp polished tools")
    elif material_to_cut == "Stainless Steel":
        recommendations.append("**Stainless Steel Tips:** Work hardens rapidly, use sharp tools, lower speeds, avoid tool rubbing")
    elif material_to_cut == "Cast Iron":
        recommendations.append("**Cast Iron Tips:** Dry cutting often preferred (graphite lubricates), higher speeds OK, carbide recommended")
    elif material_to_cut == "Brass":
        recommendations.append("**Brass Tips:** Easy machining with high speeds, excellent finish achievable, minimal lubrication")
    elif material_to_cut == "Steel":
        recommendations.append("**Steel Tips:** Balanced parameters work well, adjust speed based on hardness")
    
    # Blade type-specific advice
    if "Circular" in blade_type:
        recommendations.append("**Circular Blade:** Continuous cutting improves efficiency, monitor for uniform wear")
    elif "Insert" in blade_type or "Replaceable" in blade_type:
        recommendations.append("**Insert/Replaceable:** Cost-effective, replace inserts when worn, ensure proper clamping")
    elif "Toothed" in blade_type:
        recommendations.append("**Toothed Blade:** Multiple cutting edges, good chip evacuation, suitable for softer materials")
    elif "Straight" in blade_type:
        recommendations.append("**Straight Blade:** General purpose, simple geometry, good for standard operations")
    
    return recommendations


class BladeParameters(BaseModel):
    """Input parameters for blade performance prediction - matches training data exactly."""
    
    # Material configuration
    workpiece_material: str = Field(
        ...,
        description="Workpiece material type"
    )
    blade_material: str = Field(
        default="Carbide",
        description="Blade material type"
    )
    blade_type: str = Field(
        default="Circular Blade",
        description="Blade type/configuration"
    )
    
    # Geometric parameters
    thickness: float = Field(
        ..., 
        ge=0.5, 
        le=5.0,
        description="Blade thickness in mm"
    )
    cutting_angle: float = Field(
        ...,
        ge=15.0,
        le=45.0,
        description="Cutting angle in degrees"
    )
    
    # Operating parameters (actual training features)
    cutting_speed: float = Field(
        ...,
        ge=20.0,
        le=200.0,
        description="Cutting speed in m/min"
    )
    applied_force: float = Field(
        ...,
        ge=100.0,
        le=2000.0,
        description="Applied cutting force in N"
    )
    operating_temperature: float = Field(
        ...,
        ge=20.0,
        le=800.0,
        description="Operating temperature in °C"
    )
    lubrication: bool = Field(
        ...,
        description="Whether lubrication is used"
    )
    
    @validator('workpiece_material')
    def validate_material(cls, v):
        valid_materials = list(FRICTION_COEFFICIENTS.keys())
        if v not in valid_materials:
            raise ValueError(f"Material must be one of: {', '.join(valid_materials)}")
        return v
    
    @validator('blade_material')
    def validate_blade_material(cls, v):
        valid_blades = ["HSS", "Carbide", "Coated Carbide (TiN)", "Coated Carbide (TiAlN)", 
                       "Ceramic", "CBN", "PCD"]
        if v not in valid_blades:
            raise ValueError(f"Blade material must be one of: {', '.join(valid_blades)}")
        return v
    
    @validator('blade_type')
    def validate_blade_type(cls, v):
        valid_types = ["Straight Blade", "Circular Blade", 
                      "Insert/Replaceable Tip Blade", "Toothed Blade"]
        if v not in valid_types:
            raise ValueError(f"Blade type must be one of: {', '.join(valid_types)}")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    blade_lifespan: float = Field(..., description="Predicted blade lifespan in hours")
    wear_estimation: float = Field(..., description="Predicted wear percentage")
    cutting_efficiency: float = Field(..., description="Predicted cutting efficiency percentage")
    performance_score: float = Field(..., description="Composite performance score (0-100)")
    friction_coefficient: float = Field(..., description="Estimated friction coefficient")
    optimization_tips: List[str] = Field(..., description="Optimization recommendations")


class ModelInfo(BaseModel):
    """Information about loaded models."""
    
    num_models: int
    model_seeds: List[int]
    best_model_seed: int
    ensemble_type: str

    # Allow field names that start with "model_" without warnings (Pydantic v2)
    model_config = {
        "protected_namespaces": ()
    }


def estimate_friction_coefficient(material: str, lubrication: bool) -> float:
    """Calculate friction coefficient based on material and lubrication."""
    base_friction = FRICTION_COEFFICIENTS.get(material, 0.6)
    return base_friction * (0.6 if lubrication else 1.0)


def compute_performance_score(lifespan: float, wear: float, efficiency: float) -> float:
    """
    Compute composite performance score.
    
    Formula: 0.4 * (100 - wear) + 0.3 * efficiency + 0.3 * min(lifespan/10, 1) * 100
    """
    performance_score = (
        0.4 * (100 - wear) + 
        0.3 * efficiency + 
        0.3 * min(lifespan / 10, 1) * 100
    )
    return min(100, performance_score)


def generate_optimization_tips(
    params: BladeParameters,
    predictions: dict
) -> List[str]:
    """Generate actionable optimization recommendations."""
    tips = []
    
    wear = predictions['wear_estimation']
    efficiency = predictions['cutting_efficiency']
    lifespan = predictions['blade_lifespan']
    
    # Wear-based recommendations
    if wear > 70:
        tips.append("⚠️ High wear predicted. Consider reducing cutting speed or increasing lubrication.")
        if params.cutting_speed > 300:
            tips.append("💡 Cutting speed is high. Reduce to 250-300 m/min to extend blade life.")
    
    # Efficiency recommendations
    if efficiency < 75:
        tips.append("📉 Efficiency could be improved. Try adjusting cutting angle to 25-35°.")
        if params.cutting_angle < 20 or params.cutting_angle > 40:
            tips.append("🔧 Cutting angle is suboptimal. Target 30° for best efficiency.")
    
    # Lifespan recommendations
    if lifespan < 3:
        tips.append("⏱️ Short blade lifespan predicted. Consider using harder blade material.")
        if not params.lubrication:
            tips.append("💧 Adding lubrication could increase lifespan by 20-40%.")
    
    # Material-specific tips
    if params.workpiece_material == "Steel" and params.cutting_speed > 180:
        tips.append("🔥 For Steel, moderate speeds are safer. Consider 120-180 m/min depending on hardness.")
    
    if params.workpiece_material == "Aluminum" and params.cutting_speed < 80:
        tips.append("⚡ Aluminum can handle higher speeds. Consider increasing cutting speed for productivity.")
    
    # Default positive feedback
    if not tips:
        tips.append("✅ Parameters are well-optimized! Expect good performance.")
        if efficiency > 90:
            tips.append("🌟 Excellent efficiency predicted. Maintain these settings.")
    
    return tips


@app.on_event("startup")
async def load_models():
    """Load ensemble models and preprocessor on startup."""
    global models, preprocessor
    
    try:
        # Load preprocessor
        preprocessor_path = MODEL_DIR / "preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        
        preprocessor = joblib.load(str(preprocessor_path))
        logger.info(f"✓ Loaded preprocessor from {preprocessor_path}")
        
        # Load ensemble models
        import tensorflow as tf
        model_files = sorted(glob.glob(str(MODEL_DIR / "blade_model_seed*.h5")))
        
        if not model_files:
            raise FileNotFoundError("No model files found. Run training first.")
        
        for model_path in model_files:
            model = tf.keras.models.load_model(model_path, compile=False)
            models.append(model)
            logger.info(f"✓ Loaded model: {Path(model_path).name}")
        
        logger.info(f"🧠 Ensemble loaded successfully: {len(models)} models")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise


@app.get("/")
async def root():
    """API root endpoint with welcome message."""
    return {
        "message": "Ahlamm Blade Performance Predictor API",
        "version": "2.0",
        "models_loaded": len(models),
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "models_info": "/models/info (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/models/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded ensemble models."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Extract seed numbers from model filenames
    seeds = []
    for i, _ in enumerate(models):
        # Seeds are: 42, 1337, 2025, 7, 101
        seeds_list = [42, 1337, 2025, 7, 101]
        if i < len(seeds_list):
            seeds.append(seeds_list[i])
    
    return ModelInfo(
        num_models=len(models),
        model_seeds=seeds,
        best_model_seed=1337,  # Best performing model
        ensemble_type="averaging"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(params: BladeParameters):
    """
    Predict blade performance metrics.
    
    Uses ensemble of 5 models with averaging for robust predictions.
    """
    if not models or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Server is initializing."
        )
    
    try:
        # Calculate friction coefficient based on material and lubrication
        friction_coef = estimate_friction_coefficient(
            params.workpiece_material,
            params.lubrication
        )
        
        # Create input dataframe matching EXACT training features
        import pandas as pd
        input_df = pd.DataFrame([{
            'material_to_cut': params.workpiece_material,
            'blade_material': params.blade_material,
            'blade_type': params.blade_type,
            'cutting_angle_deg': params.cutting_angle,
            'blade_thickness_mm': params.thickness,
            'cutting_speed_m_per_min': params.cutting_speed,
            'applied_force_N': params.applied_force,
            'operating_temperature_C': params.operating_temperature,
            'friction_coefficient': friction_coef,
            'lubrication': params.lubrication
        }])
        
        # Preprocess (this will add derived features automatically)
        input_scaled = preprocessor.transform(input_df)
        
        # Ensemble prediction
        predictions = []
        for model in models:
            pred = model.predict(input_scaled, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)[0]
        
        lifespan = float(avg_prediction[0])
        wear = float(avg_prediction[1])
        efficiency = float(avg_prediction[2])
        
        # Compute performance score
        performance_score = compute_performance_score(lifespan, wear, efficiency)
        
        # Generate comprehensive optimization recommendations
        tips = generate_detailed_recommendations(
            lifespan=lifespan,
            wear=wear,
            efficiency=efficiency,
            performance_score=performance_score,
            material_to_cut=params.workpiece_material,
            blade_material=params.blade_material,
            blade_type=params.blade_type
        )
        
        return PredictionResponse(
            blade_lifespan=round(lifespan, 2),
            wear_estimation=round(wear, 2),
            cutting_efficiency=round(efficiency, 2),
            performance_score=round(performance_score, 2),
            friction_coefficient=round(friction_coef, 3),
            optimization_tips=tips
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
