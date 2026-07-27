import sys
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Body
from loguru import logger
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.causal.dml_engine import DoubleMachineLearningEngine
from src.api.schemas import CustomerProfile, ElasticityResponse
from core.config import settings

# Global reference to our loaded model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Loads the DML engine into memory so it's ready for instant inference.
    """
    model_path = "data/processed/dml_oracle_model.joblib"
    try:
        logger.info("Initializing API and loading DML Causal Engine...")
        ml_models["causal_engine"] = DoubleMachineLearningEngine.load(model_path)
        logger.success("Model loaded successfully into memory.")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    finally:
        ml_models.clear()
        logger.info("API shut down, models cleared.")

app = FastAPI(
    title=settings.app_name,
    description="Real-time Double Machine Learning API for Prescriptive Pricing Analytics.",
    version="1.0.0",
    lifespan=lifespan
)

def get_model() -> DoubleMachineLearningEngine:
    if "causal_engine" not in ml_models:
        raise HTTPException(status_code=503, detail="Causal Engine not initialized.")
    return ml_models["causal_engine"]

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Simple health check endpoint for load balancers."""
    return {"status": "healthy", "model_loaded": "causal_engine" in ml_models}

@app.post(
    "/predict_elasticity", 
    response_model=ElasticityResponse, 
    tags=["Causal Inference"],
    summary="Predict Individual Price Elasticity (CATE)"
)
async def predict_elasticity(
    profile: CustomerProfile = Body(..., description="Customer features for causal inference"),
    engine: DoubleMachineLearningEngine = Depends(get_model)
):
    """
    Receives customer features and outputs their personalized price elasticity.
    The model controls for hidden confounders via Orthogonalization.
    """
    try:
        # Convert schema to DataFrame for the model
        df_input = pd.DataFrame([profile.model_dump()])
        
        # Predict the Conditional Average Treatment Effect (CATE)
        cate_pred = engine.predict_cate(df_input)[0]
        
        # Simple business logic heuristic based on causal output
        action = "Increase Markup" if cate_pred > -1.5 else "Offer Discount"
        
        return ElasticityResponse(
            customer_id=profile.customer_id,
            predicted_elasticity=round(float(cate_pred), 4),
            recommended_action=action
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error.")