from pydantic import BaseModel, Field, ConfigDict
from typing import List

class CustomerProfile(BaseModel):
    """
    Strict validation schema for incoming pricing requests.
    """
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    x_loyalty: float = Field(..., ge=0.0, le=1.0, description="Customer loyalty score [0, 1].")
    x_spend: float = Field(..., ge=0.0, description="Historical lifetime spend.")
    x_device: int = Field(..., ge=0, le=1, description="Device type: 1 for Desktop, 0 for Mobile.")
    
    # Inject a realistic example directly into the Swagger UI
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-9921",
                "x_loyalty": 0.95,
                "x_spend": 55.4,
                "x_device": 1
            }
        }
    )

class ElasticityResponse(BaseModel):
    """
    Response schema returning the causal price elasticity.
    """
    customer_id: str
    predicted_elasticity: float
    recommended_action: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "CUST-9921",
                "predicted_elasticity": -1.2792,
                "recommended_action": "Increase Markup"
            }
        }
    )

class BatchElasticityResponse(BaseModel):
    results: List[ElasticityResponse]