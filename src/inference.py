"""
FastAPI Inference Service Stub for FraudShield AI Platform.
Demonstrates production deployment readiness.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class FraudPredictionRequest:
    transaction_amount: float
    transaction_time: float
    features: Dict[str, float]

@dataclass
class FraudPredictionResponse:
    transaction_id: str
    fraud_probability: float
    fraud_prediction: bool
    risk_level: str
    threshold_used: float
    model_version: str
    explanation: Optional[Dict[str, Any]] = None

class FraudScoringService:
    def __init__(self, model_path: str, threshold: float = 0.5, model_version: str = "1.0.0"):
        self.model_path = model_path
        self.threshold = threshold
        self.model_version = model_version
        self.model = None
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model {model_version} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(self, request: FraudPredictionRequest, transaction_id: str = "test-tx") -> FraudPredictionResponse:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        
        # Construct feature vector
        feature_dict = {'Time': request.transaction_time, 'Amount': request.transaction_amount}
        feature_dict.update(request.features)
        
        df = pd.DataFrame([feature_dict])
        
        prob = self.model.predict_proba(df)[0, 1]
        is_fraud = bool(prob >= self.threshold)
        
        return FraudPredictionResponse(
            transaction_id=transaction_id,
            fraud_probability=float(prob),
            fraud_prediction=is_fraud,
            risk_level=self._classify_risk_level(prob),
            threshold_used=self.threshold,
            model_version=self.model_version
        )
        
    def _classify_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.6:
            return 'MEDIUM'
        elif probability < 0.85:
            return 'HIGH'
        else:
            return 'CRITICAL'
            
    def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self.model is not None else 'unhealthy',
            'model_loaded': self.model is not None,
            'model_version': self.model_version,
            'threshold': self.threshold
        }

def create_fastapi_app(model_path: str = "outputs/models/champion_model.joblib", threshold: float = 0.5, model_version: str = "1.0.0") -> str:
    """Returns the source code of the FastAPI app for demonstration in the notebook."""
    return f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from src.inference import FraudScoringService, FraudPredictionRequest
import uuid

app = FastAPI(title="FraudShield AI Inference API", version="{model_version}")
scoring_service = FraudScoringService("{model_path}", {threshold}, "{model_version}")

class PredictRequest(BaseModel):
    transaction_amount: float = Field(..., gt=0)
    transaction_time: float
    features: Dict[str, float]

@app.get("/health")
def health():
    return scoring_service.health_check()

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        service_req = FraudPredictionRequest(
            transaction_amount=req.transaction_amount,
            transaction_time=req.transaction_time,
            features=req.features
        )
        tx_id = str(uuid.uuid4())
        resp = scoring_service.predict(service_req, tx_id)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
