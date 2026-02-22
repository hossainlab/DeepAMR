#!/usr/bin/env python3
"""DeepAMR API - FastAPI backend for Antimicrobial Resistance Prediction.

This API provides endpoints for:
1. AMR drug resistance prediction from genomic features
2. Model information and health checks
3. Batch predictions for multiple samples

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Or run directly:
    python -m src.api.main
"""

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.inference import DeepAMRPredictor, SklearnAMRPredictor, get_predictor, MODEL_VERSION
from src.ml.feature_extraction import KmerFeatureExtractor, get_extractor
from src.api.bangladesh_guidelines import (
    get_bangladesh_recommendations, get_bangladesh_context,
    BANGLADESH_RESISTANCE_PREVALENCE, REFERRAL_CENTERS,
)
from src.api.reports import generate_prediction_report
from src.api.database import (
    init_db, verify_password, create_user, get_user_by_email, get_user_by_id,
    list_users, delete_user, update_last_login, create_session, get_session,
    delete_session, save_prediction, get_prediction, list_predictions,
    delete_prediction, get_recent_predictions, log_activity, get_recent_activity,
    get_dashboard_stats, get_resistance_overview, get_trends, get_admin_stats,
    _sanitize_user,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Drug class display names (matches frontend DRUG_CLASS_DISPLAY_NAMES)
DRUG_CLASS_DISPLAY = {
    "aminoglycoside": "Aminoglycosides",
    "beta-lactam": "Beta-lactams",
    "fosfomycin": "Fosfomycin",
    "glycopeptide": "Glycopeptides",
    "macrolide": "Macrolides",
    "phenicol": "Phenicols",
    "quinolone": "Quinolones",
    "rifampicin": "Rifampicin",
    "sulfonamide": "Sulfonamides",
    "tetracycline": "Tetracyclines",
    "trimethoprim": "Trimethoprim",
}

# =============================================================================
# FastAPI Application
# =============================================================================

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="DeepAMR API",
    description="""
    Deep Learning API for Antimicrobial Resistance (AMR) Prediction.

    This API predicts antibiotic resistance from bacterial genomic features (k-mer frequencies).
    Designed for integration with healthcare systems in Bangladesh.

    ## Features
    - Multi-label AMR prediction across 11 drug classes
    - Deep learning and ensemble model options
    - Batch prediction support
    - Risk level assessment

    ## Drug Classes Supported
    - Aminoglycoside, Beta-lactam, Fosfomycin, Glycopeptide
    - Macrolide, Phenicol, Quinolone, Rifampicin
    - Sulfonamide, Tetracycline, Trimethoprim
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State
# =============================================================================

# Lazy-loaded predictors
_predictors: Dict[str, Union[DeepAMRPredictor, SklearnAMRPredictor]] = {}


def get_model(model_type: str = "deep_learning"):
    """Get or initialize predictor (lazy loading)."""
    if model_type not in _predictors:
        try:
            _predictors[model_type] = get_predictor(model_type)
            logger.info(f"Loaded {model_type} model")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise HTTPException(status_code=503, detail=f"Model not available: {model_type}")
    return _predictors[model_type]


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    features: List[float] = Field(
        ...,
        description="K-mer frequency features (500-dimensional vector)",
        min_length=100,
    )
    threshold: float = Field(
        default=0.5,
        description="Probability threshold for positive prediction",
        ge=0.0,
        le=1.0,
    )
    model_type: str = Field(
        default="deep_learning",
        description="Model to use: 'deep_learning' or 'sklearn'",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.01] * 500,
                "threshold": 0.5,
                "model_type": "deep_learning",
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
    )
    threshold: float = Field(default=0.5)
    model_type: str = Field(default="deep_learning")


class DrugPrediction(BaseModel):
    """Prediction result for a single drug class."""
    drug_class: str
    resistant: bool
    probability: float
    confidence: str  # "high", "medium", "low"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: Dict[str, bool] = Field(
        description="Drug class to resistance status mapping"
    )
    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Drug class to probability mapping",
    )
    resistant_count: int
    susceptible_count: int
    risk_level: str = Field(description="Overall risk assessment: MINIMAL, LOW, MODERATE, HIGH")
    risk_description: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    results: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    drug_classes: List[str]
    n_classes: int
    device: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: List[str]


# =============================================================================
# Helper Functions
# =============================================================================

def get_risk_assessment(resistant_count: int, total_classes: int) -> tuple:
    """Calculate risk level based on resistance count."""
    if resistant_count >= 5:
        return "HIGH", "Multi-drug resistant (MDR) - Requires specialist consultation"
    elif resistant_count >= 3:
        return "MODERATE", "Multiple resistance detected - Consider alternative treatments"
    elif resistant_count >= 1:
        return "LOW", "Limited resistance - Standard alternatives available"
    else:
        return "MINIMAL", "No predicted resistance - Standard treatment likely effective"


def get_confidence(probability: float) -> str:
    """Get confidence level from probability."""
    certainty = max(probability, 1 - probability)
    if certainty >= 0.8:
        return "high"
    elif certainty >= 0.6:
        return "medium"
    else:
        return "low"


# =============================================================================
# Auth dependency
# =============================================================================

async def get_current_user(authorization: Optional[str] = Header(default=None)) -> Optional[Dict]:
    """Extract user from Authorization header token."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "")
    session = get_session(token)
    if not session:
        return None
    user = get_user_by_id(session["user_id"])
    if not user:
        return None
    return _sanitize_user(user)


async def require_user(user: Optional[Dict] = Depends(get_current_user)) -> Dict:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# =============================================================================
# Auth Pydantic Models
# =============================================================================

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    organization: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str


# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/auth/register", tags=["Auth"])
@limiter.limit("3/minute")
async def auth_register(request: Request, req: RegisterRequest):
    try:
        user = create_user(req.email, req.name, req.password, organization=req.organization)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_session(user["id"])
    log_activity(user["id"], user["name"], "Registered account")
    return {"user": user, "token": token}


@app.post("/auth/login", tags=["Auth"])
@limiter.limit("5/minute")
async def auth_login(request: Request, req: LoginRequest):
    db_user = get_user_by_email(req.email)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(req.password, db_user["salt"], db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    update_last_login(db_user["id"])
    token = create_session(db_user["id"])
    user = _sanitize_user(db_user)
    log_activity(user["id"], user["name"], "Logged in")
    return {"user": user, "token": token}


@app.get("/auth/me", tags=["Auth"])
async def auth_me(user: Dict = Depends(require_user)):
    return {"user": user}


@app.post("/auth/logout", tags=["Auth"])
async def auth_logout(authorization: Optional[str] = Header(default=None)):
    if authorization:
        token = authorization.replace("Bearer ", "")
        delete_session(token)
    return {"success": True}


# =============================================================================
# Prediction History Endpoints
# =============================================================================

@app.get("/predictions", tags=["Predictions"])
async def list_predictions_endpoint(
    organism: Optional[str] = None,
    status: Optional[str] = None,
    risk: Optional[str] = None,
    search: Optional[str] = None,
    dateFrom: Optional[str] = None,
    dateTo: Optional[str] = None,
):
    return list_predictions(
        organism=organism, status=status, risk=risk,
        search=search, date_from=dateFrom, date_to=dateTo,
    )


@app.get("/predictions/recent", tags=["Predictions"])
async def recent_predictions_endpoint(limit: int = 5):
    return get_recent_predictions(limit)


@app.get("/predictions/{pred_id}", tags=["Predictions"])
async def get_prediction_endpoint(pred_id: str):
    pred = get_prediction(pred_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return pred


@app.delete("/predictions/{pred_id}", tags=["Predictions"])
async def delete_prediction_endpoint(pred_id: str):
    if not delete_prediction(pred_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"success": True}


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@app.get("/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats_endpoint():
    return get_dashboard_stats()


@app.get("/dashboard/resistance-overview", tags=["Dashboard"])
async def dashboard_resistance_overview():
    return get_resistance_overview()


@app.get("/dashboard/trends", tags=["Dashboard"])
async def dashboard_trends():
    return get_trends()


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.get("/admin/stats", tags=["Admin"])
async def admin_stats_endpoint():
    return get_admin_stats()


@app.get("/admin/users", tags=["Admin"])
async def admin_users_endpoint():
    return list_users()


@app.delete("/admin/users/{user_id}", tags=["Admin"])
async def admin_delete_user(user_id: str):
    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True}


@app.get("/admin/activity", tags=["Admin"])
async def admin_activity_endpoint():
    return get_recent_activity()


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Serve the frontend application."""
    frontend_index = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)
    return {
        "name": "DeepAMR API",
        "version": "1.0.0",
        "description": "Antimicrobial Resistance Prediction API",
        "docs": "/docs",
    }


@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint."""
    return {
        "name": "DeepAMR API",
        "version": "1.0.0",
        "description": "Antimicrobial Resistance Prediction API",
        "docs": "/docs",
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/predict/detailed",
            "/health",
            "/drug-classes",
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(_predictors.keys()),
    )


@app.get("/models/{model_type}/info", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(model_type: str = "deep_learning"):
    """Get information about a specific model."""
    try:
        predictor = get_model(model_type)
        info = predictor.model_info
        return ModelInfoResponse(
            model_type=model_type,
            drug_classes=info["drug_classes"],
            n_classes=info["n_classes"],
            device=info["device"],
            status="loaded",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drug-classes", tags=["Information"])
async def get_drug_classes():
    """Get list of supported drug classes."""
    return {
        "drug_classes": DeepAMRPredictor.DEFAULT_DRUG_CLASSES,
        "count": len(DeepAMRPredictor.DEFAULT_DRUG_CLASSES),
        "descriptions": {
            "aminoglycoside": "Antibiotics that inhibit protein synthesis (e.g., Gentamicin, Amikacin)",
            "beta-lactam": "Antibiotics with beta-lactam ring (e.g., Penicillins, Cephalosporins)",
            "fosfomycin": "Broad-spectrum antibiotic for urinary tract infections",
            "glycopeptide": "Antibiotics for Gram-positive bacteria (e.g., Vancomycin)",
            "macrolide": "Protein synthesis inhibitors (e.g., Azithromycin, Erythromycin)",
            "phenicol": "Chloramphenicol class antibiotics",
            "quinolone": "DNA synthesis inhibitors (e.g., Ciprofloxacin, Levofloxacin)",
            "rifampicin": "RNA synthesis inhibitor, used for tuberculosis",
            "sulfonamide": "Folic acid synthesis inhibitors",
            "tetracycline": "Broad-spectrum protein synthesis inhibitors",
            "trimethoprim": "Dihydrofolate reductase inhibitor",
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict(request: PredictionRequest, req: Request = None):
    """Make AMR resistance prediction for a single sample.

    Args:
        request: Prediction request with features

    Returns:
        Prediction results with resistance status and probabilities
    """
    try:
        predictor = get_model(request.model_type)

        # Make prediction
        features = np.array(request.features)
        result = predictor.predict(features, threshold=request.threshold)

        # Calculate risk level
        risk_level, risk_description = get_risk_assessment(
            result["resistant_count"],
            len(predictor.drug_classes),
        )

        return PredictionResponse(
            predictions=result["predictions"],
            probabilities=result.get("probabilities"),
            resistant_count=result["resistant_count"],
            susceptible_count=result["susceptible_count"],
            risk_level=risk_level,
            risk_description=risk_description,
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Make AMR predictions for multiple samples.

    Args:
        request: Batch prediction request with multiple feature vectors

    Returns:
        List of prediction results
    """
    import time
    start_time = time.time()

    try:
        predictor = get_model(request.model_type)

        results = []
        for features in request.samples:
            features_array = np.array(features)
            result = predictor.predict(features_array, threshold=request.threshold)

            risk_level, risk_description = get_risk_assessment(
                result["resistant_count"],
                len(predictor.drug_classes),
            )

            results.append(PredictionResponse(
                predictions=result["predictions"],
                probabilities=result.get("probabilities"),
                resistant_count=result["resistant_count"],
                susceptible_count=result["susceptible_count"],
                risk_level=risk_level,
                risk_description=risk_description,
                timestamp=datetime.now().isoformat(),
            ))

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            results=results,
            total_samples=len(request.samples),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/detailed", tags=["Prediction"])
async def predict_detailed(request: PredictionRequest):
    """Get detailed prediction with per-drug analysis.

    Returns comprehensive analysis including confidence levels
    and clinical recommendations.
    """
    try:
        predictor = get_model(request.model_type)
        features = np.array(request.features)
        result = predictor.predict(features, threshold=request.threshold)

        # Build detailed response
        drug_predictions = []
        probs = result.get("probabilities", {})

        for drug, resistant in result["predictions"].items():
            prob = probs.get(drug, 0.5)
            drug_predictions.append(DrugPrediction(
                drug_class=drug,
                resistant=resistant,
                probability=prob,
                confidence=get_confidence(prob),
            ))

        # Sort by probability (most resistant first)
        drug_predictions.sort(key=lambda x: x.probability, reverse=True)

        risk_level, risk_description = get_risk_assessment(
            result["resistant_count"],
            len(predictor.drug_classes),
        )

        return {
            "summary": {
                "resistant_count": result["resistant_count"],
                "susceptible_count": result["susceptible_count"],
                "risk_level": risk_level,
                "risk_description": risk_description,
            },
            "drug_predictions": [p.model_dump() for p in drug_predictions],
            "recommendations": get_clinical_recommendations(drug_predictions),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detailed prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fasta", tags=["Prediction"])
@limiter.limit("10/minute")
async def predict_from_fasta(
    request: Request,
    file: UploadFile = File(..., description="FASTA or FASTQ file with genomic sequence"),
    threshold: float = Form(default=0.5, description="Classification threshold"),
    model_type: str = Form(default="deep_learning", description="Model type: deep_learning or sklearn"),
    organism: str = Form(default="Unknown", description="Organism name"),
    current_user: Optional[Dict] = Depends(get_current_user),
):
    """Predict AMR resistance from a FASTA/FASTQ file upload.

    This endpoint handles the full pipeline:
    1. Reads the uploaded genomic sequence file
    2. Extracts k-mer features using the trained vocabulary
    3. Runs the prediction model
    4. Returns detailed results with clinical recommendations

    Accepted formats: .fasta, .fa, .fna, .fastq, .fq (optionally gzipped)
    Max file size: 50 MB
    """
    import gzip

    try:
        # Validate file extension
        _ALLOWED_EXTENSIONS = {".fasta", ".fa", ".fna", ".fastq", ".fq", ".gz"}
        filename = file.filename or "unknown"
        suffixes = Path(filename).suffixes
        if not any(s.lower() in _ALLOWED_EXTENSIONS for s in suffixes):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            )

        # Read file content with size limit (50 MB)
        _MAX_FILE_SIZE = 50 * 1024 * 1024
        raw_content = await file.read()
        if len(raw_content) > _MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50 MB.")

        # Decompress if gzipped
        if filename.endswith('.gz'):
            try:
                raw_content = gzip.decompress(raw_content)
                filename = filename[:-3]
            except Exception:
                raise HTTPException(status_code=400, detail="Failed to decompress gzip file")

        try:
            content = raw_content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File is not valid text/sequence data")

        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Determine format
        file_format = "fastq" if filename.endswith(('.fastq', '.fq')) else "fasta"

        # Extract k-mer features
        try:
            extractor = get_extractor()
            features, headers = extractor.extract_from_file_content(content, file_format)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Feature extraction failed: {e}")

        feature_vector = features[0]

        # Check features are not all zeros (sequence too short or invalid)
        if np.sum(feature_vector) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid k-mer features extracted. Check that the file contains valid DNA sequences (A, C, G, T)."
            )

        # Run prediction
        predictor = get_model(model_type)
        result = predictor.predict(feature_vector, threshold=threshold)

        # Build detailed response
        drug_predictions = []
        probs = result.get("probabilities", {})

        for drug, resistant in result["predictions"].items():
            prob = probs.get(drug, 0.5)
            drug_predictions.append(DrugPrediction(
                drug_class=drug,
                resistant=resistant,
                probability=prob,
                confidence=get_confidence(prob),
            ))

        drug_predictions.sort(key=lambda x: x.probability, reverse=True)

        risk_level, risk_description = get_risk_assessment(
            result["resistant_count"],
            len(predictor.drug_classes),
        )

        response_data = {
            "summary": {
                "resistant_count": result["resistant_count"],
                "susceptible_count": result["susceptible_count"],
                "risk_level": risk_level,
                "risk_description": risk_description,
            },
            "drug_predictions": [p.model_dump() for p in drug_predictions],
            "recommendations": get_clinical_recommendations(drug_predictions),
            "sequence_info": {
                "filename": file.filename,
                "format": file_format,
                "sequences_processed": len(headers),
                "sequence_headers": headers[:5],
                "n_features_extracted": int(np.count_nonzero(feature_vector)),
            },
            "model_type": model_type,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat(),
        }

        # Save prediction to database
        try:
            user_id = current_user["id"] if current_user else None
            user_name = current_user["name"] if current_user else "Anonymous"
            sample_id = f"{organism[:2].upper()}-{datetime.now().strftime('%Y')}-{__import__('uuid').uuid4().hex[:6].upper()}"
            # Store in frontend AntibioticResult format
            frontend_results = [
                {
                    "antibiotic": DRUG_CLASS_DISPLAY.get(p.drug_class, p.drug_class),
                    "class": p.drug_class,
                    "status": "R" if p.resistant else "S",
                    "confidence": p.probability,
                }
                for p in drug_predictions
            ]
            results_json = json.dumps({
                "results": frontend_results,
                "summary": {
                    "resistant": result["resistant_count"],
                    "intermediate": 0,
                    "susceptible": result["susceptible_count"],
                },
            })
            saved = save_prediction(
                sample_id=sample_id,
                user_id=user_id,
                organism=organism,
                status="completed",
                risk_level=risk_level.lower(),
                file_name=file.filename,
                file_size=len(raw_content),
                results_json=results_json,
                model_version=MODEL_VERSION,
            )
            response_data["prediction_id"] = saved["id"]
            response_data["sample_id"] = saved["sampleId"]
            log_activity(user_id, user_name, "Uploaded sample", file.filename)
        except Exception as db_err:
            logger.warning(f"Failed to save prediction to DB: {db_err}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FASTA prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_clinical_recommendations(predictions: List[DrugPrediction]) -> List[str]:
    """Generate clinical recommendations based on predictions."""
    recommendations = []

    resistant_drugs = [p for p in predictions if p.resistant]
    susceptible_drugs = [p for p in predictions if not p.resistant]

    if len(resistant_drugs) >= 5:
        recommendations.append(
            "URGENT: Multi-drug resistance detected. Recommend immediate infectious disease consultation."
        )
        recommendations.append(
            "Consider combination therapy or reserve antibiotics."
        )

    # Specific recommendations
    resistant_classes = {p.drug_class for p in resistant_drugs}

    if "beta-lactam" in resistant_classes:
        recommendations.append(
            "Beta-lactam resistance detected. Consider carbapenem or non-beta-lactam alternatives."
        )

    if "quinolone" in resistant_classes:
        recommendations.append(
            "Fluoroquinolone resistance detected. Avoid ciprofloxacin/levofloxacin."
        )

    if "aminoglycoside" in resistant_classes and "beta-lactam" in resistant_classes:
        recommendations.append(
            "Combined aminoglycoside and beta-lactam resistance. Consider colistin or tigecycline."
        )

    # Suggest susceptible options
    high_conf_susceptible = [
        p.drug_class for p in susceptible_drugs
        if p.confidence == "high"
    ]
    if high_conf_susceptible:
        recommendations.append(
            f"High-confidence susceptibility predicted for: {', '.join(high_conf_susceptible[:3])}"
        )

    if not recommendations:
        recommendations.append(
            "Standard antibiotic therapy likely effective. Monitor treatment response."
        )

    # Add Bangladesh-specific recommendations
    recommendations.extend(get_bangladesh_recommendations(resistant_classes))

    return recommendations


# =============================================================================
# Model Performance Endpoint
# =============================================================================

@app.get("/models/performance", tags=["Models"])
async def get_model_performance():
    """Get model accuracy metrics and per-class performance."""
    try:
        predictor = get_model("deep_learning")
        info = predictor.model_info
        performance = info.get("performance", {})

        per_class = {}
        if predictor.optimal_thresholds:
            for drug, data in predictor.optimal_thresholds.items():
                per_class[drug] = {
                    "optimal_threshold": data.get("threshold", 0.5),
                    "f1_score": data.get("f1", None),
                }

        return {
            "model_version": MODEL_VERSION,
            "overall": {
                "micro_f1": performance.get("micro_f1", 0.843),
                "macro_f1": performance.get("macro_f1", 0.700),
                "auc": performance.get("micro_auc", 0.986),
                "hamming_loss": performance.get("hamming_loss", 0.044),
            },
            "per_class": per_class,
            "drug_classes": info["drug_classes"],
            "has_optimal_thresholds": info.get("has_optimal_thresholds", False),
        }
    except Exception as e:
        # Return hardcoded metrics even if model not loaded
        return {
            "model_version": MODEL_VERSION,
            "overall": {
                "micro_f1": 0.843,
                "macro_f1": 0.700,
                "auc": 0.986,
                "hamming_loss": 0.044,
            },
            "per_class": {},
            "drug_classes": DeepAMRPredictor.DEFAULT_DRUG_CLASSES,
            "has_optimal_thresholds": False,
        }


# =============================================================================
# PDF Report Endpoint
# =============================================================================

@app.get("/predictions/{pred_id}/report", tags=["Reports"])
async def download_prediction_report(pred_id: str):
    """Download a PDF clinical report for a prediction."""
    pred = get_prediction(pred_id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Augment with recommendations if not present
    if not pred.get("recommendations") and pred.get("results"):
        drug_preds = [
            DrugPrediction(
                drug_class=r.get("class", ""),
                resistant=r.get("status") == "R",
                probability=r.get("confidence", 0.5),
                confidence=get_confidence(r.get("confidence", 0.5)),
            )
            for r in pred["results"]
        ]
        pred["recommendations"] = get_clinical_recommendations(drug_preds)
        resistant_classes = {r.get("class") for r in pred["results"] if r.get("status") == "R"}
        pred["bangladesh_recommendations"] = get_bangladesh_recommendations(resistant_classes)

    pdf_bytes = generate_prediction_report(pred)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="DeepAMR_Report_{pred_id}.pdf"'},
    )


# =============================================================================
# Bangladesh Guidelines Endpoint
# =============================================================================

@app.get("/guidelines/bangladesh", tags=["Information"])
async def get_bangladesh_guidelines():
    """Get Bangladesh-specific AMR guidelines and resistance data."""
    return {
        "resistance_prevalence": BANGLADESH_RESISTANCE_PREVALENCE,
        "referral_centers": REFERRAL_CENTERS,
    }


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and pre-load models on startup."""
    logger.info("DeepAMR API starting up...")

    # Initialize SQLite database
    init_db()
    logger.info("Database initialized")

    # Mount frontend static files if available (for standalone HTML builds)
    frontend_path = PROJECT_ROOT / "frontend"
    for static_dir in ["css", "js", "assets"]:
        dir_path = frontend_path / static_dir
        if dir_path.is_dir():
            app.mount(f"/{static_dir}", StaticFiles(directory=dir_path), name=static_dir)
            logger.info(f"Mounted static: /{static_dir}")
    # Note: Next.js frontend runs separately (npm run dev on port 3000)

    try:
        # Attempt to preload deep learning model
        get_model("deep_learning")
        logger.info("Deep learning model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
