from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
import json
import base64

import data_processor
import preprocessor
import model_trainer
import visualizer
import insight_generator

app = FastAPI(title="Mini AutoML API — V2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
store: Dict[str, Any] = {}


class TrainRequest(BaseModel):
    models: Optional[List[str]] = None


class PreprocessRequest(BaseModel):
    scale_features: bool = True
    scaler_type: str = "standard"  # "standard" | "minmax" | "robust"


@app.get("/")
def read_root():
    return {"message": "Mini AutoML API — V2 ready"}


# ── 1. UPLOAD ─────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), target_column: str = Form(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        df = data_processor.load_dataset(content)
        # Ensure the target column actually exists after stripping
        target_column = target_column.strip()
        analysis = data_processor.analyze_dataset(df, target_column)

        session_id = str(uuid.uuid4())
        store[session_id] = {
            "raw_df": df,
            "target_col": target_column,
            "problem_type": analysis["problem_type"],
        }

        return {
            "session_id": session_id,
            "analysis": analysis,
            "message": f"This dataset appears to be a {analysis['problem_type']} problem.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── 2. DEEP ANALYSIS ──────────────────────────────────────────────────────────
@app.post("/api/analyze/{session_id}")
def analyze_dataset(session_id: str):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")

    session = store[session_id]
    df = session["raw_df"]
    target_col = session["target_col"]

    try:
        stats = data_processor.get_full_statistics(df, target_col)
        domain = data_processor.domain_analysis(df, target_col)
        correlations = data_processor.correlation_analysis(df, target_col)
        univariate = data_processor.univariate_data(df)
        bivariate = data_processor.bivariate_data(df, target_col, session["problem_type"])

        return {
            "statistics": stats,
            "domain_analysis": domain,
            "correlations": correlations,
            "univariate": univariate,
            "bivariate": bivariate,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── 3. PREPROCESS ─────────────────────────────────────────────────────────────
@app.post("/api/preprocess/{session_id}")
async def preprocess_dataset(session_id: str, body: PreprocessRequest = None):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")

    session = store[session_id]
    raw_df = session["raw_df"]
    target_col = session["target_col"]

    scale = body.scale_features if body else True
    scaler_type = body.scaler_type if body else "standard"

    try:
        result = preprocessor.preprocess_data(raw_df, target_col, scale, scaler_type)
        session["cleaned_df"] = result["cleaned_df"]
        session["cleaned_csv_b64"] = result["cleaned_csv_b64"]

        preview_data = json.loads(result["cleaned_df"].head(10).to_json(orient="records"))

        return {
            "message": "Dataset cleaned successfully",
            "stats": result["stats"],
            "step_log": result["step_log"],
            "preview": preview_data,
            "cleaned_csv_b64": result["cleaned_csv_b64"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── 4. TRAIN ──────────────────────────────────────────────────────────────────
@app.post("/api/train/{session_id}")
async def train_models(session_id: str, body: TrainRequest = None):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")

    session = store[session_id]
    if "cleaned_df" not in session:
        raise HTTPException(status_code=400, detail="Data must be preprocessed first")

    df = session["cleaned_df"]
    target_col = session["target_col"]
    problem_type = session["problem_type"]

    try:
        selected_models = body.models if body and body.models else None
        training_output = model_trainer.train_and_evaluate(df, target_col, problem_type, selected_models)
        session["results"] = training_output["results"]
        session["best_model"] = training_output["best_model"]

        # Chart data from raw df
        charts = visualizer.generate_visualizations(session["raw_df"], target_col, problem_type)

        # Deep structured insight
        best_model_data = session["results"][0] if session["results"] else None
        insight = (
            insight_generator.generate_insight(problem_type, target_col, best_model_data)
            if best_model_data else {}
        )

        return {
            "message": "Models trained successfully",
            "results": session["results"],
            "feature_names": training_output["feature_names"],
            "charts": charts,
            "insight": insight,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
