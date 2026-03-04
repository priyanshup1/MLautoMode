from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import uuid
import json
import base64
import joblib
import os
import pandas as pd
import numpy as np

import data_processor
import feature_engineer
import preprocessor
import model_trainer
import visualizer
import insight_generator
import drift_detector

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
    pro_mode: bool = False


class PreprocessRequest(BaseModel):
    scale_features: bool = True
    scaler_type: str = "standard"
    imputation_strategy: str = "median"
    encoding_strategy: str = "onehot"
    polynomial_degree: int = 1
    feature_selection_k: int = 0
    handle_outliers: bool = False
    pro_mode: bool = False


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

    # Run advanced feature engineering
    try:
        problem_type = session.get("problem_type", "Classification")
        preprocess_output = feature_engineer.engineer_features(
            raw_df,
            target_col,
            problem_type,
            imputation_strategy=body.imputation_strategy,
            encoding_strategy=body.encoding_strategy if body.pro_mode else "onehot",
            scale_features=body.scale_features,
            scaler_type=body.scaler_type,
            polynomial_degree=body.polynomial_degree if body.pro_mode else 1,
            feature_selection_k=body.feature_selection_k if body.pro_mode else 0,
            handle_outliers=body.handle_outliers if body.pro_mode else False
        )

        session["cleaned_df"] = preprocess_output["cleaned_df"]
        session["step_log"] = preprocess_output["step_log"]
        session["stats"] = preprocess_output["stats"]
        session["cleaned_csv_b64"] = preprocess_output["cleaned_csv_b64"]

        # Sample preview
        preview_df = session["cleaned_df"].head(10).replace({np.nan: None, np.inf: None, -np.inf: None})
        preview_data = json.loads(preview_df.to_json(orient="records"))

        return {
            "message": "Dataset cleaned successfully",
            "stats": session["stats"],
            "step_log": session["step_log"],
            "preview": preview_data,
            "cleaned_csv_b64": preprocess_output["cleaned_csv_b64"],
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
        pro_mode = body.pro_mode if body else False
        training_output = model_trainer.train_and_evaluate(df, target_col, problem_type, selected_models, pro_mode)
        session["results"] = training_output["results"]
        session["best_model"] = training_output["best_model"]

        # Save model
        best_model_data = session["results"][0] if session["results"] else None
        if best_model_data and "model" in best_model_data:
            model_dir = f"models/{session_id}"
            os.makedirs(model_dir, exist_ok=True)
            # version tracking
            version = session.get("model_version", 1)
            model_path = f"{model_dir}/v{version}.pkl"
            joblib.dump(best_model_data["model"], model_path)
            session["model_version"] = version

            # Remove model from results before returning as it's not JSON serializable
            for res in session["results"]:
                res.pop("model", None)
        
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

# ── 5. PREDICT ──────────────────────────────────────────────────────────────────
@app.post("/api/predict/{session_id}")
async def predict_dataset(session_id: str, file: UploadFile = File(...), pro_mode: bool = Form(False)):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = store[session_id]
    if "best_model" not in session or "cleaned_df" not in session:
         raise HTTPException(status_code=400, detail="Models must be trained before predicting")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        new_df = data_processor.load_dataset(content)
        version = session.get("model_version", 1)
        model_path = f"models/{session_id}/v{version}.pkl"
        
        if not os.path.exists(model_path):
             raise HTTPException(status_code=404, detail="Saved model not found for this session")
             
        model = joblib.load(model_path)
        
        # We need to preprocess the new data exactly as we did the training data
        # For simplicity in this iteration, we do basic cleaning matching training pipeline
        # (A robust deployment uses sklearn Pipelines, but we adapt here)
        
        # Get expected features
        X_train = session["cleaned_df"].drop(columns=[session["target_col"]])
        expected_features = X_train.columns.tolist()
        
        # Align new data to expected features (fill missing with median, encode)
        # 1. Base imputation
        for col in new_df.columns:
             if pd.api.types.is_numeric_dtype(new_df[col]):
                 new_df[col] = new_df[col].fillna(new_df[col].median() if not pd.isna(new_df[col].median()) else 0)
             else:
                 new_df[col] = new_df[col].fillna(new_df[col].mode()[0] if not new_df[col].empty else "")
                 
        # 2. Encoding (one-hot)
        cat_cols = new_df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
             new_df = pd.get_dummies(new_df, columns=cat_cols, drop_first=True)
             
        # 3. Reindex to guarantee exact columns
        new_df = new_df.reindex(columns=expected_features, fill_value=0)
        
        # 4. In a real scenario we'd re-apply the EXACT same scaler instance here
        
        predictions = model.predict(new_df)
        
        # Optional: confidence score for classification
        confidences = None
        if session["problem_type"] == "Classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(new_df)
            confidences = [round(float(max(p)), 4) for p in proba]
            
        results_list = []
        for i, pred in enumerate(predictions):
            row_dict = {"prediction": float(pred) if pd.api.types.is_numeric_dtype(pd.Series(pred)) else str(pred)}
            if confidences:
                row_dict["confidence"] = confidences[i]
            results_list.append(row_dict)
            
        drift_report = None
        if pro_mode and "raw_df" in session:
           drift_report = drift_detector.detect_data_drift(session["raw_df"], new_df)
            
        return {
            "message": "Predictions generated successfully",
            "model_used": session["best_model"],
            "version": version,
            "predictions": results_list,
            "drift_report": drift_report
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── 6. CONTINUAL LEARNING ────────────────────────────────────────────────────────
@app.post("/api/retrain/{session_id}")
async def retrain_model(session_id: str, file: UploadFile = File(...)):
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = store[session_id]
    if "raw_df" not in session:
        raise HTTPException(status_code=400, detail="Original dataset not found in session")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    try:
        new_df = data_processor.load_dataset(content)
        
        # 1. Combine data
        combined_df = pd.concat([session["raw_df"], new_df], ignore_index=True)
        session["raw_df"] = combined_df # Update session data
        target_col = session["target_col"]
        problem_type = session["problem_type"]
        
        # 2. Re-preprocess
        # We'll use standard scaling for simplicity in retrain here
        prep_result = preprocessor.preprocess_data(combined_df, target_col, scale_features=True, scaler_type="standard")
        session["cleaned_df"] = prep_result["cleaned_df"]
        
        # 3. Re-train models
        # We reuse the previously selected models if recorded, else default
        models_to_train = None
        if "results" in session and session["results"]:
             models_to_train = [r["model_name"] for r in session["results"]]
             
        training_output = model_trainer.train_and_evaluate(session["cleaned_df"], target_col, problem_type, models_to_train)
        
        session["results"] = training_output["results"]
        session["best_model"] = training_output["best_model"]
        
        # 4. Save new model version
        best_model_data = session["results"][0] if session["results"] else None
        version = session.get("model_version", 0) + 1 # Increment version
        session["model_version"] = version
        
        if best_model_data and "model" in best_model_data:
            model_dir = f"models/{session_id}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/v{version}.pkl"
            joblib.dump(best_model_data["model"], model_path)
            
            # Remove model object from results to send JSON back
            for res in session["results"]:
                res.pop("model", None)
                
        # 5. Generate new insights
        charts = visualizer.generate_visualizations(session["raw_df"], target_col, problem_type)
        insight = (
            insight_generator.generate_insight(problem_type, target_col, best_model_data)
            if best_model_data else {}
        )
        
        return {
            "message": f"Successfully retrained models with new data. Saved as version v{version}.",
            "version": version,
            "new_dataset_size": len(combined_df),
            "results": session["results"],
            "feature_names": training_output["feature_names"],
            "charts": charts,
            "insight": insight,
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
