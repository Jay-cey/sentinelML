"""
Sentinel ML — FastAPI Backend
Endpoints:
  POST /predict  — score a single transaction
  GET  /drift    — PSI scores for all batch windows
  POST /retrain  — refit model on the latest batch, log to registry
"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Paths (relative to the project root — run uvicorn from project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "isolation_forest_v1.joblib"
DATA_DIR = ROOT / "data"
REGISTRY_PATH = ROOT / "registry.csv"
BASELINE_BATCH = DATA_DIR / "batch_01.csv"
CONTAMINATION = 0.0017

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentinel ML",
    description="Fraud detection + PSI drift monitoring API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model — loaded ONCE at startup
# ---------------------------------------------------------------------------
model: IsolationForest | None = None
baseline_scores: np.ndarray | None = None


@app.on_event("startup")
def load_model() -> None:
    global model, baseline_scores

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run scripts/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    # Pre-compute baseline anomaly scores for drift comparisons
    df_base = pd.read_csv(BASELINE_BATCH)
    X_base = df_base.drop(columns=["Class"], errors="ignore")
    baseline_scores = model.decision_function(X_base)

    print(f"[startup] Model loaded from {MODEL_PATH}")
    print(f"[startup] Baseline scores computed over {len(baseline_scores):,} rows")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two score distributions."""
    breakpoints = np.percentile(expected, np.arange(0, 101, 100 / buckets))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def append_registry(row: dict[str, Any]) -> None:
    """Append a training run record to registry.csv."""
    write_header = not REGISTRY_PATH.exists()
    with open(REGISTRY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionIn(BaseModel):
    """A single credit-card transaction.
    
    Expects the 30 numerical features of the Kaggle dataset:
    Time, V1–V28, Amount (no Class label).
    """
    features: dict[str, float]


class PredictOut(BaseModel):
    anomaly_score: float
    is_anomaly: bool


class DriftWindow(BaseModel):
    batch: int
    batch_file: str
    psi: float
    status: str
    retrain_recommended: bool


class RetrainOut(BaseModel):
    message: str
    model_path: str
    trained_on: str
    f1: float
    precision: float
    recall: float
    logged_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictOut, summary="Score a single transaction")
def predict(transaction: TransactionIn) -> PredictOut:
    """
    Accept a transaction dict under `features` and return:
    - `anomaly_score`: raw decision-function score (lower = more anomalous)
    - `is_anomaly`: True if the model flags this transaction as fraud
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        X = pd.DataFrame([transaction.features])
        score = float(model.decision_function(X)[0])
        label = int(model.predict(X)[0])  # -1 = anomaly, 1 = normal
        is_anomaly = label == -1
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    return PredictOut(anomaly_score=score, is_anomaly=is_anomaly)


@app.get("/drift", response_model=list[DriftWindow], summary="PSI scores per batch window")
def drift() -> list[DriftWindow]:
    """
    Compute PSI between Batch 01 (training baseline) and each subsequent
    batch file. Returns a list ordered by batch number.

    PSI interpretation:
      < 0.1  → No significant drift
      0.1–0.2 → Moderate drift (monitor)
      > 0.2  → Significant drift (retrain recommended)
    """
    if model is None or baseline_scores is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    results: list[DriftWindow] = []

    for i in range(2, 7):
        batch_file = DATA_DIR / f"batch_{i:02d}.csv"
        if not batch_file.exists():
            continue

        df_batch = pd.read_csv(batch_file)
        X_batch = df_batch.drop(columns=["Class"], errors="ignore")
        batch_scores = model.decision_function(X_batch)

        psi = calculate_psi(baseline_scores, batch_scores)
        retrain = psi > 0.2
        status = "DRIFT DETECTED" if retrain else "Stable"

        results.append(
            DriftWindow(
                batch=i,
                batch_file=batch_file.name,
                psi=round(psi, 6),
                status=status,
                retrain_recommended=retrain,
            )
        )

    return results


@app.post("/retrain", response_model=RetrainOut, summary="Refit model on the latest batch")
def retrain() -> RetrainOut:
    """
    Find the latest available batch file, refit the Isolation Forest,
    overwrite the saved model, and log the run to registry.csv.
    """
    global model, baseline_scores

    # Find the latest batch
    batch_files = sorted(DATA_DIR.glob("batch_*.csv"))
    if not batch_files:
        raise HTTPException(status_code=404, detail="No batch files found in data/")

    latest_batch = batch_files[-1]

    try:
        df = pd.read_csv(latest_batch)
        X = df.drop(columns=["Class"], errors="ignore")
        y_true = df["Class"] if "Class" in df.columns else None

        new_model = IsolationForest(contamination=CONTAMINATION, random_state=42)
        new_model.fit(X)

        # Evaluate if labels available
        if y_true is not None:
            y_pred = [1 if p == -1 else 0 for p in new_model.predict(X)]
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary"
            )
        else:
            precision = recall = f1 = float("nan")

        # Persist model
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)

        # Hot-swap in memory
        model = new_model
        df_base = pd.read_csv(BASELINE_BATCH)
        X_base = df_base.drop(columns=["Class"], errors="ignore")
        baseline_scores = model.decision_function(X_base)

        logged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to registry
        append_registry(
            {
                "date": logged_at,
                "trained_on": latest_batch.name,
                "f1": round(f1, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "model_path": str(MODEL_PATH.relative_to(ROOT)),
            }
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}")

    return RetrainOut(
        message=f"Model retrained on {latest_batch.name}",
        model_path=str(MODEL_PATH.relative_to(ROOT)),
        trained_on=latest_batch.name,
        f1=round(f1, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        logged_at=logged_at,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_loaded": str(model is not None),
    }
