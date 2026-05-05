import pandas as pd
import numpy as np
import joblib
import os

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate the PSI (Population Stability Index) between two distributions.
    """
    # Define breakpoints based on deciles of the expected distribution
    breakpoints = np.percentile(expected, np.arange(0, 101, 100 / buckets))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate percentages of samples in each bucket
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Clip to avoid division by zero or log(0)
    expected_percents = np.clip(expected_percents, 1e-6, None)
    actual_percents = np.clip(actual_percents, 1e-6, None)
    
    # PSI Formula: sum((actual - expected) * ln(actual / expected))
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value

def detect_drift():
    # 1. Load model and baseline data
    model_path = 'models/isolation_forest_v1.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Run training first.")
        return

    model = joblib.load(model_path)
    
    print("Calculating baseline scores (Batch 01)...")
    df_baseline = pd.read_csv('data/batch_01.csv')
    X_baseline = df_baseline.drop('Class', axis=1)
    baseline_scores = model.decision_function(X_baseline)
    
    # 2. Iterate through subsequent batches
    results = []
    for i in range(2, 7):
        batch_file = f'data/batch_{i:02d}.csv'
        print(f"Checking {batch_file} for drift...")
        
        df_batch = pd.read_csv(batch_file)
        X_batch = df_batch.drop('Class', axis=1)
        batch_scores = model.decision_function(X_batch)
        
        psi = calculate_psi(baseline_scores, batch_scores)
        status = "DRIFT DETECTED (Retrain Recommended)" if psi > 0.2 else "Stable"
        
        print(f"Batch {i:02d} PSI: {psi:.4f} - Status: {status}")
        results.append({'batch': i, 'psi': psi, 'status': status})
    
    return results

if __name__ == "__main__":
    detect_drift()
