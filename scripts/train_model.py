import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
import joblib
import os
from datetime import datetime

def train():
    # 1. Load batch 1
    print("Loading data/batch_01.csv...")
    df = pd.read_csv('data/batch_01.csv')

    # 2. Features and labels
    X = df.drop('Class', axis=1)
    y_true = df['Class']

    # 3. Fit Isolation Forest
    print("Training IsolationForest...")
    model = IsolationForest(contamination=0.0017, random_state=42)
    model.fit(X)

    # 4. Predict
    # IsolationForest returns -1 for outliers, 1 for inliers.
    # Map to 1 for fraud, 0 for normal.
    y_pred = model.predict(X)
    y_pred = [1 if p == -1 else 0 for p in y_pred]

    # 5. Evaluate
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    # 6. Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/isolation_forest_v1.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # 7. Log to registry.csv
    log_data = {
        'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'f1': [f1],
        'precision': [precision],
        'recall': [recall],
        'model_path': [model_path]
    }
    log_df = pd.DataFrame(log_data)

    registry_path = 'registry.csv'
    if not os.path.exists(registry_path):
        log_df.to_csv(registry_path, index=False)
    else:
        log_df.to_csv(registry_path, mode='a', header=False, index=False)
    print(f"Results logged to {registry_path}")

if __name__ == "__main__":
    train()
