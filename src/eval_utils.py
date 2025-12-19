import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss

def evaluate_model(
    model_path,
    eval_csv,
    model_name,
    out_dir="reports"
):
    df = pd.read_csv(eval_csv)
    model = joblib.load(model_path)

    X = df[["user_idx", "item_idx"]].values
    y = df["clicked"].values

    probs = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "auc": roc_auc_score(y, probs),
        "log_loss": log_loss(y, probs),
        "ctr": float(y.mean())
    }

    out_path = f"{out_dir}/{model_name}_offline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation → {out_path}")
    return metrics
