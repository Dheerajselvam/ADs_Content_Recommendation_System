import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, log_loss

def evaluate_candidate_model(
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


def recall_at_k(y_true, y_score, k):
    idx = np.argsort(-y_score)[:k]
    return int(np.any(y_true[idx] == 1))

def ndcg_at_k(y_true, y_score, k):
    idx = np.argsort(-y_score)[:k]
    gains = y_true[idx] / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal = np.sort(y_true)[::-1][:k]
    ideal_dcg = (ideal / np.log2(np.arange(2, k + 2))).sum()
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_ranking(df, score_col="score", label_col="clicked", k=10):
    results = {"Recall@K": [], "NDCG@K": []}

    for _, g in df.groupby("user_idx"):
        y = g[label_col].values
        s = g[score_col].values
        results["Recall@K"].append(recall_at_k(y, s, k))
        results["NDCG@K"].append(ndcg_at_k(y, s, k))

    return {k: float(np.mean(v)) for k, v in results.items()}
