import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path

MODEL_OUT = "models/ranking_lr.pkl"

def train_ranking_model(train_df, feature_cols):
    """
    Train ranking model on candidate-level features.
    """
    X = train_df[feature_cols]
    y = train_df["clicked"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print("✅ Ranking model saved:", MODEL_OUT)
    return model
