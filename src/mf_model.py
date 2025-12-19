import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib
from pathlib import Path

MODEL_OUT = "models/mf_model.pkl"

def train_mf(train_df, n_factors=32):
    """
    Matrix Factorization using SVD on user-item interaction matrix.
    """
    user_item = train_df.pivot_table(
        index="user_idx",
        columns="item_idx",
        values="clicked",
        fill_value=0
    )

    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_emb = svd.fit_transform(user_item)
    item_emb = svd.components_.T

    model = {
        "user_embeddings": user_emb,
        "item_embeddings": item_emb
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print("✅ MF model saved:", MODEL_OUT)
    return {
    "mf_model": model,
    "user_embeddings": user_emb,
    "item_embeddings": item_emb
    }

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")
    train_mf(df)
