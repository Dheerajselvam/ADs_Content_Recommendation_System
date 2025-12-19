import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from eval_utils import evaluate_model

TRAIN = "data/processed/train.csv"
MODEL_OUT = "models/baseline_lr.pkl"

def train():
    df = pd.read_csv(TRAIN)

    X = df[["user_idx", "item_idx"]].values
    y = df["clicked"].values

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    joblib.dump(model, MODEL_OUT)
    print("Saved baseline model")

if __name__ == "__main__":
    train()
    eval = evaluate_model(
    model_path=MODEL_OUT,
    eval_csv="data/processed/eval.csv",
    model_name="baseline_lr"
    )
    print(eval)

