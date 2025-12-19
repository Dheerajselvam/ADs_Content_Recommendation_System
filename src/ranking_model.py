import joblib
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(exist_ok=True)


def train_ranking_model(
    train_df,
    feature_cols,
    label_col="clicked",
    model_type="lr",
):
    """
    Train a ranking model on candidate-level features.

    model_type:
        - "lr"   : Logistic Regression (baseline)
        - "gbdt" : Gradient Boosted Decision Trees
        - "nn"   : Neural Network (MLP)
    """

    X = train_df[feature_cols].values
    y = train_df[label_col].values

    if model_type == "lr":
        model = LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight="balanced"
        )

    elif model_type == "gbdt":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8
        )

    elif model_type == "nn":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=20,
            batch_size=1024,
            early_stopping=True,
            validation_fraction=0.1
        )

    else:
        raise ValueError(
            f"Unknown ranking model_type: {model_type}. "
            f"Choose from ['lr', 'gbdt', 'nn']"
        )

    print(f"🚀 Training ranking model: {model_type}")
    model.fit(X, y)

    model_path = f"{MODEL_DIR}/ranking_{model_type}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Ranking model saved at {model_path}")

    return model
