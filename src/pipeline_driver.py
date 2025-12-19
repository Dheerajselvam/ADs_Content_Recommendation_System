import pandas as pd
import joblib
from pathlib import Path
import numpy as np

from feature_builder import featurize_and_split, build_ranking_features
from mf_model import train_mf
from candidate_gen import generate_candidates
from ranking_model import train_ranking_model
from eval_utils import evaluate_ranking
from ann_index import build_ann

# -------------------------
# Config / Paths
# -------------------------
RAW_DATA_PATH = "data/raw/interactions.csv"
PROCESSED_TRAIN = "data/processed/train.csv"
PROCESSED_EVAL = "data/processed/eval.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

Path(MODELS_DIR).mkdir(exist_ok=True)
Path(REPORTS_DIR).mkdir(exist_ok=True)

# -------------------------
# Step 1: Load raw data
# -------------------------
df = pd.read_csv(RAW_DATA_PATH)
print(f"Loaded raw data: {df.shape[0]} rows")

# -------------------------
# Step 2: Featurize & train/eval split
# -------------------------
train_df, eval_df, user_vocab, item_vocab = featurize_and_split(df)
train_df.to_csv(PROCESSED_TRAIN, index=False)
eval_df.to_csv(PROCESSED_EVAL, index=False)
print(f"Train/Eval split: {len(train_df)} / {len(eval_df)} rows")

# -------------------------
# Step 3: Train MF model
# -------------------------
mf_model_path = f"{MODELS_DIR}/mf_model.pkl"
print("Training MF model...")
mf_model = train_mf(train_df)  # returns dictionary with embeddings
joblib.dump(mf_model, mf_model_path)
print(f"MF model saved at {mf_model_path}")

# -------------------------
# Step 4: Build ANN on item embeddings
# -------------------------
item_emb = mf_model["item_embeddings"]
ann_index = build_ann(item_emb)
print("ANN index built on item embeddings.")

# -------------------------
# Step 5: Candidate Generation
# -------------------------
user_ids = eval_df["user_idx"].unique()
candidates_list = []

for user_id in user_ids:
    candidate_items = generate_candidates(
        user_index=user_id,
        mf_model=mf_model,
        ann_index=ann_index,
        top_k=50
    )
    # attach clicked label from eval_df if exists
    user_eval = eval_df[eval_df["user_idx"] == user_id][["item_idx", "clicked"]]
    candidate_items = pd.DataFrame({"user_idx": user_id, "item_idx": candidate_items})
    candidate_items = candidate_items.merge(user_eval, on="item_idx", how="left").fillna(0)
    candidates_list.append(candidate_items)

candidates_eval = pd.concat(candidates_list, ignore_index=True)
print(f"Candidate generation complete: {len(candidates_eval)} pairs.")

# -------------------------
# Step 6: Build ranking features
# -------------------------
ranking_features = []
for row in candidates_eval.itertuples():
    feats = build_ranking_features(
        user_idx=row.user_idx,
        item_idx=row.item_idx,
        mf_model=mf_model
    )
    ranking_features.append({
        "user_idx": row.user_idx,
        "item_idx": row.item_idx,
        "clicked": row.clicked,
        **feats
    })
ranking_df = pd.DataFrame(ranking_features)
print(f"Ranking features constructed: {ranking_df.shape}")

# -------------------------
# Step 7: Train ranking model
# -------------------------
ranking_model_path = f"{MODELS_DIR}/ranking_model.pkl"
print("Training ranking model...")
ranking_model = train_ranking_model(
    ranking_df,
    feature_cols=["dot", "u_norm", "i_norm"],
)
joblib.dump(ranking_model, ranking_model_path)
print(f"Ranking model saved at {ranking_model_path}")

# -------------------------
# Step 8: Evaluate ranking model
# -------------------------
print("Evaluating ranking model...")
ranking_scores = ranking_model.predict(ranking_df[["dot", "u_norm", "i_norm"]])
ranking_df["score"] = ranking_scores

metrics = evaluate_ranking(
    ranking_df,
    score_col="score",
    label_col="clicked",
    k=10
)

print("✅ Evaluation metrics:")
print(metrics)
