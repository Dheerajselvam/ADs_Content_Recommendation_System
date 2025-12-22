import pandas as pd
import joblib
from pathlib import Path
import numpy as np

from feature_builder import featurize_and_split, build_ranking_features, build_stats, create_ranking_df
from mf_model import train_mf
from candidate_gen import generate_candidates
from ranking_model import train_ranking_model
from eval_utils import evaluate_ranking
from ann_index import build_ann
from ncf_model import train_ncf
from rnn_model import train_rnn
from transformer_model import train_transformer
from cold_start import is_new_user, cold_start_candidates
from bandit import EpsilonGreedy
from online_simulator import simulate_user_feedback
from hybrid_embeddings import build_hybrid_embeddings
from online_loop import run_online_loop
pd.set_option('future.no_silent_downcasting', True)


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

def run(cf_type = "RNN", ranker_type = "lr"):
    # -------------------------
    # Step 3: Train CF model
    # -------------------------
    if cf_type == "NCF":
        ncf_model_path = f"{MODELS_DIR}/ncf_model.pkl"
        print("Training NCF model...")
        model = train_ncf(train_df)  # returns dictionary with embeddings
        joblib.dump(model, ncf_model_path)
        print(f"NCF model saved at {ncf_model_path}")
        helper(model, ranker_type)
    elif cf_type == "RNN":
        rnn_model_path = f"{MODELS_DIR}/rnn_model.pkl"
        print("Training RNN model...")
        model = train_rnn(train_df)  # returns dictionary with embeddings
        #joblib.dump(model, rnn_model_path)
        print(f"RNN model saved at {rnn_model_path}")
        helper(model, ranker_type)
    elif cf_type == "MF":
        mf_model_path = f"{MODELS_DIR}/mf_model.pkl"
        print("Training MF model...")
        model = train_mf(train_df)  # returns dictionary with embeddings
        joblib.dump(model, mf_model_path)
        print(f"MF model saved at {mf_model_path}")
        helper(model, ranker_type)
    elif cf_type == "Transformer":
        tf_model_path = f"{MODELS_DIR}/tf_model.pkl"
        print("Training TransFormer model...")
        model = train_transformer(train_df)  # returns dictionary with embeddings
        joblib.dump(model, tf_model_path)
        print(f"TF model saved at {tf_model_path}")
        helper(model, ranker_type)
    else:
        models = {
                "MF": train_mf(train_df),
                "NCF": train_ncf(train_df),
                "RNN": train_rnn(train_df),
                "Transformer": train_transformer(train_df),
            }
        hybrid_model = build_hybrid_embeddings(models)
        helper(hybrid_model, ranker_type)

    

def helper(model, ranker_type):

    # -------------------------
    # Step 4: Build ANN on item embeddings
    # -------------------------
    item_emb = model["item_embeddings"]
    ann_index = build_ann(item_emb)
    print("ANN index built on item embeddings.")

    # -------------------------
    # Step 5: Candidate Generation
    # -------------------------
    user_ids = eval_df["user_idx"].unique()
    candidates_list = []

    for user_id in user_ids:
        candidate_items_ids = generate_candidates(
            user_index=user_id,
            model=model,
            ann_index=ann_index,
            top_k=50)
        # attach clicked label from eval_df if exists
        user_eval = eval_df[eval_df["user_idx"] == user_id]
        candidate_df = pd.DataFrame({"user_idx": user_id, "item_idx": candidate_items_ids})
        candidate_df = candidate_df.merge(user_eval, on=["user_idx", "item_idx"], how="left")
        candidate_df = candidate_df.fillna(0).infer_objects(copy=False)


        candidate_df["clicked"] = candidate_df["clicked"].fillna(0)
        candidates_list.append(candidate_df)

    candidates_eval = pd.concat(candidates_list, ignore_index=True)
    print(f"Candidate generation complete: {len(candidates_eval)} pairs.")

    # -------------------------
    # Step 6: Build ranking features
    # -------------------------
    ranking_df_train = create_ranking_df(train_df, model, candidates_eval)
    ranking_df_eval = create_ranking_df(eval_df, model, candidates_eval)
    item_stats, user_stats, cat_stats = build_stats(eval_df)
    print(f"Ranking features constructed: {ranking_df_train.shape}")

    # -------------------------
    # Step 7: Train ranking model
    # -------------------------
    ranking_model_path = f"{MODELS_DIR}/ranking_model.pkl"
    print("Training ranking model...")
    ranking_model = train_ranking_model(
        ranking_df_train,
        feature_cols = [
                        "dot",
                        "u_norm",
                        "i_norm",
                        "item_ctr",
                        "cat_ctr",
                        "user_ctr",
                        ],
        model_type = ranker_type
                        )
    joblib.dump(ranking_model, ranking_model_path)
    print(f"Ranking model saved at {ranking_model_path}")

    # -------------------------
    # Step 8: Evaluate ranking model
    # -------------------------
    print("Evaluating ranking model...")
    ranking_scores = ranking_model.predict(ranking_df_eval[[
                        "dot",
                        "u_norm",
                        "i_norm",
                        "item_ctr",
                        "cat_ctr",
                        "user_ctr"
                        ]])
    ranking_df_eval["score"] = ranking_scores

    metrics = evaluate_ranking(
        ranking_df_eval,
        score_col="score",
        label_col="clicked",
        k=10
    )

    print("✅ Evaluation metrics:")
    print(metrics)

    # -------------------------
    # Step 8.5: Multi-objective re-ranking
    # -------------------------

    ALPHA = 0.5   # CTR
    BETA  = 0.2   # Engagement
    GAMMA = 0.2   # Monetization
    DELTA = 0.1   # Freshness

    ranking_df_eval["final_score"] = (
        ALPHA * ranking_df_eval["score"] +
        BETA  * ranking_df_eval["engagement"] +
        GAMMA * ranking_df_eval["monetization"] +
        DELTA * ranking_df_eval["freshness"]
    )


    # -------------------------
    # Step 9: Evaluate Online Ranking model
    # -------------------------

    bandit = EpsilonGreedy(epsilon=0.1)
    online_ctr = []
    for user_id in user_ids[:100]:
        if is_new_user(user_id, user_stats):
            ranked = cold_start_candidates(user_id, item_stats)
        else:
            ranked = (
                ranking_df_eval[ranking_df_eval.user_idx == user_id]
                .sort_values("final_score", ascending=False)
                ["item_idx"]
                .tolist()
            )

        ranked = bandit.rerank(ranked[:10])
        rewards = simulate_user_feedback(ranked, item_stats)
        online_ctr.append(sum(rewards)/len(rewards))

    print("📈 Online CTR:", np.mean(online_ctr))

    # -------------------------
    # Step 10: Online Feedback
    # -------------------------

    print("🚀 Starting Online Loop Simulation...")
    ctr_curve = run_online_loop(
            ranking_df=ranking_df_eval,
            item_stats=item_stats,
            num_steps=1000
        )
    print(ctr_curve)

if __name__ == "__main__":
    run(cf_type="Hybrid", ranker_type="gbdt")