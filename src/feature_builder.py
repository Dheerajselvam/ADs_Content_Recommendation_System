import numpy as np
import pandas as pd

def build_vocab(series):
    """Map unique values to integer IDs."""
    uniq = series.unique().tolist()
    return {v: i for i, v in enumerate(uniq)}

def apply_vocab(series, vocab):
    """Convert raw IDs to integer indices."""
    return series.map(lambda x: vocab.get(x, -1)).astype(int)

def featurize_and_split(df, test_frac=0.2, seed=42):
    # Build vocabularies
    user_vocab = build_vocab(df["user_id"])
    item_vocab = build_vocab(df["item_id"])
    cat_vocab = build_vocab(df["category"])

    # Apply vocab
    df["user_idx"] = apply_vocab(df["user_id"], user_vocab)
    df["item_idx"] = apply_vocab(df["item_id"], item_vocab)
    df["cat_idx"] = apply_vocab(df["category"], cat_vocab)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split]
    eval_df = df.iloc[split:]

    return train_df, eval_df, user_vocab, item_vocab

def build_stats(df):
    # Item popularity & CTR

    item_stats = df.groupby("item_idx").agg(
        item_clicks=("clicked", "sum"),
        item_impr=("clicked", "count")
    )
    item_stats["item_ctr"] = item_stats["item_clicks"] / item_stats["item_impr"]

    user_stats = df.groupby("user_idx").agg(
        user_clicks=("clicked", "sum"),
        user_impr=("clicked", "count")
    )
    user_stats["user_ctr"] = user_stats["user_clicks"] / user_stats["user_impr"]

    cat_stats = df.groupby("category").agg(
        cat_clicks=("clicked", "sum"),
        cat_impr=("clicked", "count")
    )
    cat_stats["cat_ctr"] = cat_stats["cat_clicks"] / cat_stats["cat_impr"]

    return item_stats, user_stats, cat_stats


def build_ranking_features(row, model, item_stats, user_stats, cat_stats):
    u = model["user_embeddings"][row.user_idx]
    i = model["item_embeddings"][row.item_idx]

    # --- Base CF signals ---
    dot = float(np.dot(u, i))

    # --- CTR stats ---
    item_ctr = item_stats.loc[row.item_idx]["item_ctr"] if row.item_idx in item_stats.index else 0.0
    user_ctr = user_stats.loc[row.user_idx]["user_ctr"] if row.user_idx in user_stats.index else 0.0
    cat_ctr = cat_stats.loc[row.category]["cat_ctr"] if row.category in cat_stats.index else 0.0

    # --- Engagement proxy ---
    engagement_score = (
        0.6 * cat_ctr +
        0.4 * np.log1p(item_stats.loc[row.item_idx]["item_impr"])
        if row.item_idx in item_stats.index else 0.0
    )

    # --- Monetization proxy ---
    monetization_score = (
        item_stats.loc[row.item_idx]["item_value"]
        if "item_value" in item_stats.columns else 0.0
    )

    # --- Freshness proxy ---
    recency = np.exp(-(pd.Timestamp.now().timestamp() - row.timestamp) / 86400)

    return {
            "dot": dot,
            "u_norm": float(np.linalg.norm(u)),
            "i_norm": float(np.linalg.norm(i)),
            "item_ctr": item_ctr,
            "user_ctr": user_ctr,
            "cat_ctr": cat_ctr,
            "engagement": engagement_score,
            "monetization": monetization_score,
            "freshness": recency,
            "clicked": row.clicked,
            "user_idx": row.user_idx,
            "item_idx": row.item_idx,
        }


