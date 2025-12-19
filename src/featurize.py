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

    # Apply vocab
    df["user_idx"] = apply_vocab(df["user_id"], user_vocab)
    df["item_idx"] = apply_vocab(df["item_id"], item_vocab)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    split = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split]
    eval_df = df.iloc[split:]

    return train_df, eval_df, user_vocab, item_vocab
