import pandas as pd
from feature_builder import featurize_and_split

RAW = "data/raw/interactions.csv"
OUT_TRAIN = "data/processed/train.csv"
OUT_EVAL = "data/processed/eval.csv"

df = pd.read_csv(RAW)
train_df, eval_df, user_vocab, item_vocab = featurize_and_split(df)

train_df.to_csv(OUT_TRAIN, index=False)
eval_df.to_csv(OUT_EVAL, index=False)

print("Saved processed datasets")
