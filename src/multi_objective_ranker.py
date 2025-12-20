import numpy as np

def combine_objectives(df, weights):

    score = np.zeros(len(df))
    for col, w in weights.items():
        score += w * df[col]

    return score