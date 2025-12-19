import numpy as np
import joblib

def generate_candidates(
    user_index,
    model,
    ann_index,
    top_k=50
):
    """
    Retrieve candidate items for a user via ANN search.
    """
    user_vec = model["user_embeddings"][user_index].reshape(1, -1)
    distances, indices = ann_index.kneighbors(user_vec, n_neighbors=top_k)
    return indices.flatten().tolist()
