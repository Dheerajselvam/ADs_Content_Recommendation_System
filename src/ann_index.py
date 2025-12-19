import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

INDEX_OUT = "models/ann_index.pkl"

def build_ann(item_embeddings, n_neighbors=50):
    """
    Build ANN index over item embeddings.
    """
    ann = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric="cosine"
    )
    ann.fit(item_embeddings)

    joblib.dump(ann, INDEX_OUT)
    print("✅ ANN index saved:", INDEX_OUT)
    return ann
