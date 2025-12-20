import numpy as np

def build_hybrid_embeddings(models, weights=None):
    """
    Concatenate or weighted-sum embeddings from multiple models.
    models: dict {name: model_dict}
    """

    if weights is None:
        weights = {k: 1.0 for k in models}

    item_embeddings = []
    user_embeddings = []

    for name, model in models.items():
        w = weights.get(name, 1.0)
        item_embeddings.append(w * model["item_embeddings"])
        user_embeddings.append(w * model["user_embeddings"])

    hybrid_item_emb = np.concatenate(item_embeddings, axis=1)
    hybrid_user_emb = np.concatenate(user_embeddings, axis=1)

    return {
        "item_embeddings": hybrid_item_emb,
        "user_embeddings": hybrid_user_emb
    }
