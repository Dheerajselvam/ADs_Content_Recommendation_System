import numpy as np

def train_mf(df, factors=32, lr=0.05, epochs=5):
    users = df["user_idx"].nunique()
    items = df["item_idx"].nunique()

    U = np.random.normal(0, 0.1, (users, factors))
    V = np.random.normal(0, 0.1, (items, factors))
    bu = np.zeros(users)
    bi = np.zeros(items)
    mu = df["clicked"].mean()

    for _ in range(epochs):
        for r in df.itertuples():
            u, i, y = r.user_idx, r.item_idx, r.clicked
            pred = mu + bu[u] + bi[i] + np.dot(U[u], V[i])
            err = y - pred

            bu[u] += lr * err
            bi[i] += lr * err
            U[u] += lr * err * V[i]
            V[i] += lr * err * U[u]

    return {
        "user_embeddings": U,
        "item_embeddings": V,
        "user_bias": bu,
        "item_bias": bi,
        "global_bias": mu
    }
