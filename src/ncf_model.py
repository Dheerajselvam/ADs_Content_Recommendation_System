import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model.
    Learns user & item embeddings jointly via an MLP.
    """

    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze()


# --------------------------------------------------
# TRAINING ENTRYPOINT (THIS WAS MISSING)
# --------------------------------------------------

def train_ncf(
    train_df,
    emb_dim=32,
    epochs=5,
    lr=1e-3,
    batch_size=256,
    device="cpu"
):
    """
    Train NCF model and return trained model + embeddings.

    Returns:
        model, user_embeddings, item_embeddings
    """
    num_users = train_df["user_idx"].nunique()
    num_items = train_df["item_idx"].nunique()
    model = NCF(num_users, num_items, emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    users = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    items = torch.tensor(train_df["item_idx"].values, dtype=torch.long)
    labels = torch.tensor(train_df["clicked"].values, dtype=torch.float)

    dataset_size = len(train_df)
    indices = np.arange(dataset_size)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            u = users[batch_idx].to(device)
            i = items[batch_idx].to(device)
            y = labels[batch_idx].to(device)

            optimizer.zero_grad()
            preds = model(u, i)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (dataset_size / batch_size)
        print(f"[NCF] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

    # Extract embeddings for retrieval / ANN
    user_emb = model.user_emb.weight.detach().cpu().numpy()
    item_emb = model.item_emb.weight.detach().cpu().numpy()

    return {
        "model": model,
        "user_embeddings": user_emb,
        "item_embeddings": item_emb
    }
