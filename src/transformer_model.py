import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle

class TransformerRec(nn.Module):
    def __init__(self, num_items, embed_dim=64, n_heads=4):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, item_seq):
        x = self.item_emb(item_seq)     # [B, T, D]
        h = self.encoder(x)             # [B, T, D]
        return h[:, -1, :]              # user embedding


MODEL = "models\tf_model.pkl"

def train_transformer(train_df, epochs=5, lr=1e-3):
    if os.path.isfile(MODEL):
        with open(MODEL, 'rb') as file:
            model = pickle.load(file)
    
    else:
        num_users = train_df["user_idx"].nunique()
        num_items = train_df["item_idx"].nunique()
        model = TransformerRec(num_items)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            total_loss = 0
            for row in train_df.itertuples():
                user_seq = torch.tensor([row.item_idx]).unsqueeze(0)
                label = torch.tensor([row.clicked], dtype=torch.float)

                user_emb = model(user_seq)               # [1, D]
                item_emb = model.item_emb(user_seq)      # [1, 1, D]
                item_emb = item_emb.squeeze(1)           # [1, D]

                score = (user_emb * item_emb).sum(dim=1) # [1]

                loss = loss_fn(score, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: loss={total_loss:.4f}")
    
    item_embeddings = model.item_emb.weight.detach().numpy()
    user_embeddings = np.random.randn(num_users, item_embeddings.shape[1])

    return {
        "model": model,
        "item_embeddings": item_embeddings,
        "user_embeddings": user_embeddings

    }

