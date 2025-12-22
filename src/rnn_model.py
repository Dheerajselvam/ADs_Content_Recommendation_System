import torch
import torch.nn as nn
import numpy as np

class RNNRec(nn.Module):
    def __init__(self, num_items, emb_dim=64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.rnn = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq):
        emb = self.item_emb(seq)
        _, h = self.rnn(emb)
        h = h.squeeze(0)
        u = self.user_emb(user_idx) 
        return h + u  


def train_rnn(train_df, emb_dim=64, epochs=5):
    sequences = train_df.groupby("user_idx")["item_idx"].apply(list)
    num_users = train_df["user_idx"].nunique()
    num_items = train_df["item_idx"].nunique()

    model = RNNRec(num_items, emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for user_idx, seq in sequences.items():
            if len(seq) < 2:
                continue
            
            x = torch.tensor(seq[:-1]).unsqueeze(0)     # [1, T]
            y = torch.tensor(seq[1:])                   # [T]
            u = torch.tensor([user_idx])                # [1]

            user_emb = model(x, u)                      # [1, D]
            logits = user_emb @ model.item_emb.weight.T # [1, num_items]

            loss = loss_fn(logits, y[-1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {
        "type": "RNN",
        "item_embeddings": model.item_emb.weight.detach().numpy(),
        "user_embeddings":model.user_emb.weight.detach().numpy(),
        "score_fn": lambda u, i: np.dot(u, i),
        "meta": {}
    }
