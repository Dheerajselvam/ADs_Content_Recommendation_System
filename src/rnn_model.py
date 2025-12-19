import torch
import torch.nn as nn
import numpy as np

class RNNRec(nn.Module):
    def __init__(self, num_items, emb_dim=64):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.rnn = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq):
        emb = self.item_emb(seq)
        _, h = self.rnn(emb)
        return h.squeeze(0)


def train_rnn(train_df, emb_dim=64, epochs=5):
    sequences = train_df.groupby("user_idx")["item_idx"].apply(list)
    num_users = train_df["user_idx"].nunique()
    num_items = train_df["item_idx"].nunique()
    model = RNNRec(num_items, emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for seq in sequences:
            if len(seq) < 2:
                continue
            x = torch.tensor(seq[:-1]).unsqueeze(0)
            y = torch.tensor(seq[1:])

            h = model(x)
            logits = h @ model.item_emb.weight.T
            loss = nn.CrossEntropyLoss()(logits, y[-1:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    item_embeddings = model.item_emb.weight.detach().numpy()
    user_embeddings = np.random.randn(num_users, item_embeddings.shape[1])
    return {
        "type": "RNN",
        "item_embeddings": item_embeddings,
        "user_embeddings":user_embeddings,
        "score_fn": lambda u, i: np.dot(u, i),
        "meta": {}
    }
