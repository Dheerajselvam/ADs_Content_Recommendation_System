import random

class OnlineBanditAdapter:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.item_rewards = {}
        self.item_counts = {}

    def update(self, item_id, reward):
        self.item_counts[item_id] = self.item_counts.get(item_id, 0) + 1
        self.item_rewards[item_id] = self.item_rewards.get(item_id, 0) + reward

    def score(self, item_id):
        if item_id not in self.item_counts:
            return 0.0
        return self.item_rewards[item_id] / self.item_counts[item_id]

    def rerank(self, items):
        if random.random() < self.epsilon:
            random.shuffle(items)
            return items

        return sorted(
            items,
            key=lambda i: self.score(i),
            reverse=True
        )
