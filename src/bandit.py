import random

class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def rerank(self, ranked_items):
        if random.random() < self.epsilon:
            random.shuffle(ranked_items)
        return ranked_items
