import numpy as np

def simulate_user_feedback(ranked_items, ctr_map):
    rewards = []
    for item in ranked_items:
        click = np.random.rand() < ctr_map.get(item, 0.01)
        rewards.append(click)
    return rewards
