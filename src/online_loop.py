import numpy as np
from event_logger import EventLogger
from streaming_stats import StreamingStats
from online_model_adapter import OnlineBanditAdapter
from online_simulator import simulate_user_feedback


def run_online_loop(ranking_df, item_stats, num_steps=1000):
    logger = EventLogger()
    stats = StreamingStats()
    bandit = OnlineBanditAdapter(epsilon=0.1)

    user_ids = ranking_df["user_idx"].unique()

    ctr_history = []

    for step in range(num_steps):
        user_id = np.random.choice(user_ids)

        ranked = (
            ranking_df[ranking_df.user_idx == user_id]
            .sort_values("score", ascending=False)["item_idx"]
            .tolist()[:10]
        )

        ranked = bandit.rerank(ranked)

        rewards = simulate_user_feedback(ranked, item_stats)

        for item, r in zip(ranked, rewards):
            logger.log(user_id, item, r)
            bandit.update(item, r)

        ctr_history.append(np.mean(rewards))

        if step % 100 == 0:
            events = logger.read_batch(200)
            stats.update(events)
            print(f"Step {step} | Online CTR: {np.mean(ctr_history[-50:]):.4f}")

    return ctr_history
