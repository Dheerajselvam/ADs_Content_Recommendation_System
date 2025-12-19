def is_new_user(user_id, user_stats):
    return user_id not in user_stats

def is_new_item(item_id, item_stats):
    return item_id not in item_stats

def cold_start_candidates(user_id, item_stats, top_k=50):
    # popularity fallback
    return (
        item_stats
        .sort_values(
            by=["item_ctr", "item_impr"],
            ascending=[False, False]
        )
        .head(top_k)
        .index
        .tolist()
    )
