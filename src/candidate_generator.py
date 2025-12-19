import pandas as pd
from collections import defaultdict

def build_user_item_sets(df):
    ui = defaultdict(set)
    for _, r in df.iterrows():
        if r.clicked == 1:
            ui[r.user_id].add(r.item_id)
    return ui

def generate_candidates(user_id, ui_map, k=50):
    seen = ui_map.get(user_id, set())
    # simple co-occurrence: items seen by similar users
    candidates = set()
    for u, items in ui_map.items():
        if user_id != u and len(seen & items) > 0:
            candidates |= items
            candidates -= seen
    return list(candidates)[:k]