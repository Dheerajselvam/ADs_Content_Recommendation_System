import random
import pandas as pd
import numpy as np


random.seed(42)
np.random.seed(42)


USERS = 500
ITEMS = 300
INTERACTIONS = 5000


AGE_BUCKETS = ["18-24","25-34","35-44"]
CATEGORIES = ["tech","sports","news","movies"]




def generate():
    rows = []
    for _ in range(INTERACTIONS):
        user = random.randint(0, USERS-1)
        item = random.randint(0, ITEMS-1)
        age = random.choice(AGE_BUCKETS)
        cat = random.choice(CATEGORIES)


        base_p = 0.02
        if age == "25-34": base_p += 0.01
        if cat == "tech": base_p += 0.015


        clicked = 1 if random.random() < base_p else 0
        rows.append({
        "user_id": user,
        "item_id": item,
        "age_bucket": age,
        "category": cat,
        "clicked": clicked
        })


    df = pd.DataFrame(rows)
    df.to_csv("data/raw/interactions.csv", index=False)
    print("Synthetic interactions generated")


if __name__ == "__main__":
    generate()