from collections import defaultdict

class StreamingStats:
    def __init__(self):
        self.item_impr = defaultdict(int)
        self.item_click = defaultdict(int)

    def update(self, events):
        for e in events:
            self.item_impr[e["item_id"]] += 1
            self.item_click[e["item_id"]] += e["clicked"]

    def get_item_ctr(self, item_id):
        impr = self.item_impr[item_id]
        if impr == 0:
            return 0.0
        return self.item_click[item_id] / impr
