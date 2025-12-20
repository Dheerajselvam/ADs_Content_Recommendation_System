from collections import deque
import time

class EventLogger:
    def __init__(self, max_events=10000):
        self.events = deque(maxlen=max_events)

    def log(self, user_id, item_id, clicked):
        event = {
            "user_id": user_id,
            "item_id": item_id,
            "clicked": clicked,
            "timestamp": time.time()
        }
        self.events.append(event)

    def read_batch(self, batch_size=100):
        batch = []
        while self.events and len(batch) < batch_size:
            batch.append(self.events.popleft())
        return batch
