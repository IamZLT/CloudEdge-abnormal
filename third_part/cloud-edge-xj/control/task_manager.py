from collections import deque
from typing import List
from common.schemas import TaskMeta, DetectionResult


class TaskManager:
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)

    def add_task(self, meta: TaskMeta, result: DetectionResult) -> None:
        self.history.append({"meta": meta, "result": result})

    def list_history(self) -> List:
        return list(self.history)
