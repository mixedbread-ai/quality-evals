import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Budget:
    per_minute: int
    spent: deque[tuple[float, int]] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, amount: int) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.spent and self.spent[0][0] < now - 60:
                    self.spent.popleft()
                if sum(t for _, t in self.spent) + amount <= self.per_minute:
                    self.spent.append((now, amount))
                    return
                wait = self.spent[0][0] + 60 - now
            time.sleep(max(wait, 0.1))
