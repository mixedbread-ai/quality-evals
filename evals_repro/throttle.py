import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Budget:
    per_minute: int
    spent: deque[tuple[float, int]] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending: deque = field(default_factory=deque, init=False)
    ready: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.ready = threading.Condition(self.lock)

    def reserve(self, amount: int) -> None:
        if not 0 <= amount <= self.per_minute:
            raise ValueError(f"Request budget {amount} exceeds per-minute limit {self.per_minute}")
        ticket = object()
        with self.ready:
            self.pending.append(ticket)
            try:
                while True:
                    now = time.monotonic()
                    while self.spent and self.spent[0][0] <= now - 60:
                        self.spent.popleft()
                    first = self.pending[0] is ticket
                    if first and sum(t for _, t in self.spent) + amount <= self.per_minute:
                        self.spent.append((now, amount))
                        return
                    timeout = max(self.spent[0][0] + 60 - now, 0.001) if first and self.spent else None
                    self.ready.wait(timeout)
            finally:
                self.pending.remove(ticket)
                self.ready.notify_all()
