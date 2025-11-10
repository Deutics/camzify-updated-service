import threading
import queue
from typing import Optional
from src.models.frame_data import FrameData
from src.utils.safe_counter import SafeCounter


class PriorityThreadSafeQueue:
    def __init__(self, maxsize=30):
        self.q = queue.PriorityQueue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.counter = SafeCounter()
        self.maxsize = maxsize
        self._closed = False  # NEW: for clean shutdown

    def put(self, item: FrameData, priority: int = 0):
        if self._closed:  # NEW: fail fast if closed
            return False

        with self.lock:
            cnt = self.counter.increment()
            # negative priority to make higher numbers pop first
            try:
                self.q.put((-priority, cnt, item), block=False)
                return True
            except queue.Full:
                # drop lowest priority by getting all and requeue
                items = []
                try:
                    while True:
                        items.append(self.q.get_nowait())
                except Exception:
                    pass
                # remove lowest priority (first after sort by negated priority)
                items_sorted = sorted(items, key=lambda x: (x[0], x[1]))
                if items_sorted:
                    items_sorted.pop(0)  # drop lowest priority (most negative = oldest/lowest frame_number)
                try:
                    for it in items_sorted:
                        self.q.put_nowait(it)
                    self.q.put_nowait((-priority, cnt, item))
                    return True
                except Exception:
                    return False

    def get(self, timeout: float = 0.01) -> Optional[FrameData]:
        try:
            pri, cnt, item = self.q.get(timeout=timeout)
            return item
        except Exception:
            return None

    def get_nowait(self) -> Optional[FrameData]:
        try:
            pri, cnt, item = self.q.get_nowait()
            return item
        except queue.Empty:
            return None

    def qsize(self) -> int:
        """NEW: expose queue size for adaptive thinning in ForwarderWorker."""
        try:
            return self.q.qsize()
        except Exception:
            return 0

    def close(self):
        """NEW: mark queue as closed to unblock producers during shutdown."""
        self._closed = True

    def get_stats(self):
        try:
            return {'size': self.q.qsize(), 'maxsize': self.maxsize}
        except Exception:
            return {'size': 0, 'maxsize': self.maxsize}