import threading


class SafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
        self._overflow_threshold = 2 ** 31 - 1000


    def increment(self, n: int = 1) -> int:
        with self._lock:
            self._value += n
            if self._value >= self._overflow_threshold:
                self._value = 1
        return self._value


    @property
    def value(self) -> int:
        with self._lock:
            return self._value