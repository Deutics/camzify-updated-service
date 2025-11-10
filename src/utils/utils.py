# utils/utils.py
import threading
import time
from typing import Callable, Optional, List
from queue import Queue
import multiprocessing as mp

def start_daemon_thread(target: Callable, args: tuple = (), name: Optional[str] = None,
                        daemon: bool = True, threads_list: Optional[List[threading.Thread]] = None):
    """
    Start a daemon thread, append to threads_list if provided and return it.
    This centralizes thread creation so we can consistently track/join threads.
    """
    t = threading.Thread(target=target, args=args, name=name, daemon=daemon)
    t.start()
    if threads_list is not None:
        threads_list.append(t)
    return t


def safe_close_queue(q):
    """
    Try to close/cleanup a queue (multiprocessing.Queue or custom queues).
    Best-effort: swallow exceptions so orchestrator keeps running.
    """
    if q is None:
        return
    try:
        # multiprocessing.Queue has close()
        close_fn = getattr(q, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        # ignore: we just want a best-effort close
        try:
            # some custom queues expose `shutdown` or `stop`
            shutdown_fn = getattr(q, "shutdown", None) or getattr(q, "stop", None)
            if callable(shutdown_fn):
                shutdown_fn()
        except Exception:
            pass


def safe_join_threads(threads: list, timeout: float = 0.5):
    """
    Join threads with a timeout and ignore errors. Useful during stop().
    """
    for t in list(threads):
        try:
            t.join(timeout=timeout)
        except Exception:
            pass


def safe_call(fn: Callable, *args, **kwargs):
    """
    Run a function and swallow exceptions (logging should be done by caller).
    Useful for event callbacks.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def format_stream_tag(sid: str) -> str:
    """
    Returns a short tag for stream logs: e.g. "[STREAM 74]".
    """
    return f"[STREAM {sid}]"
