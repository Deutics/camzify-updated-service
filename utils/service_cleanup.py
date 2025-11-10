# utils/service_cleanup.py
def safe_join_thread(t, timeout=1.0):
    try:
        t.join(timeout=timeout)
    except Exception:
        pass

def safe_join_process(p, timeout=2.0):
    try:
        p.join(timeout=timeout)
    except Exception:
        pass
    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
