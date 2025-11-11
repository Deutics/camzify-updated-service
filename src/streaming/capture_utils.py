# capture/utils.py
import queue as _queue
import cv2
import numpy as np
from typing import Optional


def safe_put_queue(q, item, timeout=0.02) -> bool:
    """
    Non-blocking queue put with overwrite-on-full behavior.
    Returns True if enqueued, False if dropped.
    """
    try:
        q.put(item, block=True, timeout=timeout)
        return True
    except _queue.Full:
        try:
            q.get_nowait()  # drop oldest
        except Exception:
            pass
        try:
            q.put(item, block=False)
            return True
        except Exception:
            return False
    except Exception:
        return False


def downscale_bgr(frame: np.ndarray, w: int = 160, h: int = 90) -> Optional[np.ndarray]:
    """Downscale BGR frame efficiently using INTER_AREA interpolation."""
    if frame is None:
        return None
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> Optional[np.ndarray]:
    """Convert image to grayscale safely (handles already-gray or BGR)."""
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# capture/utils_stream.py
# Below are the utility functions for the ImprovedStreamCapture Module

def open_opencv_stream(url: str, width: int, height: int) -> cv2.VideoCapture:
    """
    Open a new OpenCV VideoCapture stream with consistent settings.
    """
    print(f"[OpenCVCapture] Opening: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap


def toggle_stream_url(rtsp_url: str, https_url: str, using_https: bool) -> tuple[str, bool]:
    """
    Toggle between RTSP and HTTPS URLs after repeated failures.
    Returns (new_url, new_flag)
    """
    if not https_url:
        return rtsp_url, using_https

    new_using_https = not using_https
    new_url = https_url if new_using_https else rtsp_url
    print(f"[OpenCVCapture] Switched to {'HTTPS' if new_using_https else 'RTSP'}")
    return new_url, new_using_https
