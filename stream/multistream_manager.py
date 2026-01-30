"""Optimized multistream capture - minimal copying, uses StreamHandler reconnection"""

import threading
import time
import cv2
from typing import Dict, Optional
from stream.stream_handler import StreamHandler
from Utils.logger import get_logger

logger = get_logger(__name__)


class StreamFrameBuffer:
    def __init__(self, source: str, stream_id: str):
        self.source = source
        self.stream_id = stream_id
        self.stream_handler = StreamHandler(source, max_retries=5, retry_delay=2.0)
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.capture_thread = None
        self.frame_count = 0
        
    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Stream {self.stream_id} thread started")
        
    def _capture_loop(self):
        """Capture loop - StreamHandler handles all reconnection"""
        try:
            self.stream_handler.start_stream()
        except Exception as e:
            logger.error(f"Stream {self.stream_id} failed to start: {e}")
            return
        
        while self.running:
            try:
                frame, _ = self.stream_handler.read_frame()
                if frame is not None:
                    with self.frame_lock:
                        # Store reference - frame from cv2.read() is already a new array
                        self.latest_frame = frame
                        self.frame_count += 1
                        
            except Exception as e:
                logger.error(f"Stream {self.stream_id} error: {e}")
                time.sleep(1)
                
    def get_latest_frame(self) -> Optional:
        """Get latest frame - copies for thread safety since processing modifies it"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()  # copy needed for thread safety
            return None
            
    def update_annotated_frame(self, annotated_frame):
        """Store annotated frame for display"""
        with self.frame_lock:
            self.latest_annotated_frame = annotated_frame
            
    def get_display_frame(self):
        """Get frame for display - no copy needed, just for viewing"""
        with self.frame_lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame
            return self.latest_frame
            
    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        self.stream_handler.release_stream()
        logger.info(f"Stream {self.stream_id} stopped")


class MultiStreamManager:
    def __init__(self, stream_sources: Dict[str, str], enable_display: bool = True):
        self.stream_buffers = {}
        self.enable_display = enable_display
        for stream_id, source in stream_sources.items():
            self.stream_buffers[stream_id] = StreamFrameBuffer(source, stream_id)
            
    def start_all(self):
        for stream_id, buffer in self.stream_buffers.items():
            buffer.start()
        logger.info(f"Started {len(self.stream_buffers)} stream threads")
        
    def get_all_latest_frames(self) -> Dict[str, any]:
        frames = {}
        for stream_id, buffer in self.stream_buffers.items():
            frame = buffer.get_latest_frame()
            if frame is not None:
                frames[stream_id] = frame
        return frames
        
    def update_annotated_frame(self, stream_id: str, annotated_frame):
        if stream_id in self.stream_buffers:
            self.stream_buffers[stream_id].update_annotated_frame(annotated_frame)
            
    def display_all_streams(self):
        if not self.enable_display:
            return True
            
        for stream_id, buffer in self.stream_buffers.items():
            frame = buffer.get_display_frame()
            if frame is not None:
                cv2.imshow(f"Stream {stream_id}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
        
    def stop_all(self):
        for buffer in self.stream_buffers.values():
            buffer.stop()
        if self.enable_display:
            cv2.destroyAllWindows()
        logger.info("All streams stopped")
