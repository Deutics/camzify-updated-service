"""
Async Notification Handler for VAF Multistream Processing
Non-blocking alert queue system that prevents image uploads from blocking stream processing

Architecture:
- Main stream processing thread calls notify_event() → returns immediately
- Background async workers process queue → send webhook + SMS
- Proper cleanup on shutdown (waits for queue to drain)

Usage:
    handler = AsyncNotificationHandler()
    await handler.start()
    
    # In stream processing loop (non-blocking)
    await handler.notify_event(
        stream_id="74",
        object_id=1,
        instance_id="32",
        alert_type="Line Intrusion",
        direction="left",
        frame=frame
    )
    
    # On shutdown
    await handler.stop()
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from Utils.logger import get_logger
from notification.vaf_alert_service import VAFAlertService

logger = get_logger(__name__)


@dataclass
class AlertJob:
    """Alert job for queue processing"""
    stream_id: str
    object_id: int
    instance_id: str
    alert_type: str
    frame: np.ndarray
    timestamp: datetime
    direction: Optional[str] = None
    confidence: float = 1.0
    metadata: Optional[dict] = None


class AsyncNotificationHandler:
    """
    Async notification handler for VAF alerts.
    Uses queue + background workers to prevent blocking stream processing.
    """

    def __init__(
        self,
        output_dir: str = "output/vaf_alerts",
        queue_size: int = 100,
        workers: int = 2,
        config: Optional[dict] = None,
    ):
        """
        Initialize async notification handler
        
        Args:
            output_dir: Directory for saving alert images
            queue_size: Maximum queue size (drops alerts when full)
            workers: Number of background workers
            config: Optional config dict for VAFAlertService
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Alert service (handles webhook + SMS)
        self._alert_service = VAFAlertService(config or {})
        
        # Queue for async processing
        self._queue: asyncio.Queue[Optional[AlertJob]] = asyncio.Queue(maxsize=queue_size)

        # Worker tasks
        self._workers = []
        self._workers_count = workers
        self._running = False
        
        logger.info(f"AsyncNotificationHandler initialized (queue_size={queue_size}, workers={workers})")

    # -------------------- Lifecycle --------------------

    async def start(self):
        """Start background workers once"""
        if self._running:
            logger.warning("Notification handler already running")
            return
        
        self._running = True
        for i in range(self._workers_count):
            worker_task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker_task)
        
        logger.info(f" Notification workers started: {self._workers_count}")

    async def stop(self):
        """Flush queue then stop workers and close resources"""
        if not self._running:
            logger.warning("Notification handler not running")
            return

        logger.info("Stopping notification handler...")

        # Wait for all queued alerts to be processed
        logger.info(f"Waiting for {self._queue.qsize()} queued alerts to process...")
        await self._queue.join()

        # Stop workers
        self._running = False
        for _ in self._workers:
            await self._queue.put(None)  # Poison pill

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        # Close aiohttp session
        await self._alert_service.close()

        logger.info(" Notification workers stopped and resources cleaned up")

    # -------------------- Public API --------------------

    async def notify_event(
        self,
        stream_id: str,
        object_id: int,
        instance_id: str,
        alert_type: str,
        frame: np.ndarray,
        direction: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ):
        """
        Non-blocking: Queue alert and return immediately.
        
        IMPORTANT: This does NOT block stream processing!
        
        Args:
            stream_id: Stream identifier (e.g., "74")
            object_id: Tracked object ID (e.g., 1)
            instance_id: Feature instance ID (e.g., "32")
            alert_type: Type of alert (e.g., "Line Intrusion")
            frame: BGR numpy array (will be copied)
            direction: Optional direction (e.g., "left", "right")
            confidence: Alert confidence (0.0-1.0)
            metadata: Additional metadata
        """
        # Auto-start if not running
        if not self._running:
            await self.start()

        job = AlertJob(
            stream_id=str(stream_id),
            object_id=int(object_id),
            instance_id=str(instance_id),
            alert_type=alert_type,
            frame=frame.copy(),  # IMPORTANT: Copy frame (stream reuses buffer)
            timestamp=datetime.now(),
            direction=direction,
            confidence=confidence,
            metadata=metadata,
        )

        try:
            # Non-blocking put
            self._queue.put_nowait(job)
            logger.debug(
                f"Alert queued: stream={stream_id}, obj={object_id}, "
                f"type={alert_type}, queue_size={self._queue.qsize()}"
            )
        except asyncio.QueueFull:
            logger.warning(
                f" Notification queue full ({self._queue.maxsize}), "
                f"dropping alert: stream={stream_id}, obj={object_id}, type={alert_type}"
            )

    # -------------------- Worker Loop --------------------

    async def _worker_loop(self, worker_id: int):
        """Background worker that processes alert queue"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            job = await self._queue.get()
            
            # Poison pill → exit
            if job is None:
                logger.info(f"Worker {worker_id} received shutdown signal")
                self._queue.task_done()
                break
            
            try:
                await self._process_alert(worker_id, job)
            except Exception as e:
                logger.error(
                    f"[Worker {worker_id}] Failed to process alert: {e}",
                    exc_info=True
                )
            finally:
                self._queue.task_done()
        
        logger.info(f"Worker {worker_id} stopped")

    async def _process_alert(self, worker_id: int, job: AlertJob):
        """Process a single alert job"""
        logger.info(
            f"[Worker {worker_id}] Processing alert: "
            f"stream={job.stream_id}, obj={job.object_id}, type={job.alert_type}"
        )

        # Send alert via webhook + SMS
        success = await self._alert_service.send_alert(
            stream_id=job.stream_id,
            alert_type=job.alert_type,
            object_id=str(job.object_id),
            instance_id=job.instance_id,
            image=job.frame,
            direction=job.direction,
            confidence=job.confidence,
            metadata=job.metadata,
        )

        if success:
            logger.info(
                f"[Worker {worker_id}] Alert sent successfully: "
                f"stream={job.stream_id}, obj={job.object_id}"
            )
        else:
            logger.warning(
                f"[Worker {worker_id}] Alert skipped (cooldown/threshold): "
                f"stream={job.stream_id}, obj={job.object_id}"
            )

    # -------------------- Stats --------------------

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

    def is_running(self) -> bool:
        """Check if handler is running"""
        return self._running
