"""
VAF Alert Service - Webhook + SMS notification system for VAF features
Adapted from fall detection alerts for line intrusion and future features

Key features:
- Async webhook delivery with retry logic
- SMS via Twilio (offloaded to threads)
- Per-camera cooldown to prevent spam
- Base64 JPEG image encoding
- Proper resource cleanup

Usage:
    alerts = VAFAlertService(config)
    await alerts.send_alert(...)
    await alerts.close()  # at shutdown
"""

from __future__ import annotations

import os
import re
import base64
import tempfile
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import aiohttp
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from Utils.logger import get_logger
from config.constants import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM_PHONE,
    TO_PHONE_NUMBERS,
    VAF_WEBHOOK_URL,
    LOCATION,
)

LOGGER = get_logger(__name__)

E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_phone_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _jpeg_base64(image_bgr: np.ndarray, quality: int = 80) -> Tuple[str, int]:
    """
    Convert BGR numpy image -> JPEG bytes -> base64 string.
    Returns (b64_string, jpg_bytes_len)
    """
    ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to JPEG-encode image")
    jpg_bytes = enc.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8"), len(jpg_bytes)


def _json_safe(obj: Any):
    """Recursively convert non-JSON types to JSON-safe."""
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


@dataclass
class AlertConfig:
    min_confidence: float = 0.7
    cooldown_seconds: int = 120  # Per stream cooldown
    webhook_timeout_sec: int = 10
    webhook_max_retries: int = 2
    webhook_backoff_base: float = 0.7
    max_concurrent_sends: int = 3
    include_image_in_webhook: bool = True
    webhook_jpeg_quality: int = 80
    save_dir: str = "output/vaf_alerts"
    location: str = LOCATION


class VAFAlertService:
    """Service for sending VAF feature alerts via Twilio SMS + webhook push."""

    def __init__(self, config: dict = None):
        # Twilio credentials
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.from_phone = TWILIO_FROM_PHONE
        self.to_phones = _parse_phone_list(TO_PHONE_NUMBERS)
        
        # Webhook
        self.webhook_url = VAF_WEBHOOK_URL

        self._validate_phone_numbers()

        # Settings
        self.cfg = AlertConfig()
        if config:
            for k, v in config.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)

        self.last_alert_time: Dict[str, datetime] = {}  # Per-stream cooldown

        # Twilio client
        self.client: Optional[Client] = None
        if self.account_sid and self.auth_token:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                LOGGER.info("Twilio client initialized for VAF alerts")
            except Exception as e:
                LOGGER.error(f" Failed to initialize Twilio client: {e}", exc_info=True)
                self.client = None
        else:
            LOGGER.warning(" Twilio credentials missing; SMS sending will be disabled")

        # HTTP session (reused)
        self._session: Optional[aiohttp.ClientSession] = None

        # Bounded concurrency
        self._sem = asyncio.Semaphore(self.cfg.max_concurrent_sends)

        _ensure_dir(self.cfg.save_dir)

    # -------------------- Lifecycle --------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.webhook_timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            LOGGER.info("VAF alert service session closed")

    # -------------------- Validation & Cooldown --------------------

    def _validate_phone_numbers(self):
        if self.from_phone and not E164_RE.match(self.from_phone):
            LOGGER.warning(f" From phone '{self.from_phone}' may not be E.164 format")

        valid = []
        for p in self.to_phones:
            if E164_RE.match(p):
                valid.append(p)
            else:
                LOGGER.error(f"Invalid recipient phone: '{p}' (must be E.164: +1234567890)")

        if not valid and self.to_phones:
            LOGGER.warning("No valid recipient phone numbers found")

        self.to_phones = valid
        if self.to_phones:
            LOGGER.info(f" Configured {len(self.to_phones)} recipient number(s)")

    def should_send_alert(self, stream_id: str, confidence: float = 1.0) -> bool:
        """Check if alert should be sent based on confidence and cooldown"""
        if confidence < self.cfg.min_confidence:
            return False

        now = _utcnow()
        last = self.last_alert_time.get(stream_id)
        if last:
            delta = (now - last).total_seconds()
            if delta < self.cfg.cooldown_seconds:
                LOGGER.info(
                    f"Cooldown active for stream {stream_id}: "
                    f"{delta:.1f}s < {self.cfg.cooldown_seconds}s"
                )
                return False
        return True

    # -------------------- Message Formatting --------------------

    def format_alert_message(
        self,
        stream_id: str,
        alert_type: str,
        object_id: str,
        instance_id: str,
        direction: Optional[str] = None,
        timestamp: datetime = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format SMS alert message"""
        timestamp = timestamp or _utcnow()
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        msg_parts = [
            f"🚨 VAF ALERT: {alert_type}",
            f"Location: {self.cfg.location}",
            f"Stream: {stream_id}",
            f"Instance: {instance_id}",
            f"Object ID: {object_id}",
        ]
        
        if direction:
            msg_parts.append(f"Direction: {direction}")
        
        msg_parts.append(f"Time: {ts_str}")
        
        if metadata:
            extra = ", ".join(f"{k}={v}" for k, v in metadata.items() if k not in ['image'])
            if extra:
                msg_parts.append(f"Details: {extra}")
        
        return "\n".join(msg_parts)

    # -------------------- Image Saving --------------------

    async def save_alert_image(
        self,
        stream_id: str,
        object_id: str,
        instance_id: str,
        alert_type: str,
        image_bgr: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save alert image to disk with metadata"""
        timestamp = _utcnow()
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Create subdirectory for this alert type
        alert_dir = os.path.join(self.cfg.save_dir, alert_type.lower().replace(" ", "_"))
        _ensure_dir(alert_dir)
        
        filename = f"stream{stream_id}_obj{object_id}_inst{instance_id}_{ts_str}.jpg"
        filepath = os.path.join(alert_dir, filename)

        def _write():
            ok, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("Failed to encode image as JPEG")
            with open(filepath, "wb") as f:
                f.write(enc.tobytes())
            
            # Save metadata JSON
            if metadata:
                import json
                meta_path = filepath.replace(".jpg", "_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(_json_safe(metadata), f, indent=2)
            
            return filepath

        try:
            return await asyncio.to_thread(_write)
        except Exception as e:
            LOGGER.error(f" Failed to save alert image: {e}", exc_info=True)
            return ""

    # -------------------- Twilio SMS (blocking SDK -> thread) --------------------

    def _send_sms_blocking(self, body: str) -> int:
        """
        Runs in a worker thread. Returns count of successful sends.
        """
        if self.client is None:
            LOGGER.warning("Twilio client not initialized; skipping SMS")
            return 0

        success = 0
        for phone in self.to_phones:
            try:
                LOGGER.info(f"📱 Sending VAF alert SMS to {phone}")
                msg = self.client.messages.create(body=body, from_=self.from_phone, to=phone)
                LOGGER.info(f" SMS sent to {phone} SID={msg.sid}")
                success += 1
            except TwilioRestException as e:
                LOGGER.error(
                    f" Twilio SMS failed to {phone}: code={getattr(e, 'code', None)} "
                    f"status={getattr(e, 'status', None)} msg={getattr(e, 'msg', str(e))}"
                )
            except Exception as e:
                LOGGER.error(f" SMS failed to {phone}: {e}", exc_info=True)

        return success

    # -------------------- Webhook Push (async) --------------------

    async def send_webhook_notification(
        self,
        stream_id: str,
        alert_type: str,
        object_id: str,
        instance_id: str,
        direction: Optional[str] = None,
        timestamp: datetime = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """Send webhook with retries and backoff"""
        if not self.webhook_url:
            LOGGER.warning("No webhook URL configured; skipping push")
            return False

        timestamp = timestamp or _utcnow()
        
        payload = {
            "alert_type": alert_type,
            "stream_id": str(stream_id),
            "instance_id": str(instance_id),
            "object_id": str(object_id),
            "location": self.cfg.location,
            "severity": "high",
            "title": f"{alert_type} Alert",
            "message": f"{alert_type} detected on stream {stream_id} at {self.cfg.location}",
            "requires_immediate_attention": True,
            "notification_type": "vaf_alert",
            "timestamp": timestamp.isoformat(),
            "metadata": _json_safe(metadata or {}),
        }
        
        if direction:
            payload["direction"] = direction
            payload["metadata"]["direction"] = direction

        # Include image if enabled
        if self.cfg.include_image_in_webhook and image_bgr is not None:
            try:
                b64, jpg_len = _jpeg_base64(image_bgr, quality=self.cfg.webhook_jpeg_quality)
                payload["image"] = b64
                payload["image_filename"] = f"vaf_{alert_type}_{stream_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                payload["image_encoding"] = "jpeg_base64"
                payload["image_bytes"] = jpg_len
            except Exception as e:
                LOGGER.warning(f" Failed to attach image to webhook payload: {e}")

        session = await self._get_session()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VAFAlertSystem/1.0",
            "X-Alert-Type": alert_type,
            "X-Severity": "high",
            "Accept": "application/json",
        }

        for attempt in range(self.cfg.webhook_max_retries + 1):
            try:
                async with session.post(self.webhook_url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        LOGGER.info(f" Webhook delivered (status={resp.status})")
                        return True
                    LOGGER.warning(f" Webhook failed status={resp.status} body={text[:500]}")
            except asyncio.TimeoutError:
                LOGGER.warning(" Webhook timeout")
            except aiohttp.ClientError as e:
                LOGGER.warning(f" Webhook client error: {e}")
            except Exception as e:
                LOGGER.error(f" Webhook exception: {e}", exc_info=True)

            if attempt < self.cfg.webhook_max_retries:
                backoff = self.cfg.webhook_backoff_base * (2 ** attempt)
                await asyncio.sleep(backoff)

        return False

    # -------------------- Public API: Send Alert --------------------

    async def send_alert(
        self,
        stream_id: str,
        alert_type: str,
        object_id: str,
        instance_id: str,
        image: np.ndarray,
        direction: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Main entry: saves files + sends SMS + sends webhook.
        
        Args:
            stream_id: Stream identifier (e.g., "74")
            alert_type: Type of alert (e.g., "Line Intrusion")
            object_id: Tracked object ID (e.g., "1")
            instance_id: Feature instance ID (e.g., "32")
            image: BGR numpy array (OpenCV frame)
            direction: Optional direction (e.g., "left", "right")
            confidence: Alert confidence (0.0-1.0)
            metadata: Additional metadata dict
            
        Returns:
            True if alert sent successfully via SMS or webhook
        """
        async with self._sem:
            try:
                if not self.should_send_alert(stream_id, confidence):
                    return False

                timestamp = _utcnow()
                self.last_alert_time[stream_id] = timestamp
                LOGGER.info(f" Alert time recorded for stream {stream_id}: {timestamp.isoformat()}")

                # Save image to disk
                await self.save_alert_image(
                    stream_id=stream_id,
                    object_id=object_id,
                    instance_id=instance_id,
                    alert_type=alert_type,
                    image_bgr=image,
                    metadata=metadata,
                )

                # Build SMS text
                sms_body = self.format_alert_message(
                    stream_id=stream_id,
                    alert_type=alert_type,
                    object_id=object_id,
                    instance_id=instance_id,
                    direction=direction,
                    timestamp=timestamp,
                    metadata=metadata,
                )

                # Send webhook push (async)
                push_success = await self.send_webhook_notification(
                    stream_id=stream_id,
                    alert_type=alert_type,
                    object_id=object_id,
                    instance_id=instance_id,
                    direction=direction,
                    timestamp=timestamp,
                    metadata=metadata,
                    image_bgr=image,
                )

                # Send SMS (Twilio is blocking -> thread)
                sms_success = 0
                if self.client is not None and self.from_phone and self.to_phones:
                    sms_success = await asyncio.to_thread(self._send_sms_blocking, sms_body)

                if sms_success > 0:
                    LOGGER.info(f"SMS sent to {sms_success}/{len(self.to_phones)} recipients")
                else:
                    LOGGER.debug(" No SMS recipients succeeded (or SMS disabled)")

                if push_success:
                    LOGGER.info(" Push webhook sent successfully")
                else:
                    LOGGER.debug(" Push webhook failed")

                # Consider alert successful if either channel succeeds
                return (sms_success > 0) or push_success

            except Exception as e:
                LOGGER.error(f" Error sending VAF alert: {e}", exc_info=True)
                return False
