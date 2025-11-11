from typing import Dict, Any, Optional, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamParser:
    def __init__(self, default_width: int = 640, default_height: int = 480):
        self.default_width = default_width
        self.default_height = default_height

    def parse_stream_info(self, stream_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            stream_id = stream_data.get("id")
            if not stream_id:
                return None

            rtsp_url = stream_data.get("rtsp_broadcast_address")
            https_url = stream_data.get("broadcast_address")
            width, height = self._extract_resolution(stream_data)

            return {
                "stream_id": str(stream_id),
                "rtsp_url": rtsp_url,
                "https_url": https_url,
                "width": width,
                "height": height
            }

        except Exception as e:
            logger.warning(f"Failed to parse stream info: {e}")
            return None

    def _extract_resolution(self, stream_data: Dict[str, Any]) -> Tuple[int, int]:
        try:
            camera_metadata = stream_data.get("camera_metadata") or {}
            streams_metadata = camera_metadata.get("streams") or []
            
            if streams_metadata and isinstance(streams_metadata, list):
                first_stream = streams_metadata[0] or {}
                width = int(first_stream.get("width")

 or self.default_width)
                height = int(first_stream.get("height") or self.default_height)
                return width, height
        except Exception:
            pass
        
        return self.default_width, self.default_height