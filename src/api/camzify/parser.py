from typing import Dict, Any, List, Optional, Tuple
from src.api.parsers.coordinate_mapper import CoordinateMapper
from src.api.parsers.stream_parser import StreamParser
from src.api.core.crypto_utils import decrypt_aes_cbc
from src.api.camzify.config import CamzifyConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CamzifyFeatureParser:
    def __init__(self, stream_parser: StreamParser):
        self._stream_parser = stream_parser

    def parse_feature_config(self, item: Dict[str, Any], feature_type: str) -> Optional[Dict[str, Any]]:
        try:
            stream_info = item.get("stream") or {}
            parsed_stream = self._stream_parser.parse_stream_info(stream_info)
            
            if not parsed_stream:
                return None

            https_url = decrypt_aes_cbc(
                parsed_stream["https_url"],
                aes_key=CamzifyConfig.AES_KEY,
                delimiter=CamzifyConfig.AES_DELIMITER
            )
            
            if not https_url:
                logger.warning(f"Failed to decrypt URL for stream {parsed_stream['stream_id']}")
                return None

            parsed_stream["https_url"] = https_url

            mapper= CoordinateMapper(parsed_stream["width"], parsed_stream["height"])
            
            line_points_norm = self._extract_line_points(item)
            bbox_norm = self._extract_bounding_box(item)
            
            line_points_pixels = mapper.build_line_points(line_points_norm)
            bbox_pixels = mapper.build_bbox_points(bbox_norm["start"], bbox_norm["end"])

            return {
                "stream_id": parsed_stream["stream_id"],
                "rtsp_stream_url": parsed_stream["rtsp_url"],
                "https_stream_url": parsed_stream["https_url"],
                "camera_width": parsed_stream["width"],
                "camera_height": parsed_stream["height"],
                "feature_type": feature_type.replace("_detector", ""),
                "line_points_norm": line_points_norm,
                "bounding_box_norm": bbox_norm,
                "line_points": line_points_pixels,
                "bounding_box_start": bbox_pixels["bbox_start"],
                "bounding_box_end": bbox_pixels["bbox_end"],
                "alert_classes": self._extract_alert_classes(item),
                "cooldown": float(item.get("cooldown_time") or item.get("cooldown") or 1.0),
                "direction_to_use": int(item.get("boundary_line_direction_to_use") or 0),
                "precision_factor": int(item.get("bounding_box_precision_factor") or item.get("precision_factor") or 1),
                "time_bound_start": item.get("time_bound_start") or "00:00:00",
                "time_bound_end": item.get("time_bound_end") or "23:59:59",
                "is_active": bool(item.get("is_active")) if item.get("is_active") is not None else True,
                "instance_id": item.get("instance_external_id"),
            }

        except Exception as e:
            logger.warning(f"Failed to parse feature config: {e}")
            return None

    def _extract_line_points(self, item: Dict[str, Any]) -> List[Tuple[float, float]]:
        return [
            (float(item.get("boundary_line_start_x",

 0.0) or 0.0), float(item.get("boundary_line_start_y", 0.0) or 0.0)),
            (float(item.get("boundary_line_end_x", 1.0) or 1.0), float(item.get("boundary_line_end_y", 1.0) or 1.0))
        ]

    def _extract_bounding_box(self, item: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        return {
            "start": (float(item.get("bounding_box_start_x", 0.0) or 0.0), float(item.get("bounding_box_start_y", 0.0) or 0.0)),
            "end": (float(item.get("bounding_box_end_x", 1.0) or 1.0), float(item.get("bounding_box_end_y", 1.0) or 1.0))
        }

    def _extract_alert_classes(self, item: Dict[str, Any]) -> List[str]:
        raw_obj_type = item.get("object_type")
        
        if raw_obj_type is None:
            return ["person"]
        elif isinstance(raw_obj_type, str):
            classes = [cls.strip() for cls in raw_obj_type.split(",") if cls.strip()]
            return classes or ["person"]
        elif isinstance(raw_obj_type, list):
            classes = [str(cls).strip() for cls in raw_obj_type if str(cls).strip()]
            return classes or ["person"]
        else:
            return ["person"]