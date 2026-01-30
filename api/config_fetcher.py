import requests
from typing import Dict, List, Optional
from config.settings import BASE_URL, API_TOKEN
from api.camzify.config import CamzifyConfig
from api.parsers.stream_parser import StreamParser
from api.parsers.coordinate_mapper import CoordinateMapper
from api.core.crypto_utils import decrypt_aes_cbc
from Utils.logger import get_logger

logger = get_logger(__name__)


class ConfigFetcher:
    """Simple synchronous API client for fetching feature configs"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.headers = {"Authorization": f"Bearer {API_TOKEN}"}
        self.stream_parser = StreamParser(
            default_width=CamzifyConfig.DEFAULT_WIDTH,
            default_height=CamzifyConfig.DEFAULT_HEIGHT
        )
    
    def fetch_configs(
        self, 
        feature_endpoints: List[str],
        stream_ids: Optional[List[int]] = None, 
        is_active: Optional[bool] = None
    ) -> Dict[str, Dict]:
        """
        Fetch configurations for multiple features from API.
        
        Args:
            feature_endpoints: List of feature endpoint names (e.g., ["line_intrusion_detector", "zone_intrusion_detector"])
            stream_ids: Optional list of specific stream IDs to fetch
            is_active: Filter for active instances only
            
        Returns:
            Unified config dict:
            {
                "stream_id": {
                    "stream_id": str,
                    "rtsp_url": str,
                    "camera_width": int,
                    "camera_height": int,
                    "features": {
                        "line_intrusion_detector": [list of configs],
                        "zone_intrusion_detector": [list of configs],
                    }
                }
            }
        """
        unified_config = {}
        
        for feature_endpoint in feature_endpoints:
            try:
                endpoint = f"{self.base_url}/api/v1/instance/stream/analytic/{feature_endpoint}/instance/"
                params = {"is_active": str(is_active)}
                
                if stream_ids:
                    params["stream__in"] = ",".join(map(str, stream_ids))
                
                logger.info(f"Fetching {feature_endpoint} configs...")
                response = requests.get(endpoint, headers=self.headers, params=params, timeout=15)
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                logger.info(f"Fetched {len(results)} {feature_endpoint} configs")
                
                # Parse and merge configs
                self._parse_configs(results, feature_endpoint, unified_config)
                
            except Exception as e:
                logger.error(f"Failed to fetch {feature_endpoint} configs: {e}")
                continue
        
        return unified_config
    
    def _parse_configs(self, results: List[Dict], feature_type: str, unified_config: Dict):
        """Parse API results into unified stream configs"""
        
        for item in results:
            try:
                stream_info = item.get("stream", {})
                stream_id = str(stream_info.get("id"))
                
                if not stream_id:
                    continue
                
                # Initialize stream config if not exists
                if stream_id not in unified_config:
                    parsed_stream = self.stream_parser.parse_stream_info(stream_info)
                    if not parsed_stream:
                        continue
                    
                    # Decrypt RTSP URL
                    encrypted_url = parsed_stream["https_url"]
                    rtsp_url = parsed_stream["rtsp_url"]
                    https_url = decrypt_aes_cbc(
                        encrypted_url,
                        aes_key=CamzifyConfig.AES_KEY,
                        delimiter=CamzifyConfig.AES_DELIMITER
                    )
                    
                    if not (rtsp_url or https_url):
                        logger.warning(f"Failed to decrypt URL for stream {stream_id}")
                        continue
                    
                    unified_config[stream_id] = {
                        "stream_id": stream_id,
                        "rtsp_url": rtsp_url,
                        "https_url": https_url,
                        "camera_width": parsed_stream["width"],
                        "camera_height": parsed_stream["height"],
                        "features": {}
                    }
                
                # Parse feature configuration
                width = unified_config[stream_id]["camera_width"]
                height = unified_config[stream_id]["camera_height"]
                feature_config = self._parse_feature(item, width, height)
                
                if feature_config:
                    unified_config[stream_id]["features"].setdefault(feature_type, []).append(feature_config)
                
            except Exception as e:
                logger.warning(f"Failed to parse config item: {e}")
                continue
    
    def _parse_feature(self, item: Dict, width: int, height: int) -> Optional[Dict]:
        """Parse feature-specific configuration"""
        try:
            mapper = CoordinateMapper(width, height)
            
            # Extract normalized coordinates
            line_start_x = float(item.get("boundary_line_start_x", 0.0) or 0.0)
            line_start_y = float(item.get("boundary_line_start_y", 0.0) or 0.0)
            line_end_x = float(item.get("boundary_line_end_x", 1.0) or 1.0)
            line_end_y = float(item.get("boundary_line_end_y", 1.0) or 1.0)
            
            bbox_start_x = float(item.get("bounding_box_start_x", 0.0) or 0.0)
            bbox_start_y = float(item.get("bounding_box_start_y", 0.0) or 0.0)
            bbox_end_x = float(item.get("bounding_box_end_x", 1.0) or 1.0)
            bbox_end_y = float(item.get("bounding_box_end_y", 1.0) or 1.0)
            
            # Convert to pixels
            line_points = mapper.build_line_points([
                (line_start_x, line_start_y),
                (line_end_x, line_end_y)
            ])
            
            bbox_points = mapper.build_bbox_points(
                (bbox_start_x, bbox_start_y),
                (bbox_end_x, bbox_end_y)
            )
            
            # Parse object type
            object_type = item.get("object_type", "person")
            if isinstance(object_type, str):
                alert_classes = [cls.strip() for cls in object_type.split(",") if cls.strip()]
            else:
                alert_classes = ["person"]
            
            if not alert_classes:
                alert_classes = ["person"]
            
            return {
                "instance_id": item.get("instance_external_id"),
                "line_coords": line_points,
                "bounding_box_start": bbox_points["bbox_start"],
                "bounding_box_end": bbox_points["bbox_end"],
                "alert_classes": alert_classes,
                "direction_to_use": int(item.get("boundary_line_direction_to_use", 0)),
                "precision_factor": int(item.get("bounding_box_precision_factor", 20)),
                "cooldown": float(item.get("cooldown_time", 2.0) or 2.0),
                "time_bound_start": item.get("time_bound_start", "00:00:00"),
                "time_bound_end": item.get("time_bound_end", "23:59:59"),
                "is_active": bool(item.get("is_active", True))
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse feature: {e}")
            return None
