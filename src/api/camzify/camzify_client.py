# src/api/camzify/camzify_client.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from src.api.camzify.config import CamzifyConfig
from src.api.parsers.stream_parser import StreamParser
from src.api.camzify.parser import CamzifyFeatureParser
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CamzifyClient:
    """
    Async Camzify client using modular AuthModel + APIClient.
    Handles fetching and parsing feature configs for streams.
    """

    def __init__(
        self,
        api_client,
        auth_model,
        default_width: int = CamzifyConfig.DEFAULT_WIDTH,
        default_height: int = CamzifyConfig.DEFAULT_HEIGHT,
        semaphore_limit: int = CamzifyConfig.DEFAULT_SEMAPHORE_LIMIT,
    ):
        self.api_client = api_client
        self.auth_model = auth_model
        self.default_width = default_width
        self.default_height = default_height
        self.semaphore = asyncio.Semaphore(semaphore_limit)

        stream_parser = StreamParser(default_width, default_height)
        self.feature_parser = CamzifyFeatureParser(stream_parser)

    async def fetch_feature_configs(
        self,
        feature_type: str,
        is_active: Optional[bool] = True
    ) -> List[Dict[str, Any]]:
        """Fetch configs for a single feature type asynchronously."""
        endpoint = CamzifyConfig.FEATURE_ENDPOINT_TEMPLATE.format(feature=feature_type)
        params = {"is_active": str(is_active).lower()}

        async with self.semaphore:
            try:
                headers = self.auth_model.get_headers()
                response = await self.api_client.get(endpoint, headers=headers, params=params)

                results = response.get("results", []) if isinstance(response, dict) else []

                parsed_configs = []
                for item in results:
                    parsed = self.feature_parser.parse_feature_config(item, feature_type)
                    if parsed:
                        parsed_configs.append(parsed)

                logger.info(f"Fetched {len(parsed_configs)} configs for '{feature_type}'")
                return parsed_configs

            except Exception as e:
                logger.error(f"Failed to fetch '{feature_type}': {e}", exc_info=True)
                return []

    async def fetch_all_features(
        self,
        feature_types: List[str],
        is_active: Optional[bool] = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        """Fetch all feature configs concurrently and aggregate streams + rules."""
        logger.info(f"Fetching features: {feature_types}")

        tasks = [self.fetch_feature_configs(ft, is_active) for ft in feature_types]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_configs = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Feature '{feature_types[idx]}' fetch error: {result}")
            elif isinstance(result, list):
                all_configs.extend(result)

        streams, rules = self._aggregate_streams_and_rules(all_configs)
        logger.info(f" Loaded {len(streams)} streams and {len(rules)} rule sets.")
        return streams, rules

    def _aggregate_streams_and_rules(
        self, all_configs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        """Aggregate configs into structured streams and rules."""
        streams_map: Dict[str, Dict[str, Any]] = {}
        rules: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for cfg in all_configs:
            stream_id = str(cfg.get("stream_id"))
            rtsp_url = cfg.get("rtsp_stream_url")
            https_url = cfg.get("https_stream_url")

            if not stream_id or not rtsp_url or not https_url:
                continue

            width = int(cfg.get("camera_width") or self.default_width)
            height = int(cfg.get("camera_height") or self.default_height)

            if stream_id not in streams_map:
                streams_map[stream_id] = {
                    "stream_id": stream_id,
                    "rtsp_url": rtsp_url,
                    "https_url": https_url,
                    "camera_width": width,
                    "camera_height": height,
                }
            else:
                existing = streams_map[stream_id]
                # Keep higher resolution if multiple configs exist
                if (width * height) > (existing["camera_width"] * existing["camera_height"]):
                    existing["camera_width"] = width
                    existing["camera_height"] = height
                existing["rtsp_url"] = rtsp_url
                existing["https_url"] = https_url

            feature_type = cfg.get("feature_type")
            if not feature_type:
                continue

            feature_config = {
                "line_points": cfg.get("line_points"),
                "bounding_box_start": cfg.get("bounding_box_start"),
                "bounding_box_end": cfg.get("bounding_box_end"),
                "alert_classes": cfg.get("alert_classes"),
                "cooldown": cfg.get("cooldown"),
                "direction_to_use": cfg.get("direction_to_use"),
                "precision_factor": cfg.get("precision_factor"),
                "time_bound_start": cfg.get("time_bound_start"),
                "time_bound_end": cfg.get("time_bound_end"),
                "is_active": cfg.get("is_active", True),
                "line_points_norm": cfg.get("line_points_norm"),
                "bounding_box_norm": cfg.get("bounding_box_norm"),
                "instance_id": cfg.get("instance_id"),
            }
            rules.setdefault(stream_id, {}).setdefault(feature_type, []).append(feature_config)

        streams = list(streams_map.values())
        return streams, rules
