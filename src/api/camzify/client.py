import asyncio
from typing import List, Dict, Any, Optional, Tuple
from src.api.core.http_client import HttpClient
from src.api.core.auth_client import AuthClient
from src.api.parsers.stream_parser import StreamParser
from src.api.camzify.parser import CamzifyFeatureParser
from src.api.camzify.config import CamzifyConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CamzifyClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        default_width: int = CamzifyConfig.DEFAULT_WIDTH,
        default_height: int = CamzifyConfig.DEFAULT_HEIGHT,
        timeout_seconds: int = CamzifyConfig.DEFAULT_TIMEOUT,
        max_retries: int = CamzifyConfig.DEFAULT_MAX_RETRIES,
        retry_backoff: float = CamzifyConfig.DEFAULT_RETRY_BACKOFF,
        semaphore_limit: int = CamzifyConfig.DEFAULT_SEMAPHORE_LIMIT,
        debug: bool = False,
    ):
        self.default_width = default_width
        self.default_height = default_height
        self.debug = debug
        self._semaphore = asyncio.Semaphore(semaphore_limit)

        self._auth_client = AuthClient(
            base_url=base_url,
            username=username,
            password=password,
            login_endpoint=CamzifyConfig.LOGIN_ENDPOINT,
            timeout_seconds=timeout_seconds
        )

        self._http_client = HttpClient(
            base_url=base_url,
            token_provider=self._auth_client.get_token,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            debug=debug,
        )

        stream_parser = StreamParser(default_width, default_height)
        self._feature_parser = CamzifyFeatureParser(stream_parser)

        if self.debug:
            logger.setLevel("DEBUG")

        logger.info("CamzifyClient initialized with auto token refresh")

    async def __aenter__(self):
        await self._auth_client.__aenter__()
        await self._http_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._http_client.__aexit__(exc_type, exc, tb)
        await self._auth_client.__aexit__(exc_type, exc, tb)

    async def fetch_feature_configs(
        self,
        feature_type: str,
        is_active: Optional[bool] = None,
        stream_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        path = CamzifyConfig.FEATURE_ENDPOINT_TEMPLATE.format(feature=feature_type)
        
        params = {}
        if is_active is not None:
            params["is_active"] = str(is_active).lower()
        if stream_ids:
            params["stream__in"] = ",".join(map(str, stream_ids))

        if self.debug:
            logger.debug(f"Fetching {

feature_type} configs with params={params}")

        try:
            async with self._semaphore:
                response = await self._http_client.get(path, params=params)
            
            results = response.get("results", []) if isinstance(response, dict) else []
            
            parsed_configs = []
            for item in results:
                parsed = self._feature_parser.parse_feature_config(item, feature_type)
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
        is_active: Optional[bool] = None,
        stream_ids: Optional[List[int]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List

[Dict[str, Any]]]]]:
        logger.info(f"Fetching features: {feature_types}")
        
        tasks = [
            self.fetch_feature_configs(feature_type, is_active, stream_ids)
            for feature_type in feature_types
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_configs = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Feature '{feature_types[idx]}' failed: {result}")
                continue
            if isinstance(result, list):
                all_configs.extend(result)

        streams, rules = self._aggregate_streams_and_rules(all_configs)
        
        logger.info(f"Loaded {len(streams)} streams with {len(rules)} rule sets")
        return streams, rules

    def _aggregate_streams_and_rules(
        self, 
        all_configs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        streams_map = {}
        rules = {}

        for config in all_configs:
            stream_id = str(config.get("stream_id"))
            rtsp_url = config.get("rtsp_stream_url")
            https_url = config.get("https_stream_url")
            
            if not stream_id or not https_url or not rtsp_url:
                continue

            width = int(config.get("camera_width") or self.default_width)
            height = int(config.get("camera_height") or self.default_height)

            if stream_id not in streams_map:
                streams_map[stream_id] = {
                    "stream_id": stream_id,
                    "rtsp_stream_url": rtsp_url,
                    "https_stream_url": https_url,
                    "camera_width": width,
                    "camera_height": height,
                }
            else:
                existing = streams_map[stream_id]
                if (width * height) > (existing["camera_width"] * existing["camera_height"]):
                    existing["camera_width"] = width
                    existing["camera_height"] = height
                if rtsp_url:
                    existing["rtsp_stream_url"] = rtsp_url
                if https_url:
                    existing["https_stream_url"] = https_url

            feature_type = config.get("feature_type")
            if not feature_type:
                continue

            feature_config = {
                "line_points": config.get("line_points"),
                "bounding_box_start": config.get("bounding_box_start"),
                "bounding_box_end": config.get("bounding_box_end"),
                "alert_classes": config.get("alert_classes"),
                "cooldown": config.get("cooldown"),
                "direction_to_use": config.get("direction_to_use"),
                "precision_factor": config.get("precision_factor"),
                "time_bound_start": config.get("time_bound_start"),
                "time_bound_end": config.get("time_bound_end"),
                "is_active": config

.get("is_active", True),
                "line_points_norm": config.get("line_points_norm"),
                "bounding_box_norm": config.get("bounding_box_norm"),
                "instance_id": config.get("instance_id"),
            }

            rules.setdefault(stream_id, {}).setdefault(feature_type, []).append(feature_config)

        streams = [
            {
                "stream_id": stream["stream_id"],
                "rtsp_url": stream["rtsp_stream_url"],
                "https_url": stream["https_stream_url"],
                "camera_width": stream["camera_width"],
                "camera_height": stream["camera_height"],
            }
            for stream in streams_map.values()
        ]

        return streams, rules