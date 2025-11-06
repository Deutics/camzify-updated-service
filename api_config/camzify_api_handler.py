# api_config/camzify_api_handler.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from .ResolutionMapper import ResolutionMapper
from .decrypter import decrypt_data
from utils.logger import get_logger
from .api_handler import ApiHandler  # new dependency

logger = get_logger(__name__)

class CamzifyApiHandler:
    """
    Loads feature configs from Camzify API and returns fully validated,
    normalized and pixel-mapped configuration dicts ready for the service.

    Returns:
        - streams: List[{stream_id, url, camera_width, camera_height}]
        - rules: Dict[stream_id -> {feature_type -> [feature_cfg, ...]}]
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        default_width: int = 640,
        default_height: int = 480,
        session_timeout: int = 15,
        debug: bool = False,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        semaphore_limit: int = 10,
    ):
        self.default_width = default_width
        self.default_height = default_height
        self.debug = debug
        self._sem = asyncio.Semaphore(semaphore_limit)

        # Replace direct aiohttp with reusable ApiHandler
        self._api = ApiHandler(
            base_url=base_url,
            token=token,
            timeout_seconds=session_timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            debug=debug,
        )
        if self.debug:
            logger.setLevel("DEBUG")

        logger.info("Initialized CamzifyApiHandler with ApiHandler")

    def apply_partial_update(self, configs: Dict[str, Any], stream_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        new_cfgs = copy.deepcopy(configs)
        stream_cfg = new_cfgs.setdefault(stream_id, {})
        for key, value in updates.items():
            parts = key.split(".")
            target = stream_cfg
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value
        return new_cfgs

    async def _fetch_feature_configs(
        self,
        feature: str,
        is_active: Optional[bool] = None,
        stream__in: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch a single feature configs and convert to pixel space."""
        path = f"/api/v1/instance/stream/analytic/{feature}/instance"

        params: Dict[str, str] = {}
        if is_active is not None:
            params["is_active"] = str(is_active).lower()
        if stream__in:
            params["stream__in"] = ",".join(map(str, stream__in))

        if self.debug:
            logger.debug(f"Fetching {feature} configs from {path} with params={params}")

        try:
            async with self._sem:
                data = await self._api.get(path, params=params, expected_status=(200,))
        except Exception as e:
            logger.error(f"Failed to fetch {feature} configs: {e}", exc_info=True)
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
        parsed_results: List[Dict[str, Any]] = []

        for item in results:
            try:
                stream_info = item.get("stream") or {}
                stream_id = stream_info.get("id")

                rtsp_url = stream_info.get("rtsp_broadcast_address")
                https_url = decrypt_data(stream_info.get("broadcast_address"))

                width = self.default_width
                height = self.default_height
                try:
                    cam_meta = stream_info.get("camera_metadata") or {}
                    streams_meta = cam_meta.get("streams") or []
                    if streams_meta and isinstance(streams_meta, list):
                        first_stream = (streams_meta[0] or {}) if streams_meta else {}
                        width = int(first_stream.get("width") or width)
                        height = int(first_stream.get("height") or height)
                except Exception:
                    pass

                mapper = ResolutionMapper(width, height)

                ln_start_x = item.get("boundary_line_start_x", 0.0)
                ln_start_y = item.get("boundary_line_start_y", 0.0)
                ln_end_x = item.get("boundary_line_end_x", 1.0)
                ln_end_y = item.get("boundary_line_end_y", 1.0)
                line_points_norm = [
                    (float(ln_start_x or 0.0), float(ln_start_y or 0.0)),
                    (float(ln_end_x or 1.0), float(ln_end_y or 1.0)),
                ]

                bb_sx = item.get("bounding_box_start_x", 0.0) or 0.0
                bb_sy = item.get("bounding_box_start_y", 0.0) or 0.0
                bb_ex = item.get("bounding_box_end_x", 1.0) or 1.0
                bb_ey = item.get("bounding_box_end_y", 1.0) or 1.0
                bbox_norm = ((float(bb_sx), float(bb_sy)), (float(bb_ex), float(bb_ey)))

                pixel_line_points = mapper.build_line_points({"line_points_norm": line_points_norm})
                bbox_pixels = mapper.build_bbox_points({
                    "bounding_box_start": bbox_norm[0],
                    "bounding_box_end": bbox_norm[1]
                })

                instance_id = item.get("instance_external_id")
                raw_obj_type = item.get("object_type")

                if raw_obj_type is None:
                    alert_classes = ["person"]
                elif isinstance(raw_obj_type, str):
                    alert_classes = [p.strip() for p in raw_obj_type.split(",") if p.strip()] or ["person"]
                elif isinstance(raw_obj_type, list):
                    alert_classes = [str(x).strip() for x in raw_obj_type if str(x).strip()] or ["person"]
                else:
                    alert_classes = ["person"]

                feature_type = feature.replace("_detector", "")
                cooldown = float(item.get("cooldown_time") or item.get("cooldown") or 1.0)
                direction_to_use = int(item.get("boundary_line_direction_to_use") or 0)
                precision_factor = int(item.get("bounding_box_precision_factor")
                                       or item.get("precision_factor") or 1)
                time_start = item.get("time_bound_start") or "00:00:00"
                time_end = item.get("time_bound_end") or "23:59:59"
                is_active_val = bool(item.get("is_active")) if item.get("is_active") is not None else True

                parsed = {
                    "stream_id": stream_id,
                    "rtsp_stream_url": rtsp_url,
                    "https_stream_url": https_url,
                    "camera_width": width,
                    "camera_height": height,
                    "feature_type": feature_type,
                    "line_points_norm": line_points_norm,
                    "bounding_box_norm": {"start": bbox_norm[0], "end": bbox_norm[1]},
                    "line_points": pixel_line_points,
                    "bounding_box_start": bbox_pixels["bbox_start"],
                    "bounding_box_end": bbox_pixels["bbox_end"],
                    "alert_classes": alert_classes,
                    "cooldown": cooldown,
                    "direction_to_use": direction_to_use,
                    "precision_factor": precision_factor,
                    "time_bound_start": time_start,
                    "time_bound_end": time_end,
                    "is_active": is_active_val,
                    "instance_id": instance_id,
                }
                parsed_results.append(parsed)

            except Exception as e:
                logger.warning(f"Skipping malformed config item (feature={feature}): {e}", exc_info=True)
                continue

        if self.debug:
            logger.debug(f"Fetched {len(parsed_results)} {feature} configs")

        return parsed_results

    async def fetch_all_features(
        self,
        features: List[str],
        is_active: Optional[bool] = None,
        stream__in: Optional[List[int]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        """Fetch configurations for multiple features concurrently."""
        logger.info(f"Fetching all features: {features}")

        tasks = [
            self._fetch_feature_configs(feature, is_active, stream__in)
            for feature in features
        ]
        # tolerate per-feature failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        flat: List[Dict[str, Any]] = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Feature fetch failed for {features[idx]}: {r}", exc_info=True)
                continue
            if isinstance(r, list):
                flat.extend(r)

        rules: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        streams_map: Dict[str, Dict[str, Any]] = {}

        for cfg in flat:
            sid = str(cfg.get("stream_id"))
            rtsp_url = cfg.get("rtsp_stream_url")
            https_url = cfg.get("https_stream_url")
            if not sid or not https_url or not rtsp_url:
                continue

            w = int(cfg.get("camera_width") or self.default_width)
            h = int(cfg.get("camera_height") or self.default_height)

            existing = streams_map.get(sid)
            if not existing:
                streams_map[sid] = {
                    "stream_id": sid,
                    "rtsp_stream_url": rtsp_url,
                    "https_stream_url": https_url,
                    "camera_width": w,
                    "camera_height": h,
                }
            else:
                if (w * h) > (existing["camera_width"] * existing["camera_height"]):
                    existing["camera_width"] = w
                    existing["camera_height"] = h
                if rtsp_url:
                    existing["rtsp_stream_url"] = rtsp_url
                if https_url:
                    existing["https_stream_url"] = https_url

            feature_type = cfg.get("feature_type")
            if not feature_type:
                continue

            feature_cfg = {
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

            rules.setdefault(sid, {}).setdefault(feature_type, []).append(feature_cfg)

        streams = [
            {
                "stream_id": s["stream_id"],
                "rtsp_url": s["rtsp_stream_url"],
                "https_url": s["https_stream_url"],
                "camera_width": s["camera_width"],
                "camera_height": s["camera_height"],
            }
            for s in streams_map.values()
        ]

        logger.info(f"Prepared {len(streams)} streams and {len(rules)} rule sets")

        if self.debug:
            logger.debug(f"Streams={streams}")
            logger.debug(f"Rules={rules}")

        return streams, rules

    async def __aenter__(self):
        # allow: async with CamzifyApiHandler(...) as loader:
        await self._api.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._api.__aexit__(exc_type, exc, tb)