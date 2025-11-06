# api/service_controller.py
import asyncio
import threading
from typing import Optional, List, Dict, Any, Callable

from MultiStreamVideoAnalyticsService import MultiStreamVideoAnalyticsService
from api_config.camzify_api_handler import CamzifyApiHandler
from config import env_variables
from utils.logger import get_logger

logger = get_logger(__name__)


class ServiceController:
    """
    Singleton controller managing the lifecycle of MultiStreamVideoAnalyticsService.

    - start_service(): async -> fetch configs and start service (idempotent)
    - stop_service(): sync -> stop service (idempotent)
    - reload_configs(): async -> fetch latest configs and apply to running service via RuleConfigStore
    - status(): sync -> return running flag + stream info
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._svc: Optional[MultiStreamVideoAnalyticsService] = None
        self._running = False

    # ---------- lifecycle ----------
    async def start_service(self) -> bool:
        """
        Start the service if not already running.
        Fetches configs using CamzifyApiHandler and instantiates MSVAS.
        Returns True if started, False if already running.
        """
        # fast path check
        with self._lock:
            if self._running:
                logger.warning("start_service called but service already running")
                return False

        base_url = settings.BASE_URL
        token = env_variables.API_TOKEN
        features = getattr(env_variables, "DEFAULT_FEATURES", ["line_intrusion_detector", "zone_intrusion_detector"])

        if not base_url or not token:
            logger.error("Missing BASE_URL or API_TOKEN in settings")
            raise RuntimeError("Missing BASE_URL or API_TOKEN")

        logger.info("Fetching initial configs to start service...")
        loader = CamzifyApiHandler(base_url, token, debug=getattr(env_variables, "DEBUG", False))
        streams, rules = await loader.fetch_all_features(features, is_active=True)

        # instantiate service
        svc = MultiStreamVideoAnalyticsService(
            streams_config=streams,
            rules_config=rules,
            model_path=getattr(env_variables, "MODEL_PATH", "models/yolo11n_custom.pt"),
            queue_size=getattr(settings, "QUEUE_SIZE", 50),
            target_fps=getattr(env_variables, "TARGET_FPS", 20.0),
            max_mixed_batch=getattr(env_variables, "MAX_MIXED_BATCH", 16),
            width=getattr(settings, "DEFAULT_WIDTH", 640),
            height=getattr(settings, "DEFAULT_HEIGHT", 480),
        )

        # start service (this triggers internal threads/processes)
        ok = svc.start()
        if not ok:
            logger.error("Service failed to start (svc.start() returned False)")
            raise RuntimeError("Service failed to start")

        with self._lock:
            self._svc = svc
            self._running = True

        logger.info("ServiceController: service started")
        return True

    def stop_service(self) -> bool:
        """
        Stop the running service. Returns True if it was running and now stopped,
        False if it was not running.
        """
        with self._lock:
            if not self._running or self._svc is None:
                logger.warning("stop_service called but service is not running")
                return False
            svc = self._svc
            self._svc = None
            self._running = False

        try:
            svc.stop()
            logger.info("ServiceController: service stopped cleanly")
            return True
        except Exception as e:
            logger.exception(f"Error stopping service: {e}")
            return False

    # ---------- config hot-reload ----------
    async def reload_configs(self) -> bool:
        """
        Fetch fresh configs and apply them to the running service without restart.
        If service is not running, it will start it instead.
        Returns True on success.
        """
        with self._lock:
            running = self._running
            svc = self._svc

        if not running or svc is None:
            logger.info("reload_configs called while service is not running — starting service instead")
            return await self.start_service()

        # fetch new configs
        base_url = settings.BASE_URL
        token = settings.API_TOKEN
        features = getattr(env_variables, "DEFAULT_FEATURES", ["line_intrusion_detector", "zone_intrusion_detector"])

        loader = CamzifyApiHandler(base_url, token, debug=getattr(settings, "DEBUG", False))
        streams, rules = await loader.fetch_all_features(features, is_active=True)

        # Transform streams list -> dict for store mutation
        streams_by_id = {str(s["stream_id"]): dict(s) for s in (streams or [])}
        rules_by_id = {str(sid): dict(r) for sid, r in (rules or {}).items()}

        # Use RuleConfigStore.apply_mutation to atomically replace streams + rules
        try:
            def mutate_fn(streams_store: dict, rules_store: dict):
                # replace streams_store in-place
                streams_store.clear()
                streams_store.update(streams_by_id)
                # replace rules_store in-place
                rules_store.clear()
                rules_store.update(rules_by_id)

            # call apply_mutation on the service's rule store
            svc.rule_config_store.apply_mutation(mutate_fn)
            logger.info("ServiceController: configs reloaded and applied via RuleConfigStore.apply_mutation")
            return True
        except Exception as e:
            logger.exception(f"Failed to apply new configs: {e}")
            return False

    # ---------- info ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = self._running
            svc = self._svc

        if not running or svc is None:
            return {"running": False, "streams": [], "version": None}

        # derive a lightweight snapshot
        try:
            streams = list(svc.stream_cfg_map.keys())
            rules_snapshot = svc.rule_config_store.get_rules_snapshot()
            version = svc.rule_config_store.get_version()
            return {"running": True, "streams": streams, "rules_count": len(rules_snapshot), "version": version}
        except Exception as e:
            logger.exception(f"status collection failed: {e}")
            return {"running": True, "streams": [], "rules_count": 0, "version": None}


# singleton instance to import from routes
service_controller = ServiceController()
