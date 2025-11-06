# core/service_runner.py (sketch)
import asyncio
import time
import multiprocessing as mp
from typing import List, Optional
from utils.logger import get_logger
from config.env_variables import EnvVariables
from MultiStreamVideoAnalyticsService import MultiStreamVideoAnalyticsService
from api_config.camzify_api_handler import CamzifyApiHandler

logger = get_logger(__name__)

class ServiceRunner:
    def __init__(self):
        self.svc: Optional[MultiStreamVideoAnalyticsService] = None
        self._stop = asyncio.Event()

    async def setup_streams_and_rules(
        self,
        base_url: str,
        token: str,
        features: List[str],
        is_active: Optional[bool] = None,
        stream__in: Optional[List[int]] = None,
    ):
        async with CamzifyApiHandler(
            base_url,
            token,
            session_timeout=getattr(EnvVariables, "API_TIMEOUT", 15),
            debug=getattr(EnvVariables, "API_DEBUG", False),
            max_retries=getattr(EnvVariables, "API_MAX_RETRIES", 2),
            retry_backoff=getattr(EnvVariables, "API_RETRY_BACKOFF", 0.5),
            semaphore_limit=getattr(EnvVariables, "API_CONCURRENCY", 10),
        ) as loader:
            return await loader.fetch_all_features(features, is_active=is_active, stream__in=stream__in)


    async def run(self):
        try:
            mp.set_start_method('spawn')
        except Exception:
            pass

        base_url = EnvVariables.BASE_URL
        token = EnvVariables.API_TOKEN
        if not base_url or not token:
            logger.error("Missing BASE_URL or API_TOKEN. Check your .env file.")
            raise ValueError("Missing BASE_URL or API_TOKEN.")

        logger.info("Initializing MultiStreamVideoAnalyticsService...")
        logger.info(f"Using base URL: {base_url}")

        streams, rules = await self.setup_streams_and_rules(
            base_url,
            token,
            features=["line_intrusion_detector", "zone_intrusion_detector"],
            is_active=True,
            stream__in=[73,74]
        )




        if not streams:
            logger.warning("No active streams found — service will start idle.")

        self.svc = MultiStreamVideoAnalyticsService(
            streams,
            rules_config=rules,
            model_dir="yolov11",
            queue_size=50,
            target_fps=20.0,
        )

        try:
            if self.svc.start():
                logger.info("[MSVAS] Service started successfully.")
                print("Service started. Ctrl+C to stop")
                while not self._stop.is_set():
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping service...")
        except Exception as e:
            logger.error(f"Unexpected error in service loop: {e}", exc_info=True)
        finally:
            if self.svc:
                self.svc.stop()
            logger.info("[MSVAS] Service stopped cleanly.")
            print("Exited")

    def stop(self):
        self._stop.set()

    def start_service(self):
        logger.info("Launching service via asyncio runner...")
        asyncio.run(ServiceRunner().run())
