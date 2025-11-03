# core/service_runner.py
import asyncio
import time
import multiprocessing as mp
from typing import List, Optional

from MultiStreamVideoAnalyticsService import MultiStreamVideoAnalyticsService
from api_config.APIConfigLoader import APIConfigLoader
from config import settings
from utils.logger import get_logger

# Centralized logger (uses your CamzifyService namespace)
logger = get_logger(__name__)


async def setup_streams_and_rules(
    base_url: str,
    token: str,
    features: List[str],
    is_active: Optional[bool] = None,
    stream__in: Optional[List[int]] = None,
):
    """
    Fetch pre-processed configs and return formatted streams and rules.
    """
    loader = APIConfigLoader(base_url, token)
    logger.info(f"Fetching configurations for features: {features}")

    try:
        streams, rules = await loader.fetch_all_features(
            features, is_active=is_active, stream__in=stream__in
        )
        logger.info(f"Prepared {len(streams)} streams and {len(rules)} rule sets.")
    except Exception as e:
        logger.error(f"Failed to fetch feature configs: {e}", exc_info=True)
        return [], {}

    return streams, rules


async def run_service():
    """Initialize and start the video analytics service."""
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    base_url = settings.BASE_URL
    token = settings.API_TOKEN

    if not base_url or not token:
        logger.error("Missing BASE_URL or API_TOKEN. Check your .env file.")
        raise ValueError("Missing BASE_URL or API_TOKEN.")

    logger.info("Initializing MultiStreamVideoAnalyticsService...")
    logger.info(f"Using base URL: {base_url}")

    streams, rules = await setup_streams_and_rules(
        base_url,
        token,
        features=["line_intrusion_detector", "zone_intrusion_detector"],
        is_active=True,
    )

    if not streams:
        logger.warning("No active streams found — service will start idle.")

    svc = MultiStreamVideoAnalyticsService(
        streams,
        rules_config=rules,
        model_path=settings.MODEL_PATH,
        queue_size=50,
        target_fps=20.0,
    )

    try:
        if svc.start():
            logger.info("[MSVAS] Service started successfully.")
            print("Service started. Ctrl+C to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping service...")
    except Exception as e:
        logger.error(f"Unexpected error in service loop: {e}", exc_info=True)
    finally:
        svc.stop()
        logger.info("[MSVAS] Service stopped cleanly.")
        print("Exited")


def start_service():
    """Entry point to launch the async service."""
    logger.info("Launching service via asyncio runner...")
    asyncio.run(run_service())
