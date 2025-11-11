import asyncio
import multiprocessing as mp
import getpass
from typing import List, Optional
from src.utils.logger import get_logger
from config.env_variables import EnvVariables
from src.core.multi_stream_video_analytics_service import MultiStreamVideoAnalyticsService
from src.api.camzify.client import CamzifyClient

logger = get_logger(__name__)


class ServiceRunner:
    def __init__(self):
        self.svc: Optional[MultiStreamVideoAnalyticsService] = None
        self._stop = asyncio.Event()

    def get_credentials(self) -> tuple[str, str]:
        print("\n" + "="*50)
        print("Camzify Video Analytics Service")
        print("="*50)
        username = input("Enter username: ").strip()
        password = input("Enter password: ").strip()
        
        if not username or not password:
            raise ValueError("Username and password cannot be empty")
        
        return username, password

    async def setup_streams_and_rules(
        self,
        base_url: str,
        username: str,
        password: str,
        features: List[str],
        is_active: Optional[bool] = None,
        stream_ids: Optional[List[int]] = None,
    ):
        async with CamzifyClient(
            base_url=base_url,
            username=username,
            password=password,
            debug=False
        ) as client:
            return await client.fetch_all_features(features, is_active=is_active, stream_ids=stream_ids)

    async def run(self):
        try:
            mp.set_start_method('spawn')
        except Exception:
            pass

        base_url =EnvVariables.BASE_URL
        
        try:
            username, password = self.get_credentials()
        except Exception as e:
            logger.error(f"Failed to get credentials: {e}")
            print(f"\nError: {e}")
            return

        logger.info("Initializing MultiStreamVideoAnalyticsService")
        logger.info(f"Using base URL: {base_url}")
        logger.info(f"Authenticating as: {username}")

        try:
            streams, rules = await self.setup_streams_and_rules(
                base_url,
                username,
                password,
                features=["line_intrusion_detector", "zone_intrusion_detector"],
                is_active=True,
                stream_ids=[73, 74]
            )
        except Exception as e:
            logger.error(f"Failed to fetch streams and rules: {e}", exc_info=True)
            print(f"\nAuthentication or API error: {e}")
            return

        if not streams:
            logger.warning("No active streams found")


            print("\nNo active streams found. Exiting.")
            return
        # import pprint
        # pprint.pprint(streams)
        # print("="*50)
        # pprint.pprint(rules)
        # exit()

        self.svc = MultiStreamVideoAnalyticsService(
            streams,
            rules_config=rules,
            model_dir="yolov11",
            queue_size=50,
            target_fps=20.0,
        )

        try:
            if self.svc.start():
                logger.info("Service started successfully")
                print("\n" + "="*50)
                print("Service started successfully!")
                print("Press Ctrl+C to stop")
                print("="*50 + "\n")
                
                while not self._stop.is_set():
                    await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            print("\nStopping service...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\nService error: {e}")
        finally:
            if self.svc:
                self.svc.stop()
            logger.info("Service stopped")
            print("Service stopped cleanly.\n")

    def stop(self):
        self._stop.set()

    def start_service(self):
        logger.info("Launching service")
        asyncio.run(ServiceRunner().run())