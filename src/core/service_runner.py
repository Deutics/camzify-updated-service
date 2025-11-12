import asyncio
from src.api.core.api_client import APIClient
from src.api.core.auth_model import AuthModel
from src.api.camzify.camzify_client import CamzifyClient
from src.core.multi_stream_video_analytics_service import MultiStreamVideoAnalyticsService
from src.utils.logger import get_logger
from config.env_variables import EnvVariables

logger = get_logger(__name__)

class ServiceRunner:
    """Main orchestrator for Camzify Video Analytics Service (Async Version)."""

    def get_credentials(self):
        print("\n" + "=" * 50)
        print("Camzify Video Analytics Service")
        print("=" * 50)
        username = input("Enter username: ").strip()
        password = input("Enter password: ").strip()
        if not username or not password:
            raise ValueError("Username and password cannot be empty")
        return username, password

    async def run(self):
        base_url = EnvVariables.BASE_URL
        username, password = self.get_credentials()

        try:
            async with APIClient(base_url) as api_client:
                auth = AuthModel(api_client)
                await auth.authenticate(username, password)
                print("\nAuthentication successful.")
                logger.info("User authenticated successfully.")

                camzify = CamzifyClient(api_client, auth)

                feature_types = [
                    "line_intrusion_detector",
                    "zone_intrusion_detector",
                ]

                streams, rules = await camzify.fetch_all_features(feature_types, is_active=True)

                if not streams:
                    print("\n⚠️ No active streams found. Exiting.")
                    logger.warning("No active streams found. Exiting.")
                    return

                logger.info(f"Loaded {len(streams)} streams and {len(rules)} rule sets.")

                svc = MultiStreamVideoAnalyticsService(
                    streams_config=streams,
                    rules_config=rules,
                    model_dir="yolov11",
                    queue_size=50,
                    target_fps=20.0,
                )

                if svc.start():
                    print("\nService started successfully! Press Ctrl+C to stop.")
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        print("\nStopping service...")
                    finally:
                        svc.stop()
                        print("Service stopped cleanly.")

        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
            print(f"\nError: {e}")

    def start_service(self):
        """Launch the service via asyncio."""
        logger.info("Launching async service")
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            print("\nService interrupted by user")

if __name__ == "__main__":
    ServiceRunner().start_service()
