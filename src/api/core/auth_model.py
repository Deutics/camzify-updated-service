from typing import Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuthModel:
    """
    Handles asynchronous user authentication and token management.
    Works with AsyncAPIClient.
    """

    def __init__(self, api_client):
        self.api_client = api_client
        self.token: Optional[str] = None

    async def authenticate(self, username: str, password: str):
        """Authenticate the user and store the access token."""
        data = {"login": username, "password": password}
        logger.info("Authenticating user...")

        response = await self.api_client.post("/api/v1/auth/login/", data=data)
        self.token = response.get("access") if isinstance(response, dict) else None

        if not self.token:
            logger.error(f"Authentication failed: {response}")
            raise ValueError("Authentication failed. Invalid credentials or missing token.")

        logger.info("Authentication successful ")
        return self.token

    def get_headers(self) -> Dict[str, str]:
        """Return authorization headers for authenticated requests."""
        if not self.token:
            raise ValueError("No token found. Please authenticate first.")
        return {"Authorization": f"Bearer {self.token}"}
