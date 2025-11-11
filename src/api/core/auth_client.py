import aiohttp
from typing import Optional
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuthClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        login_endpoint: str = "/api/v1/auth/login/",
        token_refresh_buffer: int = 300,
        timeout_seconds: int = 15
    ):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._login_endpoint = login_endpoint
        self._token_refresh_buffer = token_refresh_buffer
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(force_close=True)
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                connector=connector
            )
        await self._ensure_valid_token()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def get_token(self) -> str:
        await self._ensure_valid_token()
        return self._access_token

    async def _ensure_valid_token(self):
        if self._is_token_valid():
            return
        
        logger.info("Token expired or missing, requesting new token")
        await self._request_token()

    def _is_token_valid(self) -> bool:
        if not self._access_token or not self._token_expiry:
            return False
        
        buffer_time = datetime.now() + timedelta(seconds=self._token_refresh_buffer)
        return buffer_time < self._token_expiry

    async def _request_token(self):
        url = f"{self._base_url}{self._login_endpoint}"
        payload = {
            "login": self._username,
            "password": self._password
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            logger.info(f"Requesting access token for user: {self._username}")
            logger.debug(f"POST {url}")
            
            async with self._session.post(
                url, 
                json=payload, 
                headers=headers,
                allow_redirects=False
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ValueError(f"Authentication failed [{response.status}]: {text}")

                data = await response.json()
                self._access_token = data.get("access")
                expires_in = data.get("expires_in", 3600)
                
                if not self._access_token:
                    raise ValueError("No access token in response")

                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
                logger.info(f"Access token obtained, expires at {self._token_expiry.strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Token request failed: {e}", exc_info=True)
            raise