import aiohttp
import asyncio
import ssl
from aiohttp import ClientSession, ClientTimeout
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIClient:
    """
    Asynchronous, generic API client for performing HTTP requests with retries and SSL support.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 15,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.ssl_context = ssl.create_default_context() if verify_ssl else False
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        retries = 0

        while True:
            try:
                async with self._session.request(method, url, ssl=self.ssl_context, **kwargs) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        logger.warning(f"{method} {url} failed [{resp.status}]: {text}")
                    try:
                        return await resp.json(content_type=None)
                    except Exception:
                        return text
            except aiohttp.ClientError as e:
                retries += 1
                if retries > self.max_retries:
                    logger.error(f"Exceeded retries for {url}: {e}")
                    raise
                logger.warning(f"Retry {retries}/{self.max_retries} for {url} due to: {e}")
                await asyncio.sleep(self.retry_backoff * retries)

    async def get(self, endpoint: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None):
        return await self._request("GET", endpoint, headers=headers, params=params)

    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, auth: Optional[Any] = None):
        return await self._request("POST", endpoint, json=data, headers=headers, auth=auth)

    async def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None):
        return await self._request("PATCH", endpoint, json=data, headers=headers)

    async def delete(self, endpoint: str, headers: Optional[Dict[str, str]] = None):
        return await self._request("DELETE", endpoint, headers=headers)
