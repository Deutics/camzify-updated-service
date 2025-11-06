# api/http_client.py
import aiohttp
import asyncio
from typing import Any, Dict, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)

class ApiHandler:
    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 15,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        debug: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.debug = debug
        self._default_headers = {
            "Accept": "application/json",
            **(default_headers or {})
        }
        if token:
            self._default_headers["Authorization"] = f"Bearer {token}"

        if self.debug:
            logger.setLevel("DEBUG")
        logger.info(f"ApiHandler initialized for base_url={self.base_url}")

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        expected_status: Union[int, set, tuple] = (200,),
        return_json: bool = True,
    ) -> Any:
        assert self.session is not None, "ApiHandler session not started. Use 'async with ApiHandler(...)'."
        url = f"{self.base_url}/{path.lstrip('/')}"
        req_headers = {**self._default_headers, **(headers or {})}

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.debug:
                    logger.debug(f"{method} {url} params={params} json={bool(json)} attempt={attempt}")

                async with self.session.request(
                    method, url, params=params, json=json, data=data, headers=req_headers
                ) as resp:
                    expected = {expected_status} if isinstance(expected_status, int) else set(expected_status)
                    if resp.status not in expected:
                        text = await resp.text()
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=f"Unexpected status {resp.status}: {text[:500]}",
                        )
                    return await resp.json() if return_json else await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                logger.error(f"HTTP {method} {url} failed after retries: {e}", exc_info=True)
                raise

        raise last_exc

    async def get(self, path: str, **kwargs) -> Any:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> Any:
        return await self._request("POST", path, **kwargs)

    async def patch(self, path: str, **kwargs) -> Any:
        return await self._request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> Any:
        return await self._request("DELETE", path, **kwargs)