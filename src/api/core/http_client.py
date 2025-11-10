import aiohttp
import asyncio
from typing import Any, Dict, Optional, Union, Callable
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HttpClient:
    def __init__(
        self,
        base_url: str,
        token_provider: Optional[Callable] = None,
        timeout_seconds: int = 15,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        debug: bool = False,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: Optional[aiohttp.ClientSession] = None
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._debug = debug
        self._token_provider = token_provider

        self._default_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if self._debug:
            logger.setLevel("DEBUG")

    async def __aenter__(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

        if self._token_provider:
            token = await self._token_provider()
            self._default_headers["Authorization"] = f"Bearer {token}"

        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _refresh_token(self):
        if self._token_provider:
            token = await self._token_provider()
            self._default_headers["Authorization"] = f"Bearer {token}"

    async def get(self, path: str, **kwargs) -> Any:
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> Any:
        return await self._request("POST", path, **kwargs)

    async def patch(self, path: str, **kwargs) -> Any:
        return await self._request("PATCH", path, **kwargs)

    async def delete(self, path: str, **kwargs) -> Any:
        return await self._request("DELETE", path, **kwargs)

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
        if self._session is None:
            raise RuntimeError("HttpClient not initialized. Use 'async with HttpClient(...)'")

        url = f"{self._base_url}/{path.lstrip('/')}"

        await self._refresh_token()
        request_headers = {**self._default_headers, **(headers or {})}

        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._debug:
                    logger.debug(f"{method} {url} | attempt={attempt + 1}")

                async with self._session.request(
                    method, url, params=params, json=json, data=data, headers=request_headers
                ) as response:
                    expected_statuses = {expected_status} if isinstance(expected_status, int) else set(expected_status)

                    if response.status not in expected_statuses:
                        text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Status {response.status}: {text[:500]}",
                        )

                    return await response.json() if return_json else await response.text()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self._max_retries:
                    wait_time = self._retry_backoff * (2 ** attempt)
                    if self._debug:
                        logger.warning(f"Request failed, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue

                logger.error(f"{method} {url} failed after {self._max_retries + 1} attempts", exc_info=True)
                raise

        raise last_exception