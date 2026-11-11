"""Async client for the AlphaRidge API, used by miners and validators. Every request is
signed with the neuron's hotkey.

    client = AlpharidgeAPIClient(base_url="https://api.alpharidge.ai", wallet=wallet)
    points = await client.ledger_points(epoch=7)
"""

from typing import Optional, Dict, Any

import httpx
import bittensor as bt

from alpharidge_ai import __version__
from alpharidge_ai.ledger import request_auth
from alpharidge_ai.schema import Claim, canonical_body


class AlpharidgeAPIError(Exception):
    """Base exception for Alpharidge API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 detail: Optional[str] = None, retry_after: Optional[float] = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(self.message)


class AuthenticationError(AlpharidgeAPIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(AlpharidgeAPIError):
    """Raised when the user is not authorized (not a validator)."""
    pass


class NotFoundError(AlpharidgeAPIError):
    """Raised when a resource is not found."""
    pass


class AlpharidgeAPIClient:
    """
    Async client for the Alpharidge AI API.

    This client handles authentication automatically and provides
    typed methods for all API endpoints.
    """

    def __init__(
        self,
        base_url: str,
        wallet: bt.Wallet,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Alpharidge API client.

        Args:
            base_url: The base URL of the API (e.g., "http://localhost:8000")
            wallet: The Bittensor wallet to use for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self.ss58_address = wallet.hotkey.ss58_address
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[httpx.AsyncClient] = None

        bt.logging.info(f"Initialized AlpharidgeAPIClient for {self.ss58_address}")

    def _sign(self, request: httpx.Request) -> None:
        """Attach signed auth headers bound to this exact request."""
        request.headers.update(request_auth.signed_headers(
            self.wallet.hotkey, request.method, str(request.url), request.content))
        request.headers["X-Validator-Version"] = __version__

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _handle_response_error(self, response: httpx.Response):
        """Handle error responses from the API."""
        status_code = response.status_code

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        if status_code == 401:
            raise AuthenticationError(
                "Authentication failed",
                status_code=status_code,
                detail=detail,
            )
        elif status_code == 403:
            raise AuthorizationError(
                "Not authorized",
                status_code=status_code,
                detail=detail,
            )
        elif status_code == 404:
            raise NotFoundError(
                "Resource not found",
                status_code=status_code,
                detail=detail,
            )
        else:
            try:
                retry_after = float(response.headers["Retry-After"])
            except (KeyError, ValueError):
                retry_after = None
            raise AlpharidgeAPIError(
                f"API request failed with status {status_code}",
                status_code=status_code,
                detail=detail,
                retry_after=retry_after,
            )

    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        content: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/claims")
            json: JSON body for POST requests
            params: Query parameters
            content: Raw JSON body, sent and signed as given

        Returns:
            Response JSON as a dictionary
        """
        client = await self._get_client()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                headers = {"Content-Type": "application/json"} if content is not None else None
                request = client.build_request(method, endpoint, json=json, params=params,
                                               content=content, headers=headers)
                self._sign(request)
                response = await client.send(request)

                if response.status_code >= 400:
                    self._handle_response_error(response)

                return response.json()

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    bt.logging.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    await self._sleep(self.retry_delay * (attempt + 1))

        raise AlpharidgeAPIError(
            f"Request failed after {self.max_retries} attempts: {last_error}"
        )

    async def _sleep(self, seconds: float):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)

    # =========================================================================
    # Event claims
    # =========================================================================

    async def submit_claim(self, claim: Claim) -> Dict[str, Any]:
        """Push one claim. The signature covers its canonical bytes exactly as sent."""
        return await self._request("POST", "/claims", content=canonical_body(claim))

    async def claim_status(self, claim_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/claims/{claim_id}")

    async def list_events(self, **query) -> Dict[str, Any]:
        """Published events, filtered by entity, type, category, since and so on."""
        params = {k: v for k, v in query.items() if v is not None}
        return await self._request("GET", "/events", params=params)

    async def event_detail(self, event_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/events/{event_id}")

    # =========================================================================
    # Event ledger
    # =========================================================================

    async def ledger_points(self, epoch: int) -> Dict[str, Any]:
        return await self._request("GET", "/ledger/points", params={"epoch": int(epoch)})

    async def ledger_leaves(self, epoch: int, cursor: Optional[str] = None,
                            limit: int = 5000) -> Dict[str, Any]:
        params = {"epoch": int(epoch), "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/ledger/leaves", params=params)

    async def ledger_sample(self, epoch: int, cursor: Optional[str] = None,
                            limit: int = 500) -> Dict[str, Any]:
        params = {"epoch": int(epoch), "validator": self.ss58_address, "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/ledger/sample", params=params)

    async def post_reverify(self, epoch: int, results: list) -> Dict[str, Any]:
        return await self._request("POST", "/ledger/reverify",
                                   json={"epoch": int(epoch), "results": results})

    # =========================================================================
    # Held scoops
    # =========================================================================

    async def scoops_open(self) -> Dict[str, Any]:
        return await self._request("GET", "/scoops/open")

    async def post_confirm(self, scoop_id: int, url: str) -> Dict[str, Any]:
        return await self._request("POST", "/scoops/confirm",
                                   json={"scoop_id": int(scoop_id), "url": url})

    # =========================================================================
    # Page judge (listed validators)
    # =========================================================================

    async def judge_pending(self) -> Dict[str, Any]:
        return await self._request("GET", "/judge/pending")

    async def post_verdict(self, body: bytes) -> Dict[str, Any]:
        """One item's verdict. The signature covers these bytes exactly as sent."""
        return await self._request("POST", "/judge/verdicts", content=body)
