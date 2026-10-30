"""Bounded HTTP transport shared by the local Ollama adapters.

The deadline cancels the client coroutine and closes its connection.  It cannot
guarantee that Ollama stops model computation after the client disconnects.
"""
from __future__ import annotations

import asyncio

import httpx


class OllamaHTTPDeadlineExceeded(Exception):
    """The client-side wall-clock request deadline elapsed."""


class OllamaHTTPUnavailable(Exception):
    """The local Ollama endpoint could not be reached or returned an error."""


class OllamaHTTPInvalidResponse(Exception):
    """The HTTP response violated a bounded transport requirement."""


class BoundedOllamaHTTPClient:
    """Synchronous boundary backed by an async, cancellable HTTP request.

    A single connect failure may be retried within the original absolute
    deadline. Timeouts, reads, writes, HTTP errors and completed generations
    are never retried at this layer because the server may already be running
    inference.
    """

    def __init__(self, base_url: str, max_response_bytes: int,
                 connect_retries: int = 1, *,
                 transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None):
        if isinstance(connect_retries, bool) or not 0 <= connect_retries <= 2:
            raise ValueError("connect_retries must be between zero and two")
        self.base_url = base_url.rstrip("/")
        self.max_response_bytes = max_response_bytes
        self.connect_retries = connect_retries
        self.transport = transport

    def post_chat(self, request_body: dict, timeout_seconds: float) -> bytes:
        if timeout_seconds <= 0:
            raise OllamaHTTPDeadlineExceeded
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise OllamaHTTPUnavailable(
                "The synchronous Ollama adapter cannot run inside an active event loop."
            )
        return asyncio.run(self._post_chat(request_body, timeout_seconds))

    async def _post_chat(self, request_body: dict, timeout_seconds: float) -> bytes:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        attempt = 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise OllamaHTTPDeadlineExceeded
            timeout = httpx.Timeout(
                connect=min(5, remaining), read=remaining, write=remaining, pool=remaining,
            )
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, trust_env=False, follow_redirects=False,
                    transport=self.transport,
                ) as client:
                    async with asyncio.timeout(remaining):
                        async with client.stream(
                            "POST", self.base_url + "/api/chat", json=request_body,
                            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
                        ) as response:
                            if response.status_code != 200:
                                raise OllamaHTTPUnavailable
                            if response.headers.get(
                                "content-encoding", "identity"
                            ).lower() != "identity":
                                raise OllamaHTTPInvalidResponse
                            length = response.headers.get("content-length")
                            if length is not None:
                                try:
                                    parsed_length = int(length)
                                except ValueError:
                                    raise OllamaHTTPInvalidResponse from None
                                if parsed_length < 0 or parsed_length > self.max_response_bytes:
                                    raise OllamaHTTPInvalidResponse
                            data = bytearray()
                            async for chunk in response.aiter_bytes(chunk_size=16384):
                                data.extend(chunk)
                                if len(data) > self.max_response_bytes:
                                    raise OllamaHTTPInvalidResponse
                            return bytes(data)
            except (TimeoutError, httpx.TimeoutException):
                raise OllamaHTTPDeadlineExceeded from None
            except httpx.ConnectError:
                if attempt >= self.connect_retries:
                    raise OllamaHTTPUnavailable from None
                attempt += 1
                # Yield once without extending the original absolute deadline.
                await asyncio.sleep(0)
            except httpx.HTTPError:
                raise OllamaHTTPUnavailable from None
