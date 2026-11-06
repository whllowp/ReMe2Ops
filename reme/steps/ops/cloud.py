"""Edge-to-cloud HTTP plumbing shared by ops collaboration steps."""

import httpx

from ...constants import normalize_connect_host
from ..base_step import BaseStep

DEFAULT_CLOUD_HOST = "127.0.0.1"
DEFAULT_CLOUD_PORT = 2444
DEFAULT_CLOUD_TIMEOUT = 8.0


class CloudClient:
    """Minimal JSON client for calling the cloud ReMe instance.

    Unlike ``HttpClient`` (display-oriented chunk streaming), this posts JSON
    and returns the raw ``Response`` dict so steps can branch on ``success``
    and read ``metadata``. An injectable httpx ``transport`` keeps it testable
    without a live server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = DEFAULT_CLOUD_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = f"http://{normalize_connect_host(host)}:{port}"
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CloudClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post(self, action: str, payload: dict) -> dict:
        """POST ``/{action}`` and decode the JSON body; raises on HTTP/transport errors."""
        if self._client is None:
            raise RuntimeError("CloudClient not started; use 'async with'")
        resp = await self._client.post(f"/{action}", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else {}


def cloud_settings(step: BaseStep) -> tuple[str, int, float]:
    """Resolve cloud coordinates: step kwargs (YAML) with job-context overrides."""
    context = step.context.data if step.context is not None else {}

    def pick(key: str, default):
        value = context.get(key)
        if value is None or value == "":
            value = step.kwargs.get(key, default)
        return value

    host = str(pick("cloud_host", DEFAULT_CLOUD_HOST))
    port = int(pick("cloud_port", DEFAULT_CLOUD_PORT))
    timeout = float(pick("cloud_timeout", DEFAULT_CLOUD_TIMEOUT))
    return host, port, timeout
