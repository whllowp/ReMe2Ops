"""Edge-side connector client: stay connected to the cloud, react in real time.

One background job keeps a WebSocket session open to the cloud connector:

- on connect it says ``hello`` and immediately reconciles (usage up, muscle
  mirror down) by running ``ops_muscle_sync``;
- on ``muscle.updated`` events it reconciles again — a promotion reaches the
  edge within seconds instead of waiting for the nightly cron;
- ``ping``/``pong`` keeps the session alive through NATs and proxies;
- disconnection is normal: the job exits, and the BackgroundJob supervisor
  reconnects with capped exponential backoff. The scheduled sync jobs remain
  as the fallback path, so the connector is pure acceleration.

Any other client can implement the same 4-message protocol
(``hello``/``welcome``/``usage.delta``/``muscle.updated``) and connect at any
time — see ``GET /ops/devices`` on the cloud for live presence.
"""

import asyncio
import json
import os
import platform

from ...components import R
from ...components.job.background_job import BackgroundJob
from .connector import PROTOCOL_VERSION

PING_TYPE = "ping"


def default_device_id() -> str:
    """A stable-enough client identity without extra configuration."""
    return os.getenv("OPS_DEVICE_ID") or f"{platform.node() or 'edge'}"


@R.register("ops_connector")
class OpsConnectorJob(BackgroundJob):
    """Persistent connector session to the cloud; the supervisor reconnects."""

    def __init__(
        self,
        cloud_host: str = "127.0.0.1",
        cloud_port: int = 2444,
        device_id: str = "",
        connector_path: str = "/ops/connector",
        ping_interval: float = 30.0,
        connect_timeout: float = 10.0,
        sync_job: str = "ops_muscle_sync",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cloud_host = cloud_host
        self.cloud_port = cloud_port
        self.device_id = device_id or default_device_id()
        self.connector_path = connector_path.rstrip("/")
        self.ping_interval = ping_interval
        self.connect_timeout = connect_timeout
        self.sync_job = sync_job

    async def _reconcile(self, reason: str) -> None:
        """Run the sync job once; failures are logged, never fatal here."""
        job = self.app_context.jobs.get(self.sync_job) if self.app_context is not None else None
        if job is None:
            self.logger.warning(f"[{self.name}] sync job '{self.sync_job}' not found; cannot reconcile ({reason})")
            return
        try:
            response = await job()
            self.logger.info(f"[{self.name}] reconcile ({reason}): {response.answer}")
        except Exception as exc:  # noqa: BLE001 - a failed reconcile is retried by cron
            self.logger.warning(f"[{self.name}] reconcile ({reason}) failed: {type(exc).__name__}: {exc}")

    async def __call__(self, **kwargs):
        """Serve one session until it drops; the supervisor reconnects us."""
        try:
            import websockets
        except ImportError as exc:
            self.logger.warning(
                f"[{self.name}] 'websockets' not installed ({exc}); connector disabled, "
                "scheduled sync jobs remain the fallback",
            )
            return

        uri = f"ws://{self.cloud_host}:{self.cloud_port}{self.connector_path}"
        self.logger.info(f"[{self.name}] connecting to {uri} as '{self.device_id}'")
        async with websockets.connect(uri, open_timeout=self.connect_timeout, ping_interval=20) as ws:
            await ws.send(json_dumps({
                "type": "hello",
                "device_id": self.device_id,
                "protocol": PROTOCOL_VERSION,
                "capabilities": ["ops.edge"],
            }))
            welcome = json_loads(await ws.recv())
            if welcome.get("type") != "welcome":
                raise RuntimeError(f"expected welcome, got: {str(welcome)[:120]}")
            self.logger.info(f"[{self.name}] connected; reconciling on welcome")
            await self._reconcile("welcome")

            while self._stop_event is None or not self._stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.ping_interval)
                except asyncio.TimeoutError:
                    await ws.send(json_dumps({"type": PING_TYPE}))
                    continue
                message = json_loads(raw) if raw else {}
                mtype = str(message.get("type") or "")
                if mtype == "muscle.updated":
                    paths = message.get("paths") or []
                    self.logger.info(f"[{self.name}] muscle.updated from cloud: {paths}")
                    await self._reconcile("muscle.updated")
                elif mtype == "pong":
                    continue
                elif mtype == "error":
                    self.logger.warning(f"[{self.name}] cloud error frame: {message}")
                else:
                    self.logger.debug(f"[{self.name}] ignored frame type {mtype!r}")
        self.logger.info(f"[{self.name}] session closed; supervisor will reconnect")


def json_dumps(payload: dict) -> str:
    """Compact JSON for the wire."""
    return json.dumps(payload, ensure_ascii=False)


def json_loads(raw: str | bytes) -> dict:
    """Tolerant frame decode; a bad frame becomes an empty mapping."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
