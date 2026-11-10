"""Cloud-side ops connector: any client can connect at any time.

The cloud instance stays a normal ReMe HTTP service; this module turns it into
a connector by subclassing the HTTP service and adding:

- ``WS /ops/connector`` — a persistent session: the client says ``hello``, the
  cloud answers ``welcome``, then the cloud pushes ``muscle.updated`` events
  the moment promotion happens, and accepts ``usage.delta`` records any time.
- ``GET /ops/devices`` — presence: which clients are connected right now.

Disconnection is a normal state, not an error: offline clients catch up on
their next connection (the ``welcome`` event makes the client reconcile) or
through the scheduled sync jobs. Everything here is additive — dropping back
to the plain ``http`` service backend removes the connector entirely.
"""

import asyncio
import datetime as dt
import json
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from ...components.service.http_service import HttpService
from ...components import R

if TYPE_CHECKING:
    from ...application import Application

PROTOCOL_VERSION = 1


def now_iso() -> str:
    """Timestamp for presence and event payloads."""
    return dt.datetime.now().isoformat(timespec="seconds")


class Connection:
    """One live client session tracked by the registry."""

    def __init__(self, device_id: str, websocket: WebSocket, meta: dict | None = None):
        self.device_id = device_id
        self.websocket = websocket
        self.meta = dict(meta or {})
        self.connected_at = now_iso()
        self.last_seen = self.connected_at
        self.messages_in = 0
        self.events_out = 0

    def touch(self) -> None:
        self.last_seen = now_iso()
        self.messages_in += 1

    def snapshot(self) -> dict:
        """Presence view for ``/ops/devices``; never exposes transport objects."""
        return {
            "device_id": self.device_id,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "messages_in": self.messages_in,
            "events_out": self.events_out,
            **self.meta,
        }


class ConnectorRegistry:
    """Thread-light registry of live client sessions on the cloud side."""

    def __init__(self):
        self._connections: dict[str, Connection] = {}
        self._lock = asyncio.Lock()

    @property
    def device_ids(self) -> list[str]:
        return list(self._connections)

    def count(self) -> int:
        return len(self._connections)

    async def register(self, websocket: WebSocket, device_id: str, meta: dict | None = None) -> Connection:
        """Accept a client; a reconnecting device replaces its stale session."""
        async with self._lock:
            stale = self._connections.get(device_id)
            if stale is not None:
                try:
                    await stale.websocket.close(code=4002, reason="replaced by a newer session")
                except Exception:  # noqa: BLE001 - the stale socket may already be gone
                    pass
            connection = Connection(device_id, websocket, meta)
            self._connections[device_id] = connection
            return connection

    async def unregister(self, device_id: str, websocket: WebSocket | None = None) -> None:
        """Drop a session; a replaced session must not unregister its successor."""
        async with self._lock:
            current = self._connections.get(device_id)
            if current is None:
                return
            if websocket is not None and current.websocket is not websocket:
                return
            self._connections.pop(device_id, None)

    async def broadcast(self, payload: dict, exclude: set[str] | None = None) -> int:
        """Send one event to every live client; dead sockets are pruned."""
        delivered = 0
        message = json.dumps(payload, ensure_ascii=False)
        async with self._lock:
            targets = [c for device_id, c in self._connections.items() if device_id not in (exclude or set())]
        for connection in targets:
            try:
                await connection.websocket.send_text(message)
                connection.events_out += 1
                delivered += 1
            except Exception:  # noqa: BLE001 - a dead socket must not break others
                await self.unregister(connection.device_id)
        return delivered

    def snapshot(self) -> list[dict]:
        """Presence list, newest activity first."""
        connections = sorted(self._connections.values(), key=lambda c: c.last_seen, reverse=True)
        return [connection.snapshot() for connection in connections]


def get_connector_registry(app_context) -> "ConnectorRegistry | None":
    """Return the cloud connector registry when this process hosts one."""
    service = getattr(app_context, "service", None)
    if isinstance(service, OpsConnectorService):
        return service.connector_registry
    return None


async def broadcast_muscle_update(step, paths: list[str], *, reason: str = "promoted") -> None:
    """Push a muscle-updated event to every connected client; failure-safe."""
    registry = get_connector_registry(step.app_context) if step.app_context is not None else None
    if registry is None or not paths:
        return
    try:
        delivered = await registry.broadcast(
            {"type": "muscle.updated", "paths": list(paths), "reason": reason, "ts": now_iso()},
        )
        if delivered:
            step.logger.info(f"[{step.name}] muscle.updated broadcast to {delivered} client(s): {paths}")
    except Exception as exc:  # noqa: BLE001 - push failures never break promotion
        step.logger.warning(f"[{step.name}] muscle.updated broadcast failed: {type(exc).__name__}: {exc}")


@R.register("ops_http")
class OpsConnectorService(HttpService):
    """Standard ReMe HTTP service plus the ops connector WebSocket endpoint."""

    def __init__(self, connector_path: str = "/ops/connector", devices_path: str = "/ops/devices", **kwargs):
        super().__init__(**kwargs)
        self.connector_path = connector_path.rstrip("/") or "/ops/connector"
        self.devices_path = devices_path.rstrip("/") or "/ops/devices"
        self.connector_registry = ConnectorRegistry()

    def finalize_service(self, app: "Application") -> None:  # noqa: F821 - typing only
        """Attach connector routes before the SPA catch-all is mounted."""
        async def devices():
            return {"success": True, "answer": f"{self.connector_registry.count()} client(s) connected", "metadata": {"devices": self.connector_registry.snapshot()}}

        async def connector_session(websocket: WebSocket):
            await self._connector_session(websocket, app)

        # Registered ahead of super()'s SPA catch-all so exact paths win.
        self.service.add_api_websocket_route(self.connector_path, connector_session)
        self.service.get(self.devices_path)(devices)
        super().finalize_service(app)

    async def _connector_session(self, websocket: WebSocket, app) -> None:  # noqa: F821 - typing only
        """One client session: hello -> welcome, then events and deltas."""
        await websocket.accept()
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            hello = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - no valid hello in time
            await websocket.close(code=4001, reason=f"hello expected: {type(exc).__name__}")
            return
        device_id = str(hello.get("device_id") or "").strip()
        if not device_id:
            await websocket.close(code=4001, reason="device_id required")
            return
        meta = {"capabilities": hello.get("capabilities") or [], "protocol": hello.get("protocol") or PROTOCOL_VERSION}
        connection = await self.connector_registry.register(websocket, device_id, meta)
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "device_id": device_id,
            "protocol": PROTOCOL_VERSION,
            "hint": "reconcile now; muscle.updated events will follow",
        }))
        try:
            while True:
                raw = await websocket.receive_text()
                connection.touch()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "error": "message must be JSON"}))
                    continue
                mtype = str(message.get("type") or "").strip()
                if mtype == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "ts": now_iso()}))
                elif mtype == "usage.delta":
                    await self._ingest_usage(app, websocket, device_id, message)
                elif mtype == "bye":
                    break
                else:
                    await websocket.send_text(json.dumps({"type": "error", "error": f"unknown type {mtype!r}"}))
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - a broken session is a disconnect
            connection_meta = {"last_error": f"{type(exc).__name__}: {exc}"}
            connection.meta.update(connection_meta)
        finally:
            await self.connector_registry.unregister(device_id, websocket)

    async def _ingest_usage(self, app, websocket: WebSocket, device_id: str, message: dict) -> None:
        """Run the usage ingest job in-process; fast-track judge follows inside it."""
        records = message.get("records")
        job = app.context.jobs.get("ops_usage_ingest")
        if job is None:
            await websocket.send_text(json.dumps({"type": "error", "error": "ops_usage_ingest job not available"}))
            return
        if not isinstance(records, list):
            await websocket.send_text(json.dumps({"type": "error", "error": "records must be a list"}))
            return
        response = await job(records=records, source_device=device_id)
        await websocket.send_text(json.dumps({
            "type": "usage.acked",
            "accepted": (response.metadata.get("usage_ingest") or {}).get("accepted", 0),
            "fast_track_candidates": (response.metadata.get("usage_ingest") or {}).get("fast_track_candidates", []),
            "success": response.success,
        }))
