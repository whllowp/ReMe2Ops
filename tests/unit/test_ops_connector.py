"""Ops connector: registry, wire protocol, broadcast hook, and the client job."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.steps.ops.connector import (
    ConnectorRegistry,
    OpsConnectorService,
    get_connector_registry,
)
from reme.steps.ops.connector_client import OpsConnectorJob
from reme.steps.ops.promote import OpsMusclePromoteStep


class FakeWebSocket:
    """WebSocket double: records sends, replays queued receives."""

    def __init__(self, incoming=(), *, fail_send_after=None):
        self.sent: list[str] = []
        self._incoming = list(incoming)
        self._fail_send_after = fail_send_after
        self.closed = None
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text):
        if self._fail_send_after is not None and len(self.sent) >= self._fail_send_after:
            raise ConnectionError("socket dead")
        self.sent.append(text)

    async def recv(self):
        if not self._incoming:
            raise ConnectionError("closed")
        return self._incoming.pop(0)

    async def receive_text(self):
        return await self.recv()

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)

    def frames(self):
        return [json.loads(frame) for frame in self.sent]


def _service() -> OpsConnectorService:
    return OpsConnectorService()


@pytest.mark.asyncio
async def test_registry_register_replace_and_broadcast():
    registry = ConnectorRegistry()
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    await registry.register(ws1, "edge-a")
    await registry.register(FakeWebSocket(), "edge-b")

    delivered = await registry.broadcast({"type": "muscle.updated", "paths": ["x"]}, exclude={"edge-b"})
    assert delivered == 1
    assert json.loads(ws1.sent[0])["type"] == "muscle.updated"
    assert registry.count() == 2

    # A reconnecting device replaces (and closes) its stale session.
    ws3 = FakeWebSocket()
    await registry.register(ws3, "edge-a")
    assert ws1.closed and ws1.closed[0] == 4002
    assert registry.device_ids == ["edge-b", "edge-a"] or set(registry.device_ids) == {"edge-b", "edge-a"}


@pytest.mark.asyncio
async def test_registry_broadcast_prunes_dead_sockets():
    registry = ConnectorRegistry()
    dead, alive = FakeWebSocket(fail_send_after=0), FakeWebSocket()
    await registry.register(dead, "dead-1")
    await registry.register(alive, "alive-1")

    delivered = await registry.broadcast({"type": "muscle.updated"})
    assert delivered == 1
    assert "dead-1" not in registry.device_ids
    assert alive.frames()[0]["type"] == "muscle.updated"


@pytest.mark.asyncio
async def test_session_protocol_hello_welcome_delta_ping_bye(monkeypatch):
    service = _service()
    ws = FakeWebSocket(incoming=[
        json.dumps({"type": "hello", "device_id": "edge-a"}),
        json.dumps({"type": "usage.delta", "records": [{"path": "p.md", "outcome": "solved"}]}),
        json.dumps({"type": "ping"}),
        json.dumps({"type": "bogus"}),
        json.dumps({"type": "bye"}),
    ])
    ingest_job = AsyncMock(return_value=Mock(success=True, metadata={"usage_ingest": {"accepted": 1, "fast_track_candidates": []}}))
    app = SimpleNamespace(context=SimpleNamespace(jobs={"ops_usage_ingest": ingest_job}))

    await service._connector_session(ws, app)

    frames = ws.frames()
    assert frames[0]["type"] == "welcome"
    assert frames[0]["device_id"] == "edge-a"
    types = [frame["type"] for frame in frames]
    assert "usage.acked" in types
    assert "pong" in types
    assert "error" in types  # bogus type rejected
    ingest_job.assert_awaited_once()
    assert ingest_job.await_args.kwargs["records"][0]["path"] == "p.md"
    assert service.connector_registry.count() == 0  # bye unregistered


@pytest.mark.asyncio
async def test_usage_delta_duplicate_is_acked_without_reingest(tmp_path, monkeypatch):
    import tempfile

    workspace = tempfile.mkdtemp()
    service = _service()
    # point the ledger at a tmp workspace
    from reme.steps.ops.connector import EventLedger

    ledger = EventLedger(workspace, "metadata")
    service.event_ledger = ledger
    ws = FakeWebSocket(incoming=[
        json.dumps({"type": "hello", "device_id": "edge-a"}),
        json.dumps({"type": "usage.delta", "event_id": "evt-1", "records": [{"path": "p.md", "outcome": "solved"}]}),
        json.dumps({"type": "usage.delta", "event_id": "evt-1", "records": [{"path": "p.md", "outcome": "solved"}]}),
        json.dumps({"type": "bye"}),
    ])
    ingest_job = AsyncMock(return_value=Mock(success=True, metadata={"usage_ingest": {"accepted": 1, "fast_track_candidates": []}}))
    app = SimpleNamespace(context=SimpleNamespace(jobs={"ops_usage_ingest": ingest_job}))

    await service._connector_session(ws, app)
    ingest_job.assert_awaited_once()  # second frame deduped
    acks = [f for f in ws.frames() if f["type"] == "usage.acked"]
    assert len(acks) == 2
    assert acks[0].get("duplicate") is None
    assert acks[1].get("duplicate") is True and acks[1]["event_id"] == "evt-1"
    # the operational log persisted the delivered frame (Apiary)
    log_lines = ledger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("usage.delta" in line for line in log_lines)


@pytest.mark.asyncio
async def test_session_requires_hello_with_device_id():
    service = _service()
    ws = FakeWebSocket(incoming=[json.dumps({"type": "ping"})])
    await service._connector_session(ws, SimpleNamespace(context=SimpleNamespace(jobs={})))
    assert ws.closed and ws.closed[0] == 4001

    ws2 = FakeWebSocket(incoming=[json.dumps({"type": "hello"})])
    await service._connector_session(ws2, SimpleNamespace(context=SimpleNamespace(jobs={})))
    assert ws2.closed and ws2.closed[0] == 4001


def test_get_connector_registry_detects_connector_service():
    ctx = ApplicationContext(workspace_dir=".")
    assert get_connector_registry(ctx) is None
    ctx.service = _service()
    assert get_connector_registry(ctx) is ctx.service.connector_registry


@pytest.mark.asyncio
async def test_promote_step_broadcasts_muscle_updated(tmp_path):
    source = tmp_path / "digest" / "procedure" / "a.md"
    source.parent.mkdir(parents=True)
    source.write_text("body", encoding="utf-8")

    ctx = ApplicationContext(workspace_dir=str(tmp_path))
    service = _service()
    ws = FakeWebSocket()
    await service.connector_registry.register(ws, "edge-a")
    ctx.service = service

    step = OpsMusclePromoteStep(app_context=ctx, file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path))
    step.run_job = AsyncMock(
        side_effect=[
            Mock(success=True, answer="Moved"),
            Mock(success=True, answer="stamped"),
        ],
    )
    response = await step(RuntimeContext(path="digest/procedure/a.md"))

    assert response.success is True
    frames = ws.frames()
    assert frames[-1]["type"] == "muscle.updated"
    assert frames[-1]["paths"] == ["digest/muscle/a.md"]


class _FakeWebsocketsModule:
    """Minimal 'websockets' module double for the client job."""

    closed = type("ConnectionClosed", (Exception,), {})
    sessions = []

    def __init__(self, incoming):
        self._incoming = list(incoming)

    def connect(self, uri, open_timeout=None, ping_interval=None):
        session = _FakeSession(uri, self._incoming)
        _FakeWebsocketsModule.sessions.append(session)
        return session


class _FakeSession:
    def __init__(self, uri, incoming):
        self.uri = uri
        self.sent = []
        self._incoming = list(incoming)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        if not self._incoming:
            raise ConnectionError("closed")
        return self._incoming.pop(0)

    async def receive_text(self):
        return await self.recv()


@pytest.mark.asyncio
async def test_connector_job_reconciles_on_welcome_and_events(tmp_path, monkeypatch):
    sync_job = AsyncMock(return_value=Mock(success=True, answer="synced"))
    incoming = [
        json.dumps({"type": "welcome", "device_id": "edge-x"}),
        json.dumps({"type": "muscle.updated", "paths": ["digest/muscle/a.md"]}),
    ]
    module = _FakeWebsocketsModule(incoming)
    monkeypatch.setitem(__import__("sys").modules, "websockets", module)

    ctx = ApplicationContext(workspace_dir=str(tmp_path))
    ctx.jobs = {"ops_muscle_sync": sync_job}
    job = OpsConnectorJob(
        app_context=ctx,
        cloud_host="127.0.0.1",
        cloud_port=2444,
        device_id="edge-x",
        ping_interval=5.0,
    )
    with pytest.raises(ConnectionError):  # incoming exhausted -> session drops, supervisor reconnects
        await job()

    session = _FakeWebsocketsModule.sessions[-1]
    assert session.uri == "ws://127.0.0.1:2444/ops/connector"
    assert session.sent[0]["type"] == "hello" and session.sent[0]["device_id"] == "edge-x"
    assert sync_job.await_count == 2  # once on welcome, once on muscle.updated
