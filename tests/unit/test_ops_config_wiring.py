"""Ops config wiring smoke tests: new jobs exist, steps are anchored, hints live.

Catches the regressions step-level unit tests cannot see: a mistyped backend
name, a YAML anchor that silently dropped a step, or a job-level hint kwarg
lost in an edit. Loads the built-in configs the same way the parser tests do.
"""

from reme.config.config_parser import _load_config


def _edge() -> dict:
    return _load_config("ops_edge.yaml")


def _cloud() -> dict:
    return _load_config("ops_cloud.yaml")


def _backends(job: dict) -> list[str]:
    return [str(step.get("backend") or "") for step in job.get("steps") or []]


def test_edge_has_dream_audit_job_and_cron():
    edge = _edge()
    job = edge["jobs"]["ops_dream_audit"]
    assert job["backend"] == "base"
    assert _backends(job) == ["ops_dream_audit_step"]
    cron = edge["jobs"]["ops_dream_audit_cron"]
    assert cron["backend"] == "cron"
    assert cron["cron"] == "30 21 * * *"  # audits before the 22:30 dream push
    assert _backends(cron) == ["ops_dream_audit_step"]  # shared anchor survives edits


def test_dream_push_jobs_carry_job_level_salience_hint():
    """The hint must live on the job itself: step-entry kwargs never reach the
    dream-extract context (_run_steps awaits step(context) without kwargs)."""
    edge = _edge()
    for name in ("ops_dream_push", "ops_dream_push_cron"):
        job = edge["jobs"][name]
        hint = str(job.get("hint") or "").strip()
        assert hint, f"{name} lost its job-level hint kwarg"
        assert "salience" in hint.lower()
        assert "failure" in hint.lower()


def test_muscle_sync_rebuilds_the_reflex_index():
    edge = _edge()
    sync = edge["jobs"]["ops_muscle_sync"]
    assert _backends(sync)[-1] == "ops_reflex_index_step"
    assert _backends(edge["jobs"]["ops_muscle_sync_cron"]) == _backends(sync)
    assert edge["jobs"]["ops_reflex_index"]["backend"] == "base"
    assert _backends(edge["jobs"]["ops_reflex_index"]) == ["ops_reflex_index_step"]


def test_edge_search_weights_verified_status():
    step = _edge()["jobs"]["ops_search"]["steps"][0]
    assert step["backend"] == "ops_search_step"
    assert step["confirmed_boost"] == 0.05
    assert step["conflict_penalty"] == 0.15


def test_cloud_has_link_repair_job_and_maintenance_slot():
    cloud = _cloud()
    job = cloud["jobs"]["ops_link_repair"]
    assert job["backend"] == "base"
    assert job["enable_serve"] is True
    assert _backends(job) == ["ops_link_repair_step"]
    maintenance = _backends(cloud["jobs"]["ops_maintenance_cron"])
    assert "ops_link_repair_step" in maintenance
    assert maintenance.index("ops_link_repair_step") > maintenance.index("ops_transitions_step")
    assert maintenance.index("ops_link_repair_step") > maintenance.index("ops_decay_scan_step")
