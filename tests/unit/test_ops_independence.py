"""Evidence-independence dedupe: same-session solves count once for promotion.

Sage-Taiyi (arXiv 2611.12762v2): several observations sharing one unexamined
premise do not independently support each other. Here: repeated solves of the
same SOP inside one session must not pile up as multi-session promotion
evidence. Two layers are kept separate (review finding 6):

- counting layer: the online aggregate and the replay simulator share the
  exact same distinct-session pure function (``solved_session_key``), so their
  counts agree by construction — asserted behaviorally below;
- decision layer: replay and scan can only be compared when the ideal point is
  pinned (replay uses the fixed ``IDEAL_FLOOR``; the scan persists a
  monotonic ideal), so those tests write ``ideal_point.json`` explicitly.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.steps.ops.multi_objective import (
    IDEAL_FLOOR,
    ideal_point_path,
    objective_vector,
    rank_promotion_candidates,
)
from reme.steps.ops.promote import OpsPromotionScanStep
from reme.steps.ops.replay import build_session_trees, policy_params, replay_once
from reme.steps.ops.usage import aggregate_usage, solved_session_key

HERO = "digest/procedure/hero.md"
WEIGHTS = {"rate": 0.5, "solved": 0.5, "breadth": 0.0}


def _record(ts, path, outcome, session=None, device="d1", query=None):
    record = {"ts": ts, "path": path, "outcome": outcome, "source_device": device}
    if session is not None:
        record["session_id"] = session
    if query is not None:
        record["query"] = query
    return record


def _stats_shape():
    """The historical 7-key entry shape (byte-identity contract, finding 3)."""
    return {
        "solved": 0,
        "failed": 0,
        "total": 0,
        "success_rate": 0.0,
        "last_used": "",
        "devices": 0,
        "device_list": [],
    }


# ----- counting layer (usage.py) -----------------------------------------------


def test_same_session_three_solves_count_once():
    records = [_record("t1", HERO, "solved", "s1"), _record("t2", HERO, "solved", "s1"), _record("t3", HERO, "solved", "s1")]
    stats = aggregate_usage(records, session_dedupe=True)
    assert stats[HERO]["solved"] == 3  # raw count untouched
    assert stats[HERO]["distinct_session_solved"] == 1  # one evidence unit


def test_session_dedupe_default_off_is_byte_identical():
    records = [
        _record("t1", HERO, "solved", "s1", device="d1"),
        _record("t2", HERO, "solved", "s1", device="d2"),
        _record("t3", HERO, "failed", "s2"),
    ]
    implicit = aggregate_usage(records)
    explicit = aggregate_usage(records, session_dedupe=False)
    expected = {
        **_stats_shape(),
        "solved": 2,
        "failed": 1,
        "total": 3,
        "success_rate": 0.6667,
        "last_used": "t3",
        "devices": 2,
        "device_list": ["d1", "d2"],
    }
    # exact 7-key equality: no new key leaks into the default output
    assert implicit[HERO] == expected
    assert implicit == explicit
    assert set(implicit[HERO]) == set(_stats_shape())
    assert "distinct_session_solved" not in implicit[HERO]


def test_unsessioned_records_fall_back_to_raw_count():
    records = [_record("t1", HERO, "solved"), _record("t2", HERO, "solved"), _record("t3", HERO, "solved")]
    stats = aggregate_usage(records, session_dedupe=True)
    assert stats[HERO]["distinct_session_solved"] == 3  # each counts as its own evidence


def test_mixed_sessioned_and_unsessioned_solves():
    records = [
        _record("t1", HERO, "solved", "s1"),
        _record("t2", HERO, "solved", "s1"),
        _record("t3", HERO, "solved"),  # session-less: its own evidence
    ]
    stats = aggregate_usage(records, session_dedupe=True)
    assert stats[HERO]["distinct_session_solved"] == 2


def test_failed_and_invalid_records_do_not_enter_the_session_count():
    records = [
        _record("t1", HERO, "failed", "s1"),
        _record("t2", HERO, "solved", "s2"),
        _record("t3", HERO, "maybe", "s3"),  # invalid outcome ignored
        "junk",
    ]
    stats = aggregate_usage(records, session_dedupe=True)
    assert stats[HERO]["distinct_session_solved"] == 1
    assert stats[HERO]["failed"] == 1


def test_breadth_is_never_session_deduped():
    records = [
        _record("t1", HERO, "solved", "s1", device="d1"),
        _record("t2", HERO, "solved", "s1", device="d2"),  # same session, other device
    ]
    stats = aggregate_usage(records, session_dedupe=True)
    assert stats[HERO]["devices"] == 2  # breadth untouched
    assert stats[HERO]["distinct_session_solved"] == 1


def test_solved_session_key_is_the_shared_rule():
    assert solved_session_key({"session_id": " s1 "}, "fb") == "s1"
    assert solved_session_key({"session_id": ""}, "fb") == "fb"
    assert solved_session_key({}, "record:7") == "record:7"


# ----- objective layer (multi_objective.py) ------------------------------------


def test_objective_vector_reads_distinct_count_with_fallback():
    deduped = {**_stats_shape(), "solved": 3, "total": 3, "success_rate": 1.0, "distinct_session_solved": 1}
    assert objective_vector(deduped, independent_only=True)["solved"] == 1
    assert objective_vector(deduped, independent_only=False)["solved"] == 3  # flag off reads raw
    raw = {**_stats_shape(), "solved": 3, "total": 3, "success_rate": 1.0}
    assert objective_vector(raw, independent_only=True)["solved"] == 3  # key absent -> fall back


def test_rank_prefilter_stays_raw_but_objective_dedupes():
    stats = {HERO: {**_stats_shape(), "solved": 3, "failed": 0, "total": 3, "success_rate": 1.0, "distinct_session_solved": 1}}
    # a grown ideal (solved=3) is what exposes the shortfall difference
    candidates = rank_promotion_candidates(
        stats,
        ideal={"rate": 1.0, "solved": 3.0, "breadth": 1.0},
        weights=WEIGHTS,
        min_solved=2,  # raw floor: passes with raw solved=3
        min_success_rate=0.5,
        score_threshold=0.15,
        procedure_dir="digest/procedure",
        independent_only=True,
    )
    assert len(candidates) == 1  # raw prefilter still admits it
    candidate = candidates[0]
    assert candidate["solved"] == 3  # reported evidence stays raw
    assert candidate["rate"] == 1.0 and candidate["breadth"] == 0.0
    # Tchebycheff worst shortfall = 0.5 * (1 - 1/3) = 0.3333 > 0.15 -> not promotable
    assert candidate["promotable"] is False and candidate["score"] == pytest.approx(0.3333, abs=1e-3)
    # same stats, raw objective: worst shortfall 0 <= 0.15 -> promotable
    raw = rank_promotion_candidates(
        stats,
        ideal={"rate": 1.0, "solved": 3.0, "breadth": 1.0},
        weights=WEIGHTS,
        min_solved=2,
        min_success_rate=0.5,
        score_threshold=0.15,
        procedure_dir="digest/procedure",
    )
    assert raw[0]["promotable"] is True


# ----- replay layer (replay.py) ------------------------------------------------


def test_replay_independent_mode_counts_distinct_sessions():
    solo = build_session_trees([_record("t1", HERO, "solved", "s1"), _record("t2", HERO, "solved", "s1"), _record("t3", HERO, "solved", "s1")])
    spread = build_session_trees([_record("t1", HERO, "solved", "s1"), _record("t2", HERO, "solved", "s2"), _record("t3", HERO, "solved", "s3")])
    params = policy_params({"min_solved": 2, "min_success_rate": 0.5, "score_bar": 1.0, "weights": WEIGHTS})
    counts = replay_once(solo, params, independent_only=True)
    assert counts["distinct_solved_by_path"] == {HERO: 1}  # three same-session solves = one evidence
    assert replay_once(spread, params, independent_only=True)["distinct_solved_by_path"] == {HERO: 3}
    # raw decisions unchanged by the flag: the raw floor and floor ideal score
    # identically here, which is exactly why finding6 pins the ideal for
    # decision-layer comparison (replay uses fixed IDEAL_FLOOR).
    assert counts["promotions"] == 1
    assert "distinct_solved_by_path" not in replay_once(solo, params)  # default shape unchanged


def test_replay_and_scan_share_the_counting_rule():
    """Counting-layer equivalence: both paths call solved_session_key.

    For the same records, the replay's per-path distinct count equals the
    online aggregate's ``distinct_session_solved`` — including the unsessioned
    fallback path.
    """
    records = [
        _record("t1", HERO, "solved", "s1"),
        _record("t2", HERO, "solved", "s1"),
        _record("t3", HERO, "solved", "s2"),
        _record("t4", HERO, "failed", "s3"),
        # unsessioned path: build_session_trees forms its world from the query
        _record("t5", "digest/procedure/other.md", "solved", query="ad-hoc"),
        _record("t6", "digest/procedure/other.md", "solved", query="ad-hoc"),
    ]
    aggregate = aggregate_usage(records, session_dedupe=True)
    counts = replay_once(build_session_trees(records), policy_params({"min_solved": 1, "min_success_rate": 0.0, "score_bar": 1.0, "weights": {"rate": 0.0, "solved": 0.0, "breadth": 0.0}}), independent_only=True)
    assert counts["distinct_solved_by_path"] == {
        path: stats["distinct_session_solved"] for path, stats in aggregate.items()
    }
    assert counts["distinct_solved_by_path"] == {HERO: 2, "digest/procedure/other.md": 2}


# ----- decision layer: scan vs replay, ideal pinned -----------------------------


def _pin_ideal(tmp_path: Path, ideal: dict) -> None:
    path = ideal_point_path(tmp_path, "metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ideal), encoding="utf-8")


def _seed_usage(tmp_path: Path, records: list[dict]) -> None:
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _scan(tmp_path: Path, **kwargs) -> OpsPromotionScanStep:
    (tmp_path / HERO).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / HERO).write_text("runbook body", encoding="utf-8")
    return OpsPromotionScanStep(
        app_context=ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=Mock(spec=BaseFileStore, workspace_path=tmp_path),
        procedure_dir="digest/procedure",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_scan_and_replay_agree_with_ideal_pinned(tmp_path):
    """Decision-layer equivalence, only with the ideal point explicitly pinned."""
    records = [
        _record("t1", HERO, "solved", "s1"),
        _record("t2", HERO, "solved", "s2"),
        _record("t3", HERO, "failed", "s3"),
    ]
    _pin_ideal(tmp_path, dict(IDEAL_FLOOR))  # replay always uses IDEAL_FLOOR
    _seed_usage(tmp_path, records)
    step = _scan(
        tmp_path,
        independent_only=True,
        min_solved=1,
        min_success_rate=0.5,
        score_threshold=0.45,
        lambda_rate=0.4,
        lambda_solved=0.3,
        lambda_breadth=0.3,
    )
    step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))
    response = await step(RuntimeContext())

    assert response.metadata["promotion"]["independent_only"] is True
    assert [entry["path"] for entry in response.metadata["promotion"]["promoted"]] == [HERO]
    replay_counts = replay_once(
        build_session_trees(records),
        policy_params({"min_solved": 1, "min_success_rate": 0.5, "score_bar": 0.45, "weights": {"rate": 0.4, "solved": 0.3, "breadth": 0.3}}),
        independent_only=True,
    )
    assert replay_counts["promotions"] == 1  # same verdict as the pinned scan


@pytest.mark.asyncio
async def test_scan_repeated_session_solves_no_longer_stack(tmp_path):
    """The systematic bias: raw mode stacks same-session solves, dedupe does not."""
    records = [_record(f"t{i}", HERO, "solved", "s1") for i in range(3)]  # raw solved=3, distinct=1
    _pin_ideal(tmp_path, {"rate": 1.0, "solved": 3.0, "breadth": 1.0})  # grown ideal exposes the shortfall
    _seed_usage(tmp_path, records)

    raw_step = _scan(
        tmp_path,
        min_solved=1,
        min_success_rate=1.0,
        score_threshold=0.15,
        lambda_rate=0.5,
        lambda_solved=0.5,
        lambda_breadth=0.0,
    )
    raw_step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))
    raw_response = await raw_step(RuntimeContext())
    assert [entry["path"] for entry in raw_response.metadata["promotion"]["promoted"]] == [HERO]

    _pin_ideal(tmp_path, {"rate": 1.0, "solved": 3.0, "breadth": 1.0})  # reset the grown ideal
    dedupe_step = _scan(
        tmp_path,
        independent_only=True,
        min_solved=1,
        min_success_rate=1.0,
        score_threshold=0.15,
        lambda_rate=0.5,
        lambda_solved=0.5,
        lambda_breadth=0.0,
    )
    dedupe_step.run_job = AsyncMock(return_value=Mock(success=True, answer="promoted"))
    dedupe_response = await dedupe_step(RuntimeContext())

    assert dedupe_response.metadata["promotion"]["promoted"] == []  # one session is one evidence
    assert dedupe_response.metadata["promotion"]["candidates"][0]["promotable"] is False


# ----- wiring -------------------------------------------------------------------


def test_promotion_cron_enables_independent_only():
    cloud = _load_config("ops_cloud.yaml")
    scan = cloud["jobs"]["ops_promotion_cron"]["steps"][0]
    assert scan["backend"] == "ops_promotion_scan_step"
    assert scan["independent_only"] is True
