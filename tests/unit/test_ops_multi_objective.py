"""Multi-objective Tchebycheff promotion: objectives, ideal point, ranking."""

import json

import pytest

from reme.steps.ops.multi_objective import (
    IDEAL_FLOOR,
    ideal_point_path,
    load_ideal_point,
    objective_vector,
    rank_promotion_candidates,
    save_ideal_point,
    tchebycheff_score,
    total_device_population,
    update_ideal_point,
)


def stats(solved=5, failed=1, devices=1):
    return {
        "solved": solved,
        "failed": failed,
        "total": solved + failed,
        "success_rate": round(solved / (solved + failed), 4),
        "devices": devices,
        "device_list": [f"dev{i}" for i in range(devices)],
    }


def test_objective_vector_extracts_raw_axes():
    vector = objective_vector(stats(solved=10, failed=0, devices=2))
    assert vector == {"rate": 1.0, "solved": 10, "breadth": 2}
    vector = objective_vector(stats(solved=3, failed=1, devices=1))
    assert vector == {"rate": 0.75, "solved": 3, "breadth": 1}


def test_tchebycheff_takes_worst_weighted_shortfall():
    ideal = {"rate": 1.0, "solved": 10.0, "breadth": 2.0}
    weights = {"rate": 0.4, "solved": 0.3, "breadth": 0.3}
    vector = {"rate": 0.8, "solved": 8.0, "breadth": 2.0}
    # shortfalls: rate 0.4*0.2=0.08, solved 0.3*(1-0.8)=0.06, breadth 0
    assert tchebycheff_score(vector, ideal, weights) == pytest.approx(0.08)


def test_tchebycheff_unlike_weighted_sum_ignores_fine_other_objectives():
    ideal = {"rate": 1.0, "solved": 10.0, "breadth": 2.0}
    weights = {"rate": 0.6, "solved": 0.4, "breadth": 0.0}
    weak_rate = tchebycheff_score({"rate": 0.6, "solved": 10.0, "breadth": 2.0}, ideal, weights)
    assert weak_rate == pytest.approx(0.6 * 0.4)


def test_ideal_point_updates_monotonically_and_persists(tmp_path):
    ideal = load_ideal_point(tmp_path / "missing.json")
    assert ideal == IDEAL_FLOOR
    grown = update_ideal_point(ideal, [{"rate": 1.0, "solved": 12.0, "breadth": 0.5}])
    assert grown["solved"] == pytest.approx(12.0) and grown["rate"] == 1.0
    shrunk = update_ideal_point(grown, [{"rate": 0.2, "solved": 1.0, "breadth": 0.1}])
    assert shrunk == grown  # z* never shrinks
    path = ideal_point_path(tmp_path, "metadata")
    save_ideal_point(path, grown)
    assert load_ideal_point(path) == grown
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    assert load_ideal_point(corrupt) == IDEAL_FLOOR


def test_total_device_population():
    usage = {
        "a.md": {"device_list": ["d1", "d2"]},
        "b.md": {"device_list": ["d2"]},
        "c.md": {"device_list": []},
    }
    assert total_device_population(usage) == 2


def _seed(tmp_path, records):
    usage = tmp_path / "metadata" / "ops_usage" / "usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_rank_promotion_candidates_orders_by_tchebycheff(tmp_path):
    records = []
    for device in ("d1", "d2"):
        for i in range(8):
            records.append({"ts": f"{device}-{i}", "path": "digest/procedure/star.md",
                            "outcome": "solved", "source_device": device})
    for i in range(3):
        records.append({"ts": f"one-{i}", "path": "digest/procedure/single.md",
                        "outcome": "solved", "source_device": "d1"})
    _seed(tmp_path, records)

    from reme.steps.ops.usage import aggregate_usage, read_usage_records
    from reme.steps.ops.usage import usage_file_path
    usage_stats = aggregate_usage(read_usage_records(usage_file_path(tmp_path, "metadata", cloud=True)))

    candidates = rank_promotion_candidates(
        usage_stats,
        ideal={"rate": 1.0, "solved": 8.0, "breadth": 2.0},
        procedure_dir="digest/procedure",
        min_solved=2,
        min_success_rate=0.5,
        score_threshold=0.45,
        weights={"rate": 0.4, "solved": 0.3, "breadth": 0.3},
    )
    assert [c["path"] for c in candidates] == ["digest/procedure/star.md", "digest/procedure/single.md"]
    star, single = candidates
    assert star["promotable"] is True and single["promotable"] is True
    assert star["score"] < single["score"]  # breadth + volume beat the single-device SOP
    assert star["breadth"] == 2  # raw distinct-device count


def test_rank_respects_floor_and_threshold(tmp_path):
    _seed(tmp_path, [{"ts": "t", "path": "digest/procedure/weak.md", "outcome": "failed", "source_device": "d1"}])
    from reme.steps.ops.usage import aggregate_usage, read_usage_records, usage_file_path
    usage_stats = aggregate_usage(read_usage_records(usage_file_path(tmp_path, "metadata", cloud=True)))
    candidates = rank_promotion_candidates(
        usage_stats,
        ideal={"rate": 1.0, "solved": 8.0, "breadth": 2.0},
        procedure_dir="digest/procedure",
        min_solved=2,
        min_success_rate=0.8,
        score_threshold=0.1,
        weights={"rate": 0.4, "solved": 0.3, "breadth": 0.3},
    )
    assert candidates == []  # floor filters the all-failed SOP before scoring
