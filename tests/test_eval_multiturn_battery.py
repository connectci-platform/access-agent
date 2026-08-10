"""Thread-battery loading: YAML/JSON by extension + fail-fast validation."""

import json

import pytest

VALID = [
    {
        "thread_id": "mt-x-01",
        "description": "d",
        "questions": [
            {"turn_id": "t1", "question": "One?"},
            {"turn_id": "t2", "question": "Two?"},
        ],
    }
]


def _write(tmp_path, name, data):
    p = tmp_path / name
    if name.endswith(".json"):
        p.write_text(json.dumps(data))
    else:
        import yaml

        p.write_text(yaml.safe_dump(data))
    return str(p)


def test_loads_json(tmp_path):
    from src.eval.multiturn import load_thread_battery

    assert load_thread_battery(_write(tmp_path, "b.json", VALID))[0]["thread_id"] == "mt-x-01"


def test_loads_yaml(tmp_path):
    from src.eval.multiturn import load_thread_battery

    assert load_thread_battery(_write(tmp_path, "b.yaml", VALID))[0]["thread_id"] == "mt-x-01"


def test_duplicate_thread_ids_rejected(tmp_path):
    from src.eval.multiturn import load_thread_battery

    data = [VALID[0], dict(VALID[0])]
    with pytest.raises(ValueError, match="duplicate thread_id"):
        load_thread_battery(_write(tmp_path, "b.yaml", data))


def test_duplicate_turn_ids_rejected(tmp_path):
    from src.eval.multiturn import load_thread_battery

    bad = [
        {
            "thread_id": "t",
            "questions": [{"turn_id": "t1", "question": "a?"}, {"turn_id": "t1", "question": "b?"}],
        }
    ]
    with pytest.raises(ValueError, match="duplicate turn_id"):
        load_thread_battery(_write(tmp_path, "b.yaml", bad))


def test_question_id_length_rejected(tmp_path):
    from src.eval.multiturn import load_thread_battery

    bad = [{"thread_id": "x" * 63, "questions": [{"turn_id": "t1", "question": "a?"}]}]
    with pytest.raises(ValueError, match="exceeds 64"):
        load_thread_battery(_write(tmp_path, "b.yaml", bad))


def test_empty_question_rejected(tmp_path):
    from src.eval.multiturn import load_thread_battery

    bad = [{"thread_id": "t", "questions": [{"turn_id": "t1", "question": "  "}]}]
    with pytest.raises(ValueError, match="empty question"):
        load_thread_battery(_write(tmp_path, "b.yaml", bad))


def test_top_level_mapping_rejected(tmp_path):
    from src.eval.multiturn import load_thread_battery

    bad = {"thread_id": "t", "questions": [{"turn_id": "t1", "question": "a?"}]}
    with pytest.raises(ValueError, match="top-level list"):
        load_thread_battery(_write(tmp_path, "b.yaml", bad))


STATE = {"final_answer": "ok", "tools_used": [], "messages": []}


@pytest.mark.asyncio
async def test_session_ids_disjoint_across_battery_runs(tmp_path):
    from unittest.mock import AsyncMock, patch

    from src.eval.multiturn import run_battery

    battery = _write(tmp_path, "b.json", VALID)
    seen: list[str] = []

    async def fake_run_agent(**kwargs):
        seen.append(kwargs["session_id"])
        return STATE

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=fake_run_agent)),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        await run_battery(battery_path=battery)
        first = set(seen)
        seen.clear()
        await run_battery(battery_path=battery)
        second = set(seen)

    assert first and second
    assert first.isdisjoint(second)  # D2a: runs never resume each other's checkpoints
    assert all(s.startswith("eval_") and s.endswith("_mt-x-01") for s in first | second)


def test_cli_multiturn_flags_reach_run_battery(monkeypatch):
    import sys
    from unittest.mock import AsyncMock

    import src.eval.__main__ as cli
    import src.eval.multiturn as mt

    mock = AsyncMock(return_value=([], None))
    monkeypatch.setattr(mt, "run_battery", mock)
    monkeypatch.setattr(mt, "print_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval", "multiturn", "--threads", "b.yaml", "--score", "--judge-model", "m"],
    )
    cli.main()
    assert mock.call_args.kwargs["score"] is True
    assert mock.call_args.kwargs["judge_model"] == "m"


ACTING_USER_REQUIRED = [
    {
        "thread_id": "mt-auth-01",
        "description": "d",
        "acting_user_required": True,
        "questions": [{"turn_id": "t1", "question": "Am I registered?"}],
    }
]


@pytest.mark.asyncio
async def test_acting_user_required_without_acting_user_raises(tmp_path):
    from unittest.mock import AsyncMock, patch

    from src.eval.multiturn import run_battery

    battery = _write(tmp_path, "b.json", ACTING_USER_REQUIRED)

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        with pytest.raises(ValueError, match="mt-auth-01"):
            await run_battery(battery_path=battery)


@pytest.mark.asyncio
async def test_acting_user_required_with_acting_user_runs(tmp_path):
    from unittest.mock import AsyncMock, patch

    from src.eval.multiturn import run_battery

    battery = _write(tmp_path, "b.json", ACTING_USER_REQUIRED)

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        results, _ = await run_battery(battery_path=battery, acting_user="test_user")

    assert len(results) == 1
    assert results[0].thread_id == "mt-auth-01"


def test_support_battery_file_is_valid():
    from src.eval.multiturn import load_thread_battery

    threads = load_thread_battery("eval/questions/multiturn_support_battery.yaml")
    assert len(threads) == 9
    scenarios = [t.get("scenario") for t in threads]
    assert scenarios.count("followup") == 2
    assert scenarios.count("clarification") == 2
    assert scenarios.count("action") == 2
    assert scenarios.count("topic_switch") == 2
    assert scenarios.count("mixed") == 1
