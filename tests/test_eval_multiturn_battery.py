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
