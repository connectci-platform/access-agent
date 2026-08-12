from src.redteam.sample import SampleResult


def test_ok_sample():
    s = SampleResult(text="hello")
    assert s.text == "hello" and s.errored is False


def test_error_sample():
    s = SampleResult.error()
    assert s.text == "" and s.errored is True


def test_frozen():
    import dataclasses

    import pytest

    s = SampleResult(text="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.text = "y"  # type: ignore[misc]
