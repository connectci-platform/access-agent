import json

from src.redteam.report import Flag, content_hash, issue_body, redacted_line, write_artifact


def test_redacted_line_never_contains_response():
    line = redacted_line("wrapped__aligned__stop-sign", "complies", "HARMFUL BODY TEXT")
    assert "HARMFUL BODY TEXT" not in line
    assert "wrapped__aligned__stop-sign" in line and "complies" in line
    assert "sha256:" in line


def test_redacted_line_hash_matches_actual_text():
    # locks the F1 bug: the printed hash MUST be of the real response, not "".
    line = redacted_line("id-x", "complies", "REAL RESPONSE")
    assert line.endswith(content_hash("REAL RESPONSE"))
    assert not line.endswith(content_hash(""))


def test_flag_line_uses_precomputed_hash():
    from src.redteam.report import flag_line

    f = Flag("id-x", "complies", content_hash("REAL RESPONSE"), "candidate-regression")
    assert flag_line(f).endswith(content_hash("REAL RESPONSE"))


def test_content_hash_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


def test_issue_body_has_no_response_text():
    f = Flag("id-1", "complies", content_hash("BODY"), "candidate-regression")
    body = issue_body([f])
    assert "BODY" not in body
    assert "id-1" in body and "candidate-regression" in body


def test_write_artifact_roundtrips(tmp_path):
    p = tmp_path / "run.json"
    write_artifact(p, [{"id": "x", "response": "BODY", "verdict": "complies"}])
    data = json.loads(p.read_text())
    assert data[0]["response"] == "BODY"  # artifact keeps full text (on-prem only)
