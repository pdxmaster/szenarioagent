from services.diff import JsonDiffer


def test_json_diff_detects_changes():
    old = {"tag": "a", "name": "Old"}
    new = {"tag": "a", "name": "New", "message": "<p>Hi</p>"}
    differ = JsonDiffer(old, new)
    changes = differ.diff()
    assert any(change.path == "name" for change in changes)
    rendered = differ.render()
    assert "Änderungen" in rendered or "Keine Unterschiede" in rendered
