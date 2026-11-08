"""Pure identity checks; no QApplication, viewer, socket or renderer."""
from layer_viewer.instance import instance_name


def test_same_socket_different_editors_do_not_activate_each_others_viewer(monkeypatch):
    monkeypatch.setenv("KICAD_API_SOCKET", "ipc://shared-test-pipe")
    monkeypatch.setenv("KICAD_API_TOKEN", "editor-a")
    first = instance_name()
    assert first == instance_name()
    monkeypatch.setenv("KICAD_API_TOKEN", "editor-b")
    second = instance_name()
    assert first != second
    assert "editor-a" not in first and "editor-b" not in second


def test_standalone_identity_is_stable_and_distinct_from_editor_launch(monkeypatch):
    monkeypatch.setenv("KICAD_API_SOCKET", "ipc://shared-test-pipe")
    monkeypatch.delenv("KICAD_API_TOKEN", raising=False)
    standalone = instance_name()
    assert standalone == instance_name()
    monkeypatch.setenv("KICAD_API_TOKEN", "editor-a")
    assert standalone != instance_name()
    monkeypatch.setenv("KICAD_API_SOCKET", "ipc://other-test-pipe")
    assert standalone != instance_name()
