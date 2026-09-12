"""Agent capability permissions (Phase 2)."""
import os
from agent_permissions import PermissionStore


def _store(tmp_path):
    return PermissionStore(path=os.path.join(str(tmp_path), "permissions.json"))


def test_defaults_all_enabled(tmp_path):
    p = _store(tmp_path)
    for dom in ("comms", "web", "files", "screen"):
        assert p.is_domain_enabled(dom)
    assert p.is_tool_allowed("search_web")
    assert p.is_tool_allowed("read_email")


def test_ungated_tool_always_allowed(tmp_path):
    p = _store(tmp_path)
    p.update({"domains": {"comms": {"enabled": False}, "web": {"enabled": False}}})
    assert p.is_tool_allowed("move_avatar") is True  # not in TOOL_DOMAINS


def test_disabling_domain_blocks_its_tools(tmp_path):
    p = _store(tmp_path)
    p.update({"domains": {"web": {"enabled": False}}})
    assert p.is_tool_allowed("search_web") is False
    reason = p.check_tool("search_web", {})
    assert reason and "turned off" in reason
    # other domains untouched
    assert p.is_tool_allowed("read_email") is True


def test_file_path_allowlist(tmp_path):
    p = _store(tmp_path)
    allowed = os.path.realpath(str(tmp_path))
    p.update({"domains": {"files": {"enabled": True, "allowed_dirs": [allowed]}}})
    assert p.check_tool("read_file", {"file_path": os.path.join(allowed, "a.txt")}) is None
    blocked = p.check_tool("read_file", {"file_path": "C:\\Windows\\system.ini"})
    assert blocked and "allowed directories" in blocked


def test_empty_allowlist_is_unrestricted(tmp_path):
    p = _store(tmp_path)
    assert p.check_tool("read_file", {"file_path": "C:\\anywhere\\x.txt"}) is None


def test_persistence_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), "permissions.json")
    PermissionStore(path=path).update({"domains": {"screen": {"enabled": False}}})
    # a fresh store reads the persisted file
    p2 = PermissionStore(path=path)
    assert p2.is_domain_enabled("screen") is False


def test_partial_config_merges_defaults(tmp_path):
    p = _store(tmp_path)
    p.update({"domains": {"web": {"enabled": False}}})  # omits other domains
    cfg = p.get()
    assert cfg["domains"]["comms"]["enabled"] is True   # default preserved
    assert cfg["domains"]["web"]["enabled"] is False
