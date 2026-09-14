"""Code workspace: path jail, read/write, run, and domain gating (Phase 5)."""
import asyncio
import os

import pytest
import tools.code_tool as ct


@pytest.fixture(autouse=True)
def jail(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "WORKSPACE", str(tmp_path))
    return tmp_path


def test_write_read_roundtrip():
    assert "Saved" in ct.write_code("hello.py", "print('hi')")
    assert "print('hi')" in ct.read_code("hello.py")


def test_path_jail_blocks_traversal():
    with pytest.raises(ValueError):
        ct._safe_path("../escape.txt")
    with pytest.raises(ValueError):
        ct._safe_path("C:\\Windows\\system.ini")


def test_list_workspace():
    ct.write_code("a.py", "x=1")
    ct.write_code("sub/b.py", "y=2")
    files = ct.list_workspace()
    assert "a.py" in files
    assert any(f.replace("\\", "/") == "sub/b.py" for f in files)


def test_read_missing():
    assert "No file" in ct.read_code("nope.py")


def test_run_python_snippet():
    out = asyncio.run(ct.run_python(code="print(6*7)"))
    assert "42" in out
    assert "exit 0" in out


def test_run_python_file():
    ct.write_code("script.py", "print('from file')")
    out = asyncio.run(ct.run_python(filename="script.py"))
    assert "from file" in out


def test_run_python_nothing():
    out = asyncio.run(ct.run_python())
    assert "Nothing to run" in out


def test_code_domain_gating(tmp_path):
    from agent_permissions import PermissionStore
    p = PermissionStore(path=os.path.join(str(tmp_path), "perms.json"))
    assert p.is_tool_allowed("run_python") is True
    assert p.is_tool_allowed("web_navigate") is True  # web domain
    p.update({"domains": {"code": {"enabled": False}}})
    assert p.is_tool_allowed("run_python") is False
    assert p.is_tool_allowed("web_navigate") is True   # web untouched
