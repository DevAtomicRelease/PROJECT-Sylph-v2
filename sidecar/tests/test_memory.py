"""Short-term SQLite memory round-trip (temp db, no external state)."""
import os
from memory.short_term import ShortTermMemory


def _mem(tmp_path):
    m = ShortTermMemory(db_path=os.path.join(str(tmp_path), "sessions.db"))
    m.initialize()
    return m


def test_add_and_recent_roundtrip(tmp_path):
    m = _mem(tmp_path)
    tid = "thread-a"
    m.add_message(tid, "user", "hello")
    m.add_message(tid, "assistant", "hi there")
    recent = m.get_recent_messages(tid, limit=10)
    assert [r["role"] for r in recent] == ["user", "assistant"]
    assert recent[0]["content"] == "hello"
    m.close()


def test_recent_limit_and_chronological_order(tmp_path):
    m = _mem(tmp_path)
    tid = "thread-b"
    for i in range(5):
        m.add_message(tid, "user", f"m{i}")
    recent = m.get_recent_messages(tid, limit=3)
    # newest 3, returned oldest-first
    assert [r["content"] for r in recent] == ["m2", "m3", "m4"]
    m.close()


def test_list_threads_orders_by_recency(tmp_path):
    import time
    m = _mem(tmp_path)
    m.add_message("old", "user", "x")
    # updated_at has ~ms resolution; on Windows two fast inserts can tie. A real
    # deployment's threads span sessions, so make the two timestamps distinct.
    time.sleep(0.05)
    m.add_message("new", "user", "y")
    threads = m.list_threads(limit=5)
    assert threads[0]["thread_id"] == "new"
    m.close()


def test_clear_all(tmp_path):
    m = _mem(tmp_path)
    m.add_message("t", "user", "x")
    m.clear_all()
    assert m.get_all_messages() == []
    m.close()
