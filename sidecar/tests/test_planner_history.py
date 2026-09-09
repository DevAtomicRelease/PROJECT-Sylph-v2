"""Planner history restore filtering + windowing (no LLM calls)."""
from llm.planner import ConversationPlanner, MAX_TOOL_ITERATIONS


def _planner():
    # ollama_client is only used when a turn runs; load_history never calls it.
    return ConversationPlanner(ollama_client=None)


def test_load_history_filters_non_conversational_roles():
    p = _planner()
    p.load_history([
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "tool output"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": ""},        # empty dropped
        {"role": "system", "content": "ignored"},
    ])
    roles = [m["role"] for m in p.history]
    assert roles == ["user", "assistant"]


def test_load_history_caps_to_window():
    p = _planner()
    msgs = [{"role": "user", "content": str(i)} for i in range(100)]
    p.load_history(msgs)
    assert p.history_length <= p._max_history
    # keeps the most recent
    assert p.history[-1]["content"] == "99"


def test_load_history_empty_is_noop():
    p = _planner()
    p.load_history([])
    assert p.history_length == 0


def test_tool_iteration_cap_is_bounded():
    assert 1 <= MAX_TOOL_ITERATIONS <= 20
