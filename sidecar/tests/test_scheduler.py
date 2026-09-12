"""Reminder time-phrase parsing (Phase 4). Pure logic, no live scheduler."""
import re
from datetime import datetime

import pytest
from scheduler import _parse_when, _parse_clock, _to_seconds, ScheduleError


def test_relative_minutes():
    spec = _parse_when("in 30 minutes")
    assert spec["kind"] == "once"
    run_at = datetime.fromisoformat(spec["run_at"])
    assert 25 * 60 <= (run_at - datetime.now()).total_seconds() <= 35 * 60


def test_relative_hours_and_seconds():
    assert _to_seconds(2, "hours") == 7200
    assert _to_seconds(45, "seconds") == 45
    assert _parse_when("in 2 hours")["kind"] == "once"


def test_interval():
    spec = _parse_when("every 2 hours")
    assert spec["kind"] == "interval"
    assert spec["seconds"] == 7200


def test_daily_cron():
    spec = _parse_when("daily at 09:00")
    assert spec["kind"] == "cron"
    assert spec["hour"] == 9 and spec["minute"] == 0


def test_every_day_at_pm():
    spec = _parse_when("every day at 5 pm")
    assert spec["kind"] == "cron"
    assert spec["hour"] == 17 and spec["minute"] == 0


def test_at_clock_rolls_to_future():
    spec = _parse_when("at 09:30")
    assert spec["kind"] == "once"
    assert datetime.fromisoformat(spec["run_at"]) > datetime.now()


def test_parse_clock_am_pm():
    assert _parse_clock("5:30 pm") == (17, 30)
    assert _parse_clock("12 am") == (0, 0)
    assert _parse_clock("09:00") == (9, 0)


def test_bad_phrase_raises():
    with pytest.raises(ScheduleError):
        _parse_when("whenever you feel like it")
    with pytest.raises(ScheduleError):
        _parse_when("")


def test_schedule_domain_gating(tmp_path):
    import os
    from agent_permissions import PermissionStore
    p = PermissionStore(path=os.path.join(str(tmp_path), "perms.json"))
    assert p.is_tool_allowed("schedule_reminder") is True
    assert p.is_tool_allowed("open_app") is True
    p.update({"domains": {"schedule": {"enabled": False}}})
    assert p.is_tool_allowed("schedule_reminder") is False
    assert p.is_tool_allowed("open_app") is True  # system domain untouched
