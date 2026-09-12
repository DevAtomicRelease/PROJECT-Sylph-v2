"""
Reminders & routines — Phase 4.

An AsyncIOScheduler drives time-based reminders. When one fires, a callback
(wired in main.py) makes Sylph speak the reminder and notifies the frontend.
Task specs are persisted to ~/.sylph/config/schedules.json so recurring
reminders survive restarts; one-shots that already passed while the app was
closed are dropped rather than fired late.

Non-destructive: reminders only speak/notify. Gated by the 'schedule' domain.

`when` grammar (parsed from a short phrase the model fills in):
  "in 30 minutes" | "in 2 hours" | "in 45 seconds"     -> one-shot (relative)
  "at 09:00" | "at 5:30 pm"                             -> one-shot (today/next)
  "every 30 minutes" | "every 2 hours"                 -> recurring interval
  "daily at 09:00" | "every day at 5 pm"               -> recurring daily
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("sylph.scheduler")

_STORE_PATH = os.path.expanduser("~/.sylph/config/schedules.json")


class ScheduleError(ValueError):
    """Raised when a `when` phrase can't be understood."""


def _parse_when(when: str) -> dict:
    """Parse a `when` phrase into a persistable spec dict. Raises ScheduleError."""
    s = (when or "").strip().lower()
    if not s:
        raise ScheduleError("no time given")

    # daily at HH:MM  /  every day at H[:MM] [am|pm]
    m = re.search(r"(?:daily|every\s+day)\s+at\s+(.+)", s)
    if m:
        h, mi = _parse_clock(m.group(1))
        return {"kind": "cron", "hour": h, "minute": mi, "desc": f"daily at {h:02d}:{mi:02d}"}

    # every N unit
    m = re.search(r"every\s+(\d+)\s*(second|sec|s|minute|min|m|hour|hr|h)s?\b", s)
    if m:
        secs = _to_seconds(int(m.group(1)), m.group(2))
        return {"kind": "interval", "seconds": secs, "desc": f"every {_humanize(secs)}"}

    # in N unit
    m = re.search(r"in\s+(\d+)\s*(second|sec|s|minute|min|m|hour|hr|h)s?\b", s)
    if m:
        secs = _to_seconds(int(m.group(1)), m.group(2))
        run_at = datetime.now() + timedelta(seconds=secs)
        return {"kind": "once", "run_at": run_at.isoformat(), "desc": f"in {_humanize(secs)}"}

    # at HH:MM [am|pm]
    m = re.search(r"at\s+(.+)", s)
    if m:
        h, mi = _parse_clock(m.group(1))
        now = datetime.now()
        run_at = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        if run_at <= now:
            run_at += timedelta(days=1)
        return {"kind": "once", "run_at": run_at.isoformat(), "desc": f"at {h:02d}:{mi:02d}"}

    raise ScheduleError(f"couldn't understand the time '{when}'")


def _parse_clock(text: str) -> tuple[int, int]:
    text = text.strip()
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        raise ScheduleError(f"bad time '{text}'")
    h = int(m.group(1))
    mi = int(m.group(2) or 0)
    ap = m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        raise ScheduleError(f"bad time '{text}'")
    return h, mi


def _to_seconds(n: int, unit: str) -> int:
    if unit.startswith("s"):
        return n
    if unit.startswith("h") or unit == "hr":
        return n * 3600
    return n * 60  # minute default


def _humanize(secs: int) -> str:
    if secs % 3600 == 0:
        return f"{secs // 3600} hour(s)"
    if secs % 60 == 0:
        return f"{secs // 60} minute(s)"
    return f"{secs} second(s)"


class ScheduleManager:
    def __init__(self, store_path: str = _STORE_PATH):
        self._store_path = store_path
        self._sched = AsyncIOScheduler()
        self._tasks: dict[str, dict] = {}
        self._fire: Optional[Callable[[str], Awaitable[None]]] = None

    def set_fire_callback(self, cb: Callable[[str], Awaitable[None]]) -> None:
        self._fire = cb

    async def start(self) -> None:
        self._tasks = self._load()
        self._sched.start()
        # Re-register surviving tasks; drop one-shots whose time has passed.
        for tid, spec in list(self._tasks.items()):
            if spec.get("kind") == "once":
                try:
                    if datetime.fromisoformat(spec["run_at"]) <= datetime.now():
                        del self._tasks[tid]
                        continue
                except Exception:
                    del self._tasks[tid]
                    continue
            self._register_job(tid, spec)
        self._save()
        logger.info("Scheduler started with %d task(s)", len(self._tasks))

    async def stop(self) -> None:
        try:
            self._sched.shutdown(wait=False)
        except Exception:
            pass

    # -- job wiring ----------------------------------------------------------

    def _build_trigger(self, spec: dict):
        kind = spec["kind"]
        if kind == "once":
            return DateTrigger(run_date=datetime.fromisoformat(spec["run_at"]))
        if kind == "interval":
            return IntervalTrigger(seconds=int(spec["seconds"]))
        if kind == "cron":
            return CronTrigger(hour=int(spec["hour"]), minute=int(spec["minute"]))
        raise ScheduleError(f"unknown kind {kind}")

    def _register_job(self, tid: str, spec: dict) -> None:
        self._sched.add_job(self._on_fire, self._build_trigger(spec), args=[tid], id=tid, replace_existing=True)

    async def _on_fire(self, tid: str) -> None:
        spec = self._tasks.get(tid)
        if not spec:
            return
        message = spec.get("message", "Reminder")
        logger.info("Reminder fired [%s]: %s", tid, message)
        if self._fire:
            try:
                await self._fire(message)
            except Exception as e:
                logger.error("reminder fire callback failed: %s", e)
        # One-shots are done after firing.
        if spec.get("kind") == "once":
            self._tasks.pop(tid, None)
            self._save()

    # -- public API (tool-facing) -------------------------------------------

    def add(self, message: str, when: str) -> str:
        message = (message or "").strip()
        if not message:
            return "What should I remind you about?"
        spec = _parse_when(when)  # may raise ScheduleError
        tid = uuid.uuid4().hex[:8]
        spec["message"] = message
        self._tasks[tid] = spec
        self._register_job(tid, spec)
        self._save()
        logger.info("Reminder added [%s] '%s' %s", tid, message[:40], spec["desc"])
        return f"Okay — I'll remind you to '{message}' {spec['desc']} (id {tid})."

    def list(self) -> list[dict]:
        return [
            {"id": tid, "message": s.get("message", ""), "when": s.get("desc", "")}
            for tid, s in self._tasks.items()
        ]

    def cancel(self, reminder_id: str) -> str:
        tid = (reminder_id or "").strip()
        if tid not in self._tasks:
            return f"No reminder with id '{tid}'."
        try:
            self._sched.remove_job(tid)
        except Exception:
            pass
        msg = self._tasks.pop(tid).get("message", "")
        self._save()
        return f"Cancelled the reminder to '{msg}'."

    # -- persistence ---------------------------------------------------------

    def _load(self) -> dict:
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning("schedules.json unreadable (%s) — starting empty", e)
            return {}

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._tasks, f, indent=2)
        except Exception as e:
            logger.error("failed to persist schedules: %s", e)


# Process-wide singleton.
schedule_manager = ScheduleManager()
