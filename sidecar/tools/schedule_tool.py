"""Reminder tools (Phase 4) — thin wrappers over the ScheduleManager singleton."""

from scheduler import schedule_manager, ScheduleError


def schedule_reminder(message: str, when: str) -> str:
    """Set a reminder. `when` e.g. 'in 30 minutes', 'at 09:00', 'every 2 hours',
    'daily at 5 pm'. Sylph will speak the reminder when it fires."""
    try:
        return schedule_manager.add(message, when)
    except ScheduleError as e:
        return f"I couldn't set that reminder — {e}. Try 'in 30 minutes' or 'daily at 9am'."


def list_reminders() -> list[dict]:
    """List the user's active reminders."""
    return schedule_manager.list()


def cancel_reminder(reminder_id: str) -> str:
    """Cancel a reminder by its id (from list_reminders)."""
    return schedule_manager.cancel(reminder_id)
