import logging
from datetime import datetime, time
from googleapiclient.discovery import build
from .google_auth import get_google_credentials

logger = logging.getLogger("sylph.tools.calendar")

def get_calendar_service():
    creds = get_google_credentials()
    return build("calendar", "v3", credentials=creds)

def list_events_today() -> list[dict]:
    """
    List Google Calendar events for the current day.

    Returns:
        A list of events with details (id, summary, start, end, description).
    """
    try:
        service = get_calendar_service()
        now = datetime.now()
        start_of_day = datetime.combine(now.date(), time.min).isoformat() + "Z"
        end_of_day = datetime.combine(now.date(), time.max).isoformat() + "Z"

        logger.info("Listing events between %s and %s", start_of_day, end_of_day)
        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day,
            timeMax=end_of_day,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        formatted_events = []
        for event in events:
            formatted_events.append({
                "id": event["id"],
                "summary": event.get("summary", "(No Title)"),
                "start": event["start"].get("dateTime") or event["start"].get("date"),
                "end": event["end"].get("dateTime") or event["end"].get("date"),
                "description": event.get("description", ""),
            })

        logger.info("Retrieved %d events for today", len(formatted_events))
        return formatted_events
    except Exception as e:
        logger.error("Failed to list today's events: %s", e)
        raise e

def list_events_range(start_iso: str, end_iso: str) -> list[dict]:
    """
    List Google Calendar events within a specified ISO time range.

    Args:
        start_iso: Start date-time string in ISO format (e.g. 2026-06-11T00:00:00Z).
        end_iso: End date-time string in ISO format (e.g. 2026-06-11T23:59:59Z).

    Returns:
        A list of events.
    """
    try:
        service = get_calendar_service()
        events_result = service.events().list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        formatted_events = []
        for event in events:
            formatted_events.append({
                "id": event["id"],
                "summary": event.get("summary", "(No Title)"),
                "start": event["start"].get("dateTime") or event["start"].get("date"),
                "end": event["end"].get("dateTime") or event["end"].get("date"),
                "description": event.get("description", ""),
            })

        logger.info("Retrieved %d events between %s and %s", len(formatted_events), start_iso, end_iso)
        return formatted_events
    except Exception as e:
        logger.error("Failed to list events in range: %s", e)
        raise e

def create_event(title: str, start_iso: str, end_iso: str, description: str = "") -> dict:
    """
    Create a new calendar event. Requires user confirmation before calling.

    Args:
        title: Title/summary of the event.
        start_iso: Start ISO datetime (e.g., 2026-06-11T14:00:00).
        end_iso: End ISO datetime (e.g., 2026-06-11T15:00:00).
        description: Description of the event.

    Returns:
        The created event details.
    """
    try:
        service = get_calendar_service()
        event_body = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": start_iso,
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end_iso,
                "timeZone": "UTC",
            }
        }
        event = service.events().insert(calendarId="primary", body=event_body).execute()
        logger.info("Created calendar event '%s' with ID %s", title, event["id"])
        return event
    except Exception as e:
        logger.error("Failed to create calendar event: %s", e)
        raise e

def delete_event(event_id: str) -> dict:
    """
    Delete a calendar event by its ID. Requires user confirmation before calling.

    Args:
        event_id: The Google Calendar event ID.

    Returns:
        A dict confirming deletion.
    """
    try:
        service = get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        logger.info("Deleted calendar event %s", event_id)
        return {"status": "success", "event_id": event_id}
    except Exception as e:
        logger.error("Failed to delete calendar event %s: %s", event_id, e)
        raise e
