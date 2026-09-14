# sidecar/tools/__init__.py — External tool integrations
# Components: Gmail, Calendar, Web (Playwright), Files (Tika/pypdf)
# Implementation: Phase 9

from .gmail_tool import list_recent_emails, read_email, draft_email, send_email
from .calendar_tool import list_events_today, list_events_range, create_event, delete_event
from .web_tool import search_web, read_page
from .file_tool import list_files, read_file, search_files
from .system import open_app, focus_app, list_open_windows
from .schedule_tool import schedule_reminder, list_reminders, cancel_reminder
from .code_tool import write_code, read_code, list_workspace, run_python
from .web_session import web_navigate, web_click, web_type, web_page_text, close_web_session
