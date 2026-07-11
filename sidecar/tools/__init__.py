# sidecar/tools/__init__.py — External tool integrations
# Components: Gmail, Calendar, Web (Playwright), Files (Tika/pypdf)
# Implementation: Phase 9

from .gmail_tool import list_recent_emails, read_email, draft_email, send_email
from .calendar_tool import list_events_today, list_events_range, create_event, delete_event
from .web_tool import search_web, read_page
from .file_tool import list_files, read_file, search_files
