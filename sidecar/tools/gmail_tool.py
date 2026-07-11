import logging
import base64
from email.message import EmailMessage
from googleapiclient.discovery import build
from .google_auth import get_google_credentials

logger = logging.getLogger("sylph.tools.gmail")

def get_gmail_service():
    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)

def list_recent_emails(count: int = 3) -> list[dict]:
    """
    List the recent email messages from the user's Gmail inbox.

    Args:
        count: The number of recent emails to retrieve.

    Returns:
        A list of dictionaries containing email info: id, threadId, sender, subject, date, snippet.
    """
    try:
        service = get_gmail_service()
        results = service.users().messages().list(userId="me", maxResults=count, labelIds=["INBOX"]).execute()
        messages = results.get("messages", [])

        emails = []
        for msg in messages:
            msg_id = msg["id"]
            # Fetch message details
            detail = service.users().messages().get(userId="me", id=msg_id, format="metadata",
                                                     metadataHeaders=["From", "Subject", "Date"]).execute()
            headers = detail.get("payload", {}).get("headers", [])
            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
            date = next((h["value"] for h in headers if h["name"] == "Date"), "")
            snippet = detail.get("snippet", "")

            emails.append({
                "id": msg_id,
                "threadId": msg.get("threadId"),
                "sender": sender,
                "subject": subject,
                "date": date,
                "snippet": snippet,
            })
        logger.info("Retrieved %d recent emails", len(emails))
        return emails
    except Exception as e:
        logger.error("Failed to list recent emails: %s", e)
        raise e

def read_email(msg_id: str) -> dict:
    """
    Get the full body and details of a specific email by ID.

    Args:
        msg_id: The Gmail message ID.

    Returns:
        A dictionary containing sender, subject, date, snippet, and body.
    """
    try:
        service = get_gmail_service()
        detail = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        payload = detail.get("payload", {})
        headers = payload.get("headers", [])
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No Subject)")
        date = next((h["value"] for h in headers if h["name"] == "Date"), "")
        snippet = detail.get("snippet", "")

        # Extract body text
        body = ""
        parts = [payload]
        while parts:
            part = parts.pop()
            mime_type = part.get("mimeType", "")
            if mime_type.startswith("text/plain"):
                data = part.get("body", {}).get("data", "")
                if data:
                    body += base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            elif "parts" in part:
                parts.extend(part["parts"])

        if not body:
            body = snippet  # Fallback to snippet if body extraction fails

        logger.info("Successfully read email %s", msg_id)
        return {
            "id": msg_id,
            "sender": sender,
            "subject": subject,
            "date": date,
            "snippet": snippet,
            "body": body.strip(),
        }
    except Exception as e:
        logger.error("Failed to read email %s: %s", msg_id, e)
        raise e

def draft_email(to: str, subject: str, body: str) -> dict:
    """
    Draft an email.

    Args:
        to: Recipient email address.
        subject: Email subject.
        body: Email body text.

    Returns:
        The created draft info containing draft id and message details.
    """
    try:
        service = get_gmail_service()
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to
        message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        create_message = {"message": {"raw": encoded_message}}

        draft = service.users().drafts().create(userId="me", body=create_message).execute()
        logger.info("Created email draft with ID %s", draft["id"])
        return draft
    except Exception as e:
        logger.error("Failed to create email draft: %s", e)
        raise e

def send_email(draft_id: str) -> dict:
    """
    Send an email draft by its ID. Requires user confirmation before calling.

    Args:
        draft_id: The ID of the draft to send.

    Returns:
        The sent message details.
    """
    try:
        service = get_gmail_service()
        sent = service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        logger.info("Successfully sent draft %s", draft_id)
        return sent
    except Exception as e:
        logger.error("Failed to send draft %s: %s", draft_id, e)
        raise e
