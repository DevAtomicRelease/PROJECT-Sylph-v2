import os
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger("sylph.tools.google_auth")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

# Save credentials in a standard path under the user's home directory
CONFIG_DIR = os.path.expanduser("~/.sylph/config")
TOKEN_PATH = os.path.join(CONFIG_DIR, "token.json")
CREDENTIALS_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "credentials.json"))

def get_google_credentials() -> Credentials:
    """Load or retrieve Google OAuth2 credentials."""
    creds = None
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Load token from ~/.sylph/config/token.json if it exists
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
            logger.info("Loaded Google OAuth credentials from token.json")
        except Exception as e:
            logger.warning("Failed to load credentials from token.json: %s", e)

    # If credentials don't exist or are invalid, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Refreshed Google OAuth credentials")
            except Exception as e:
                logger.warning("Failed to refresh Google OAuth credentials: %s", e)
                creds = None

        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Google API credentials file not found at {CREDENTIALS_PATH}. "
                    "Please download credentials.json from Google Cloud Console and place it at the project root."
                )

            logger.info("Starting local Google OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
            logger.info("Google OAuth flow successful")

        # Save credentials for future use
        try:
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())
            logger.info("Saved Google OAuth credentials to %s", TOKEN_PATH)
        except Exception as e:
            logger.error("Failed to save credentials to %s: %s", TOKEN_PATH, e)

    return creds
