"""
core/google_service.py — Google Calendar & Meet Integration
Handles token refresh automatically to prevent "Token Expired" errors.
"""
import os
import logging
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from django.conf import settings

logger = logging.getLogger('innovaite')

SCOPES = [
    'https://www.googleapis.com/auth/calendar.events',
    'https://www.googleapis.com/auth/calendar.settings.readonly'
]


def _build_flow():
    """Helper to build a reusable OAuth flow from Django settings."""
    return Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=SCOPES
    )


def get_google_auth_url():
    """Generate the URL for the recruiter to sync their Google account."""
    flow = _build_flow()
    # Force prompt for offline access to get a refresh token every time
    auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')
    return auth_url


def exchange_code_for_tokens(code):
    """Exchange auth code for access + refresh tokens."""
    flow = _build_flow()
    flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
    flow.fetch_token(code=code)
    creds = flow.credentials
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': list(creds.scopes) if creds.scopes else SCOPES,
        'expiry': creds.expiry.isoformat() if creds.expiry else None,
    }


def _build_credentials(user_tokens: dict) -> Credentials | None:
    """
    Build a Credentials object from stored tokens.
    Automatically refreshes the access token if it has expired.
    Returns None if refresh_token is missing (cannot recover).
    """
    refresh_token = user_tokens.get('refresh_token')
    if not refresh_token:
        logger.warning('[Google] No refresh_token stored — recruiter must re-sync Google account.')
        return None

    # Parse expiry if present
    expiry = None
    expiry_str = user_tokens.get('expiry')
    if expiry_str:
        try:
            expiry = datetime.fromisoformat(expiry_str)
            # Make timezone-aware for proper comparison
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            expiry = None

    creds = Credentials(
        token=user_tokens.get('token'),
        refresh_token=refresh_token,
        token_uri=user_tokens.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=user_tokens.get('client_id', settings.GOOGLE_CLIENT_ID),
        client_secret=user_tokens.get('client_secret', settings.GOOGLE_CLIENT_SECRET),
        scopes=user_tokens.get('scopes', SCOPES),
        expiry=expiry,
    )

    # Auto-refresh if the token is expired or missing
    if not creds.valid:
        try:
            logger.info('[Google] Access token expired — refreshing automatically...')
            creds.refresh(GoogleRequest())
            logger.info('[Google] Token refreshed successfully.')
        except Exception as e:
            logger.error(f'[Google] Token refresh failed: {e}')
            return None

    return creds


def create_google_meet_link(
    user_tokens: dict,
    interview_title: str,
    start_time: str,
    duration_minutes: int
) -> str | None:
    """
    Create a Google Calendar event with a Google Meet link.
    Handles token refresh automatically.

    Returns:
        str: The Meet link (e.g. https://meet.google.com/xxx-xxxx-xxx)
        None: If token is invalid/missing or API call fails
    """
    creds = _build_credentials(user_tokens)
    if not creds:
        return None

    try:
        service = build('calendar', 'v3', credentials=creds)

        from datetime import timedelta
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        event = {
            'summary': f'InnovAIte Interview: {interview_title}',
            'description': 'Live AI-assisted technical interview via InnovAIte Guardian.',
            'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'UTC'},
            'conferenceData': {
                'createRequest': {
                    'requestId': f"iv_{os.urandom(8).hex()}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeeting'}
                }
            },
        }

        created_event = service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1
        ).execute()

        meet_link = created_event.get('hangoutLink')
        if meet_link:
            logger.info(f'[Google Meet] Created link: {meet_link}')
        else:
            logger.warning('[Google Meet] Event created but no hangoutLink returned.')
        return meet_link

    except Exception as e:
        logger.error(f'[Google Meet] Error creating calendar event: {e}')
        return None


def get_refreshed_tokens(user_tokens: dict) -> dict | None:
    """
    Refresh tokens and return the updated token dict (for saving back to the user document).
    Returns None if refresh fails.
    """
    creds = _build_credentials(user_tokens)
    if not creds:
        return None
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token or user_tokens.get('refresh_token'),
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': list(creds.scopes) if creds.scopes else SCOPES,
        'expiry': creds.expiry.isoformat() if creds.expiry else None,
    }
