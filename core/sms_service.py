"""
SMS Service — Send SMS notifications via Twilio
Fallback to console logging if Twilio is not configured
"""
import logging
from django.conf import settings

logger = logging.getLogger('innovaite')


def send_sms(to_phone: str, message: str):
    """
    Send SMS using Twilio.
    Falls back to console logging if Twilio is not configured.
    """
    try:
        # Check if Twilio is configured
        if not hasattr(settings, 'TWILIO_ACCOUNT_SID') or not settings.TWILIO_ACCOUNT_SID:
            logger.info(f"[SMS] Twilio not configured. Skipping SMS to {to_phone}.")
            return False

        from twilio.rest import Client
        
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        
        sms = client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        
        logger.info(f"[SMS] Sent to {to_phone}: {sms.sid}")
        return True
    except Exception as e:
        logger.warning(f"[SMS] Failed to send to {to_phone}: {str(e)}")
        return False


def send_interview_scheduled_sms(candidate_phone: str, interview_data: dict):
    """Send interview scheduled SMS to candidate."""
    message = f"""
InnovAIte Interview Guardian

Interview Scheduled:
{interview_data['title']}

Date: {interview_data['scheduled_at']}
Duration: {interview_data['duration_minutes']} min

Join: {settings.FRONTEND_URL}/interview/room/{interview_data['room_id']}

Good luck! 🚀
    """.strip()
    
    return send_sms(candidate_phone, message)


def send_interview_reminder_sms(candidate_phone: str, interview_data: dict):
    """Send interview reminder SMS 1 hour before."""
    message = f"""
⏰ REMINDER: Your interview starts in 1 hour!

{interview_data['title']}
Time: {interview_data['scheduled_at']}

Join: {settings.FRONTEND_URL}/interview/room/{interview_data['room_id']}

- InnovAIte
    """.strip()
    
    return send_sms(candidate_phone, message)
