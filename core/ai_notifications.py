"""
core/ai_notifications.py — Notify all admin users when AI quota events occur.
Uses lazy imports to avoid circular dependencies.
Called from gemini.py inside _call() — must be fast and non-blocking.
"""
import threading
import logging
from datetime import date

logger = logging.getLogger('innovaite')

_ALERT_LABELS = {
    'AI_QUOTA_WARNING': {
        'title': '⚠️ AI Quota Warning',
        'message': (
            'Gemini API daily call count has reached the warning threshold. '
            'AI features are still working but approaching the daily limit. '
            'Consider upgrading your Gemini plan or monitoring usage.'
        ),
        'notification_type': 'ai_quota_warning',
        'email_subject': '⚠️ InnovAIte — AI Quota Warning',
    },
    'AI_QUOTA_EXHAUSTED': {
        'title': '🔴 AI Quota Exhausted',
        'message': (
            'Gemini API daily quota is EXHAUSTED. All AI features are now unavailable '
            'until the quota resets (midnight UTC). '
            'Fix: Visit Google AI Studio to upgrade your plan or wait for reset.'
        ),
        'notification_type': 'ai_quota_exhausted',
        'email_subject': '🔴 InnovAIte — AI Service DOWN: Quota Exhausted',
    },
}


def notify_admins_async(alert_type: str):
    """
    Fire-and-forget: send in-app + email notifications to all admins.
    Runs in a background thread so it never blocks an API response.
    """
    thread = threading.Thread(
        target=_notify_admins,
        args=(alert_type,),
        daemon=True,
    )
    thread.start()


def _notify_admins(alert_type: str):
    """Actual notification logic — runs in background thread."""
    try:
        # Lazy imports to avoid circular dependency at module load time
        from accounts.models import User
        from notifications.models import Notification
        from core.email_service import send_ai_quota_alert_email

        alert = _ALERT_LABELS.get(alert_type)
        if not alert:
            return

        # Get all admin users
        admins = User.objects(role='admin', is_active=True)
        if not admins:
            logger.warning('[AI Notify] No active admin users found to notify.')
            return

        today = str(date.today())

        for admin in admins:
            # Avoid duplicate notifications: skip if one already sent today for this alert
            existing = Notification.objects(
                recipient_id=str(admin.id),
                notification_type=alert['notification_type'],
                title__contains=today,
            ).first()
            if existing:
                continue

            # Create in-app notification
            Notification(
                recipient_id=str(admin.id),
                notification_type=alert['notification_type'],
                title=f"{alert['title']} — {today}",
                message=alert['message'],
                link='/admin/dashboard',
            ).save()

            # Send email
            if admin.email:
                send_ai_quota_alert_email(
                    admin_email=admin.email,
                    admin_name=admin.name or 'Admin',
                    alert_type=alert_type,
                    alert=alert,
                )

        logger.info(f'[AI Notify] Sent {alert_type} notifications to {admins.count()} admin(s).')

    except Exception as e:
        logger.error(f'[AI Notify] Failed to notify admins: {e}')
