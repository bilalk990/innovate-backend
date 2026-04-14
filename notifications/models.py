"""
Notifications app models — MongoDB Documents via mongoengine
"""
import mongoengine as me
from datetime import datetime


class Notification(me.Document):
    recipient_id = me.StringField(required=True)   # User ObjectId
    sender_id = me.StringField(default=None)        # Optional
    notification_type = me.StringField(
        choices=[
            'interview_scheduled',
            'interview_starting',
            'evaluation_ready',
            'resume_parsed',
            'candidate_invited',
            'new_application',
            'application_status',
            'system',
            'ai_quota_warning',
            'ai_quota_exhausted',
        ],
        default='system'
    )
    title = me.StringField(required=True)
    message = me.StringField(required=True)
    link = me.StringField(default='')    # Frontend route to navigate to
    is_read = me.BooleanField(default=False)
    created_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'notifications',
        'indexes': ['recipient_id', 'is_read'],
        'ordering': ['-created_at'],
    }

    def to_dict(self):
        return {
            'id': str(self.id),
            'recipient_id': self.recipient_id,
            'sender_id': self.sender_id,
            'notification_type': self.notification_type,
            'title': self.title,
            'message': self.message,
            'link': self.link,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat(),
        }
