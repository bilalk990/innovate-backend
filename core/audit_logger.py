"""
Audit Logger — Comprehensive activity logging for security and compliance
Tracks all critical user actions with timestamps and metadata
"""
import logging
import mongoengine as me
from datetime import datetime

logger = logging.getLogger('innovaite')


class AuditLog(me.Document):
    """Audit log entry for tracking user activities."""
    user_id = me.StringField(required=True)
    user_email = me.StringField(default='')
    user_role = me.StringField(default='')
    action = me.StringField(required=True)  # login, logout, create_interview, etc.
    resource_type = me.StringField(default='')  # interview, evaluation, user, etc.
    resource_id = me.StringField(default='')
    ip_address = me.StringField(default='')
    user_agent = me.StringField(default='')
    status = me.StringField(choices=['success', 'failure', 'warning'], default='success')
    details = me.DictField(default={})  # Additional context
    timestamp = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'audit_logs',
        'indexes': ['user_id', 'action', 'timestamp', 'status'],
        'ordering': ['-timestamp'],
    }

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': self.user_id,
            'user_email': self.user_email,
            'user_role': self.user_role,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'ip_address': self.ip_address,
            'status': self.status,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


def log_audit(user, action: str, resource_type: str = '', resource_id: str = '', 
              status: str = 'success', details: dict = None, request=None):
    """
    Create an audit log entry.
    
    Args:
        user: User object or user_id string
        action: Action performed (e.g., 'login', 'create_interview')
        resource_type: Type of resource affected
        resource_id: ID of the resource
        status: 'success', 'failure', or 'warning'
        details: Additional context dictionary
        request: Django request object for IP and user agent
    """
    try:
        user_id = str(user.id) if hasattr(user, 'id') else str(user)
        user_email = user.email if hasattr(user, 'email') else ''
        user_role = user.role if hasattr(user, 'role') else ''
        
        ip_address = ''
        user_agent = ''
        if request:
            ip_address = request.META.get('REMOTE_ADDR', '')
            user_agent = request.META.get('HTTP_USER_AGENT', '')[:200]
        
        log_entry = AuditLog(
            user_id=user_id,
            user_email=user_email,
            user_role=user_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            details=details or {},
        )
        log_entry.save()
        
        logger.info(f"[Audit] {user_email} - {action} - {status}")
    except Exception as e:
        logger.error(f"[Audit] Failed to log: {str(e)}")


# Convenience functions for common actions
def log_login(user, request, status='success'):
    log_audit(user, 'login', status=status, request=request)


def log_logout(user, request):
    log_audit(user, 'logout', request=request)


def log_interview_created(user, interview_id, request):
    log_audit(user, 'create_interview', 'interview', interview_id, request=request)


def log_interview_joined(user, interview_id, request):
    log_audit(user, 'join_interview', 'interview', interview_id, request=request)


def log_evaluation_triggered(user, evaluation_id, request):
    log_audit(user, 'trigger_evaluation', 'evaluation', evaluation_id, request=request)


def log_resume_uploaded(user, resume_id, request):
    log_audit(user, 'upload_resume', 'resume', resume_id, request=request)


def log_security_violation(user, violation_type, details, request):
    log_audit(user, f'security_violation_{violation_type}', status='warning', 
              details=details, request=request)
