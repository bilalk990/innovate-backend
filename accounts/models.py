"""
Accounts app models — MongoDB Documents via mongoengine
Roles: candidate, recruiter, admin
"""
import mongoengine as me
from datetime import datetime


class User(me.Document):
    name = me.StringField(required=True, max_length=100)
    email = me.EmailField(required=True, unique=True)
    password = me.StringField(required=True)  # bcrypt hash
    role = me.StringField(
        required=True,
        choices=['candidate', 'recruiter', 'admin'],
        default='candidate'
    )
    profile_pic = me.StringField(default='')
    phone = me.StringField(default='')
    company = me.StringField(default='')  # for recruiters
    company_name = me.StringField(default='')
    company_values = me.ListField(me.StringField())  # For Culture Fit analysis
    bio = me.StringField(default='')
    is_active = me.BooleanField(default=True)
    google_tokens = me.StringField(default='')  # encrypted JSON — #46

    # MFA fields
    mfa_enabled = me.BooleanField(default=False)
    mfa_secret = me.StringField(default='')
    mfa_backup_codes = me.ListField(me.StringField(), default=[])  # Hashed backup codes
    token_version = me.IntField(default=0)  # For invalidating tokens on password change
    
    # Email verification
    email_verified = me.BooleanField(default=False)
    verification_token = me.StringField(default='')
    verification_token_created = me.DateTimeField(default=None)
    
    # Profile fields
    headline = me.StringField(max_length=200, default='')
    detailed_skills = me.ListField(me.StringField(), default=[])
    work_history = me.ListField(me.DictField(), default=[]) # {title, company, duration, description}
    education_history = me.ListField(me.DictField(), default=[]) # {degree, institution, year, details}
    projects = me.ListField(me.DictField(), default=[]) # {name, description, technologies, link}
    certifications = me.ListField(me.DictField(), default=[]) # {name, platform, year}
    achievements = me.ListField(me.StringField(), default=[]) # List of achievements/activities
    languages = me.ListField(me.StringField(), default=[]) # ['English', 'Urdu', etc.]
    is_profile_complete = me.BooleanField(default=False)
    
    # Location
    location = me.StringField(default='')

    # User timezone for proper scheduling
    timezone = me.StringField(default='UTC')

    created_at = me.DateTimeField(default=datetime.utcnow)
    updated_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'users',
        'indexes': ['email', 'role'],
        'ordering': ['-created_at'],
    }

    @property
    def is_authenticated(self):
        """Always True for logged-in User objects. Required for Django compatibility."""
        return True

    @property
    def is_anonymous(self):
        """Always False for User documents. Required for Django compatibility."""
        return False

    @property
    def is_staff(self):
        """Admins get staff status for Django/DRF admin area access."""
        return self.role == 'admin'

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'profile_pic': self.profile_pic,
            'phone': self.phone,
            'company': self.company,
            'company_name': self.company_name,
            'company_values': self.company_values,
            'bio': self.bio,
            'headline': self.headline,
            'location': self.location,
            'is_active': self.is_active,
            'is_profile_complete': self.is_profile_complete,
            'detailed_skills': self.detailed_skills if self.detailed_skills else [],
            'work_history': self.work_history if self.work_history else [],
            'education_history': self.education_history if self.education_history else [],
            'projects': getattr(self, 'projects', []) or [],
            'certifications': getattr(self, 'certifications', []) or [],
            'achievements': getattr(self, 'achievements', []) or [],
            'languages': getattr(self, 'languages', []) or [],
            'has_google_sync': bool(self._get_google_tokens().get('refresh_token')),
            'mfa_enabled': self.mfa_enabled,
            'created_at': self.created_at.isoformat(),
        }

    def _get_google_tokens(self) -> dict:
        """Decrypt and return google tokens dict."""
        if not self.google_tokens:
            return {}
        try:
            from core.encryption import decrypt_dict
            return decrypt_dict(self.google_tokens)
        except Exception:
            return {}

    def set_google_tokens(self, tokens: dict):
        """Encrypt and store google tokens."""
        try:
            from core.encryption import encrypt_dict
            self.google_tokens = encrypt_dict(tokens)
        except Exception:
            pass


class SystemConfiguration(me.Document):

    site_name = me.StringField(default='InnovAIte Interview Guardian')
    maintenance_mode = me.BooleanField(default=False)
    allow_registration = me.BooleanField(default=True)
    email_notifications = me.BooleanField(default=True)
    mfa_required = me.BooleanField(default=False)
    updated_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'system_settings',
    }

    def to_dict(self):
        return {
            'siteName': self.site_name,
            'maintenanceMode': self.maintenance_mode,
            'allowRegistration': self.allow_registration,
            'emailNotifications': self.email_notifications,
            'mfaRequired': self.mfa_required,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

