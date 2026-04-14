"""
Django settings for InnovAIte Interview Guardian
Backend: Django + MongoDB (mongoengine) | No ORM
"""

import os
import warnings
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config('SECRET_KEY')
if not SECRET_KEY or SECRET_KEY.startswith('django-insecure'):
    raise ValueError('SECRET_KEY must be set in .env and cannot use default insecure key!')

DEBUG = config('DEBUG', default=False, cast=bool)
ALLOWED_HOSTS = ['*']  # Temporarily allow all hosts for debugging

# ─── Diagnostic Logs (Visible in Railway logs) ───────────────────────────────
print(f"[DEBUG] ALLOWED_HOSTS: {ALLOWED_HOSTS}")
print(f"[DEBUG] CORS_ORIGINS: {config('CORS_ALLOWED_ORIGINS', default='Not Set')}")
print(f"[DEBUG] CSRF_ORIGINS: {config('CSRF_TRUSTED_ORIGINS', default='Not Set')}")
print(f"[DEBUG] MONGODB_URI starts with: {config('MONGODB_URI', default='')[:20]}...")

# ─── Installed Apps ───────────────────────────────────────────────────────────
INSTALLED_APPS = [
    'daphne',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'channels',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'accounts.apps.AccountsConfig',
    'interviews.apps.InterviewsConfig',
    'resumes.apps.ResumesConfig',
    'evaluations.apps.EvaluationsConfig',
    'notifications.apps.NotificationsConfig',
    'admin_monitoring.apps.AdminMonitoringConfig',
    'jobs.apps.JobsConfig',
]

# ─── Middleware ───────────────────────────────────────────────────────────────
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # MUST be first
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Added for stability
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Added for stability
    'django.contrib.messages.middleware.MessageMiddleware',      # Added for stability
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'core.request_id_middleware.RequestIDMiddleware',
    'core.performance_monitor.PerformanceMiddleware',
    'core.middleware.SecurityHeadersMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'
ASGI_APPLICATION = 'core.asgi.application'

# ─── Database — MongoDB via mongoengine ───────────────────────────────────────
DATABASES = {}

MONGODB_URI = config('MONGODB_URI', default='mongodb://localhost:27017/innovaite_db')
MONGODB_DB_NAME = 'innovaite_db'

# CRITICAL FIX: Skip MongoDB connection during build phase or collectstatic
import sys
IS_MANAGEMENT_COMMAND = any(arg in sys.argv for arg in ['collectstatic', 'makemigrations', 'migrate', 'test'])

if not IS_MANAGEMENT_COMMAND:
    try:
        import mongoengine
        mongoengine.connect(
            host=MONGODB_URI, 
            db=MONGODB_DB_NAME, 
            serverSelectionTimeoutMS=5000,
            maxPoolSize=50,
            minPoolSize=10
        )
        # Test connection
        mongoengine.connection.get_db().command('ping')
    except Exception as e:
        if not DEBUG:
            # In production, we usually want this to fail, 
            # but during build $MONGODB_URI might be missing.
            print(f"[WARNING] MongoDB connection failed: {e}")
            if MONGODB_URI and 'localhost' not in MONGODB_URI:
                 raise RuntimeError(f'[InnovAIte] MongoDB connection failed: {e}. Cannot start in production without DB!')
        else:
            warnings.warn(f'[InnovAIte] MongoDB connection failed: {e}. Start MongoDB and restart.')

# ─── Django Channels ──────────────────────────────────────────────────────────
REDIS_URL = config('REDIS_URL', default='')

CHANNEL_LAYERS = {
    'default': {
        # #62 — Use Redis in production for multi-process WebSocket support
        # Set REDIS_URL in .env to enable: redis://localhost:6379
        'BACKEND': 'channels_redis.core.RedisChannelLayer' if REDIS_URL else 'channels.layers.InMemoryChannelLayer',
        **({'CONFIG': {'hosts': [REDIS_URL]}} if REDIS_URL else {}),
    }
}

# ─── Static & Media Files ─────────────────────────────────────────────────────
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
os.makedirs(BASE_DIR / 'media' / 'resumes', exist_ok=True)

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ─── JWT ──────────────────────────────────────────────────────────────────────
JWT_SECRET = config('JWT_SECRET')
if not JWT_SECRET:
    raise ValueError('[InnovAIte] JWT_SECRET must be set in .env for security!')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRY_HOURS = 24
JWT_REFRESH_EXPIRY_DAYS = 7

# ─── AI Configuration ─────────────────────────────────────────────────────────
# OpenAI GPT (Primary AI Service - GPT-4o-mini)
OPENAI_API_KEY = config('OPENAI_API_KEY', default='')
if not OPENAI_API_KEY:
    warnings.warn('[InnovAIte] OPENAI_API_KEY not set. AI features disabled.')

# Daily AI usage limits for $10 budget (30-day usage)
# ~2,200 total calls possible with GPT-4o-mini
# Set to ~73 calls/day for 30-day usage
AI_DAILY_SOFT_LIMIT = config('AI_DAILY_SOFT_LIMIT', default=73, cast=int)
AI_WARNING_THRESHOLD = config('AI_WARNING_THRESHOLD', default=50, cast=int)  # 70% of daily limit


# ─── Encryption ───────────────────────────────────────────────────────────────
ENCRYPTION_KEY = config('ENCRYPTION_KEY', default='')

# CRITICAL FIX: Require explicit ENCRYPTION_KEY in production
if not ENCRYPTION_KEY:
    if not DEBUG:
        raise ValueError('[InnovAIte] ENCRYPTION_KEY must be set in .env for production!')
    else:
        warnings.warn('[InnovAIte] ENCRYPTION_KEY not set. Using SECRET_KEY derivation (not recommended for production).')
        # Fallback for development only
        import hashlib, base64
        raw = hashlib.sha256(SECRET_KEY.encode()).digest()
        ENCRYPTION_KEY = base64.urlsafe_b64encode(raw).decode()

# ─── Google OAuth ─────────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID = config('GOOGLE_CLIENT_ID', default='')
GOOGLE_CLIENT_SECRET = config('GOOGLE_CLIENT_SECRET', default='')
GOOGLE_REDIRECT_URI = config('GOOGLE_REDIRECT_URI', default='http://localhost:8000/api/auth/google/callback/')
if not GOOGLE_CLIENT_ID:
    warnings.warn('[InnovAIte] GOOGLE_CLIENT_ID not set. Google OAuth disabled.')

# ─── Email ────────────────────────────────────────────────────────────────────
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = config('EMAIL_HOST', default='smtp.gmail.com')
EMAIL_PORT = config('EMAIL_PORT', default=587, cast=int)
EMAIL_USE_TLS = config('EMAIL_USE_TLS', default=True, cast=bool)
EMAIL_HOST_USER = config('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = config('EMAIL_HOST_PASSWORD', default='')
EMAIL_FROM = config('EMAIL_FROM', default='noreply@innovaite.com')

# ─── SMS (Twilio) ─────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = config('TWILIO_ACCOUNT_SID', default='')
TWILIO_AUTH_TOKEN = config('TWILIO_AUTH_TOKEN', default='')
TWILIO_PHONE_NUMBER = config('TWILIO_PHONE_NUMBER', default='')

# ─── Frontend URL ─────────────────────────────────────────────────────────────
FRONTEND_URL = config('FRONTEND_URL', default='http://localhost:5173')

# CRITICAL FIX: Validate FRONTEND_URL to prevent open redirect vulnerabilities
from urllib.parse import urlparse
try:
    parsed_frontend_url = urlparse(FRONTEND_URL)
    if not parsed_frontend_url.scheme or parsed_frontend_url.scheme not in ['http', 'https']:
        raise ValueError('FRONTEND_URL must be a valid HTTP(S) URL')
    if not parsed_frontend_url.netloc:
        raise ValueError('FRONTEND_URL must include a domain')
except Exception as e:
    if not DEBUG:
        raise ValueError(f'Invalid FRONTEND_URL configuration: {e}')
    else:
        warnings.warn(f'Invalid FRONTEND_URL: {e}. Using default.')

# ─── CORS ─────────────────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS = config(
    'CORS_ALLOWED_ORIGINS',
    default='http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174'
).split(',')
CORS_ALLOW_CREDENTIALS = True
# Allow all origins in development for easier testing
CORS_ALLOW_ALL_ORIGINS = DEBUG

# WebSocket CORS — defined AFTER CORS_ALLOWED_ORIGINS to avoid NameError
CORS_ALLOW_WEBSOCKET_ORIGINS = CORS_ALLOWED_ORIGINS

# CSRF Trusted Origins for production
CSRF_TRUSTED_ORIGINS = config('CSRF_TRUSTED_ORIGINS', default='http://localhost:5173').split(',')

# ─── Django REST Framework ────────────────────────────────────────────────────
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'accounts.authentication.MongoJWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
        'rest_framework.throttling.ScopedRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '1000/hour',        # Increased from 100/hour
        'user': '100000/hour',      # Increased from 1000/hour
        'login': '5/minute',        # CRITICAL FIX: Reduced from 60/minute for brute force protection
        'evaluation': '200/hour',    # prevents mass spam but allows testing
    },
    'EXCEPTION_HANDLER': 'rest_framework.views.exception_handler',
    # Disable CSRF for API endpoints (using JWT authentication)
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# ─── Security ─────────────────────────────────────────────────────────────────
SECURE_SSL_REDIRECT = config('SECURE_SSL_REDIRECT', default=False, cast=bool)
SECURE_HSTS_SECONDS = config('SECURE_HSTS_SECONDS', default=0, cast=int)
SECURE_HSTS_INCLUDE_SUBDOMAINS = config('SECURE_HSTS_INCLUDE_SUBDOMAINS', default=False, cast=bool)
SECURE_HSTS_PRELOAD = config('SECURE_HSTS_PRELOAD', default=False, cast=bool)
SESSION_COOKIE_SECURE = config('SESSION_COOKIE_SECURE', default=False, cast=bool)
CSRF_COOKIE_SECURE = config('CSRF_COOKIE_SECURE', default=False, cast=bool)

# Required for Railway/Render SSL termination
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True
USE_X_FORWARDED_PORT = True

# ─── Internationalization ─────────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ─── File Upload Limits ───────────────────────────────────────────────────────
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # 10 MB

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {'format': '[{levelname}] {asctime} {module}: {message}', 'style': '{'},
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'verbose'},
    },
    'root': {'handlers': ['console'], 'level': 'WARNING'},
    'loggers': {
        'django': {'handlers': ['console'], 'level': 'WARNING', 'propagate': False},
        'innovaite': {'handlers': ['console'], 'level': 'DEBUG' if DEBUG else 'INFO', 'propagate': False},
    },
}
