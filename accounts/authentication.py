"""
Custom JWT Authentication for Django REST Framework
Works with mongoengine User documents (no Django ORM)
"""
import jwt
import mongoengine
from django.conf import settings
from rest_framework import authentication, exceptions
from accounts.models import User


class MongoJWTAuthentication(authentication.BaseAuthentication):
    """Validate Bearer JWT tokens and attach the mongoengine User document."""

    def authenticate(self, request):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return None

        token = auth_header.split(' ', 1)[1].strip()
        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('Token has expired.')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token.')

        user_id = payload.get('user_id')
        if not user_id:
            raise exceptions.AuthenticationFailed('Invalid token payload.')

        try:
            user = User.objects.get(id=user_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            # mongoengine raises DoesNotExist (not User.DoesNotExist like Django ORM)
            raise exceptions.AuthenticationFailed('User not found.')
        except Exception:
            raise exceptions.AuthenticationFailed('Authentication error.')

        if not user.is_active:
            raise exceptions.AuthenticationFailed('User account is disabled.')

        # Check token version for invalidation (e.g., after password change)
        token_version = payload.get('token_version', 0)
        if token_version != getattr(user, 'token_version', 0):
            raise exceptions.AuthenticationFailed('Token has been invalidated. Please log in again.')

        return (user, token)
