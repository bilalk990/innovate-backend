import jwt
import mongoengine
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from channels.db import database_sync_to_async
from accounts.models import User
from urllib.parse import parse_qs

@database_sync_to_async
def get_user_from_token(token):
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get('user_id')
        if not user_id:
            return AnonymousUser()

        user = User.objects.get(id=user_id)
        if not user.is_active:
            return AnonymousUser()

        # Token version check
        token_version = payload.get('token_version', 0)
        if token_version != getattr(user, 'token_version', 0):
            return AnonymousUser()

        return user
    except Exception:
        return AnonymousUser()

class JWTAuthMiddleware:
    """
    Custom middleware that takes a token from the query string and authenticates the user.
    Usage: ws://.../?token=<token>
    """
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        query_string = parse_qs(scope['query_string'].decode())
        token = query_string.get('token', [None])[0]

        # CRITICAL FIX: Require token for WebSocket connections
        if token:
            scope['user'] = await get_user_from_token(token)
        else:
            # No token provided - set AnonymousUser (consumer will reject)
            scope['user'] = AnonymousUser()
            # Note: Consumer must check is_authenticated and close connection

        return await self.inner(scope, receive, send)

def JWTAuthMiddlewareStack(inner):
    return JWTAuthMiddleware(inner)
