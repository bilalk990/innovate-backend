import jwt
import mongoengine
from django.conf import settings
from channels.db import database_sync_to_async
from urllib.parse import parse_qs

@database_sync_to_async
def get_user_from_token(token):
    # Import here to avoid AppRegistryNotReady error
    from django.contrib.auth.models import AnonymousUser
    from accounts.models import User
    import logging
    logger = logging.getLogger('innovaite')
    
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id = payload.get('user_id')
        if not user_id:
            logger.error(f'[WS Auth] No user_id in JWT payload')
            return AnonymousUser()

        user = User.objects.get(id=user_id)
        if not user.is_active:
            logger.error(f'[WS Auth] User {user_id} is not active')
            return AnonymousUser()

        # Token version check
        token_version = payload.get('token_version', 0)
        if token_version != getattr(user, 'token_version', 0):
            logger.error(f'[WS Auth] Token version mismatch for user {user_id}')
            return AnonymousUser()

        logger.info(f'[WS Auth] Successfully authenticated user: {user.email} (role: {user.role})')
        return user
    except jwt.ExpiredSignatureError as e:
        logger.error(f'[WS Auth] JWT expired: {e}')
        return AnonymousUser()
    except jwt.InvalidTokenError as e:
        logger.error(f'[WS Auth] Invalid JWT: {e}')
        return AnonymousUser()
    except User.DoesNotExist:
        logger.error(f'[WS Auth] User not found in database')
        return AnonymousUser()
    except Exception as e:
        logger.error(f'[WS Auth] Unexpected error: {type(e).__name__}: {e}')
        return AnonymousUser()

class JWTAuthMiddleware:
    """
    Custom middleware that takes a token from the query string and authenticates the user.
    Usage: ws://.../?token=<token>
    """
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        # Import here to avoid AppRegistryNotReady error
        from django.contrib.auth.models import AnonymousUser
        import logging
        logger = logging.getLogger('innovaite')
        
        query_string = parse_qs(scope['query_string'].decode())
        token = query_string.get('token', [None])[0]

        # CRITICAL FIX: Require token for WebSocket connections
        if token:
            logger.info(f'[WS Auth] Token received, length: {len(token)} chars')
            scope['user'] = await get_user_from_token(token)
            if scope['user'].is_authenticated:
                logger.info(f'[WS Auth] User authenticated: {scope["user"].email}')
            else:
                logger.warning(f'[WS Auth] Token provided but authentication failed')
        else:
            # No token provided - set AnonymousUser (consumer will reject)
            logger.warning(f'[WS Auth] No token in query string')
            scope['user'] = AnonymousUser()
            # Note: Consumer must check is_authenticated and close connection

        return await self.inner(scope, receive, send)

def JWTAuthMiddlewareStack(inner):
    return JWTAuthMiddleware(inner)
