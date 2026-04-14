"""
ASGI config for InnovAIte Interview Guardian (supports WebSockets via Channels)
"""
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from core.websocket_auth import JWTAuthMiddleware
import interviews.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': JWTAuthMiddleware(
        AuthMiddlewareStack(
            URLRouter(
                interviews.routing.websocket_urlpatterns
            )
        )
    ),
})
