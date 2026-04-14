"""
ASGI config for InnovAIte Interview Guardian (supports WebSockets via Channels)
"""
import os
import django

# Set Django settings module FIRST
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Setup Django BEFORE importing anything else
django.setup()

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from core.websocket_auth import JWTAuthMiddleware
import interviews.routing

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
