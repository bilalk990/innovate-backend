from django.urls import re_path
from interviews.consumers import SignalingConsumer

websocket_urlpatterns = [
    re_path(r'ws/room/(?P<room_id>[a-zA-Z0-9]+)/$', SignalingConsumer.as_asgi()),
]
