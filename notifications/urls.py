from django.urls import path
from notifications.views import (
    NotificationListView, NotificationMarkReadView,
    MarkAllReadView, UnreadCountView,
)

urlpatterns = [
    path('', NotificationListView.as_view(), name='notification-list'),
    path('unread-count/', UnreadCountView.as_view(), name='notification-unread-count'),
    path('mark-all-read/', MarkAllReadView.as_view(), name='notification-mark-all-read'),
    path('<str:notif_id>/read/', NotificationMarkReadView.as_view(), name='notification-mark-read'),
]
