"""
Notifications app views — create and fetch in-app notifications
"""
import mongoengine
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from notifications.models import Notification


class NotificationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        notifs = Notification.objects(recipient_id=str(request.user.id)).order_by('-created_at')[:50]
        return Response([n.to_dict() for n in notifs])


class NotificationMarkReadView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, notif_id):
        try:
            notif = Notification.objects.get(id=notif_id, recipient_id=str(request.user.id))
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Notification not found.'}, status=404)
        notif.is_read = True
        notif.save()
        return Response(notif.to_dict())


class MarkAllReadView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        Notification.objects(
            recipient_id=str(request.user.id), is_read=False
        ).update(is_read=True)
        return Response({'message': 'All notifications marked as read.'})


class UnreadCountView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        count = Notification.objects(
            recipient_id=str(request.user.id), is_read=False
        ).count()
        return Response({'unread_count': count})
