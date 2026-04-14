"""
Admin Monitoring Dashboard Views
System health, performance metrics, audit logs
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from datetime import datetime, timedelta
from core.performance_monitor import get_performance_summary, capture_system_health, PerformanceMetric, SystemHealth
from core.audit_logger import AuditLog
from accounts.models import User
from interviews.models import Interview
from evaluations.models import Evaluation


class SystemHealthView(APIView):
    """Get current system health metrics."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        
        health = capture_system_health()
        return Response(health)


class PerformanceDashboardView(APIView):
    """Get performance metrics dashboard data."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        
        hours = int(request.query_params.get('hours', 24))
        summary = get_performance_summary(hours=hours)
        
        # Get recent metrics
        since = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = PerformanceMetric.objects(timestamp__gte=since).order_by('-timestamp')[:100]
        
        return Response({
            'summary': summary,
            'recent_metrics': [
                {
                    'metric_type': m.metric_type,
                    'endpoint': m.endpoint,
                    'value': m.value,
                    'timestamp': m.timestamp.isoformat(),
                }
                for m in recent_metrics
            ]
        })


class AuditLogsView(APIView):
    """Get audit logs with filtering."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        
        # Filters
        user_id = request.query_params.get('user_id')
        action = request.query_params.get('action')
        status = request.query_params.get('status')
        hours = int(request.query_params.get('hours', 24))
        limit = int(request.query_params.get('limit', 100))
        
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Build query
        query = {'timestamp__gte': since}
        if user_id:
            query['user_id'] = user_id
        if action:
            query['action'] = action
        if status:
            query['status'] = status
        
        logs = AuditLog.objects(**query).order_by('-timestamp')[:limit]
        
        return Response({
            'total': len(logs),
            'logs': [log.to_dict() for log in logs]
        })


class SystemStatsView(APIView):
    """Get overall system statistics."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        
        # User stats
        total_users = User.objects.count()
        active_users = User.objects(is_active=True).count()
        candidates = User.objects(role='candidate').count()
        recruiters = User.objects(role='recruiter').count()
        admins = User.objects(role='admin').count()
        
        # Interview stats
        total_interviews = Interview.objects.count()
        scheduled_interviews = Interview.objects(status='scheduled').count()
        active_interviews = Interview.objects(status='active').count()
        completed_interviews = Interview.objects(status='completed').count()
        
        # Evaluation stats
        total_evaluations = Evaluation.objects.count()
        pending_evaluations = Evaluation.objects(status='pending').count()
        
        # Recent activity (last 24 hours)
        since_24h = datetime.utcnow() - timedelta(hours=24)
        new_users_24h = User.objects(created_at__gte=since_24h).count()
        new_interviews_24h = Interview.objects(created_at__gte=since_24h).count()
        new_evaluations_24h = Evaluation.objects(created_at__gte=since_24h).count()
        
        # Average scores
        evaluations = Evaluation.objects(status='complete')
        avg_score = sum(e.overall_score for e in evaluations) / len(evaluations) if evaluations else 0
        
        return Response({
            'users': {
                'total': total_users,
                'active': active_users,
                'candidates': candidates,
                'recruiters': recruiters,
                'admins': admins,
                'new_24h': new_users_24h,
            },
            'interviews': {
                'total': total_interviews,
                'scheduled': scheduled_interviews,
                'active': active_interviews,
                'completed': completed_interviews,
                'new_24h': new_interviews_24h,
            },
            'evaluations': {
                'total': total_evaluations,
                'pending': pending_evaluations,
                'avg_score': round(avg_score, 1),
                'new_24h': new_evaluations_24h,
            }
        })


class SecurityAlertsView(APIView):
    """Get security-related audit logs and alerts."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        
        hours = int(request.query_params.get('hours', 24))
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get failed login attempts
        failed_logins = AuditLog.objects(
            action='login',
            status='failure',
            timestamp__gte=since
        ).order_by('-timestamp')[:50]
        
        # Get security violations
        violations = AuditLog.objects(
            action__startswith='security_violation',
            timestamp__gte=since
        ).order_by('-timestamp')[:50]
        
        # Get MFA events
        mfa_events = AuditLog.objects(
            action__startswith='mfa',
            timestamp__gte=since
        ).order_by('-timestamp')[:50]
        
        return Response({
            'failed_logins': [log.to_dict() for log in failed_logins],
            'security_violations': [log.to_dict() for log in violations],
            'mfa_events': [log.to_dict() for log in mfa_events],
        })
