"""
Performance Monitoring Service
Tracks system performance metrics, latency, and health
"""
import logging
import mongoengine as me
from datetime import datetime, timedelta
import time

logger = logging.getLogger('innovaite')


class PerformanceMetric(me.Document):
    """Store performance metrics for monitoring."""
    metric_type = me.StringField(required=True)  # api_latency, video_latency, page_load, etc.
    endpoint = me.StringField(default='')
    value = me.FloatField(required=True)  # milliseconds or percentage
    user_id = me.StringField(default='')
    status_code = me.IntField(default=200)
    metadata = me.DictField(default={})
    timestamp = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'performance_metrics',
        'indexes': ['metric_type', 'endpoint', 'timestamp'],
        'ordering': ['-timestamp'],
    }


class SystemHealth(me.Document):
    """System health snapshot."""
    cpu_usage = me.FloatField(default=0)
    memory_usage = me.FloatField(default=0)
    active_users = me.IntField(default=0)
    active_interviews = me.IntField(default=0)
    avg_api_latency = me.FloatField(default=0)
    error_rate = me.FloatField(default=0)
    timestamp = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'system_health',
        'ordering': ['-timestamp'],
    }


def track_api_latency(endpoint: str, latency_ms: float, user_id: str = '', status_code: int = 200):
    """Track API endpoint latency."""
    try:
        PerformanceMetric(
            metric_type='api_latency',
            endpoint=endpoint,
            value=latency_ms,
            user_id=user_id,
            status_code=status_code,
        ).save()
        
        # Alert if latency exceeds SRS requirement (2000ms)
        if latency_ms > 2000:
            logger.warning(f"[Performance] High latency detected: {endpoint} - {latency_ms}ms")
    except Exception as e:
        logger.error(f"[Performance] Failed to track latency: {str(e)}")


def track_video_latency(room_id: str, latency_ms: float, user_id: str = ''):
    """Track video streaming latency."""
    try:
        PerformanceMetric(
            metric_type='video_latency',
            endpoint=f'room_{room_id}',
            value=latency_ms,
            user_id=user_id,
        ).save()
        
        # Alert if latency exceeds SRS requirement (200ms)
        if latency_ms > 200:
            logger.warning(f"[Performance] High video latency: Room {room_id} - {latency_ms}ms")
    except Exception as e:
        logger.error(f"[Performance] Failed to track video latency: {str(e)}")


def track_page_load(page: str, load_time_ms: float, user_id: str = ''):
    """Track frontend page load time."""
    try:
        PerformanceMetric(
            metric_type='page_load',
            endpoint=page,
            value=load_time_ms,
            user_id=user_id,
        ).save()
    except Exception as e:
        logger.error(f"[Performance] Failed to track page load: {str(e)}")


def get_performance_summary(hours: int = 24):
    """Get performance summary for the last N hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    metrics = PerformanceMetric.objects(timestamp__gte=since)
    
    api_metrics = [m for m in metrics if m.metric_type == 'api_latency']
    video_metrics = [m for m in metrics if m.metric_type == 'video_latency']
    
    summary = {
        'period_hours': hours,
        'total_requests': len(api_metrics),
        'avg_api_latency': sum(m.value for m in api_metrics) / len(api_metrics) if api_metrics else 0,
        'max_api_latency': max((m.value for m in api_metrics), default=0),
        'avg_video_latency': sum(m.value for m in video_metrics) / len(video_metrics) if video_metrics else 0,
        'max_video_latency': max((m.value for m in video_metrics), default=0),
        'error_count': len([m for m in api_metrics if m.status_code >= 400]),
        'error_rate': len([m for m in api_metrics if m.status_code >= 400]) / len(api_metrics) * 100 if api_metrics else 0,
    }
    
    return summary


def capture_system_health():
    """Capture current system health snapshot."""
    try:
        import psutil
        from interviews.models import Interview
        from accounts.models import User
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get application metrics
        active_interviews = Interview.objects(status='active').count()
        
        # Get recent performance
        summary = get_performance_summary(hours=1)
        
        health = SystemHealth(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_interviews=active_interviews,
            avg_api_latency=summary['avg_api_latency'],
            error_rate=summary['error_rate'],
        )
        health.save()
        
        logger.info(f"[Health] CPU: {cpu_usage}% | Memory: {memory_usage}% | Active Interviews: {active_interviews}")
        
        return health.to_dict()
    except Exception as e:
        logger.error(f"[Health] Failed to capture: {str(e)}")
        return {}


# Middleware for automatic latency tracking
class PerformanceMiddleware:
    """Django middleware to automatically track API latency."""
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        response = self.get_response(request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Track API latency
        user_id = ''
        if hasattr(request, 'user') and hasattr(request.user, 'id'):
            user_id = str(request.user.id)
        
        track_api_latency(
            endpoint=request.path,
            latency_ms=latency_ms,
            user_id=user_id,
            status_code=response.status_code
        )
        
        return response
