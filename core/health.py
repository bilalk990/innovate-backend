"""
Health check endpoint for monitoring and load balancers
"""
import mongoengine
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny


class HealthCheckView(APIView):
    """GET /api/health/ - System health check"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        health = {
            'status': 'healthy',
            'checks': {}
        }
        
        # Check MongoDB connection
        try:
            mongoengine.connection.get_db().command('ping')
            health['checks']['mongodb'] = 'connected'
        except Exception as e:
            health['checks']['mongodb'] = f'error: {str(e)}'
            health['status'] = 'unhealthy'
        
        # Check Redis (if configured)
        if settings.REDIS_URL:
            try:
                from django.core.cache import cache
                cache.set('health_check', 'ok', 10)
                if cache.get('health_check') == 'ok':
                    health['checks']['redis'] = 'connected'
                else:
                    health['checks']['redis'] = 'error: cache test failed'
                    health['status'] = 'degraded'
            except Exception as e:
                health['checks']['redis'] = f'error: {str(e)}'
                health['status'] = 'degraded'
        else:
            health['checks']['redis'] = 'not configured'
        
        # Check OpenAI GPT (Primary AI Service)
        if settings.OPENAI_API_KEY:
            health['checks']['openai_gpt'] = 'configured'
        else:
            health['checks']['openai_gpt'] = 'not configured'
        
        status_code = 200 if health['status'] == 'healthy' else 503
        return Response(health, status=status_code)


class AIStatusView(APIView):
    """GET /api/ai-status/ - Check if OpenAI GPT is working + return daily usage stats"""
    permission_classes = [AllowAny]

    def get(self, request):
        from core.openai_client import check_ai_health, get_ai_usage_stats
        result = check_ai_health()
        result['usage'] = get_ai_usage_stats()
        status_code = 200 if result['status'] == 'ok' else 503
        return Response(result, status=status_code)
