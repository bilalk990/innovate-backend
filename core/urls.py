"""
Core URL Configuration — InnovAIte Interview Guardian
"""
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core.health import HealthCheckView, AIStatusView

urlpatterns = [
    # Health check endpoint (no auth required)
    path('api/health/', HealthCheckView.as_view(), name='health-check'),
    path('api/ai-status/', AIStatusView.as_view(), name='ai-status'),
    
    # API v1 endpoints
    path('api/v1/auth/', include('accounts.urls')),
    path('api/v1/interviews/', include('interviews.urls')),
    path('api/v1/resumes/', include('resumes.urls')),
    path('api/v1/evaluations/', include('evaluations.urls')),
    path('api/v1/notifications/', include('notifications.urls')),
    path('api/v1/jobs/', include('jobs.urls')),
    path('api/v1/admin-monitoring/', include('admin_monitoring.urls')),
    
    # Legacy endpoints (backwards compatibility - remove in v2)
    path('api/auth/', include('accounts.urls')),
    path('api/interviews/', include('interviews.urls')),
    path('api/resumes/', include('resumes.urls')),
    path('api/evaluations/', include('evaluations.urls')),
    path('api/notifications/', include('notifications.urls')),
    path('api/jobs/', include('jobs.urls')),
    path('api/admin-monitoring/', include('admin_monitoring.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
