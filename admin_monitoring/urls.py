"""
Admin Monitoring URL patterns
"""
from django.urls import path
from admin_monitoring import views

urlpatterns = [
    path('health/', views.SystemHealthView.as_view()),
    path('performance/', views.PerformanceDashboardView.as_view()),
    path('audit-logs/', views.AuditLogsView.as_view()),
    path('stats/', views.SystemStatsView.as_view()),
    path('security-alerts/', views.SecurityAlertsView.as_view()),
]
