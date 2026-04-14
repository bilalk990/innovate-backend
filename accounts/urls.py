from django.urls import path
from accounts.views import (
    RegisterView, LoginView, ProfileView,
    ChangePasswordView, UsersListView, UserDetailView,
    GoogleLoginView, AuditLogListView, BulkUserImportView,
    SystemSettingsView
)
from accounts.google_views import GoogleAuthURLView, GoogleCallbackView
from accounts.mfa_views import (
    MFAStatusView, MFASetupView, MFAEnableView,
    MFADisableView, MFAVerifyView
)

urlpatterns = [
    # Authentication
    path('register/', RegisterView.as_view(), name='auth-register'),
    path('login/', LoginView.as_view(), name='auth-login'),
    path('profile/', ProfileView.as_view(), name='auth-profile'),
    path('change-password/', ChangePasswordView.as_view(), name='auth-change-password'),
    
    # MFA (Multi-Factor Authentication)
    path('mfa/status/', MFAStatusView.as_view(), name='mfa-status'),
    path('mfa/setup/', MFASetupView.as_view(), name='mfa-setup'),
    path('mfa/enable/', MFAEnableView.as_view(), name='mfa-enable'),
    path('mfa/disable/', MFADisableView.as_view(), name='mfa-disable'),
    path('mfa/verify/', MFAVerifyView.as_view(), name='mfa-verify'),
    
    # User Management
    path('users/', UsersListView.as_view(), name='users-list'),
    path('users/bulk-import/', BulkUserImportView.as_view(), name='users-bulk-import'),  # #59
    path('users/<str:user_id>/', UserDetailView.as_view(), name='user-detail'),
    
    # Google OAuth
    path('google/url/', GoogleAuthURLView.as_view(), name='google-auth-url'),
    path('google/callback/', GoogleCallbackView.as_view(), name='google-auth-callback'),
    path('google-login/', GoogleLoginView.as_view(), name='google-login'),
    
    # Audit Logs
    path('audit-logs/', AuditLogListView.as_view(), name='audit-logs'),

    # System Settings
    path('system-settings/', SystemSettingsView.as_view(), name='system-settings'),
]
