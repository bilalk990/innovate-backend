from django.urls import path
from accounts.views import (
    RegisterView, LoginView, ProfileView,
    ChangePasswordView, UsersListView, UserDetailView,
    GoogleLoginView, AuditLogListView, BulkUserImportView,
    SystemSettingsView, ProfileImprovementsView,
    SalaryNegotiationView, CareerPathView,
    InterviewPrepPlanView, InterviewPrepQuizView, InterviewPrepReportView,
    CandidateComparisonView, BiasDetectorView, ReferenceCheckView,
    OfferPredictorView, FunnelAnalyzerView, TeamFitView, InterviewerCoachView,
    AnxietyCoachView, BulkResumeScreenerView, EmailCampaignView, SentimentTrackerView,
    CandidateDNAView, TalentRediscoveryView, InterviewQualityIntelligenceView,
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

    # AI Profile Improvement Suggestions
    path('profile-suggestions/', ProfileImprovementsView.as_view(), name='profile-suggestions'),

    # Category 1: New Candidate AI Features
    path('salary-negotiation/', SalaryNegotiationView.as_view(), name='salary-negotiation'),
    path('career-path/', CareerPathView.as_view(), name='career-path'),

    # Interview Prep Lab
    path('interview-prep/plan/', InterviewPrepPlanView.as_view(), name='interview-prep-plan'),
    path('interview-prep/quiz/', InterviewPrepQuizView.as_view(), name='interview-prep-quiz'),
    path('interview-prep/report/', InterviewPrepReportView.as_view(), name='interview-prep-report'),

    # HR AI Power Tools (Recruiter)
    path('hr/compare-candidates/', CandidateComparisonView.as_view(), name='hr-compare-candidates'),
    path('hr/bias-detector/', BiasDetectorView.as_view(), name='hr-bias-detector'),
    path('hr/reference-check/', ReferenceCheckView.as_view(), name='hr-reference-check'),
    path('hr/offer-predictor/', OfferPredictorView.as_view(), name='hr-offer-predictor'),
    path('hr/funnel-analyzer/', FunnelAnalyzerView.as_view(), name='hr-funnel-analyzer'),
    path('hr/team-fit/', TeamFitView.as_view(), name='hr-team-fit'),
    path('hr/interviewer-coach/', InterviewerCoachView.as_view(), name='hr-interviewer-coach'),

    # Feature Set 3
    path('anxiety-coach/', AnxietyCoachView.as_view(), name='anxiety-coach'),
    path('hr/bulk-resume-screen/', BulkResumeScreenerView.as_view(), name='hr-bulk-resume-screen'),
    path('hr/email-campaign/', EmailCampaignView.as_view(), name='hr-email-campaign'),
    path('hr/sentiment-tracker/', SentimentTrackerView.as_view(), name='hr-sentiment-tracker'),

    # Feature Set 4
    path('hr/candidate-dna/', CandidateDNAView.as_view(), name='hr-candidate-dna'),
    path('hr/talent-rediscovery/', TalentRediscoveryView.as_view(), name='hr-talent-rediscovery'),
    path('hr/interview-quality/', InterviewQualityIntelligenceView.as_view(), name='hr-interview-quality'),
]
