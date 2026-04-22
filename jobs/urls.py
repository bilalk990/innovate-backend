from django.urls import path
from .views import (
    JobListView, JobDetailView, ApplicationView, ApplicationDetailView, 
    JobApplicantsView, GapAnalysisView,
    # NEW AI FEATURES
    JobFitmentAnalysisView, AdvancedGapAnalysisView
)

urlpatterns = [
    path('', JobListView.as_view(), name='job-list'),
    path('applications/', ApplicationView.as_view(), name='applications'),
    path('applications/<str:app_id>/', ApplicationDetailView.as_view(), name='application-detail'),
    # NEW AI FEATURES - Must come before <job_id>/ wildcard
    path('fitment-analysis/', JobFitmentAnalysisView.as_view(), name='job-fitment-analysis'),
    path('advanced-gap-analysis/', AdvancedGapAnalysisView.as_view(), name='advanced-gap-analysis'),
    # Feature 4 — Gap Analysis must come before <job_id>/ wildcard
    path('<str:job_id>/gap-analysis/', GapAnalysisView.as_view(), name='job-gap-analysis'),
    path('<str:job_id>/applicants/', JobApplicantsView.as_view(), name='job-applicants'),
    path('<str:job_id>/', JobDetailView.as_view(), name='job-detail'),
]
