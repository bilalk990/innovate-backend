from django.urls import path
from resumes.views import (
    ResumeUploadView, ResumeListView, ResumeDetailView, 
    GenerateResumeView,
    # NEW AI FEATURE
    GenerateAdvancedResumeView
)

urlpatterns = [
    path('', ResumeListView.as_view(), name='resume-list'),
    path('upload/', ResumeUploadView.as_view(), name='resume-upload'),
    path('generate/', GenerateResumeView.as_view(), name='resume-generate'),
    path('generate-advanced/', GenerateAdvancedResumeView.as_view(), name='resume-generate-advanced'),
    path('<str:resume_id>/', ResumeDetailView.as_view(), name='resume-detail'),
]
