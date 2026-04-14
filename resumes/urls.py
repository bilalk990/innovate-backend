from django.urls import path
from resumes.views import ResumeUploadView, ResumeListView, ResumeDetailView, GenerateResumeView

urlpatterns = [
    path('', ResumeListView.as_view(), name='resume-list'),
    path('upload/', ResumeUploadView.as_view(), name='resume-upload'),
    path('generate/', GenerateResumeView.as_view(), name='resume-generate'),
    path('<str:resume_id>/', ResumeDetailView.as_view(), name='resume-detail'),
]
