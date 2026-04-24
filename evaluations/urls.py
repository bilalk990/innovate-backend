from django.urls import path
from evaluations.views import (
    TriggerEvaluationView, EvaluationListView,
    EvaluationDetailView, OfferLetterView, ExportEvaluationPDFView,
    EvaluationShareView, ExportEvaluationsCSVView,
    CandidateRankingView,
    InterviewDebriefView,
    HireProbabilityView,
    FollowUpEmailView,
    # NEW AI FEATURES
    BehavioralTraitsView,
    IntegrityCheckView,
    CultureFitView,
    ExecutiveSummaryView,
    PredictHireView,
    ReadinessScoreView,
)

urlpatterns = [
    path('', EvaluationListView.as_view(), name='evaluation-list'),
    path('trigger/', TriggerEvaluationView.as_view(), name='evaluation-trigger'),
    path('offer/', OfferLetterView.as_view(), name='evaluation-offer'),
    path('export/', ExportEvaluationsCSVView.as_view(), name='evaluation-export-csv'),
    path('rank/', CandidateRankingView.as_view(), name='evaluation-rank'),              # Feature 6
    # NEW AI FEATURES - Must come before <eval_id> wildcard
    path('readiness/', ReadinessScoreView.as_view(), name='readiness-score'),
    path('behavioral-traits/', BehavioralTraitsView.as_view(), name='behavioral-traits'),
    path('check-integrity/', IntegrityCheckView.as_view(), name='check-integrity'),
    path('culture-fit/', CultureFitView.as_view(), name='culture-fit'),
    path('executive-summary/', ExecutiveSummaryView.as_view(), name='executive-summary'),
    path('<str:eval_id>/', EvaluationDetailView.as_view(), name='evaluation-detail'),
    path('<str:eval_id>/export-pdf/', ExportEvaluationPDFView.as_view(), name='evaluation-export-pdf'),
    path('<str:eval_id>/share/', EvaluationShareView.as_view(), name='evaluation-share'),
    path('<str:eval_id>/debrief/', InterviewDebriefView.as_view(), name='evaluation-debrief'),
    path('<str:eval_id>/hire-probability/', HireProbabilityView.as_view(), name='hire-probability'),
    path('<str:eval_id>/predict-hire/', PredictHireView.as_view(), name='predict-hire'),
    path('<str:eval_id>/followup-email/', FollowUpEmailView.as_view(), name='followup-email'),
]
