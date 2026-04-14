from django.urls import path
from interviews.views import (
    InterviewListCreateView, InterviewDetailView,
    EndInterviewView,
    SubmitResponseView, JoinRoomView, RecordViolationView, RescheduleInterviewView
)
from interviews.ai_views import (
    GenerateQuestionsView, CandidateHintsView,
    # Feature 1 — Emotion Proctoring
    EmotionProctoringView,
    # Feature 2 — Live Transcript Analysis
    LiveTranscriptAnalysisView,
    # Feature 3 — Adaptive Questions
    AdaptiveQuestionView,
    # Feature 5 — Slot Suggestions
    SuggestSlotsView,
    # Feature 8 — Question Bank
    QuestionBankListCreateView, QuestionBankDetailView, QuestionBankAIGenerateView,
    # Feature 9 — Live Question Suggester
    LiveQuestionSuggesterView,
)

urlpatterns = [
    path('', InterviewListCreateView.as_view(), name='interview-list-create'),

    # ── Static/prefixed routes MUST come before wildcard <str:interview_id> patterns ──
    path('generate-questions/', GenerateQuestionsView.as_view(), name='interview-generate-questions'),
    path('hints/', CandidateHintsView.as_view(), name='interview-candidate-hints'),
    path('suggest-slots/', SuggestSlotsView.as_view(), name='interview-suggest-slots'),   # Feature 5
    path('room/<str:room_id>/', JoinRoomView.as_view(), name='join-room'),

    # ── Question Bank routes (Feature 8) ──
    path('question-banks/', QuestionBankListCreateView.as_view(), name='qbank-list-create'),
    path('question-banks/ai-generate/', QuestionBankAIGenerateView.as_view(), name='qbank-ai-generate'),
    path('question-banks/<str:bank_id>/', QuestionBankDetailView.as_view(), name='qbank-detail'),

    # ── Wildcard interview_id routes LAST ──
    path('<str:interview_id>/', InterviewDetailView.as_view(), name='interview-detail'),
    path('<str:interview_id>/respond/', SubmitResponseView.as_view(), name='interview-respond'),
    path('<str:interview_id>/end/', EndInterviewView.as_view(), name='interview-end'),
    path('<str:interview_id>/reschedule/', RescheduleInterviewView.as_view(), name='interview-reschedule'),
    path('<str:interview_id>/violation/', RecordViolationView.as_view(), name='record-violation'),
    path('<str:interview_id>/proctoring-emotion/', EmotionProctoringView.as_view(), name='emotion-proctoring'),    # Feature 1
    path('<str:interview_id>/transcript-analysis/', LiveTranscriptAnalysisView.as_view(), name='live-transcript'),  # Feature 2
    path('<str:interview_id>/adaptive-question/', AdaptiveQuestionView.as_view(), name='adaptive-question'),        # Feature 3
    path('<str:interview_id>/suggest-questions/', LiveQuestionSuggesterView.as_view(), name='suggest-questions'),  # Feature 9
]
