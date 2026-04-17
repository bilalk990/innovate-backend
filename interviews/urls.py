from django.urls import path
from interviews.views import (
    InterviewListCreateView, InterviewDetailView,
    EndInterviewView,
    SubmitResponseView, JoinRoomView, RecordViolationView, RescheduleInterviewView
)
from interviews.ai_views import (
    GenerateQuestionsView, CandidateHintsView,
    EmotionProctoringView,
    LiveTranscriptAnalysisView,
    AdaptiveQuestionView,
    SuggestSlotsView,
    QuestionBankListCreateView, QuestionBankDetailView, QuestionBankAIGenerateView,
    LiveQuestionSuggesterView,
    # New Features
    VoiceToneAnalysisView,
    LiveQualityMeterView,
    WhisperTranscribeView,
    InconsistencyDetectionView,
    RecruiterCoachView,
    JDAnalyzerView,
    DifficultyCalibrationView,
)

urlpatterns = [
    path('', InterviewListCreateView.as_view(), name='interview-list-create'),

    # ── Static/prefixed routes MUST come before wildcard <str:interview_id> patterns ──
    path('generate-questions/', GenerateQuestionsView.as_view(), name='interview-generate-questions'),
    path('hints/', CandidateHintsView.as_view(), name='interview-candidate-hints'),
    path('suggest-slots/', SuggestSlotsView.as_view(), name='interview-suggest-slots'),
    path('analyze-jd/', JDAnalyzerView.as_view(), name='analyze-jd'),
    path('calibrate-difficulty/', DifficultyCalibrationView.as_view(), name='calibrate-difficulty'),
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
    path('<str:interview_id>/proctoring-emotion/', EmotionProctoringView.as_view(), name='emotion-proctoring'),
    path('<str:interview_id>/transcript-analysis/', LiveTranscriptAnalysisView.as_view(), name='live-transcript'),
    path('<str:interview_id>/adaptive-question/', AdaptiveQuestionView.as_view(), name='adaptive-question'),
    path('<str:interview_id>/suggest-questions/', LiveQuestionSuggesterView.as_view(), name='suggest-questions'),
    # New Features
    path('<str:interview_id>/voice-tone/', VoiceToneAnalysisView.as_view(), name='voice-tone'),
    path('<str:interview_id>/live-quality/', LiveQualityMeterView.as_view(), name='live-quality'),
    path('<str:interview_id>/transcribe/', WhisperTranscribeView.as_view(), name='whisper-transcribe'),
    path('<str:interview_id>/inconsistency-check/', InconsistencyDetectionView.as_view(), name='inconsistency-check'),
    path('<str:interview_id>/recruiter-coach/', RecruiterCoachView.as_view(), name='recruiter-coach'),
]
