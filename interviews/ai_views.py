"""
interviews/ai_views.py — Candidate AI Assistance + Advanced AI Features
"""
import logging
import mongoengine
from datetime import datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from interviews.models import Interview, Question, QuestionBank
from core.openai_client import (
    generate_candidate_hints, generate_interview_questions,
    analyze_emotion_confidence, analyze_live_transcript_chunk,
    suggest_adaptive_question, suggest_interview_slots,
    generate_question_bank_suggestions,
    suggest_next_question,
)
from resumes.models import Resume

logger = logging.getLogger('innovaite')


# ── Centralized AI error handler ──
AI_ERROR_MESSAGES = {
    'AI_QUOTA_EXHAUSTED': ('OpenAI API quota exhausted. Please add more credits to your account.', 503),
    'AI_KEY_INVALID': ('AI service is not configured correctly. Please contact the administrator.', 503),
    'AI_BILLING_ISSUE': ('AI service billing issue. Please contact the administrator.', 503),
}

# Alert types that should notify admins (quota exhausted is handled inside _call itself,
# but KEY_INVALID / BILLING may surface here from views that don't go through _call)
_ADMIN_ALERT_TYPES = {'AI_KEY_INVALID', 'AI_BILLING_ISSUE'}

def handle_ai_error(e: Exception):
    """Return a proper Response for AI errors, and notify admins for critical ones."""
    error_type = str(e)
    if error_type in AI_ERROR_MESSAGES:
        msg, status = AI_ERROR_MESSAGES[error_type]
        # Notify admins for key/billing issues (quota exhausted is already handled in _call)
        if error_type in _ADMIN_ALERT_TYPES:
            try:
                from core.ai_notifications import notify_admins_async
                notify_admins_async(error_type)
            except Exception as notify_err:
                logger.warning(f'[AI] Could not send admin notification: {notify_err}')
        return Response({'error': msg, 'error_type': error_type}, status=status)
    if 'rate limit' in error_type.lower():
        return Response({'error': error_type, 'error_type': 'RATE_LIMITED'}, status=429)
    return Response({'error': 'AI service temporarily unavailable. Please try again.', 'error_type': 'AI_ERROR'}, status=503)


class GenerateQuestionsView(APIView):
    """HR generates questions via OpenAI GPT"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        job_title = request.data.get('job_title', '').strip()
        job_desc = request.data.get('job_description', '').strip()
        num = min(int(request.data.get('num_questions', 8)), 20)
        candidate_id = request.data.get('candidate_id')

        resume_data = None
        if candidate_id:
            res = Resume.objects(candidate_id=candidate_id, is_active=True).first()
            if res:
                resume_data = res.parsed_data

        try:
            # CRITICAL FIX: Pass user_id for rate limiting
            questions = generate_interview_questions(
                job_title=job_title, 
                job_description=job_desc, 
                num_questions=num,
                resume_data=resume_data,
                user_id=str(request.user.id)
            )
            return Response({'questions': questions, 'count': len(questions)})
        except Exception as e:
            return handle_ai_error(e)


class CandidateHintsView(APIView):
    """
    POST /api/interviews/hints/
    Candidate requests strategic AI tips for a specific question.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        question_text = request.data.get('question_text', '').strip()
        category = request.data.get('category', 'general')

        if not question_text:
            return Response({'error': 'question_text is required.'}, status=400)

        try:
            hints = generate_candidate_hints(question_text, category, user_id=str(request.user.id))
            return Response({'hints': hints})
        except Exception as e:
            return handle_ai_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Real-Time Emotion & Confidence Proctoring
# ─────────────────────────────────────────────────────────────────────────────

class EmotionProctoringView(APIView):
    """
    POST /api/interviews/<id>/proctoring-emotion/
    Candidate/system sends batched face snapshots; AI returns proctoring intelligence.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        # Only candidate or recruiter of this interview may post
        uid = str(request.user.id)
        if uid not in [interview.candidate_id, interview.recruiter_id] and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        face_snapshots = request.data.get('snapshots', [])
        if not isinstance(face_snapshots, list):
            return Response({'error': 'snapshots must be a list.'}, status=400)

        result = analyze_emotion_confidence(face_snapshots)
        return Response({'proctoring': result})


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Live Transcript Semantic Analysis
# ─────────────────────────────────────────────────────────────────────────────

class LiveTranscriptAnalysisView(APIView):
    """
    POST /api/interviews/<id>/transcript-analysis/
    Recruiter sends a live transcript chunk for real-time semantic coaching.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        uid = str(request.user.id)
        if uid not in [interview.recruiter_id, interview.candidate_id] and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        transcript = request.data.get('transcript', '').strip()
        question_index = request.data.get('question_index', 0)

        if not transcript:
            return Response({'error': 'transcript is required.'}, status=400)

        # Get the question text if available
        question_text = ''
        if 0 <= question_index < len(interview.questions):
            question_text = interview.questions[question_index].text

        try:
            analysis = analyze_live_transcript_chunk(
                transcript=transcript,
                question=question_text,
                job_title=interview.job_title,
                user_id=str(request.user.id),
            )
            return Response({'analysis': analysis})
        except Exception as e:
            return handle_ai_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Adaptive Question Engine
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveQuestionView(APIView):
    """
    POST /api/interviews/<id>/adaptive-question/
    Based on candidate's latest response, AI suggests next adaptive question.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        if interview.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Only recruiters can use adaptive questions.'}, status=403)

        current_question = request.data.get('current_question', '').strip()
        candidate_response = request.data.get('candidate_response', '').strip()
        current_difficulty = request.data.get('current_difficulty', 'medium')
        category = request.data.get('category', 'technical')

        if not current_question or not candidate_response:
            return Response({'error': 'current_question and candidate_response are required.'}, status=400)

        try:
            suggestion = suggest_adaptive_question(
                current_question=current_question,
                candidate_response=candidate_response,
                current_difficulty=current_difficulty,
                job_title=interview.job_title,
                category=category,
                user_id=str(request.user.id),
            )
            return Response({'adaptive': suggestion})
        except Exception as e:
            return handle_ai_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Smart Scheduling Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class SuggestSlotsView(APIView):
    """
    POST /api/interviews/suggest-slots/
    AI suggests optimal interview time slots based on context.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can use slot suggestions.'}, status=403)

        job_title = request.data.get('job_title', '').strip()
        duration = int(request.data.get('duration_minutes', 45))
        recruiter_tz = request.data.get('recruiter_timezone', 'UTC')
        candidate_tz = request.data.get('candidate_timezone', 'UTC')
        preferred_days = request.data.get('preferred_days', [])

        # Get recruiter's existing interviews this week for context
        existing = Interview.objects(
            recruiter_id=str(request.user.id),
            status__in=['scheduled', 'pending']
        ).count()

        try:
            result = suggest_interview_slots(
                job_title=job_title or 'General Role',
                duration_minutes=duration,
                recruiter_timezone=recruiter_tz,
                candidate_timezone=candidate_tz,
                existing_interviews=[{}] * existing,
                preferred_days=preferred_days or None,
            )
            return Response(result)
        except Exception as e:
            return handle_ai_error(e)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8 — Question Bank CRUD
# ─────────────────────────────────────────────────────────────────────────────

class QuestionBankListCreateView(APIView):
    """
    GET  /api/interviews/question-banks/   — list recruiter's banks
    POST /api/interviews/question-banks/   — create new bank
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        banks = QuestionBank.objects(recruiter_id=str(request.user.id))
        return Response([b.to_dict() for b in banks])

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        name = request.data.get('name', '').strip()
        if not name:
            return Response({'error': 'name is required.'}, status=400)

        job_title = request.data.get('job_title', '').strip()
        description = request.data.get('description', '').strip()
        raw_questions = request.data.get('questions', [])
        is_public = bool(request.data.get('is_public', False))

        questions = []
        for q in raw_questions:
            if q.get('text'):
                questions.append(Question(
                    text=q['text'],
                    category=q.get('category', 'general'),
                    expected_keywords=q.get('expected_keywords', []),
                    ideal_answer=q.get('ideal_answer', ''),
                    difficulty=q.get('difficulty', 'medium'),
                    time_estimate_minutes=int(q.get('time_estimate_minutes', 3)),
                    tags=q.get('tags', []),
                ))

        bank = QuestionBank(
            name=name,
            recruiter_id=str(request.user.id),
            job_title=job_title,
            description=description,
            questions=questions,
            is_public=is_public,
        )
        bank.save()
        return Response(bank.to_dict(), status=201)


class QuestionBankDetailView(APIView):
    """
    GET    /api/interviews/question-banks/<bank_id>/  — retrieve
    PATCH  /api/interviews/question-banks/<bank_id>/  — update
    DELETE /api/interviews/question-banks/<bank_id>/  — delete
    """
    permission_classes = [IsAuthenticated]

    def _get_bank(self, bank_id, user):
        try:
            bank = QuestionBank.objects.get(id=bank_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return None, Response({'error': 'Question bank not found.'}, status=404)

        if bank.recruiter_id != str(user.id) and user.role != 'admin':
            return None, Response({'error': 'Forbidden.'}, status=403)

        return bank, None

    def get(self, request, bank_id):
        bank, err = self._get_bank(bank_id, request.user)
        if err:
            return err
        return Response(bank.to_dict())

    def patch(self, request, bank_id):
        bank, err = self._get_bank(bank_id, request.user)
        if err:
            return err

        for field in ['name', 'job_title', 'description']:
            if field in request.data:
                setattr(bank, field, request.data[field])

        if 'is_public' in request.data:
            bank.is_public = bool(request.data['is_public'])

        if 'questions' in request.data:
            questions = []
            for q in request.data['questions']:
                if q.get('text'):
                    questions.append(Question(
                        text=q['text'],
                        category=q.get('category', 'general'),
                        expected_keywords=q.get('expected_keywords', []),
                        ideal_answer=q.get('ideal_answer', ''),
                        difficulty=q.get('difficulty', 'medium'),
                        time_estimate_minutes=int(q.get('time_estimate_minutes', 3)),
                        tags=q.get('tags', []),
                    ))
            bank.questions = questions

        bank.updated_at = datetime.utcnow()
        bank.save()
        return Response(bank.to_dict())

    def delete(self, request, bank_id):
        bank, err = self._get_bank(bank_id, request.user)
        if err:
            return err
        bank.delete()
        return Response({'deleted': True})


class QuestionBankAIGenerateView(APIView):
    """
    POST /api/interviews/question-banks/ai-generate/
    AI generates a full question bank for a given role.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        job_title = request.data.get('job_title', '').strip()
        job_description = request.data.get('job_description', '').strip()
        count = min(int(request.data.get('count', 15)), 30)
        categories = request.data.get('categories', [])

        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)

        try:
            questions = generate_question_bank_suggestions(
                job_title=job_title,
                job_description=job_description,
                categories=categories or None,
                count=count,
                user_id=str(request.user.id),
            )
            return Response({'questions': questions, 'count': len(questions)})
        except Exception as e:
            logger.error(f'[QuestionBank] AI generation failed: {e}')
            return Response({'error': 'AI generation failed.'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 9 — Live Question Recommender
# ─────────────────────────────────────────────────────────────────────────────

class LiveQuestionSuggesterView(APIView):
    """
    POST /api/interviews/<interview_id>/suggest-questions/
    Recruiter sends live transcript → AI suggests next best questions.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        transcript = request.data.get('transcript', '').strip()
        current_question = request.data.get('current_question', '').strip()
        questions_asked = request.data.get('questions_asked', [])

        if not transcript:
            return Response({'error': 'transcript is required.'}, status=400)

        try:
            result = suggest_next_question(
                transcript=transcript,
                job_title=interview.title or '',
                current_question=current_question,
                questions_asked=questions_asked,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            return handle_ai_error(e)
