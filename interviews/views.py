"""
Interviews app views — CRUD for scheduling, managing, and running interviews
"""
import uuid
import secrets
import logging
import threading
import mongoengine
import bleach
from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from interviews.models import Interview, Question
from accounts.models import User
from jobs.models import Application
from core.google_service import create_google_meet_link
from core.conflict_detector import check_scheduling_conflicts, validate_interview_time
from core.email_service import send_interview_scheduled_email, send_recruiter_notification_email
from core.sms_service import send_interview_scheduled_sms
from core.audit_logger import log_interview_created, log_interview_joined
from notifications.models import Notification

logger = logging.getLogger('innovaite')

# Check if AI is available (same guard as engine.py)
try:
    from core.openai_client import analyze_response_semantics as _test_import  # noqa
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False


# XSS sanitization configuration
ALLOWED_TAGS = []  # No HTML tags allowed in responses
ALLOWED_ATTRIBUTES = {}


def sanitize_response(text):
    """Sanitize user input to prevent XSS attacks"""
    if not text:
        return text
    return bleach.clean(text, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, strip=True)


class InterviewListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role == 'recruiter':
            interviews = Interview.objects(recruiter_id=str(user.id))
        elif user.role == 'candidate':
            interviews = Interview.objects(candidate_id=str(user.id))
        else:  # admin
            interviews = Interview.objects.all()

        status_filter = request.query_params.get('status')
        if status_filter:
            interviews = interviews.filter(status=status_filter)

        # #65 — pagination to avoid loading all interviews
        try:
            limit = min(int(request.query_params.get('limit', 50)), 200)
            offset = int(request.query_params.get('offset', 0))
        except (ValueError, TypeError):
            limit, offset = 50, 0

        total = interviews.count()
        interviews = interviews.skip(offset).limit(limit)
        return Response({
            'results': [i.to_dict() for i in interviews],
            'total': total,
            'limit': limit,
            'offset': offset,
        })

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can create interviews.'}, status=403)

        data = request.data
        required = ['title', 'scheduled_at']
        for f in required:
            if not data.get(f):
                return Response({'error': f'{f} is required.'}, status=400)

        # ─── ROBUST ID VALIDATION ───
        # Ensure IDs are valid hex strings before querying to prevent 500 crashes
        import bson
        target_candidate_id = data.get('candidate_id')
        if target_candidate_id and not bson.ObjectId.is_valid(target_candidate_id):
            return Response({'error': 'Invalid Candidate ID format.'}, status=400)
            
        target_job_id = data.get('job_id')
        if target_job_id and not bson.ObjectId.is_valid(target_job_id):
            return Response({'error': 'Invalid Job ID format.'}, status=400)

        try:
            # ─── TIME & CONFLICT VALIDATION ───
            try:
                scheduled_at = datetime.fromisoformat(data['scheduled_at'].replace('Z', '+00:00'))
            except ValueError:
                return Response({'error': 'Invalid scheduled_at format. Use ISO 8601.'}, status=400)

            duration_minutes = data.get('duration_minutes', 45)
            try:
                duration_minutes = int(duration_minutes)
                if duration_minutes < 15 or duration_minutes > 180:
                    return Response({'error': 'Duration must be between 15 and 180 minutes.'}, status=400)
            except (ValueError, TypeError):
                return Response({'error': 'Duration must be a valid number.'}, status=400)

            # Validate interview timing & get NORMALISED aware datetime
            time_validation = validate_interview_time(scheduled_at, duration_minutes)
            if not time_validation['valid']:
                return Response({'error': time_validation['message']}, status=400)
            
            # Use the NORMALISED UTC aware datetime for all subsequent steps
            scheduled_at = time_validation['scheduled_at']

            # Validate candidate exists
            candidate_id = data.get('candidate_id')
            if candidate_id:
                try:
                    User.objects.get(id=candidate_id, role='candidate')
                except (mongoengine.DoesNotExist, mongoengine.ValidationError):
                    return Response({'error': 'Candidate not found.'}, status=400)

            # Check for scheduling conflicts
            conflict_check = check_scheduling_conflicts(
                recruiter_id=str(request.user.id),
                candidate_id=candidate_id,
                scheduled_at=scheduled_at,
                duration_minutes=duration_minutes
            )
            
            if conflict_check['has_conflict']:
                return Response({
                    'error': conflict_check['message'],
                    'conflicts': conflict_check['conflicts']
                }, status=409)

            # ─── OBJECT CREATION ───
            questions = []
            for q in data.get('questions', []):
                questions.append(Question(
                    text=q.get('text', ''),
                    category=q.get('category', 'general'),
                    expected_keywords=q.get('expected_keywords', []),
                    ideal_answer=q.get('ideal_answer', ''),
                    difficulty=q.get('difficulty', 'medium')
                ))

            room_token = secrets.token_urlsafe(32)
            token_expires_at = scheduled_at + timedelta(minutes=duration_minutes + 60)

            interview = Interview(
                title=data['title'],
                recruiter_id=str(request.user.id),
                candidate_id=candidate_id,
                room_id=str(uuid.uuid4()).replace('-', '')[:16],
                room_token=room_token,
                token_expires_at=token_expires_at,
                scheduled_at=scheduled_at,
                duration_minutes=duration_minutes,
                job_id=data.get('job_id'),
                job_title=data.get('job_title', ''),
                job_description=data.get('job_description', ''),
                questions=questions,
                notes=data.get('notes', ''),
            )

            # Automated Google Meet link generation
            if data.get('generate_meet_link'):
                user_doc = User.objects(id=request.user.id).first()
                if user_doc and user_doc.google_tokens:
                    google_tokens = user_doc._get_google_tokens()
                    if google_tokens:
                        meet_link = create_google_meet_link(
                            user_tokens=google_tokens,
                            interview_title=interview.title,
                            start_time=scheduled_at.isoformat(),
                            duration_minutes=interview.duration_minutes
                        )
                        if meet_link:
                            interview.meet_link = meet_link
                            from core.google_service import get_refreshed_tokens
                            refreshed = get_refreshed_tokens(google_tokens)
                            if refreshed:
                                user_doc.set_google_tokens(refreshed)
                                user_doc.save()

            interview.save()
            log_interview_created(request.user, str(interview.id), request)

            # Auto-sync Application status
            if candidate_id:
                try:
                    job_id = data.get('job_id')
                    query = {'candidate_id': candidate_id, 'status__in': ['pending', 'reviewed']}
                    if job_id:
                        query['job_id'] = job_id
                    else:
                        query['recruiter_id'] = str(request.user.id)
                    
                    matching_app = Application.objects(**query).order_by('-applied_at').first()
                    if matching_app:
                        matching_app.status = 'interview_scheduled'
                        matching_app.save()
                except Exception as sync_err:
                    logger.warning(f'[Interview] Application status sync failed: {sync_err}')

            # Send notifications ASYNC so HTTP response is instant (email was causing 30s+ timeouts)
            if candidate_id:
                def _notify_candidate(interview_dict, cand_id, recruiter_id):
                    try:
                        candidate = User.objects.get(id=cand_id)
                        send_interview_scheduled_email(
                            candidate_email=candidate.email,
                            candidate_name=candidate.name,
                            interview_data=interview_dict
                        )
                        if candidate.phone:
                            send_interview_scheduled_sms(
                                candidate_phone=candidate.phone,
                                interview_data=interview_dict
                            )
                        Notification(
                            recipient_id=cand_id,
                            sender_id=recruiter_id,
                            notification_type='interview_scheduled',
                            title='Interview Scheduled',
                            message=f'You have been scheduled for an interview: {interview_dict.get("title")}',
                            link=f'/interview/room/{interview_dict.get("room_id")}'
                        ).save()
                    except Exception as e:
                        logger.warning(f'[Notification] Failed to notify candidate: {str(e)}')

                threading.Thread(
                    target=_notify_candidate,
                    args=(interview.to_dict(), candidate_id, str(request.user.id)),
                    daemon=True
                ).start()

            result = interview.to_dict()
            result['notification_sent'] = bool(candidate_id)
            return Response(result, status=201)

        except Exception as global_err:
            logger.error(f'[Interview] Critical failure during creation: {str(global_err)}')
            return Response({'error': 'A critical error occurred while scheduling. Please check all fields and try again.'}, status=500)


class InterviewDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def _get_interview(self, interview_id, user):
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return None, Response({'error': 'Interview not found.'}, status=404)
        # Access control
        if user.role == 'recruiter' and interview.recruiter_id != str(user.id):
            return None, Response({'error': 'Forbidden.'}, status=403)
        if user.role == 'candidate' and interview.candidate_id != str(user.id):
            return None, Response({'error': 'Forbidden.'}, status=403)
        return interview, None

    def get(self, request, interview_id):
        interview, err = self._get_interview(interview_id, request.user)
        if err:
            return err
        return Response(interview.to_dict())

    def patch(self, request, interview_id):
        interview, err = self._get_interview(interview_id, request.user)
        if err:
            return err
        if request.user.role == 'candidate':
            return Response({'error': 'Forbidden.'}, status=403)

        data = request.data
        old_scheduled_at = interview.scheduled_at
        
        # Check for conflicts if scheduled_at is being changed
        if 'scheduled_at' in data:
            try:
                new_scheduled_at = datetime.fromisoformat(data['scheduled_at'].replace('Z', '+00:00'))
            except ValueError:
                return Response({'error': 'Invalid scheduled_at format.'}, status=400)
            
            # Validate new time
            duration = data.get('duration_minutes', interview.duration_minutes)
            time_validation = validate_interview_time(new_scheduled_at, duration)
            if not time_validation['valid']:
                return Response({'error': time_validation['message']}, status=400)
            
            # Check conflicts (exclude current interview)
            conflict_check = check_scheduling_conflicts(
                recruiter_id=interview.recruiter_id,
                candidate_id=data.get('candidate_id', interview.candidate_id),
                scheduled_at=new_scheduled_at,
                duration_minutes=duration,
                exclude_interview_id=str(interview.id)
            )
            if conflict_check['has_conflict']:
                return Response({
                    'error': conflict_check['message'],
                    'conflicts': conflict_check['conflicts']
                }, status=409)
            
            interview.scheduled_at = new_scheduled_at
        
        editable = ['title', 'candidate_id', 'status', 'notes', 'duration_minutes', 'job_id', 'job_title', 'job_description', 'meet_link']
        for field in editable:
            if field in data and field != 'scheduled_at':  # Already handled above
                setattr(interview, field, data[field])
        interview.updated_at = datetime.utcnow()
        interview.save()

        # Notify candidate if schedule changed
        if 'scheduled_at' in data and interview.candidate_id and interview.scheduled_at != old_scheduled_at:
            try:
                from core.email_service import send_interview_scheduled_email
                candidate = User.objects.get(id=interview.candidate_id)
                send_interview_scheduled_email(
                    candidate_email=candidate.email,
                    candidate_name=candidate.name,
                    interview_data=interview.to_dict()
                )
                from notifications.models import Notification
                Notification(
                    recipient_id=interview.candidate_id,
                    sender_id=str(request.user.id),
                    notification_type='interview_scheduled',
                    title='Interview Rescheduled',
                    message=f'Your interview "{interview.title}" has been rescheduled.',
                    link=f'/interview/room/{interview.room_id}'
                ).save()
            except Exception as e:
                logger.warning(f'[Notification] Reschedule notify failed: {e}')

        return Response(interview.to_dict())

    def delete(self, request, interview_id):
        interview, err = self._get_interview(interview_id, request.user)
        if err:
            return err
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)
        interview.delete()
        return Response({'message': 'Interview deleted.'})


class EndInterviewView(APIView):
    """POST /api/interviews/<id>/end/ — Recruiter/Admin ends the interview"""
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except Exception:
            return Response({'error': 'Interview not found.'}, status=404)

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can end interviews.'}, status=403)

        if interview.status == 'completed':
            return Response({'message': 'Interview already completed.', 'status': 'completed'})

        interview.status = 'completed'
        interview.updated_at = datetime.utcnow()
        interview.save()

        logger.info(f'[Interview] Ended by {request.user.role}: {interview_id}')
        return Response({'message': 'Interview ended.', 'status': 'completed'})


class SubmitResponseView(APIView):
    """Candidate submits their response to a question during interview"""
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can submit responses.'}, status=403)
        try:
            interview = Interview.objects.get(id=interview_id, candidate_id=str(request.user.id))
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        question_index = str(request.data.get('question_index', '0'))
        response_text = request.data.get('response', '').strip()
        if not response_text:
            return Response({'error': 'response is required.'}, status=400)

        # Validate question index — allow free-form index 0 when no questions defined
        try:
            q_idx = int(question_index)
            if q_idx < 0:
                return Response({'error': 'question_index must be >= 0.'}, status=400)
            # Only enforce upper bound if the interview has structured questions
            if len(interview.questions) > 0 and q_idx >= len(interview.questions):
                return Response({'error': 'Invalid question_index.'}, status=400)
        except (ValueError, TypeError):
            return Response({'error': 'question_index must be a number.'}, status=400)

        # Validate response length
        if len(response_text) < 5:
            return Response({'error': 'Response is too short.'}, status=400)
        if len(response_text) > 5000:
            return Response({'error': 'Response exceeds maximum length of 5000 characters.'}, status=400)

        # Sanitize response to prevent XSS attacks
        response_text = sanitize_response(response_text)

        responses = dict(interview.candidate_responses or {})
        responses[question_index] = response_text
        interview.candidate_responses = responses
        
        # Save full transcript if provided (usually on final submission)
        full_transcript = request.data.get('full_transcript')
        if full_transcript:
            interview.full_transcript = full_transcript

        interview.updated_at = datetime.utcnow()
        interview.save()

        # ── Real-time semantic pre-scoring (Solution 1: evaluate DURING interview) ──
        # Kick off AI semantic analysis for THIS answer in a background thread.
        # By the time the interview ends, all questions are already scored → engine skips AI calls.
        if AI_AVAILABLE and q_idx < len(interview.questions):
            question_obj = interview.questions[q_idx]
            if getattr(question_obj, 'ideal_answer', ''):
                def _precompute_semantic(iview_id, q_index, q_text, ideal_ans, resp_text, uid):
                    try:
                        from core.openai_client import analyze_response_semantics
                        from interviews.models import Interview as _Interview
                        ai_res = analyze_response_semantics(q_text, ideal_ans, resp_text, user_id=uid)
                        # Reload + update atomically
                        fresh = _Interview.objects.get(id=iview_id)
                        scores = dict(fresh.semantic_scores or {})
                        scores[str(q_index)] = {
                            'score': ai_res.get('score', 5.0),
                            'explanation': ai_res.get('explanation', ''),
                            'missing_points': ai_res.get('missing_points', []),
                            'computed_at': datetime.utcnow().isoformat(),
                        }
                        fresh.semantic_scores = scores
                        fresh.save()
                        logger.info(f'[SemanticPre] Q{q_index} scored {ai_res.get("score")} for interview {iview_id}')
                    except Exception as sem_err:
                        logger.warning(f'[SemanticPre] Failed for Q{q_index} interview {iview_id}: {sem_err}')

                _user_id = getattr(request.user, 'id', None)
                threading.Thread(
                    target=_precompute_semantic,
                    args=(str(interview.id), q_idx, question_obj.text,
                          question_obj.ideal_answer, response_text, str(_user_id)),
                    daemon=True
                ).start()

        # --- Auto-trigger XAI evaluation when ALL questions are answered ---
        all_answered = (
            len(interview.questions) > 0 and
            len(responses) >= len(interview.questions) and
            all(responses.get(str(i), '').strip() for i in range(len(interview.questions)))
        )
        evaluation_triggered = False
        if all_answered:
            # CRITICAL FIX: Set status atomically BEFORE spawning thread to prevent race condition
            interview.status = 'evaluating'
            interview.save()
            
            def _run_evaluation():
                """Run evaluation in background thread so API responds instantly."""
                try:
                    from evaluations.engine import run_xai_evaluation
                    from evaluations.models import Evaluation, CriterionResult
                    from resumes.models import Resume
                    from core.email_service import send_evaluation_ready_email

                    # Reload interview with fresh data
                    fresh_interview = Interview.objects.get(id=interview.id)

                    # Prevent duplicate evaluations
                    if Evaluation.objects(interview_id=str(fresh_interview.id)).first():
                        return

                    # Get candidate's resume
                    resume = Resume.objects(
                        candidate_id=fresh_interview.candidate_id,
                        is_active=True
                    ).order_by('-uploaded_at').first()
                    resume_data = resume.parsed_data if resume else {}

                    # Run the XAI engine
                    result = run_xai_evaluation(fresh_interview, resume_data)

                    # Build criterion result documents
                    criterion_docs = [
                        CriterionResult(
                            criterion=cr['criterion'],
                            score=cr['score'],
                            max_score=cr['max_score'],
                            weight=cr['weight'],
                            explanation=cr['explanation'],
                            rules_applied=cr['rules_applied'],
                            evidence=cr['evidence'],
                        )
                        for cr in result['criterion_results']
                    ]

                    evaluation = Evaluation(
                        interview_id=str(fresh_interview.id),
                        candidate_id=fresh_interview.candidate_id,
                        recruiter_id=fresh_interview.recruiter_id,
                        criterion_results=criterion_docs,
                        overall_score=result['overall_score'],
                        recommendation=result['recommendation'],
                        summary=result['summary'],
                        strengths=result.get('strengths', []),
                        weaknesses=result.get('weaknesses', []),
                        resume_alignment_score=result.get('resume_alignment_score', 0),
                        confidence_score=result.get('confidence_score', 50),
                        fluency_score=result.get('fluency_score', 50),
                        behavioral_summary=result.get('behavioral_summary', ''),
                        proctoring_score=result.get('proctoring_score', 100),
                        integrity_notes=result.get('integrity_notes', ''),
                        tab_switch_count=fresh_interview.tab_switch_count or 0,
                        culture_fit_score=result.get('culture_fit_score', 0),
                        ai_summary_used=result.get('ai_summary_used', False),
                        status='complete',
                    )
                    evaluation.save()

                    # Link evaluation back to interview so frontend can navigate directly
                    fresh_interview.status = 'completed'
                    fresh_interview.evaluation_id = str(evaluation.id)
                    fresh_interview.save()

                    # Notify recruiter (in-app)
                    Notification(
                        recipient_id=fresh_interview.recruiter_id,
                        sender_id='system',
                        notification_type='evaluation_ready',
                        title='⚡ AI Evaluation Ready',
                        message=f'XAI Report for "{fresh_interview.title}" is ready to review.',
                        link=f'/recruiter/evaluations/{str(evaluation.id)}'
                    ).save()

                    # Notify candidate (in-app)
                    Notification(
                        recipient_id=fresh_interview.candidate_id,
                        sender_id='system',
                        notification_type='evaluation_ready',
                        title='Your Interview is Complete',
                        message='Thank you! Your interview has been submitted and is being reviewed.',
                        link=f'/candidate/evaluations/{str(evaluation.id)}'
                    ).save()

                    logger.info(f'[AutoEval] Evaluation {evaluation.id} created for interview {fresh_interview.id}')

                except Exception as eval_err:
                    logger.error(f'[AutoEval] Background evaluation failed for interview {interview.id}: {eval_err}')

            thread = threading.Thread(target=_run_evaluation, daemon=True)
            thread.start()
            evaluation_triggered = True
            logger.info(f'[AutoEval] All {len(interview.questions)} answers received — evaluation thread started.')
        # -------------------------------------------------------------------

        return Response({
            'message': 'Response saved.',
            'responses': interview.candidate_responses,
            'all_answered': all_answered,
            'evaluation_triggered': evaluation_triggered,
        })


class JoinRoomView(APIView):
    """Get room details for a WebRTC session"""
    permission_classes = [IsAuthenticated]

    def get(self, request, room_id):
        try:
            interview = Interview.objects.get(room_id=room_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Room not found.'}, status=404)
        
        user = request.user
        allowed = [interview.recruiter_id, interview.candidate_id]
        if user.role != 'admin' and str(user.id) not in allowed:
            return Response({'error': 'Forbidden.'}, status=403)
        
        # Validate token expiry (CRITICAL FIX: >= instead of > to prevent off-by-one bypass)
        now = datetime.utcnow()
        if interview.token_expires_at and now >= interview.token_expires_at:
            return Response({'error': 'Interview link has expired.'}, status=410)

        # Auto-set status to active when someone joins
        if interview.status == 'scheduled':
            interview.status = 'active'
            interview.save()

        # Audit log
        log_interview_joined(user, str(interview.id), request)
        
        return Response(interview.to_dict())

class RecordViolationView(APIView):
    """Log a proctoring violation (e.g. tab switch) during interview.
    Allowed callers: candidate (self), or the recruiter who owns the interview.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found'}, status=404)

        uid = str(request.user.id)
        role = request.user.role
        # Candidate can record their own violations; recruiter can record for their interview
        if role == 'candidate' and interview.candidate_id != uid:
            return Response({'error': 'Forbidden.'}, status=403)
        if role == 'recruiter' and interview.recruiter_id != uid:
            return Response({'error': 'Forbidden.'}, status=403)
        if role not in ('candidate', 'recruiter', 'admin'):
            return Response({'error': 'Forbidden.'}, status=403)

        interview.tab_switch_count = (interview.tab_switch_count or 0) + 1
        interview.save()
        return Response({'success': True, 'tab_switch_count': interview.tab_switch_count})


class RescheduleInterviewView(APIView):
    """POST /api/interviews/<id>/reschedule/ — reschedule with conflict check (#58)"""
    permission_classes = [IsAuthenticated]

    def post(self, request, interview_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)
        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        if interview.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        new_time_str = request.data.get('scheduled_at')
        if not new_time_str:
            return Response({'error': 'scheduled_at is required.'}, status=400)

        try:
            new_time = datetime.fromisoformat(new_time_str.replace('Z', '+00:00'))
        except ValueError:
            return Response({'error': 'Invalid scheduled_at format.'}, status=400)

        # Validate new time
        time_validation = validate_interview_time(new_time, interview.duration_minutes)
        if not time_validation['valid']:
            return Response({'error': time_validation['message']}, status=400)

        # Check conflicts (exclude current interview)
        from core.conflict_detector import check_scheduling_conflicts
        conflict_check = check_scheduling_conflicts(
            recruiter_id=str(request.user.id),
            candidate_id=interview.candidate_id,
            scheduled_at=new_time,
            duration_minutes=interview.duration_minutes,
            exclude_interview_id=str(interview.id)
        )
        if conflict_check['has_conflict']:
            return Response({'error': conflict_check['message'], 'conflicts': conflict_check['conflicts']}, status=409)

        old_time = interview.scheduled_at
        interview.scheduled_at = new_time
        # Extend token expiry: duration + 60 min buffer
        interview.token_expires_at = new_time + timedelta(minutes=interview.duration_minutes + 60)
        interview.updated_at = datetime.utcnow()
        interview.save()

        # Notify candidate
        if interview.candidate_id:
            try:
                candidate = User.objects.get(id=interview.candidate_id)
                from core.email_service import send_interview_scheduled_email
                send_interview_scheduled_email(
                    candidate_email=candidate.email,
                    candidate_name=candidate.name,
                    interview_data=interview.to_dict()
                )
                Notification(
                    recipient_id=interview.candidate_id,
                    sender_id=str(request.user.id),
                    notification_type='interview_scheduled',
                    title='Interview Rescheduled',
                    message=f'Your interview "{interview.title}" has been rescheduled to {new_time.strftime("%Y-%m-%d %H:%M UTC")}.',
                    link=f'/interview/room/{interview.room_id}'
                ).save()
            except Exception as e:
                logger.warning(f'[Reschedule] Notify failed: {e}')

        return Response({**interview.to_dict(), 'rescheduled_from': old_time.isoformat()})
