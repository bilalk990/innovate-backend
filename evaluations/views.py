"""
Evaluations app views — trigger and retrieve XAI evaluations
"""
import logging
import mongoengine
from datetime import datetime

logger = logging.getLogger('innovaite')
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from evaluations.models import Evaluation, CriterionResult, MockInterviewSession
from evaluations.engine import run_xai_evaluation
from interviews.models import Interview
from resumes.models import Resume
from notifications.models import Notification
from accounts.models import User
from core.openai_client import (
    generate_offer_letter, rank_candidates_for_job, generate_interview_debrief,
    predict_hire_probability, generate_followup_email, calculate_readiness_score,
    generate_mock_interview_question, evaluate_mock_answer, generate_mock_interview_report,
    analyze_anxiety_signals,
)
from core.email_service import send_evaluation_ready_email
from core.audit_logger import log_evaluation_triggered
from core.pdf_generator import generate_evaluation_pdf


class TriggerEvaluationView(APIView):
    """POST /api/evaluations/ — recruiter triggers evaluation for a completed interview"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can trigger evaluations.'}, status=403)

        interview_id = request.data.get('interview_id')
        if not interview_id:
            return Response({'error': 'interview_id is required.'}, status=400)

        try:
            interview = Interview.objects.get(id=interview_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Interview not found.'}, status=404)

        if interview.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        if not interview.candidate_id:
            return Response({'error': 'No candidate assigned to this interview.'}, status=400)

        # Get candidate's latest resume
        resume = Resume.objects(
            candidate_id=interview.candidate_id,
            is_active=True
        ).order_by('-uploaded_at').first()
        resume_data = resume.parsed_data if resume else {}

        # Return existing evaluation if already computed
        existing = Evaluation.objects(interview_id=str(interview.id)).first()
        if existing:
            return Response(existing.to_dict(), status=200)

        # Create a placeholder evaluation with 'processing' status
        evaluation = Evaluation(
            interview_id=str(interview.id),
            candidate_id=interview.candidate_id,
            recruiter_id=str(request.user.id),
            status='processing',
        )
        evaluation.save()

        # Run evaluation in background thread — no Celery needed
        import threading

        def _run_eval():
            try:
                result = run_xai_evaluation(interview, resume_data, user_id=str(request.user.id))

                criterion_docs = []
                for cr in result['criterion_results']:
                    criterion_docs.append(CriterionResult(
                        criterion=cr['criterion'],
                        score=cr['score'],
                        max_score=cr['max_score'],
                        weight=cr['weight'],
                        explanation=cr['explanation'],
                        rules_applied=cr['rules_applied'],
                        evidence=cr['evidence'],
                    ))

                evaluation.criterion_results = criterion_docs
                evaluation.overall_score = result['overall_score']
                evaluation.recommendation = result['recommendation']
                evaluation.summary = result['summary']
                evaluation.strengths = result['strengths']
                evaluation.weaknesses = result['weaknesses']
                evaluation.resume_alignment_score = result.get('resume_alignment_score', 0)
                evaluation.confidence_score = result.get('confidence_score', 50)
                evaluation.fluency_score = result.get('fluency_score', 50)
                evaluation.behavioral_summary = result.get('behavioral_summary', '')
                evaluation.proctoring_score = result.get('proctoring_score', 100)
                evaluation.integrity_notes = result.get('integrity_notes', '')
                evaluation.tab_switch_count = result.get('tab_switch_count', 0)
                evaluation.culture_fit_score = result.get('culture_fit_score', 0)
                evaluation.ai_summary_used = result.get('ai_summary_used', False)
                evaluation.question_analysis = result.get('question_analysis', {})
                evaluation.emotion_timeline = result.get('emotion_timeline', {})
                evaluation.recommendation_reason = result.get('recommendation_reason', '')
                evaluation.performance_stats = result.get('performance_stats', {})
                evaluation.status = 'complete'
                evaluation.updated_at = datetime.utcnow()
                evaluation.save()

                # Mark interview as completed
                interview.status = 'completed'
                interview.save()

                # Audit log
                log_evaluation_triggered(None, str(evaluation.id), None)

                # Notify candidate via email
                try:
                    candidate = User.objects.get(id=interview.candidate_id)
                    send_evaluation_ready_email(
                        candidate_email=candidate.email,
                        candidate_name=candidate.name,
                        evaluation_data=evaluation.to_dict()
                    )
                except Exception as e:
                    logger.warning(f"[Email] Failed to notify candidate: {str(e)}")

                # Notify recruiter
                Notification(
                    recipient_id=str(request.user.id),
                    notification_type='evaluation_ready',
                    title='Evaluation Complete',
                    message=f'XAI Evaluation for "{interview.title}" is ready for review.',
                    link=f'/evaluation/{str(evaluation.id)}'
                ).save()

                logger.info(f'[EVAL] Background evaluation complete for interview {interview.id}')

            except Exception as e:
                logger.error(f'[EVAL] Background evaluation failed for interview {interview.id}: {e}')
                evaluation.status = 'failed'
                evaluation.summary = f'Evaluation failed: {str(e)}'
                evaluation.save()

        thread = threading.Thread(target=_run_eval, daemon=True)
        thread.start()

        # Return 202 immediately — frontend should poll
        return Response({
            **evaluation.to_dict(),
            'status': 'processing',
            'message': 'Evaluation is being generated. Poll this endpoint to check status.',
        }, status=202)


class EvaluationShareView(APIView):
    """PATCH /api/evaluations/<id>/share/ — toggle candidate visibility (#57)"""
    permission_classes = [IsAuthenticated]

    def patch(self, request, eval_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        if evaluation.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        evaluation.candidate_visible = not getattr(evaluation, 'candidate_visible', False)
        evaluation.save()

        # Notify candidate if now visible
        if evaluation.candidate_visible:
            try:
                Notification(
                    recipient_id=evaluation.candidate_id,
                    notification_type='evaluation_ready',
                    title='Your Evaluation is Available',
                    message='Your interview evaluation report is now available to view.',
                    link=f'/candidate/evaluations/{str(evaluation.id)}'
                ).save()
            except Exception:
                pass

        return Response({'candidate_visible': evaluation.candidate_visible})


class ExportEvaluationsCSVView(APIView):
    """GET /api/evaluations/export/ — ATS export as CSV (#60)"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        import csv
        from django.http import HttpResponse as DjangoHttpResponse

        evals = Evaluation.objects(recruiter_id=str(request.user.id)).order_by('-created_at')

        response = DjangoHttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="evaluations_export.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Evaluation ID', 'Candidate Name', 'Candidate ID', 'Interview ID',
            'Overall Score', 'Recommendation', 'Confidence', 'Fluency',
            'Proctoring Score', 'Culture Fit', 'Resume Alignment',
            'Tab Switches', 'HR Reviewed', 'Created At'
        ])

        # Bulk fetch all candidates to avoid N+1 query problem
        candidate_ids = [ev.candidate_id for ev in evals]
        candidates = User.objects(id__in=candidate_ids).only('id', 'name')
        candidate_map = {str(c.id): c.name for c in candidates}

        for ev in evals:
            candidate_name = candidate_map.get(ev.candidate_id, '')

            writer.writerow([
                str(ev.id), candidate_name, ev.candidate_id, ev.interview_id,
                ev.overall_score, ev.recommendation, ev.confidence_score, ev.fluency_score,
                ev.proctoring_score, ev.culture_fit_score, ev.resume_alignment_score,
                ev.tab_switch_count, ev.reviewed_by_hr, ev.created_at.isoformat() if ev.created_at else ''
            ])

        return response


class EvaluationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role == 'recruiter':
            evals = Evaluation.objects(recruiter_id=str(user.id))
            # Allow filtering by candidate for recruiter's profile view
            candidate_id = request.query_params.get('candidate_id')
            if candidate_id:
                evals = evals.filter(candidate_id=candidate_id)
        elif user.role == 'candidate':
            evals = Evaluation.objects(candidate_id=str(user.id))
        else:
            evals = Evaluation.objects.all()

        # Pagination
        try:
            limit = min(int(request.query_params.get('limit', 50)), 100)
            offset = int(request.query_params.get('offset', 0))
        except (ValueError, TypeError):
            limit, offset = 50, 0

        total = evals.count()
        evals = evals.order_by('-created_at').skip(offset).limit(limit)
        return Response({
            'results': [e.to_dict() for e in evals],
            'total': total,
            'limit': limit,
            'offset': offset,
        })


class EvaluationDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, eval_id):
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        user = request.user
        if user.role == 'candidate' and evaluation.candidate_id != str(user.id):
            return Response({'error': 'Forbidden.'}, status=403)
        if user.role == 'recruiter' and evaluation.recruiter_id != str(user.id):
            return Response({'error': 'Forbidden.'}, status=403)

        return Response(evaluation.to_dict())

    def patch(self, request, eval_id):
        """HR submits notes/review"""
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        if 'hr_notes' in request.data:
            evaluation.hr_notes = request.data['hr_notes']
        evaluation.reviewed_by_hr = True
        evaluation.updated_at = datetime.utcnow()
        evaluation.save()
        return Response(evaluation.to_dict())

class OfferLetterView(APIView):
    """GET /api/evaluations/offer/?eval_id=... — HR generates an offer draft"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)
        
        eval_id = request.query_params.get('eval_id')
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
            user_doc = User.objects(id=evaluation.candidate_id).first()
            candidate_name = user_doc.name if user_doc else "Candidate"
            
            interview = Interview.objects.get(id=evaluation.interview_id)
            
            draft = generate_offer_letter(candidate_name, interview.job_title, evaluation.overall_score)
            return Response({'draft': draft})
        except (mongoengine.DoesNotExist, mongoengine.ValidationError, AttributeError) as e:
            return Response({'error': f'Failed to generate offer: {str(e)}'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Candidate Ranking AI Engine
# ─────────────────────────────────────────────────────────────────────────────

class CandidateRankingView(APIView):
    """
    GET /api/evaluations/rank/?job_id=<job_id>
    Recruiter gets AI-ranked list of all candidates for a specific job.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can use candidate ranking.'}, status=403)

        job_id = request.query_params.get('job_id', '').strip()
        if not job_id:
            return Response({'error': 'job_id is required.'}, status=400)

        # Get all completed evaluations for interviews linked to this job
        try:
            from jobs.models import Job
            job = Job.objects.get(id=job_id)
        except Exception:
            return Response({'error': 'Job not found.'}, status=404)

        if job.posted_by != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        # Find all completed interviews for this job
        from interviews.models import Interview
        interviews = Interview.objects(
            job_id=job_id,
            status='completed'
        )

        candidates_data = []
        for interview in interviews:
            eval_obj = Evaluation.objects(interview_id=str(interview.id)).first()
            if not eval_obj:
                continue

            # Get candidate info
            candidate = User.objects(id=interview.candidate_id).first()
            candidate_name = candidate.name if candidate else 'Unknown'

            # Get resume skills
            resume = None
            try:
                from resumes.models import Resume as ResumeModel
                resume = ResumeModel.objects(
                    candidate_id=interview.candidate_id,
                    is_active=True
                ).first()
            except Exception:
                pass

            candidates_data.append({
                'candidate_id': interview.candidate_id,
                'name': candidate_name,
                'overall_score': eval_obj.overall_score,
                'recommendation': eval_obj.recommendation,
                'confidence_score': eval_obj.confidence_score,
                'culture_fit_score': eval_obj.culture_fit_score,
                'proctoring_score': eval_obj.proctoring_score,
                'skills': resume.parsed_data.get('skills', []) if resume and resume.parsed_data else [],
                'experience_years': resume.parsed_data.get('total_experience_years', 0) if resume and resume.parsed_data else 0,
                'eval_id': str(eval_obj.id),
                'interview_id': str(interview.id),
            })

        if not candidates_data:
            return Response({'ranked': [], 'ranking_rationale': 'No completed evaluations for this job yet.', 'top_recommendation': ''})

        try:
            result = rank_candidates_for_job(
                job_title=job.title,
                job_description=job.description,
                candidates=candidates_data,
            )
            # Attach eval_id for frontend links
            eval_map = {c['candidate_id']: c['eval_id'] for c in candidates_data}
            for r in result.get('ranked', []):
                r['eval_id'] = eval_map.get(r.get('candidate_id', ''), '')
            return Response(result)
        except Exception as e:
            logger.error(f'[CandidateRanking] Failed: {e}')
            return Response({'error': 'Ranking service unavailable.'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — AI Interview Debrief Generator
# ─────────────────────────────────────────────────────────────────────────────

class InterviewDebriefView(APIView):
    """
    GET /api/evaluations/<eval_id>/debrief/
    Candidate gets their personalized post-interview coaching debrief.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, eval_id):
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        # Access control: candidate can only see their own, recruiter sees their evaluations
        if request.user.role == 'candidate' and evaluation.candidate_id != str(request.user.id):
            return Response({'error': 'Forbidden.'}, status=403)
        if request.user.role == 'recruiter' and evaluation.recruiter_id != str(request.user.id):
            return Response({'error': 'Forbidden.'}, status=403)

        # Candidate visibility check
        if request.user.role == 'candidate' and not evaluation.candidate_visible:
            return Response({'error': 'Debrief not yet released by your recruiter.'}, status=403)

        # Get interview data
        try:
            interview = Interview.objects.get(id=evaluation.interview_id)
        except Exception:
            interview = None

        candidate_name = ''
        try:
            candidate = User.objects.get(id=evaluation.candidate_id)
            candidate_name = candidate.name
        except Exception:
            pass

        try:
            debrief = generate_interview_debrief(
                evaluation=evaluation.to_dict(),
                interview=interview.to_dict() if interview else {},
                candidate_name=candidate_name,
            )
            debrief['eval_id'] = eval_id
            debrief['candidate_name'] = candidate_name
            return Response(debrief)
        except Exception as e:
            logger.error(f'[Debrief] Generation failed: {e}')
            return Response({'error': 'Debrief service unavailable.'}, status=500)


class ExportEvaluationPDFView(APIView):
    """Export evaluation as PDF."""
    permission_classes = [IsAuthenticated]

    def get(self, request, eval_id):
        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        user = request.user
        if user.role == 'candidate' and evaluation.candidate_id != str(user.id):
            return Response({'error': 'Forbidden.'}, status=403)
        if user.role == 'recruiter' and evaluation.recruiter_id != str(user.id):
            return Response({'error': 'Forbidden.'}, status=403)

        # Get related data
        try:
            interview = Interview.objects.get(id=evaluation.interview_id)
            candidate = User.objects.get(id=evaluation.candidate_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Related data not found.'}, status=404)

        # Generate PDF
        pdf_buffer = generate_evaluation_pdf(
            evaluation_data=evaluation.to_dict(),
            interview_data=interview.to_dict(),
            candidate_data=candidate.to_dict()
        )

        # Return as downloadable file
        response = HttpResponse(pdf_buffer, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="evaluation_{eval_id}.pdf"'

        return response


# ─────────────────────────────────────────────────────────────────────────────
# Feature 10 — Predictive Hiring Score
# ─────────────────────────────────────────────────────────────────────────────

class HireProbabilityView(APIView):
    """
    GET /api/evaluations/<eval_id>/hire-probability/
    Recruiter/Admin gets AI-predicted hire probability for a candidate.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, eval_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        if request.user.role == 'recruiter' and evaluation.recruiter_id != str(request.user.id):
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            interview = Interview.objects.get(id=evaluation.interview_id)
            job_title = getattr(interview, 'job_title', None) or getattr(interview, 'title', None) or 'Unknown Role'
        except Exception:
            job_title = 'Unknown Role'

        try:
            result = predict_hire_probability(
                overall_score=evaluation.overall_score or 0,
                confidence_score=evaluation.confidence_score or 0,
                proctoring_score=evaluation.proctoring_score or 0,
                fluency_score=evaluation.fluency_score or 0,
                culture_fit_score=evaluation.culture_fit_score or 0,
                violations=evaluation.tab_switch_count or 0,
                recommendation=evaluation.recommendation or 'maybe',
                job_title=job_title,
                criterion_results=[cr.to_dict() for cr in (evaluation.criterion_results or [])],
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[HireProbability] Failed: {e}')
            return Response({'error': 'Prediction service unavailable.'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Automated Follow-up Email Generation
# ─────────────────────────────────────────────────────────────────────────────

class FollowUpEmailView(APIView):
    """
    POST /api/evaluations/<eval_id>/followup-email/
    Generate personalized post-interview email for candidate (selection/rejection/hold/next_round).
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, eval_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Only recruiters can generate follow-up emails.'}, status=403)

        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        if evaluation.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        decision = request.data.get('decision', 'hold')
        if decision not in ('selected', 'rejected', 'hold', 'next_round'):
            return Response({'error': "decision must be 'selected', 'rejected', 'hold', or 'next_round'."}, status=400)

        company_name = request.data.get('company_name', 'InnovAIte')

        # Resolve candidate name and job_title from interview
        candidate_name = 'Candidate'
        interview = None  # initialize before try so it's always bound
        try:
            interview = Interview.objects.get(id=evaluation.interview_id)
            if interview.candidate_id:
                candidate_user = User.objects(id=interview.candidate_id).first()
                if candidate_user:
                    candidate_name = candidate_user.name or candidate_user.email or 'Candidate'
        except Exception:
            pass

        job_title = (
            getattr(evaluation, 'job_title', None)
            or (getattr(interview, 'job_title', None) if interview else None)
            or 'the position'
        )

        # Build evaluation summary for AI
        eval_data = {
            'overall_score': evaluation.overall_score or 0,
            'recommendation': evaluation.recommendation or 'maybe',
            'strengths': list(evaluation.strengths or [])[:3],
            'weaknesses': list(evaluation.weaknesses or [])[:3],
            'summary': evaluation.summary or '',
        }

        try:
            result = generate_followup_email(
                candidate_name=candidate_name,
                job_title=job_title,
                decision=decision,
                evaluation_data=eval_data,
                company_name=company_name,
                user_id=str(request.user.id),
            )
            return Response({'email': result, 'candidate_name': candidate_name})
        except Exception as e:
            logger.error(f'[FollowUpEmail] Generation failed: {e}')
            return Response({'error': 'Email generation failed.'}, status=500)


# ═════════════════════════════════════════════════════════════════════════════
# NEW AI FEATURES - DYNAMIC ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

class BehavioralTraitsView(APIView):
    """
    POST /api/evaluations/behavioral-traits/
    Analyze transcript for behavioral traits using AI.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        transcript = request.data.get('transcript', '').strip()
        if not transcript or len(transcript) < 20:
            return Response({'error': 'Transcript is required and must be at least 20 characters.'}, status=400)

        try:
            from core.gemini import analyze_behavioral_traits
            result = analyze_behavioral_traits(transcript, user_id=str(request.user.id))
            
            # Normalize response format
            if isinstance(result, dict):
                # Ensure traits field exists
                if 'traits' not in result and 'confidence_score' in result:
                    result['traits'] = {
                        'confidence': result.get('confidence_score', 70),
                        'fluency': result.get('fluency_score', 70),
                        'filler_words': result.get('filler_count', 0)
                    }
                return Response(result)
            else:
                return Response({'error': 'Invalid AI response format.'}, status=500)
        except Exception as e:
            logger.error(f'[BehavioralTraits] Failed: {e}')
            return Response({'error': f'Analysis failed: {str(e)}'}, status=500)


class IntegrityCheckView(APIView):
    """
    POST /api/evaluations/check-integrity/
    Check responses for plagiarism and AI-generated content.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        responses = request.data.get('responses', [])
        if not responses or not isinstance(responses, list):
            return Response({'error': 'Responses array is required.'}, status=400)

        try:
            from core.gemini import check_integrity_plagiarism
            result = check_integrity_plagiarism(responses, user_id=str(request.user.id))
            
            # Normalize response format
            if isinstance(result, dict):
                # Ensure required fields exist
                if 'integrity_score' not in result:
                    result['integrity_score'] = result.get('score', 90)
                if 'red_flags' not in result:
                    result['red_flags'] = result.get('notes', '').split('. ') if result.get('notes') else []
                return Response(result)
            else:
                return Response({'error': 'Invalid AI response format.'}, status=500)
        except Exception as e:
            logger.error(f'[IntegrityCheck] Failed: {e}')
            return Response({'error': f'Integrity check failed: {str(e)}'}, status=500)


class CultureFitView(APIView):
    """
    POST /api/evaluations/culture-fit/
    Analyze candidate's culture fit based on transcript and company values.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        transcript = request.data.get('transcript', '').strip()
        company_values = request.data.get('company_values', [])
        
        if not transcript or len(transcript) < 20:
            return Response({'error': 'Transcript is required and must be at least 20 characters.'}, status=400)
        
        if not company_values:
            company_values = ['Innovation', 'Collaboration', 'Excellence', 'Integrity']

        try:
            from core.gemini import analyze_culture_fit
            result = analyze_culture_fit(transcript, company_values)
            
            # Normalize response format
            if isinstance(result, dict):
                # Ensure required fields exist
                if 'culture_fit_score' not in result:
                    result['culture_fit_score'] = result.get('culture_score', 70)
                if 'aligned_values' not in result:
                    result['aligned_values'] = result.get('aligned', [])
                if 'misaligned_values' not in result:
                    result['misaligned_values'] = result.get('red_flags', [])
                return Response(result)
            else:
                return Response({'error': 'Invalid AI response format.'}, status=500)
        except Exception as e:
            logger.error(f'[CultureFit] Failed: {e}')
            return Response({'error': f'Culture fit analysis failed: {str(e)}'}, status=500)


class ExecutiveSummaryView(APIView):
    """
    POST /api/evaluations/executive-summary/
    Generate executive summary for leadership review.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        interview_data = request.data.get('interview_data', {})
        evaluation_results = request.data.get('evaluation_results', {})
        
        if not interview_data or not evaluation_results:
            return Response({'error': 'Both interview_data and evaluation_results are required.'}, status=400)

        try:
            from core.gemini import generate_executive_summary
            summary = generate_executive_summary(interview_data, evaluation_results)
            
            # Return as structured response
            if isinstance(summary, str):
                return Response({'summary': summary, 'executive_summary': summary})
            elif isinstance(summary, dict):
                return Response(summary)
            else:
                return Response({'error': 'Invalid AI response format.'}, status=500)
        except Exception as e:
            logger.error(f'[ExecutiveSummary] Failed: {e}')
            return Response({'error': f'Summary generation failed: {str(e)}'}, status=500)


class PredictHireView(APIView):
    """
    GET /api/evaluations/<eval_id>/predict-hire/
    Advanced predictive hiring score using ML-based analysis.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, eval_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            evaluation = Evaluation.objects.get(id=eval_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Evaluation not found.'}, status=404)

        if request.user.role == 'recruiter' and evaluation.recruiter_id != str(request.user.id):
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            # Get interview for job title
            interview = Interview.objects.get(id=evaluation.interview_id)
            job_title = getattr(interview, 'job_title', None) or getattr(interview, 'title', None) or 'Unknown Role'
        except Exception:
            job_title = 'Unknown Role'

        try:
            from core.openai_client import predict_hire_probability
            result = predict_hire_probability(
                overall_score=evaluation.overall_score or 0,
                confidence_score=evaluation.confidence_score or 0,
                proctoring_score=evaluation.proctoring_score or 0,
                fluency_score=evaluation.fluency_score or 0,
                culture_fit_score=evaluation.culture_fit_score or 0,
                violations=evaluation.tab_switch_count or 0,
                recommendation=evaluation.recommendation or 'maybe',
                job_title=job_title,
                criterion_results=[cr.to_dict() for cr in (evaluation.criterion_results or [])],
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PredictHire] Failed: {e}')
            return Response({'error': f'Prediction failed: {str(e)}'}, status=500)


class ReadinessScoreView(APIView):
    """GET /api/evaluations/readiness/ — AI-powered interview readiness score for candidate."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            profile_data = {
                'skills': getattr(user, 'detailed_skills', []) or [],
                'work_history': getattr(user, 'work_history', []) or [],
                'bio': getattr(user, 'bio', '') or '',
                'education': getattr(user, 'education', '') or '',
                'resume_uploaded': False,
            }
            try:
                from resumes.models import Resume
                profile_data['resume_uploaded'] = Resume.objects.filter(candidate_id=str(user.id)).count() > 0
            except Exception:
                pass

            practice_history = []
            try:
                evals = Evaluation.objects.filter(candidate_id=str(user.id)).order_by('-created_at')[:5]
                for ev in evals:
                    practice_history.append({
                        'overall_score': getattr(ev, 'overall_score', 0),
                        'recommendation': getattr(ev, 'recommendation', 'MAYBE'),
                        'job_title': getattr(ev, 'job_title', ''),
                    })
            except Exception:
                pass

            result = calculate_readiness_score(profile_data, practice_history, user_id=str(user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[ReadinessScore] Failed: {e}')
            return Response({'error': f'Readiness score failed: {str(e)}'}, status=500)


# ═════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: MOCK INTERVIEW + ANXIETY DETECTION VIEWS
# ═════════════════════════════════════════════════════════════════════════════

class MockInterviewStartView(APIView):
    """POST /api/evaluations/mock/start/ — Start a new AI mock interview session."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can use mock interviews.'}, status=403)

        role = request.data.get('role', '').strip()
        level = request.data.get('level', 'mid').strip()

        if not role:
            return Response({'error': 'Job role is required.'}, status=400)
        if level not in ['junior', 'mid', 'senior']:
            level = 'mid'

        # Abandon any existing active session for this user
        try:
            MockInterviewSession.objects(
                user_id=str(request.user.id), status='active'
            ).update(status='abandoned')
        except Exception:
            pass

        # Generate first question
        try:
            q_data = generate_mock_interview_question(
                role=role, level=level, history=[], question_number=1,
                user_id=str(request.user.id)
            )
        except Exception as e:
            logger.error(f'[MockInterview] Question generation failed: {e}')
            return Response({'error': 'Failed to generate interview question.'}, status=500)

        # Store first question in history (without answer yet) so answer submission can find it
        first_entry = {
            'question': q_data.get('question', ''),
            'question_type': q_data.get('question_type', 'behavioral'),
            'what_to_assess': q_data.get('what_to_assess', ''),
            'tip_for_candidate': q_data.get('tip_for_candidate', ''),
        }

        # Create session with first question already in history
        session = MockInterviewSession(
            user_id=str(request.user.id),
            role=role,
            level=level,
            history=[first_entry],
            current_question=1,
            total_questions=5,
            status='active',
        )
        session.save()

        return Response({
            'session_id': str(session.id),
            'role': role,
            'level': level,
            'total_questions': 5,
            'current_question': 1,
            'question_data': q_data,
        })


class MockInterviewAnswerView(APIView):
    """POST /api/evaluations/mock/answer/ — Submit answer, get feedback + next question or final report."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'candidate':
            return Response({'error': 'Forbidden.'}, status=403)

        session_id = request.data.get('session_id', '').strip()
        answer = request.data.get('answer', '').strip()

        if not session_id:
            return Response({'error': 'session_id is required.'}, status=400)
        if not answer or len(answer) < 5:
            return Response({'error': 'Answer must be at least 5 characters.'}, status=400)

        # Load session
        try:
            session = MockInterviewSession.objects.get(id=session_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Session not found.'}, status=404)

        if session.user_id != str(request.user.id):
            return Response({'error': 'Forbidden.'}, status=403)
        if session.status != 'active':
            return Response({'error': 'Session is no longer active.'}, status=400)

        # Get current question (last item pending answer in history, or from start)
        # The question was stored when generating, answer now
        history = session.history or []
        current_q_num = session.current_question

        # The last question data (if history has an entry without answer)
        if history and 'answer' not in history[-1]:
            current_q_data = history[-1]
        else:
            # Shouldn't happen, but fallback
            return Response({'error': 'No pending question found.'}, status=400)

        # Evaluate the answer
        try:
            eval_result = evaluate_mock_answer(
                question=current_q_data.get('question', ''),
                answer=answer,
                role=session.role,
                question_type=current_q_data.get('question_type', 'behavioral'),
                user_id=str(request.user.id)
            )
        except Exception as e:
            logger.error(f'[MockInterview] Answer evaluation failed: {e}')
            eval_result = {'score': 6, 'grade': 'Average', 'feedback': 'Answer recorded.', 'strengths': [], 'improvements': [], 'better_answer_hint': '', 'keywords_used': [], 'keywords_missed': []}

        # Update the last history entry with answer + feedback
        current_q_data['answer'] = answer
        current_q_data['score'] = eval_result.get('score', 6)
        current_q_data['grade'] = eval_result.get('grade', 'Average')
        current_q_data['feedback'] = eval_result.get('feedback', '')
        current_q_data['strengths'] = eval_result.get('strengths', [])
        current_q_data['improvements'] = eval_result.get('improvements', [])
        current_q_data['better_answer_hint'] = eval_result.get('better_answer_hint', '')
        current_q_data['keywords_used'] = eval_result.get('keywords_used', [])
        current_q_data['keywords_missed'] = eval_result.get('keywords_missed', [])
        history[-1] = current_q_data

        # Check if this was the last question
        if current_q_num >= session.total_questions:
            # Generate final report
            try:
                report = generate_mock_interview_report(
                    role=session.role, level=session.level, history=history,
                    user_id=str(request.user.id)
                )
            except Exception as e:
                logger.error(f'[MockInterview] Report generation failed: {e}')
                report = {'overall_score': 70, 'performance_grade': 'Average', 'interview_summary': 'Interview completed.', 'top_strengths': [], 'critical_improvements': [], 'recommended_resources': [], 'readiness_for_real_interview': 'Almost Ready', 'next_steps': [], 'motivational_note': 'Keep practicing!'}

            session.history = history
            session.status = 'completed'
            session.final_report = report
            session.completed_at = datetime.utcnow()
            session.save()

            return Response({
                'status': 'completed',
                'feedback': eval_result,
                'final_report': report,
                'session': session.to_dict(),
            })
        else:
            # Generate next question
            next_q_num = current_q_num + 1
            try:
                next_q = generate_mock_interview_question(
                    role=session.role, level=session.level,
                    history=[{k: v for k, v in h.items() if k in ('question', 'answer')} for h in history],
                    question_number=next_q_num,
                    user_id=str(request.user.id)
                )
            except Exception as e:
                logger.error(f'[MockInterview] Next question failed: {e}')
                next_q = {'question': 'Tell me about a recent professional achievement.', 'question_type': 'behavioral', 'what_to_assess': 'Achievement orientation', 'tip_for_candidate': 'Be specific and quantify results.'}

            # Add next question to history (without answer yet)
            history.append({
                'question': next_q.get('question', ''),
                'question_type': next_q.get('question_type', 'behavioral'),
                'what_to_assess': next_q.get('what_to_assess', ''),
                'tip_for_candidate': next_q.get('tip_for_candidate', ''),
            })

            session.history = history
            session.current_question = next_q_num
            session.save()

            return Response({
                'status': 'active',
                'feedback': eval_result,
                'next_question': next_q_num,
                'total_questions': session.total_questions,
                'question_data': next_q,
            })


class MockInterviewSessionView(APIView):
    """GET /api/evaluations/mock/<session_id>/ — Get a mock interview session."""
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            session = MockInterviewSession.objects.get(id=session_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Session not found.'}, status=404)

        if session.user_id != str(request.user.id) and request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        return Response(session.to_dict())


class MockInterviewListView(APIView):
    """GET /api/evaluations/mock/ — List candidate's mock interview sessions."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            sessions = MockInterviewSession.objects(
                user_id=str(request.user.id)
            ).order_by('-created_at')[:10]
            return Response([s.to_dict() for s in sessions])
        except Exception as e:
            logger.error(f'[MockInterview] List failed: {e}')
            return Response({'error': str(e)}, status=500)


class AnxietyDetectionView(APIView):
    """POST /api/evaluations/anxiety-check/ — Analyze speech features for anxiety during interview."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Accept speech feature data from client-side audio analysis
        speech_features = {
            'speech_rate_wpm': request.data.get('speech_rate_wpm', 120),
            'pause_frequency': request.data.get('pause_frequency', 0),
            'long_pauses': request.data.get('long_pauses', 0),
            'volume_variance': request.data.get('volume_variance', 0.2),
            'filler_word_count': request.data.get('filler_word_count', 0),
            'speaking_duration_seconds': request.data.get('speaking_duration_seconds', 30),
            'silence_ratio': request.data.get('silence_ratio', 0.1),
            'pitch_variance': request.data.get('pitch_variance', 0),
        }

        try:
            result = analyze_anxiety_signals(speech_features, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[AnxietyDetection] Failed: {e}')
            return Response({
                'anxiety_score': 30,
                'anxiety_level': 'Mild',
                'detected_signals': ['Analysis unavailable'],
                'calm_message': 'You are doing great. Stay focused and breathe.',
                'breathing_exercise': 'Inhale for 4 counts, hold for 4, exhale for 6.',
                'quick_tip': 'Take a brief pause before answering each question.',
                'positive_affirmation': 'You are prepared. Trust yourself.'
            })
