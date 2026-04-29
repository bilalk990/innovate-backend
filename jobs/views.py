import logging
import mongoengine
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import Job, Application
from accounts.models import User
from resumes.models import Resume
from interviews.models import Interview
from evaluations.models import Evaluation
from notifications.models import Notification
from core.email_service import send_recruiter_notification_email
from core.openai_client import analyze_resume_jd_gap, predict_application_status
from django.conf import settings

logger = logging.getLogger('innovaite')

class JobListView(APIView):
    def get_permissions(self):
        if self.request.method == 'GET':
            return [AllowAny()]
        return [IsAuthenticated()]

    def get(self, request):
        # Candidates and guests see active jobs
        # Recruiters see their own jobs
        user_role = getattr(request.user, 'role', None)
        search_query = request.query_params.get('search', '').strip()

        if not request.user.is_authenticated or user_role == 'candidate':
            jobs = Job.objects(is_active=True)
        elif user_role == 'recruiter':
            jobs = Job.objects(posted_by=str(request.user.id))
        else:
            jobs = Job.objects(is_active=True)

        if search_query:
            from mongoengine import Q
            jobs = jobs.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(company_name__icontains=search_query) |
                Q(requirements__icontains=search_query)
            )
            logger.info(f'[JobSearch] Query: "{search_query}", Results: {jobs.count()}')

        results = []
        for j in jobs:
            jd = j.to_dict()
            try:
                recruiter = User.objects(id=j.posted_by).first()
                if recruiter:
                    jd['company_description'] = recruiter.bio
            except Exception:
                jd['company_description'] = None
            results.append(jd)

        return Response(results)

    def post(self, request):
        if request.user.role != 'recruiter':
            return Response({'error': 'Only recruiters can post jobs.'}, status=403)

        # Enforce Profile Completion
        if not getattr(request.user, 'is_profile_complete', False):
            return Response({'error': 'Please complete your recruiter profile before posting jobs.', 'redirect': '/recruiter/profile-setup'}, status=400)
        
        data = request.data
        job = Job(
            title=data.get('title'),
            company_name=request.user.company_name or 'Independent Recruiter',
            location=data.get('location', 'Remote'),
            job_type=data.get('job_type', 'full-time'),
            salary_range=data.get('salary_range'),
            description=data.get('description'),
            requirements=data.get('requirements', []),
            posted_by=str(request.user.id)
        )
        job.save()
        return Response(job.to_dict(), status=201)

class JobDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        try:
            job = Job.objects.get(id=job_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Job not found.'}, status=404)
        jd = job.to_dict()
        try:
            recruiter = User.objects(id=job.posted_by).first()
            if recruiter:
                jd['company_description'] = recruiter.bio
        except Exception:
            jd['company_description'] = None
        return Response(jd)

    def patch(self, request, job_id):
        try:
            job = Job.objects.get(id=job_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Job not found.'}, status=404)
            
        if job.posted_by != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)
            
        for key, val in request.data.items():
            if hasattr(job, key):
                setattr(job, key, val)
        job.save()
        return Response(job.to_dict())

class ApplicationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Candidates see their applications
        # Recruiters see applications for their jobs
        if request.user.role == 'candidate':
            apps = Application.objects(candidate_id=str(request.user.id))
        elif request.user.role == 'recruiter':
            apps = Application.objects(recruiter_id=str(request.user.id))
        else:
            apps = Application.objects.all()
            
        return Response([a.to_dict() for a in apps])

    def post(self, request):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can apply.'}, status=403)

        # Resume is required before applying
        resume = Resume.objects(candidate_id=str(request.user.id), is_active=True).first()
        if not resume:
            return Response({
                'error': 'Please upload or generate your resume in the Resume Builder before applying to jobs.',
                'code': 'RESUME_REQUIRED'
            }, status=400)

        job_id = request.data.get('job_id')

        if not job_id:
            return Response({'error': 'job_id is required.'}, status=400)
        try:
            job = Job.objects.get(id=job_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Job not found.'}, status=404)
            
        # Check if already applied
        if Application.objects(job_id=job_id, candidate_id=str(request.user.id)).first():
            return Response({'error': 'You have already applied for this job.'}, status=400)
            
        app = Application(
            job_id=job_id,
            candidate_id=str(request.user.id),
            recruiter_id=job.posted_by,
            candidate_name=request.user.name,
            candidate_headline=getattr(request.user, 'headline', '')
        )
        app.save()

        # --- Notify recruiter in background thread so HTTP response is instant ---
        import threading
        from django.conf import settings as django_settings

        def _notify(job_id, job_title, recruiter_id, applicant_id, applicant_name):
            try:
                recruiter = User.objects(id=recruiter_id).first()
                if recruiter:
                    Notification(
                        recipient_id=recruiter_id,
                        sender_id=applicant_id,
                        notification_type='new_application',
                        title='New Application Received',
                        message=f'{applicant_name} applied for your posting: "{job_title}".',
                        link=f'/recruiter/jobs/{job_id}/applicants'
                    ).save()
                    frontend_url = getattr(django_settings, 'FRONTEND_URL', '')
                    send_recruiter_notification_email(
                        recruiter_email=recruiter.email,
                        recruiter_name=recruiter.name,
                        message=(
                            f'<strong>{applicant_name}</strong> has applied for your job posting '
                            f'<strong>"{job_title}"</strong>. Review their application.'
                        ),
                        link=f'{frontend_url}/recruiter/jobs/{job_id}/applicants'
                    )
            except Exception as e:
                logger.warning(f'[Application] Recruiter notification failed: {e}')

        threading.Thread(
            target=_notify,
            args=(job_id, job.title, job.posted_by, str(request.user.id), request.user.name),
            daemon=True
        ).start()
        # -------------------------------------------------------------------------

        return Response(app.to_dict(), status=201)

# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — AI Resume vs JD Gap Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class GapAnalysisView(APIView):
    """
    GET /api/jobs/<job_id>/gap-analysis/
    Candidate gets AI-powered gap analysis between their resume and this job.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        if request.user.role != 'candidate':
            return Response({'error': 'Only candidates can use gap analysis.'}, status=403)

        try:
            job = Job.objects.get(id=job_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Job not found.'}, status=404)

        # Get candidate's active resume
        resume = Resume.objects(
            candidate_id=str(request.user.id),
            is_active=True
        ).order_by('-uploaded_at').first()

        if not resume or not resume.parsed_data:
            return Response({'error': 'Please upload your resume first to use gap analysis.'}, status=400)

        try:
            result = analyze_resume_jd_gap(
                resume_data=resume.parsed_data,
                job_description=job.description,
                job_title=job.title,
                requirements=job.requirements,
            )
            result['job_title'] = job.title
            result['job_id'] = str(job.id)
            return Response(result)
        except Exception as e:
            logger.error(f'[GapAnalysis] Failed: {e}')
            return Response({'error': 'Gap analysis service unavailable.'}, status=500)


class ApplicationDetailView(APIView):
    """
    PATCH /api/jobs/applications/<app_id>/
    Recruiter updates application status: reviewed, shortlisted, rejected, offer_sent, hired.
    Triggers candidate notification on status change.
    """
    permission_classes = [IsAuthenticated]

    # CRITICAL FIX: Define valid state transitions to prevent invalid status changes
    VALID_TRANSITIONS = {
        'pending': ['reviewed', 'rejected'],
        'reviewed': ['shortlisted', 'rejected'],
        'shortlisted': ['interview_scheduled', 'rejected'],
        'interview_scheduled': ['offer_sent', 'rejected'],
        'offer_sent': ['hired', 'rejected'],
        'hired': [],  # Terminal state
        'rejected': [],  # Terminal state
    }

    CANDIDATE_MESSAGES = {
        'reviewed':             ('Application Reviewed', 'Your application has been reviewed by the recruiter. We will be in touch soon.'),
        'shortlisted':          ('You\'ve Been Shortlisted! 🎉', 'Great news! You have been shortlisted for the next stage. Expect an interview invitation shortly.'),
        'rejected':             ('Application Update', 'Thank you for your interest. After careful consideration, we have decided to move forward with other candidates at this time.'),
        'interview_scheduled':  ('Interview Scheduled', 'An interview has been scheduled for your application. Please check your email for details.'),
        'offer_sent':           ('Offer Letter Sent! 🎊', 'Congratulations! An offer letter has been generated for you. Please check your evaluation report to view and respond.'),
        'hired':                ('Welcome Aboard! 🚀', 'Congratulations! You have officially been hired. Welcome to the team!'),
    }

    def patch(self, request, app_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)

        try:
            app = Application.objects.get(id=app_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Application not found.'}, status=404)

        if app.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        new_status = request.data.get('status', '').strip()
        old_status = app.status
        
        # CRITICAL FIX: Validate state transitions
        valid_next_states = self.VALID_TRANSITIONS.get(old_status, [])
        if new_status not in valid_next_states:
            return Response({
                'error': f'Invalid status transition from "{old_status}" to "{new_status}". Valid transitions: {", ".join(valid_next_states) if valid_next_states else "none (terminal state)"}'
            }, status=400)

        app.status = new_status
        app.save()

        # CRITICAL FIX: Audit log for status changes
        from core.audit_logger import log_audit
        log_audit(
            request.user, 
            'application_status_changed',
            status='success',
            details={'app_id': str(app.id), 'old_status': old_status, 'new_status': new_status},
            request=request
        )

        # Notify candidate about status change
        if new_status != old_status:
            try:
                title, message = self.CANDIDATE_MESSAGES.get(new_status, ('Application Update', 'Your application status has been updated.'))
                Notification(
                    recipient_id=app.candidate_id,
                    sender_id=str(request.user.id),
                    notification_type='application_status',
                    title=title,
                    message=message,
                    link='/candidate/applications',
                ).save()

                # Email the candidate
                candidate = User.objects(id=app.candidate_id).first()
                if candidate and candidate.email:
                    try:
                        job = Job.objects.get(id=app.job_id)
                        job_title = job.title
                    except Exception:
                        job_title = 'the position'

                    from core.email_service import send_application_status_email
                    send_application_status_email(
                        candidate_email=candidate.email,
                        candidate_name=candidate.name or 'Candidate',
                        job_title=job_title,
                        new_status=new_status,
                        message=message,
                    )
            except Exception as notify_err:
                logger.warning(f'[Application] Candidate notification failed: {notify_err}')

        return Response(app.to_dict())


class JobApplicantsView(APIView):
    """View to list all applicants for a specific job posting (Recruiter only)"""
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Forbidden.'}, status=403)
        
        try:
            job = Job.objects.get(id=job_id)
            if job.posted_by != str(request.user.id) and request.user.role != 'admin':
                return Response({'error': 'Forbidden.'}, status=403)
            
            apps = Application.objects(job_id=job_id).order_by('-applied_at')
            # Enrichment: Add a flag if they have a resume and their latest status
            result = []
            for a in apps:
                d = a.to_dict()
                latest_resume = Resume.objects(candidate_id=a.candidate_id, is_active=True).first()
                d['has_resume'] = latest_resume is not None
                # CRITICAL FIX: Only return resume data if recruiter owns this job (IDOR protection)
                if latest_resume and job.posted_by == str(request.user.id):
                    d['resume_id'] = str(latest_resume.id)
                    d['resume_skills'] = latest_resume.parsed_data.get('skills', [])
                
                # Check for evaluation result — query Evaluation model (not Interview)
                latest_interview = Interview.objects(
                    candidate_id=a.candidate_id,
                    job_id=job_id,
                    status='completed'
                ).order_by('-scheduled_at').first()
                if latest_interview:
                    latest_eval = Evaluation.objects(
                        interview_id=str(latest_interview.id)
                    ).first()
                    if latest_eval:
                        d['evaluation_score'] = latest_eval.overall_score
                        d['recommendation'] = latest_eval.recommendation
                
                result.append(d)
                
            return Response(result)
        except (Job.DoesNotExist, Exception) as e:
            logger.error(f'[JobApplicantsView] Error: {e}')
            return Response({'error': 'Job not found or error fetching applicants.'}, status=404)




# ═════════════════════════════════════════════════════════════════════════════
# NEW AI FEATURES - JOB FITMENT & ADVANCED GAP ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

class JobFitmentAnalysisView(APIView):
    """
    POST /api/jobs/fitment-analysis/
    Deep semantic job fitment analysis using AI.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        resume_data = request.data.get('resume_data', {})
        job_description = request.data.get('job_description', '').strip()
        
        if not resume_data or not job_description:
            return Response({'error': 'Both resume_data and job_description are required.'}, status=400)

        try:
            from core.gemini import analyze_job_fitment
            result = analyze_job_fitment(resume_data, job_description)
            
            # Normalize response format
            if isinstance(result, dict):
                # Ensure required fields exist
                if 'fitment_score' not in result:
                    result['fitment_score'] = result.get('match_percentage', 60)
                if 'matched_skills' not in result:
                    result['matched_skills'] = result.get('matched_dimensions', [])
                if 'missing_skills' not in result:
                    result['missing_skills'] = result.get('missing_relevance', [])
                return Response(result)
            else:
                return Response({'error': 'Invalid AI response format.'}, status=500)
        except Exception as e:
            logger.error(f'[JobFitment] Failed: {e}')
            return Response({'error': f'Fitment analysis failed: {str(e)}'}, status=500)


class AdvancedGapAnalysisView(APIView):
    """
    POST /api/jobs/advanced-gap-analysis/
    Advanced gap analysis with detailed recommendations.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        resume_data = request.data.get('resume_data', {})
        job_id = request.data.get('job_id', '').strip()
        
        if not resume_data or not job_id:
            return Response({'error': 'Both resume_data and job_id are required.'}, status=400)

        try:
            job = Job.objects.get(id=job_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'Job not found.'}, status=404)

        try:
            from core.openai_client import analyze_resume_jd_gap
            result = analyze_resume_jd_gap(
                resume_data=resume_data,
                job_description=job.description,
                job_title=job.title,
                requirements=job.requirements,
            )
            result['job_title'] = job.title
            result['job_id'] = str(job.id)
            return Response(result)
        except Exception as e:
            logger.error(f'[AdvancedGapAnalysis] Failed: {e}')
            return Response({'error': f'Gap analysis failed: {str(e)}'}, status=500)


class PredictApplicationStatusView(APIView):
    """
    POST /api/jobs/<job_id>/predict-status/
    Predict if the candidate's application will be shortlisted for this job.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, job_id):
        user = request.user
        if getattr(user, 'role', '') != 'candidate':
            return Response({'error': 'Only candidates can use this feature.'}, status=403)

        try:
            job = Job.objects.get(id=job_id)
        except (Job.DoesNotExist, Exception):
            return Response({'error': 'Job not found.'}, status=404)

        # Get candidate's active resume
        resume_data = {}
        try:
            resumes = Resume.objects.filter(candidate_id=str(user.id))
            active_resume = next((r for r in resumes if getattr(r, 'is_active', False)), None) or (resumes[0] if resumes else None)
            if active_resume and getattr(active_resume, 'parsed_data', None):
                resume_data = active_resume.parsed_data
        except Exception:
            pass

        # Fallback to user profile skills
        if not resume_data:
            resume_data = {
                'skills': getattr(user, 'detailed_skills', []) or [],
                'experience': getattr(user, 'work_history', []) or [],
                'headline': getattr(user, 'headline', '') or '',
                'total_experience_years': 0,
            }

        try:
            result = predict_application_status(
                resume_data=resume_data,
                job_title=getattr(job, 'title', 'Unknown Role'),
                job_description=getattr(job, 'description', '') or '',
                requirements=getattr(job, 'requirements', []) or [],
                user_id=str(user.id),
            )
            result['job_title'] = getattr(job, 'title', '')
            result['job_id'] = str(job.id)
            return Response(result)
        except Exception as e:
            logger.error(f'[PredictApplicationStatus] Failed: {e}')
            return Response({'error': f'Prediction failed: {str(e)}'}, status=500)
