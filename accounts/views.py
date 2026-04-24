"""
Accounts app views — register, login, profile management
JWT-based authentication against MongoDB users
"""
import jwt
import bcrypt
import uuid
import mongoengine
from datetime import datetime, timedelta
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from accounts.models import User, SystemConfiguration
from core.audit_logger import AuditLog, log_login, log_audit
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import logging

logger = logging.getLogger(__name__)


def generate_token(user):
    """Generate a JWT access token for the given user."""
    payload = {
        'user_id': str(user.id),
        'email': user.email,
        'role': user.role,
        'token_version': getattr(user, 'token_version', 0),  # For token invalidation
        'exp': datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS),
        'iat': datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


class RegisterView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []  # Disable authentication for register endpoint

    def post(self, request):
        # Removed print statement - never log registration data (contains passwords)
        try:
            data = request.data
            required = ['name', 'email', 'password', 'role']
            for field in required:
                if not data.get(field):
                    return Response({'error': f'{field} is required.'}, status=400)

            if data['role'] not in ['candidate', 'recruiter']:
                return Response({'error': f"Invalid role. Must be 'candidate' or 'recruiter'."}, status=400)

            if len(data['password']) < 8:
                return Response({'error': 'Password must be at least 8 characters.'}, status=400)

            email_clean = data['email'].lower().strip()
            if not email_clean or '@' not in email_clean:
                return Response({'error': 'Invalid email address.'}, status=400)

            if User.objects(email=email_clean).first():
                return Response({'error': 'Email already registered.'}, status=400)

            hashed = bcrypt.hashpw(data['password'].encode(), bcrypt.gensalt()).decode()
            user = User(
                name=data['name'].strip()[:100],
                email=email_clean,
                password=hashed,
                role=data['role'],
                phone=data.get('phone', ''),
                company=data.get('company', ''),
                company_name=data.get('company', ''),
                bio=data.get('bio', ''),
            )
            user.save()

            log_audit(user, 'register', status='success', details={'role': user.role}, request=request)
            token = generate_token(user)
            return Response({'token': token, 'user': user.to_dict()}, status=201)
        except mongoengine.ValidationError as e:
            return Response({'error': f'Validation error: {str(e)}'}, status=400)
        except Exception as e:
            return Response({'error': 'Registration failed. Please try again.'}, status=500)

class AuditLogListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        logs = AuditLog.objects.all().order_by('-timestamp')[:100]
        return Response([log.to_dict() for log in logs])


class LoginView(APIView):
    permission_classes = [AllowAny]
    throttle_scope = 'login'  # #45 — 10 attempts per minute max
    authentication_classes = []  # Disable authentication for login endpoint

    def post(self, request):
        email = request.data.get('email', '').lower().strip()
        password = request.data.get('password', '')
        mfa_token = request.data.get('mfa_token', '')  # Optional MFA token
        
        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=400)

        user = User.objects(email=email).first()
        if not user:
            return Response({'error': 'Invalid credentials.'}, status=401)

        try:
            if not bcrypt.checkpw(password.encode(), user.password.encode()):
                return Response({'error': 'Invalid credentials.'}, status=401)
        except Exception:
            return Response({'error': 'Invalid credentials.'}, status=401)

        if not user.is_active:
            return Response({'error': 'Account is disabled. Contact admin.'}, status=403)

        # Check if MFA is enabled
        if user.mfa_enabled:
            if not mfa_token:
                # First step: password verified, now need MFA token
                return Response({
                    'mfa_required': True,
                    'message': 'Please provide your MFA token.',
                    'user_id': str(user.id)  # Temporary identifier for next step
                }, status=200)
            
            # Verify MFA token
            from core.mfa_service import verify_mfa_token
            if not verify_mfa_token(user.mfa_secret, mfa_token):
                log_audit(user, 'login_mfa_failed', status='failure', request=request)
                return Response({'error': 'Invalid MFA token.'}, status=401)

        # Audit Log
        log_login(user, request, status='success')

        token = generate_token(user)
        return Response({'token': token, 'user': user.to_dict()})

class GoogleLoginView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []  # Disable authentication for Google login endpoint

    def post(self, request):
        token = request.data.get('token')
        role = request.data.get('role', 'candidate')
        if not token:
            return Response({'error': 'Google token is required.'}, status=400)

        # Security: Only allow candidate/recruiter roles via OAuth, never admin
        if role not in ['candidate', 'recruiter']:
            return Response({'error': 'Invalid role. Only candidate or recruiter allowed.'}, status=400)

        try:
            # Verify the token with Google (clock_skew_in_seconds handles minor server clock drift)
            # Increased to 300 seconds (5 minutes) to handle system clock synchronization issues
            idinfo = id_token.verify_oauth2_token(
                token, google_requests.Request(), settings.GOOGLE_CLIENT_ID,
                clock_skew_in_seconds=300
            )
            
            email = idinfo['email'].lower().strip()
            name = idinfo.get('name', 'Google User')
            profile_pic = idinfo.get('picture', '')

            user = User.objects(email=email).first()
            if not user:
                # Auto-register if not exists - only as candidate/recruiter
                logger.info(f"[GoogleLogin] Creating new user: {email} with role: {role}")
                user = User(
                    name=name,
                    email=email,
                    role=role,
                    profile_pic=profile_pic,
                    password=bcrypt.hashpw(str(uuid.uuid4()).encode(), bcrypt.gensalt()).decode()
                )
                user.save()
            else:
                logger.info(f"[GoogleLogin] Existing user login: {email} with role: {user.role}")
            
            if not user.is_active:
                return Response({'error': 'Account is disabled.'}, status=403)

            # Audit Log
            log_audit(user, 'google_login', status='success', request=request)

            access_token = generate_token(user)
            user_dict = user.to_dict()
            
            # CRITICAL: Ensure role is in response
            logger.info(f"[GoogleLogin] Response for {email}: role={user_dict.get('role')}, token_length={len(access_token)}")
            
            return Response({'token': access_token, 'user': user_dict})

        except ValueError as e:
            error_msg = str(e).upper()
            if 'TOKEN USED TOO EARLY' in error_msg or 'CLOCK' in error_msg:
                return Response({
                    'error': 'Clock synchronization issue detected. Please sync your system clock and try again.',
                    'details': str(e)
                }, status=400)
            logger.error(f"[GoogleLogin] Token verification failed: {str(e)}")
            return Response({'error': f'Invalid Google token: {str(e)}'}, status=400)
        except Exception as e:
            logger.error(f"[GoogleLogin] Unexpected error: {str(e)}")
            return Response({'error': f'Google authentication error: {str(e)}'}, status=500)


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(request.user.to_dict())

    def patch(self, request):
        user = request.user
        logger.info(f"[Profile] PATCH request for {user.email}. Fields: {list(request.data.keys())}")
        
        allowed = [
            'name', 'phone', 'company', 'bio', 'profile_pic',
            'company_name', 'company_values',
            'headline', 'detailed_skills', 'work_history', 'education_history',
            'projects', 'certifications', 'languages', 'achievements',
            'location', 'timezone',
        ]
        
        updated_fields = []
        for field in allowed:
            if field in request.data:
                setattr(user, field, request.data[field])
                updated_fields.append(field)
        
        # Handle profile completeness separately if needed
        if 'is_profile_complete' in request.data:
            user.is_profile_complete = bool(request.data['is_profile_complete'])
            updated_fields.append('is_profile_complete')
            
        user.updated_at = datetime.utcnow()
        user.save()
        
        logger.info(f"[Profile] Successfully updated {user.email}. Fields changed: {updated_fields}. Skills count: {len(getattr(user, 'detailed_skills', []))}")
        return Response(user.to_dict())



class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        current = request.data.get('current_password', '')
        new_pass = request.data.get('new_password', '')
        if not current or not new_pass:
            return Response({'error': 'Both current_password and new_password are required.'}, status=400)
        if len(new_pass) < 6:
            return Response({'error': 'New password must be at least 6 characters.'}, status=400)

        user = request.user
        try:
            if not bcrypt.checkpw(current.encode(), user.password.encode()):
                return Response({'error': 'Current password is incorrect.'}, status=400)
        except Exception:
            return Response({'error': 'Password verification failed.'}, status=400)

        hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
        user.password = hashed
        user.token_version = getattr(user, 'token_version', 0) + 1  # Invalidate all existing tokens
        user.updated_at = datetime.utcnow()
        user.save()
        
        # Log password change
        log_audit(user, 'password_changed', status='success', request=request)
        
        return Response({'message': 'Password updated successfully. Please log in again.'})


class UsersListView(APIView):
    """Admin & Recruiter — list users (admin sees all, recruiter sees candidates)"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role not in ['admin', 'recruiter']:
            return Response({'error': 'Forbidden.'}, status=403)

        role_filter = request.query_params.get('role')
        qs = User.objects.all()

        # Recruiters can only see candidates
        if user.role == 'recruiter':
            qs = qs.filter(role='candidate')
        elif role_filter:
            qs = qs.filter(role=role_filter)

        # CRITICAL FIX: Add pagination to prevent loading all users
        try:
            limit = min(int(request.query_params.get('limit', 50)), 200)
            offset = int(request.query_params.get('offset', 0))
        except (ValueError, TypeError):
            limit, offset = 50, 0

        total = qs.count()
        users = qs.skip(offset).limit(limit)
        
        return Response({
            'results': [u.to_dict() for u in users],
            'total': total,
            'limit': limit,
            'offset': offset,
        })


class BulkUserImportView(APIView):
    """POST /api/auth/users/bulk-import/ — CSV bulk user import (#59)"""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)

        import csv
        import io

        file = request.FILES.get('file')
        if not file:
            return Response({'error': 'CSV file is required.'}, status=400)

        if not file.name.endswith('.csv'):
            return Response({'error': 'Only CSV files are supported.'}, status=400)

        content = file.read().decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(content))

        created, skipped, errors = [], [], []

        for i, row in enumerate(reader, start=2):  # row 1 is header
            name = row.get('name', '').strip()
            email = row.get('email', '').strip().lower()
            role = row.get('role', 'candidate').strip().lower()
            password = row.get('password', '').strip()

            if not name or not email:
                errors.append(f'Row {i}: name and email are required.')
                continue

            if role not in ['candidate', 'recruiter']:
                errors.append(f'Row {i}: invalid role "{role}".')
                continue

            if User.objects(email=email).first():
                skipped.append(email)
                continue

            if not password:
                import secrets as _s
                password = _s.token_urlsafe(10)

            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            try:
                user = User(
                    name=name[:100],
                    email=email,
                    password=hashed,
                    role=role,
                    company=row.get('company', ''),
                )
                user.save()
                created.append(email)
            except Exception as e:
                errors.append(f'Row {i}: {str(e)}')

        return Response({
            'created': len(created),
            'skipped': len(skipped),
            'errors': errors,
            'created_emails': created,
        }, status=201)

class UserDetailView(APIView):
    """Admin can get/update/deactivate any user"""
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id):
        try:
            user = User.objects.get(id=user_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'User not found.'}, status=404)

        # Admin sees anyone; recruiter may view candidates only
        if request.user.role == 'recruiter':
            if user.role != 'candidate':
                return Response({'error': 'Forbidden.'}, status=403)
        elif request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        return Response(user.to_dict())

    def patch(self, request, user_id):
        if request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)
        try:
            user = User.objects.get(id=user_id)
        except (mongoengine.DoesNotExist, mongoengine.ValidationError):
            return Response({'error': 'User not found.'}, status=404)

        old_role = user.role
        old_active = user.is_active

        if 'is_active' in request.data:
            # Support toggle functionality
            if request.data['is_active'] == 'toggle':
                user.is_active = not user.is_active
            else:
                user.is_active = bool(request.data['is_active'])
        if 'role' in request.data:
            if request.data['role'] not in ['candidate', 'recruiter', 'admin']:
                return Response({'error': 'Invalid role.'}, status=400)
            user.role = request.data['role']
        user.updated_at = datetime.utcnow()
        user.save()

        # CRITICAL FIX: Audit log for role changes and account status changes
        if old_role != user.role:
            log_audit(
                request.user,
                'user_role_changed',
                status='success',
                details={'target_user': str(user.id), 'old_role': old_role, 'new_role': user.role},
                request=request
            )
        if old_active != user.is_active:
            log_audit(
                request.user,
                'user_status_changed',
                status='success',
                details={'target_user': str(user.id), 'is_active': user.is_active},
                request=request
            )

        return Response(user.to_dict())


class SystemSettingsView(APIView):
    """Admin — manage system configurations"""
    permission_classes = [IsAuthenticated]

    def _get_config(self):
        config = SystemConfiguration.objects.first()
        if not config:
            config = SystemConfiguration()
            config.save()
        return config

    def get(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        config = self._get_config()
        return Response(config.to_dict())

    def patch(self, request):
        if request.user.role != 'admin':
            return Response({'error': 'Admin access required.'}, status=403)
        config = self._get_config()
        
        mapping = {
            'siteName': 'site_name',
            'maintenanceMode': 'maintenance_mode',
            'allowRegistration': 'allow_registration',
            'emailNotifications': 'email_notifications',
            'mfaRequired': 'mfa_required',
        }
        
        for json_key, model_key in mapping.items():
            if json_key in request.data:
                setattr(config, model_key, request.data[json_key])
        
        config.updated_at = datetime.utcnow()
        config.save()
        return Response(config.to_dict())


class ProfileImprovementsView(APIView):
    """
    GET /api/auth/profile-suggestions/
    AI-powered suggestions to improve candidate profile for better job matching.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        try:
            from core.openai_client import suggest_profile_improvements
        except ImportError:
            return Response({'error': 'AI service unavailable.'}, status=503)

        # Build profile_data
        profile_data = {
            'skills': getattr(user, 'detailed_skills', []) or [],
            'work_history': getattr(user, 'work_history', []) or [],
            'bio': getattr(user, 'bio', '') or '',
            'headline': getattr(user, 'headline', '') or '',
            'education': getattr(user, 'education', '') or '',
            'resume_uploaded': False,
        }

        # Check if resume exists
        try:
            from resumes.models import Resume
            profile_data['resume_uploaded'] = Resume.objects.filter(candidate_id=str(user.id)).count() > 0
        except Exception:
            pass

        # Fetch evaluation history for context
        evaluation_history = []
        try:
            from evaluations.models import Evaluation
            evals = Evaluation.objects.filter(candidate_id=str(user.id)).order_by('-created_at')[:5]
            for ev in evals:
                evaluation_history.append({
                    'overall_score': getattr(ev, 'overall_score', 0),
                    'strengths': getattr(ev, 'strengths', []),
                    'weaknesses': getattr(ev, 'weaknesses', []),
                })
        except Exception:
            pass

        try:
            result = suggest_profile_improvements(
                profile_data=profile_data,
                evaluation_history=evaluation_history,
                user_id=str(user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[ProfileImprovements] Failed: {e}')
            return Response({'error': f'Suggestion failed: {str(e)}'}, status=500)


class SalaryNegotiationView(APIView):
    """POST /api/auth/salary-negotiation/ — AI salary negotiation strategy."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import suggest_salary_negotiation

        job_title = request.data.get('job_title', '').strip()
        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)

        user = request.user
        skills = getattr(user, 'detailed_skills', []) or []
        work_history = getattr(user, 'work_history', []) or []
        experience_years = request.data.get('experience_years') or len(work_history)
        location = request.data.get('location') or getattr(user, 'location', 'United States') or 'United States'
        current_offer = request.data.get('current_offer')
        company_size = request.data.get('company_size', 'medium')

        # Parse numeric values safely
        try:
            experience_years = int(experience_years)
        except (TypeError, ValueError):
            experience_years = max(len(work_history), 0)

        try:
            current_offer = float(current_offer) if current_offer else None
        except (TypeError, ValueError):
            current_offer = None

        try:
            result = suggest_salary_negotiation(
                job_title=job_title,
                skills=skills,
                experience_years=experience_years,
                location=location,
                current_offer=current_offer,
                company_size=company_size,
                user_id=str(user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[SalaryNegotiation] Failed: {e}')
            return Response({'error': f'Salary analysis failed: {str(e)}'}, status=500)


class CareerPathView(APIView):
    """GET /api/auth/career-path/ — AI career path recommendations."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from core.openai_client import recommend_career_paths

        user = request.user
        profile_data = {
            'skills': getattr(user, 'detailed_skills', []) or [],
            'work_history': getattr(user, 'work_history', []) or [],
            'bio': getattr(user, 'bio', '') or '',
            'headline': getattr(user, 'headline', '') or '',
        }

        evaluation_history = []
        try:
            from evaluations.models import Evaluation
            evals = Evaluation.objects.filter(candidate_id=str(user.id)).order_by('-created_at')[:5]
            for ev in evals:
                evaluation_history.append({
                    'overall_score': getattr(ev, 'overall_score', 0),
                    'job_title': '',
                    'recommendation': getattr(ev, 'recommendation', ''),
                })
        except Exception:
            pass

        try:
            result = recommend_career_paths(
                profile_data=profile_data,
                evaluation_history=evaluation_history,
                user_id=str(user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[CareerPath] Failed: {e}')
            return Response({'error': f'Career path recommendation failed: {str(e)}'}, status=500)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW PREP LAB VIEWS
# ═══════════════════════════════════════════════════════════════════════════════

class InterviewPrepPlanView(APIView):
    """POST /api/auth/interview-prep/plan/ — Generate AI interview prep roadmap."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_interview_prep_plan
        role = request.data.get('role', '').strip()
        stacks = request.data.get('stacks', [])
        level = request.data.get('level', 'mid').strip()

        if not role:
            return Response({'error': 'role is required.'}, status=400)
        if not isinstance(stacks, list):
            stacks = [stacks] if stacks else []
        if level not in ['junior', 'mid', 'senior']:
            level = 'mid'

        try:
            result = generate_interview_prep_plan(
                role=role, stacks=stacks, level=level, user_id=str(request.user.id)
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PrepPlan] Failed: {e}')
            return Response({'error': f'Plan generation failed: {str(e)}'}, status=500)


class InterviewPrepQuizView(APIView):
    """POST /api/auth/interview-prep/quiz/ — Generate stack-specific MCQ quiz."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_interview_mcq_quiz
        role = request.data.get('role', '').strip()
        stacks = request.data.get('stacks', [])
        level = request.data.get('level', 'mid').strip()
        count = int(request.data.get('count', 10))

        if not role:
            return Response({'error': 'role is required.'}, status=400)
        if not isinstance(stacks, list):
            stacks = [stacks] if stacks else []
        count = max(5, min(count, 15))

        try:
            result = generate_interview_mcq_quiz(
                role=role, stacks=stacks, level=level, count=count, user_id=str(request.user.id)
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PrepQuiz] Failed: {e}')
            return Response({'error': f'Quiz generation failed: {str(e)}'}, status=500)


class InterviewPrepReportView(APIView):
    """POST /api/auth/interview-prep/report/ — Generate final readiness report."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_prep_final_report
        role = request.data.get('role', '').strip()
        stacks = request.data.get('stacks', [])
        level = request.data.get('level', 'mid').strip()
        quiz_score = int(request.data.get('quiz_score', 0))
        total_questions = int(request.data.get('total_questions', 10))
        tab_switches = int(request.data.get('tab_switches', 0))
        wrong_topics = request.data.get('wrong_topics', [])
        time_per_q_avg = float(request.data.get('time_per_q_avg', 30))

        if not role:
            return Response({'error': 'role is required.'}, status=400)
        if not isinstance(stacks, list):
            stacks = [stacks] if stacks else []
        if not isinstance(wrong_topics, list):
            wrong_topics = []

        try:
            result = generate_prep_final_report(
                role=role, stacks=stacks, level=level,
                quiz_score=quiz_score, total_questions=total_questions,
                tab_switches=tab_switches, wrong_topics=wrong_topics,
                time_per_q_avg=time_per_q_avg, user_id=str(request.user.id)
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PrepReport] Failed: {e}')
            return Response({'error': f'Report generation failed: {str(e)}'}, status=500)


# ═══════════════════════════════════════════════════════════════════════════════
# HR AI POWER TOOLS — 7 Professional Features for Recruiters
# ═══════════════════════════════════════════════════════════════════════════════

class CandidateComparisonView(APIView):
    """POST /api/auth/hr/compare-candidates/ — AI side-by-side candidate comparison."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import compare_candidates
        from evaluations.models import Evaluation
        from resumes.models import Resume

        candidate_ids = request.data.get('candidate_ids', [])
        job_title = request.data.get('job_title', '').strip()
        blind_mode = bool(request.data.get('blind_mode', False))

        if len(candidate_ids) < 2:
            return Response({'error': 'At least 2 candidates required.'}, status=400)
        if len(candidate_ids) > 5:
            return Response({'error': 'Maximum 5 candidates.'}, status=400)
        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)

        candidates_data = []
        for cid in candidate_ids:
            try:
                candidate = User.objects.get(id=cid)
                eval_obj = Evaluation.objects(candidate_id=str(cid)).order_by('-created_at').first()
                resume = Resume.objects(candidate_id=str(cid), is_active=True).order_by('-uploaded_at').first()
                resume_data = resume.parsed_data if resume else {}
                candidates_data.append({
                    'name': candidate.name if not blind_mode else f'Candidate {chr(65 + len(candidates_data))}',
                    'overall_score': eval_obj.overall_score if eval_obj else 0,
                    'skills': resume_data.get('skills', []),
                    'experience_years': resume_data.get('total_experience_years', 0),
                    'education': str(resume_data.get('education', [{}])[0].get('degree', 'N/A')) if resume_data.get('education') else 'N/A',
                    'strengths': eval_obj.strengths if eval_obj else [],
                    'weaknesses': eval_obj.weaknesses if eval_obj else [],
                    'summary': eval_obj.summary if eval_obj else '',
                    'recommendation': eval_obj.recommendation if eval_obj else 'N/A',
                })
            except Exception as ex:
                logger.warning(f'[Compare] Could not load candidate {cid}: {ex}')
                continue

        if len(candidates_data) < 2:
            return Response({'error': 'Could not load enough candidate data.'}, status=400)

        try:
            result = compare_candidates(candidates_data, job_title, blind_mode, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[Compare] Failed: {e}')
            return Response({'error': f'Comparison failed: {str(e)}'}, status=500)


class BiasDetectorView(APIView):
    """POST /api/auth/hr/bias-detector/ — Detect and rewrite biased job descriptions."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import detect_jd_bias
        jd_text = request.data.get('jd_text', '').strip()
        if not jd_text:
            return Response({'error': 'jd_text is required.'}, status=400)
        if len(jd_text) < 50:
            return Response({'error': 'JD too short — provide full job description.'}, status=400)
        try:
            result = detect_jd_bias(jd_text, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[BiasDetector] Failed: {e}')
            return Response({'error': f'Bias detection failed: {str(e)}'}, status=500)


class ReferenceCheckView(APIView):
    """POST /api/auth/hr/reference-check/ — Generate targeted AI reference check questions."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import generate_reference_questions
        from resumes.models import Resume
        from evaluations.models import Evaluation

        candidate_id = request.data.get('candidate_id', '').strip()
        job_title = request.data.get('job_title', '').strip()

        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)

        resume_data = {}
        eval_summary = ''
        if candidate_id:
            try:
                resume = Resume.objects(candidate_id=candidate_id, is_active=True).order_by('-uploaded_at').first()
                resume_data = resume.parsed_data if resume else {}
                eval_obj = Evaluation.objects(candidate_id=candidate_id).order_by('-created_at').first()
                eval_summary = eval_obj.summary if eval_obj else ''
            except Exception:
                pass

        try:
            result = generate_reference_questions(resume_data, job_title, eval_summary, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[ReferenceCheck] Failed: {e}')
            return Response({'error': f'Reference question generation failed: {str(e)}'}, status=500)


class OfferPredictorView(APIView):
    """POST /api/auth/hr/offer-predictor/ — Predict offer acceptance probability."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import predict_offer_acceptance
        from resumes.models import Resume
        from evaluations.models import Evaluation

        candidate_id = request.data.get('candidate_id', '').strip()
        offer_data = {
            'base_salary': request.data.get('base_salary', 'Not specified'),
            'total_package': request.data.get('total_package', 'Not specified'),
            'benefits': request.data.get('benefits', 'Standard benefits'),
            'remote_policy': request.data.get('remote_policy', 'On-site'),
            'start_date': request.data.get('start_date', 'Flexible'),
            'role_level': request.data.get('role_level', 'Mid'),
        }

        candidate_data = {
            'label': request.data.get('candidate_name', 'Candidate'),
            'current_salary': request.data.get('current_salary', 'Unknown'),
            'expected_salary': request.data.get('expected_salary', 'Unknown'),
            'enthusiasm_score': int(request.data.get('enthusiasm_score', 7)),
            'has_competing_offers': bool(request.data.get('has_competing_offers', False)),
            'notes': request.data.get('notes', ''),
        }

        if candidate_id:
            try:
                resume = Resume.objects(candidate_id=candidate_id, is_active=True).order_by('-uploaded_at').first()
                if resume:
                    rd = resume.parsed_data or {}
                    candidate_data['skills'] = rd.get('skills', [])
                    candidate_data['experience_years'] = rd.get('total_experience_years', 0)
                    candidate_data['location'] = rd.get('location', 'Unknown')
                eval_obj = Evaluation.objects(candidate_id=candidate_id).order_by('-created_at').first()
                if eval_obj:
                    candidate_data['notes'] += f' Evaluation: {eval_obj.summary}'
            except Exception:
                pass

        try:
            result = predict_offer_acceptance(candidate_data, offer_data, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[OfferPredictor] Failed: {e}')
            return Response({'error': f'Offer prediction failed: {str(e)}'}, status=500)


class FunnelAnalyzerView(APIView):
    """POST /api/auth/hr/funnel-analyzer/ — AI hiring funnel drop-off analysis."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import analyze_hiring_funnel
        job_title = request.data.get('job_title', '').strip()
        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)

        funnel_stats = {
            'applications': int(request.data.get('applications', 0)),
            'screened': int(request.data.get('screened', 0)),
            'phone_screened': int(request.data.get('phone_screened', 0)),
            'assessed': int(request.data.get('assessed', 0)),
            'final_interview': int(request.data.get('final_interview', 0)),
            'offers_made': int(request.data.get('offers_made', 0)),
            'offers_accepted': int(request.data.get('offers_accepted', 0)),
            'hired': int(request.data.get('hired', 0)),
            'time_to_fill': int(request.data.get('time_to_fill', 0)),
            'cost_per_hire': int(request.data.get('cost_per_hire', 0)),
        }

        try:
            result = analyze_hiring_funnel(funnel_stats, job_title, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[FunnelAnalyzer] Failed: {e}')
            return Response({'error': f'Funnel analysis failed: {str(e)}'}, status=500)


class TeamFitView(APIView):
    """POST /api/auth/hr/team-fit/ — Predict candidate fit with existing team."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import predict_team_fit
        from resumes.models import Resume
        from evaluations.models import Evaluation

        candidate_id = request.data.get('candidate_id', '').strip()
        team_description = {
            'size': int(request.data.get('team_size', 5)),
            'skills': [s.strip() for s in request.data.get('team_skills', '').split(',') if s.strip()],
            'gaps': [g.strip() for g in request.data.get('team_gaps', '').split(',') if g.strip()],
            'work_style': request.data.get('work_style', 'Collaborative'),
            'culture': request.data.get('team_culture', 'Professional'),
            'challenges': request.data.get('team_challenges', 'None specified'),
            'management_style': request.data.get('management_style', 'Flat'),
        }

        candidate_profile = {'skills': [], 'experience_years': 0, 'strengths': [], 'weaknesses': [], 'summary': ''}
        if candidate_id:
            try:
                resume = Resume.objects(candidate_id=candidate_id, is_active=True).order_by('-uploaded_at').first()
                if resume:
                    rd = resume.parsed_data or {}
                    candidate_profile['skills'] = rd.get('skills', [])
                    candidate_profile['experience_years'] = rd.get('total_experience_years', 0)
                eval_obj = Evaluation.objects(candidate_id=candidate_id).order_by('-created_at').first()
                if eval_obj:
                    candidate_profile['strengths'] = eval_obj.strengths or []
                    candidate_profile['weaknesses'] = eval_obj.weaknesses or []
                    candidate_profile['summary'] = eval_obj.summary or ''
            except Exception:
                pass

        try:
            result = predict_team_fit(team_description, candidate_profile, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[TeamFit] Failed: {e}')
            return Response({'error': f'Team fit analysis failed: {str(e)}'}, status=500)


class InterviewerCoachView(APIView):
    """POST /api/auth/hr/interviewer-coach/ — AI coaching for recruiters on interview technique."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiters only.'}, status=403)
        from core.openai_client import coach_interviewer
        from interviews.models import Interview

        interview_id = request.data.get('interview_id', '').strip()
        if not interview_id:
            return Response({'error': 'interview_id is required.'}, status=400)

        try:
            interview = Interview.objects.get(id=interview_id)
        except Exception:
            return Response({'error': 'Interview not found.'}, status=404)

        if interview.recruiter_id != str(request.user.id) and request.user.role != 'admin':
            return Response({'error': 'Forbidden.'}, status=403)

        transcript = interview.full_transcript or ' '.join(str(v) for v in (interview.candidate_responses or {}).values())
        questions = [q.text for q in (interview.questions or [])]
        interviewer_name = request.user.name or 'Recruiter'

        try:
            result = coach_interviewer(transcript, questions, interviewer_name, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[InterviewerCoach] Failed: {e}')
            return Response({'error': f'Coaching analysis failed: {str(e)}'}, status=500)
