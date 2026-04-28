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
            return Response({'error': f'Registration failed: {str(e)}'}, status=500)

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
            try:
                # Primary verification with configured Client ID
                idinfo = id_token.verify_oauth2_token(
                    token, google_requests.Request(), settings.GOOGLE_CLIENT_ID,
                    clock_skew_in_seconds=300
                )
            except ValueError as ve:
                if 'Wrong audience' in str(ve):
                    logger.warning(f"[GoogleLogin] Audience mismatch. Configured: {settings.GOOGLE_CLIENT_ID}. Falling back to permissive verification.")
                    # Fallback: verify WITHOUT explicit audience to see what's in the token
                    idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), clock_skew_in_seconds=300)
                    
                    # Safety check: ensure the audience starts with the same project number
                    token_aud = idinfo.get('aud', '')
                    project_number = settings.GOOGLE_CLIENT_ID.split('-')[0]
                    if token_aud.startswith(project_number):
                        logger.info(f"[GoogleLogin] Permissive match successful for audience: {token_aud}")
                    else:
                        logger.error(f"[GoogleLogin] Audience prefix mismatch. Found: {token_aud}, Expected prefix: {project_number}")
                        raise ve
                else:
                    raise ve

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
        # Support both jd_text and text for robustness
        jd_text = (request.data.get('jd_text') or request.data.get('text') or '').strip()
        if not jd_text:
            return Response({'error': 'Job description text (jd_text) is required.'}, status=400)

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
        # Map flat payload from frontend to backend structure
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
                    notes_prefix = str(candidate_data.get('notes', ''))
                    candidate_data['notes'] = notes_prefix + f' Evaluation: {eval_obj.summary}'
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

        def safe_int(val, default=0):
            try:
                if val is None or str(val).strip() == '':
                    return default
                return int(float(val))
            except (ValueError, TypeError):
                return default

        funnel_stats = {
            'applications': safe_int(request.data.get('applications', request.data.get('total_applicants'))),
            'screened': safe_int(request.data.get('screened', request.data.get('shortlisted'))),
            'phone_screened': safe_int(request.data.get('phone_screened')),
            'assessed': safe_int(request.data.get('assessed', request.data.get('interviewed'))),
            'final_interview': safe_int(request.data.get('final_interview')),
            'offers_made': safe_int(request.data.get('offers_made', request.data.get('offered'))),
            'offers_accepted': safe_int(request.data.get('offers_accepted')),
            'hired': safe_int(request.data.get('hired')),
            'time_to_fill': safe_int(request.data.get('time_to_fill')),
            'cost_per_hire': safe_int(request.data.get('cost_per_hire')),

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

        transcript = interview.full_transcript or ''
        if not transcript and hasattr(interview, 'candidate_responses'):
             transcript = ' '.join(str(v) for v in (getattr(interview, 'candidate_responses', {}) or {}).values())
        
        if not transcript or len(transcript.strip()) < 10:
             return Response({'error': 'Interview transcript is too short for analysis. Complete the interview first.'}, status=400)
             
        # Handle potential list of strings or list of objects for questions
        raw_questions = getattr(interview, 'questions', []) or []
        questions = []
        for q in raw_questions:
            if isinstance(q, str):
                questions.append(q)
            elif hasattr(q, 'text'):
                questions.append(q.text)
            elif isinstance(q, dict):
                questions.append(q.get('question', q.get('text', '')))

        interviewer_name = request.user.name or 'Recruiter'

        try:
            result = coach_interviewer(transcript, questions, interviewer_name, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[InterviewerCoach] Failed: {e}')
            return Response({'error': f'Coaching analysis failed: {str(e)}'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET 3 VIEWS — Anxiety Coach, Bulk Screener, Email Campaign, Sentiment
# ─────────────────────────────────────────────────────────────────────────────

class AnxietyCoachView(APIView):
    """POST /auth/anxiety-coach/ — Personalized pre-interview anxiety coaching."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_anxiety_coaching
        role = request.data.get('role', 'Software Engineer')
        experience_level = request.data.get('experience_level', 'Mid-level')
        concerns = request.data.get('concerns', '')
        try:
            result = generate_anxiety_coaching(role, experience_level, concerns, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[AnxietyCoach] Failed: {e}')
            return Response({'error': f'Coaching generation failed: {str(e)}'}, status=500)


class BulkResumeScreenerView(APIView):
    """POST /auth/hr/bulk-resume-screen/ — Screen + rank multiple candidates against JD."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import screen_resumes_bulk
        from resumes.models import Resume

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        jd_text = request.data.get('jd_text', '')
        job_title = request.data.get('job_title', 'Open Position')
        candidate_ids = request.data.get('candidate_ids', [])

        if not jd_text.strip():
            return Response({'error': 'Job description is required.'}, status=400)
        if not candidate_ids:
            return Response({'error': 'Provide at least 1 candidate_id.'}, status=400)

        resumes_data = []
        for cid in candidate_ids[:15]:
            try:
                resume = Resume.objects.filter(candidate_id=str(cid), is_active=True).first()
                if resume and resume.parsed_data:
                    pd = resume.parsed_data
                    resumes_data.append({
                        'name': pd.get('name', f'Candidate {cid[:6]}'),
                        'skills': pd.get('skills', []),
                        'experience_years': pd.get('total_experience_years', 0),
                        'education_level': pd.get('education', [{}])[0].get('degree', 'Unknown') if pd.get('education') else 'Unknown',
                        'summary': pd.get('summary', ''),
                        'candidate_id': str(cid),
                    })
                else:
                    resumes_data.append({'name': f'Candidate {str(cid)[:6]}', 'skills': [], 'experience_years': 0, 'education_level': 'Unknown', 'summary': 'No resume uploaded', 'candidate_id': str(cid)})
            except Exception:
                resumes_data.append({'name': f'Candidate {str(cid)[:6]}', 'skills': [], 'experience_years': 0, 'education_level': 'Unknown', 'summary': 'Resume unavailable', 'candidate_id': str(cid)})

        try:
            result = screen_resumes_bulk(resumes_data, jd_text, job_title, user_id=str(request.user.id))
            # Attach candidate_ids back to ranked results
            for rc in result.get('ranked_candidates', []):
                idx = rc.get('candidate_index', 0)
                if idx < len(resumes_data):
                    rc['candidate_id'] = resumes_data[idx].get('candidate_id', '')
            return Response(result)
        except Exception as e:
            logger.error(f'[BulkScreener] Failed: {e}')
            return Response({'error': f'Bulk screening failed: {str(e)}'}, status=500)


class EmailCampaignView(APIView):
    """POST /auth/hr/email-campaign/ — Generate personalized bulk email campaign."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_email_campaign

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        email_type = request.data.get('email_type', 'follow_up')
        job_title = request.data.get('job_title', 'Open Position')
        company_name = request.data.get('company_name', '')
        custom_message = request.data.get('custom_message', '')
        candidates_raw = request.data.get('candidates', [])

        if not candidates_raw:
            return Response({'error': 'Provide at least 1 candidate.'}, status=400)

        # Enrich candidate data from DB if ids provided
        candidates = []
        for c in candidates_raw[:20]:
            candidates.append({'name': c.get('name', 'Candidate'), 'note': c.get('note', '')})

        try:
            result = generate_email_campaign(email_type, candidates, job_title, company_name, custom_message, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[EmailCampaign] Failed: {e}')
            return Response({'error': f'Email campaign generation failed: {str(e)}'}, status=500)


class SentimentTrackerView(APIView):
    """POST /auth/hr/sentiment-tracker/ — Analyze candidate sentiment + engagement."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import analyze_candidate_sentiment
        from accounts.models import User

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        candidate_id = request.data.get('candidate_id', request.data.get('id', ''))
        job_title = request.data.get('job_title', request.data.get('title', 'Open Position'))
        interactions = request.data.get('interactions', request.data.get('transcript', []))


        candidate_name = 'Candidate'
        if candidate_id:
            try:
                cand = User.objects.get(id=candidate_id)
                candidate_name = cand.name or cand.email
            except Exception:
                pass

        try:
            result = analyze_candidate_sentiment(interactions, candidate_name, job_title, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[SentimentTracker] Failed: {e}')
            return Response({'error': f'Sentiment analysis failed: {str(e)}'}, status=500)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET 4 VIEWS — DNA Profiler, Talent Rediscovery, Interview Quality Intel
# ─────────────────────────────────────────────────────────────────────────────

class CandidateDNAView(APIView):
    """POST /auth/hr/candidate-dna/ — Deep personality + behavioral profiling."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import profile_candidate_dna
        from resumes.models import Resume

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        candidate_id = request.data.get('candidate_id', '')
        if not candidate_id:
            return Response({'error': 'candidate_id is required.'}, status=400)

        # Build candidate data from DB
        candidate_data = {'name': 'Candidate', 'skills': [], 'experience_years': 0}
        interview_data = {'transcript_sample': ''}
        evaluation_data = {'overall_score': 0, 'strengths': [], 'gaps': []}

        try:
            cand = User.objects.get(id=candidate_id)
            candidate_data['name'] = cand.name or cand.email
            candidate_data['skills'] = list(getattr(cand, 'skills', []) or [])
        except Exception:
            pass

        try:
            resume = Resume.objects.filter(user_id=str(candidate_id), is_active=True).first()
            if resume and resume.parsed_data:
                pd = resume.parsed_data
                candidate_data['skills'] = pd.get('skills', candidate_data['skills'])
                candidate_data['experience_years'] = pd.get('total_experience_years', 0)
        except Exception:
            pass

        try:
            from interviews.models import Interview
            from evaluations.models import Evaluation
            ivs = Interview.objects.filter(candidate_id=str(candidate_id)).order_by('-created_at')[:3]
            transcripts = []
            for iv in ivs:
                if hasattr(iv, 'full_transcript') and isinstance(iv.full_transcript, list):
                    for entry in (iv.full_transcript or [])[:5]:
                        if entry.get('role') == 'candidate':
                            transcripts.append(entry.get('content', ''))
            interview_data['transcript_sample'] = ' '.join(transcripts)[:500]

            evals = Evaluation.objects.filter(candidate_id=str(candidate_id)).order_by('-created_at')[:1]
            if evals:
                ev = evals[0]
                evaluation_data['overall_score'] = getattr(ev, 'overall_score', 0) or 0
                if hasattr(ev, 'ai_summary') and ev.ai_summary:
                    evaluation_data['strengths'] = ev.ai_summary.get('strengths', [])
                    evaluation_data['gaps'] = ev.ai_summary.get('gaps', [])
        except Exception:
            pass

        try:
            result = profile_candidate_dna(candidate_data, interview_data, evaluation_data, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[CandidateDNA] Failed: {e}')
            return Response({'error': f'DNA profiling failed: {str(e)}'}, status=500)


class TalentRediscoveryView(APIView):
    """POST /auth/hr/talent-rediscovery/ — Find past candidates for new openings."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import rediscover_talent
        from resumes.models import Resume
        from evaluations.models import Evaluation

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        new_job_title = (request.data.get('job_title') or request.data.get('role') or '').strip()
        new_jd = (request.data.get('jd_text') or request.data.get('text') or request.data.get('jd') or '').strip()
        if not new_job_title:
            return Response({'error': 'job_title is required.'}, status=400)


        # Gather past candidates who were evaluated (not currently active)
        past_candidates = []
        try:
            evals = Evaluation.objects.filter(recruiter_id=str(request.user.id)).order_by('-created_at')[:30]
            seen_ids = set()
            for ev in evals:
                cid = str(ev.candidate_id)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                cname = 'Candidate'
                skills = []
                try:
                    cand = User.objects.get(id=cid)
                    cname = cand.name or cand.email
                except Exception:
                    pass
                try:
                    resume = Resume.objects.filter(candidate_id=str(cid), is_active=True).first()
                    if resume and resume.parsed_data:
                        skills = resume.parsed_data.get('skills', [])
                except Exception:
                    pass

                prev_role = getattr(ev, 'job_title', '') or ''
                score = getattr(ev, 'overall_score', 0) or 0
                decision = getattr(ev, 'hire_decision', '') or ''
                rejection = 'Role mismatch' if not decision or decision.lower() in ['no', 'reject', 'pass'] else 'Considered'

                past_candidates.append({
                    'candidate_id': cid,
                    'name': cname,
                    'skills': skills,
                    'prev_applied_role': prev_role,
                    'rejection_reason': rejection,
                    'score': score,
                    'experience_years': 0,
                })
        except Exception as e:
            logger.warning(f'[TalentRediscovery] DB fetch error: {e}')

        if not past_candidates:
            # Use manually provided candidate_ids if no evals found
            provided_ids = request.data.get('candidate_ids', [])
            for cid in provided_ids[:20]:
                cname = 'Candidate'
                skills = []
                try:
                    cand = User.objects.get(id=cid)
                    cname = cand.name or cand.email
                    skills = list(getattr(cand, 'skills', []) or [])
                except Exception:
                    pass
                past_candidates.append({'candidate_id': str(cid), 'name': cname, 'skills': skills, 'prev_applied_role': 'Previous Role', 'rejection_reason': 'Role mismatch', 'score': 0, 'experience_years': 0})

        if not past_candidates:
            return Response({'error': 'No past candidate data found. Complete some evaluations first or provide candidate_ids.'}, status=400)

        try:
            result = rediscover_talent(past_candidates, new_job_title, new_jd, user_id=str(request.user.id))
            # Attach candidate_ids back to results
            for rd in result.get('rediscovered', []):
                idx = rd.get('candidate_index', 0)
                if idx < len(past_candidates):
                    rd['candidate_id'] = past_candidates[idx].get('candidate_id', '')
            return Response(result)
        except Exception as e:
            logger.error(f'[TalentRediscovery] Failed: {e}')
            return Response({'error': f'Talent rediscovery failed: {str(e)}'}, status=500)


class InterviewQualityIntelligenceView(APIView):
    """POST /auth/hr/interview-quality/ — Analyze interview patterns across all past interviews."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import analyze_interview_quality_intelligence
        from interviews.models import Interview
        from evaluations.models import Evaluation

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        interviews_summary = []
        try:
            ivs = Interview.objects.filter(recruiter_id=str(request.user.id)).order_by('-created_at')[:20]
            for iv in ivs:
                questions = []
                if hasattr(iv, 'questions') and iv.questions:
                    for q in (iv.questions or [])[:6]:
                        if isinstance(q, dict):
                            questions.append(q.get('question', q.get('text', '')))
                        elif isinstance(q, str):
                            questions.append(q)

                eval_score = 0
                was_hired = False
                try:
                    ev = Evaluation.objects.filter(interview_id=str(iv.id)).first()
                    if ev:
                        eval_score = getattr(ev, 'overall_score', 0) or 0
                        hire_dec = getattr(ev, 'hire_decision', '') or ''
                        was_hired = hire_dec.lower() in ['yes', 'hire', 'strong hire']
                except Exception:
                    pass

                duration = 0
                if hasattr(iv, 'started_at') and hasattr(iv, 'ended_at') and iv.started_at and iv.ended_at:
                    try:
                        delta = iv.ended_at - iv.started_at
                        duration = int(delta.total_seconds() / 60)
                    except Exception:
                        pass

                interviews_summary.append({
                    'title': iv.title or 'Interview',
                    'questions': questions,
                    'was_hired': was_hired,
                    'eval_score': eval_score,
                    'interviewer': getattr(request.user, 'name', 'Recruiter'),
                    'duration_mins': duration,
                })
        except Exception as e:
            logger.warning(f'[InterviewQualityIntel] DB fetch: {e}')

        try:
            result = analyze_interview_quality_intelligence(interviews_summary, user_id=str(request.user.id))
            return Response(result)
        except Exception as e:
            logger.error(f'[InterviewQualityIntel] Failed: {e}')
            return Response({'error': f'Interview quality analysis failed: {str(e)}'}, status=500)


class HRDocumentGeneratorView(APIView):
    """POST /auth/hr/generate-document/ — Generate any professional HR document."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_hr_document

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        document_type = (request.data.get('document_type') or request.data.get('type') or '').strip()
        company_name = (request.data.get('company_name') or request.data.get('company') or '').strip()
        employee_name = (request.data.get('employee_name') or request.data.get('name') or '').strip()
        employee_designation = request.data.get('employee_designation', request.data.get('role', '')).strip()

        employee_department = request.data.get('employee_department', '').strip()
        employee_id = request.data.get('employee_id', '').strip()
        additional_details = request.data.get('additional_details', '').strip()
        hr_name = request.data.get('hr_name', getattr(request.user, 'name', 'HR Manager')).strip()
        hr_designation = request.data.get('hr_designation', 'HR Manager').strip()
        country = request.data.get('country', 'Pakistan').strip()

        if not document_type:
            return Response({'error': 'document_type is required.'}, status=400)
        if not employee_name:
            return Response({'error': 'employee_name is required.'}, status=400)
        if not company_name:
            return Response({'error': 'company_name is required.'}, status=400)

        try:
            result = generate_hr_document(
                document_type=document_type,
                company_name=company_name,
                employee_name=employee_name,
                employee_designation=employee_designation or 'Employee',
                employee_department=employee_department or 'General',
                employee_id=employee_id,
                additional_details=additional_details,
                hr_name=hr_name,
                hr_designation=hr_designation,
                country=country,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[HRDocumentGenerator] Failed: {e}')
            return Response({'error': f'Document generation failed: {str(e)}'}, status=500)


class EmployeeHandbookBuilderView(APIView):
    """POST /auth/hr/handbook-builder/ — Generate a complete employee handbook."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_employee_handbook

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        company_name = (request.data.get('company_name') or request.data.get('company') or '').strip()
        industry = (request.data.get('industry') or request.data.get('sector') or '').strip()

        company_size = request.data.get('company_size', '').strip()
        country = request.data.get('country', 'Pakistan').strip()
        culture_type = request.data.get('culture_type', 'Professional').strip()
        work_model = request.data.get('work_model', 'On-site').strip()
        selected_sections = request.data.get('selected_sections', [])
        additional_notes = request.data.get('additional_notes', '').strip()

        if not company_name:
            return Response({'error': 'company_name is required.'}, status=400)
        if not industry:
            return Response({'error': 'industry is required.'}, status=400)

        try:
            result = generate_employee_handbook(
                company_name=company_name,
                industry=industry,
                company_size=company_size or '50-200 employees',
                country=country,
                culture_type=culture_type,
                work_model=work_model,
                selected_sections=selected_sections,
                additional_notes=additional_notes,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[HandbookBuilder] Failed: {e}')
            return Response({'error': f'Handbook generation failed: {str(e)}'}, status=500)


class LDRoadmapView(APIView):
    """POST /auth/hr/ld-roadmap/ — Generate personalized L&D training roadmap."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_ld_roadmap

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        employee_name = (request.data.get('employee_name') or request.data.get('name') or '').strip()
        current_role = (request.data.get('current_role') or request.data.get('role') or '').strip()
        target_role = (request.data.get('target_role') or request.data.get('target', '')).strip()
        current_skills_raw = request.data.get('current_skills', request.data.get('skills', ''))
        
        def safe_int(val, default=0):
            try:
                if val is None or str(val).strip() == '': return default
                return int(float(val))
            except: return default
            
        experience_years = safe_int(request.data.get('experience_years'))
        learning_style = request.data.get('learning_style', 'Blended').strip()
        budget_range = request.data.get('budget_range', '$200-500').strip()
        timeline_months = safe_int(request.data.get('timeline_months'), 6)

        industry = request.data.get('industry', '').strip()
        company_name = request.data.get('company_name', '').strip()

        if not employee_name:
            return Response({'error': 'employee_name is required.'}, status=400)
        if not current_role:
            return Response({'error': 'current_role is required.'}, status=400)
        if not target_role:
            return Response({'error': 'target_role is required.'}, status=400)

        # Parse skills — accept comma-separated string or list
        if isinstance(current_skills_raw, list):
            current_skills = [s.strip() for s in current_skills_raw if s.strip()]
        else:
            current_skills = [s.strip() for s in str(current_skills_raw).split(',') if s.strip()]

        try:
            result = generate_ld_roadmap(
                employee_name=employee_name,
                current_role=current_role,
                target_role=target_role,
                current_skills=current_skills,
                experience_years=experience_years,
                learning_style=learning_style,
                budget_range=budget_range,
                timeline_months=min(timeline_months, 24),
                industry=industry or 'General',
                company_name=company_name,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[LDRoadmap] Failed: {e}')
            return Response({'error': f'Roadmap generation failed: {str(e)}'}, status=500)


class PolicyComplianceView(APIView):
    """POST /auth/hr/policy-compliance/ — AI compliance check of HR policy text."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import check_policy_compliance

        if request.user.role not in ['recruiter', 'admin']:
            return Response({'error': 'Recruiter access required.'}, status=403)

        policy_text = request.data.get('policy_text', '').strip()
        country = request.data.get('country', 'Pakistan').strip()
        industry = request.data.get('industry', 'General').strip()
        company_size = request.data.get('company_size', '50-200 employees').strip()
        policy_type = request.data.get('policy_type', '').strip()

        if not policy_text or len(policy_text) < 50:
            return Response({'error': 'Please provide the full policy text (minimum 50 characters).'}, status=400)

        try:
            result = check_policy_compliance(
                policy_text=policy_text,
                country=country,
                industry=industry,
                company_size=company_size,
                policy_type=policy_type,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PolicyCompliance] Failed: {e}')
            return Response({'error': f'Compliance check failed: {str(e)}'}, status=500)


# ──────────────────────────────────────────────────────────────
# Feature Set 7 — Candidate Career AI Tools
# ──────────────────────────────────────────────────────────────

class CoverLetterView(APIView):
    """POST /auth/cover-letter/ — Generate a tailored cover letter for any job."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_cover_letter
        from resumes.models import Resume

        if request.user.role != 'candidate':
            return Response({'error': 'Candidate access required.'}, status=403)

        job_title    = request.data.get('job_title', '').strip()
        company_name = request.data.get('company_name', '').strip()
        jd_text      = request.data.get('jd_text', '').strip()
        tone         = request.data.get('tone', 'Professional').strip()

        if not job_title:
            return Response({'error': 'job_title is required.'}, status=400)
        if not company_name:
            return Response({'error': 'company_name is required.'}, status=400)
        if not jd_text:
            return Response({'error': 'jd_text (job description) is required.'}, status=400)

        # Pull candidate details from profile & resume
        candidate_name = getattr(request.user, 'name', 'Candidate')
        candidate_skills    = []
        experience_summary  = ''
        try:
            resume = Resume.objects.filter(candidate_id=str(request.user.id), is_active=True).first()
            if resume:
                raw_skills = getattr(resume, 'skills', []) or []
                candidate_skills = [s.strip() for s in raw_skills if isinstance(s, str) and s.strip()]
                exps = getattr(resume, 'experience', []) or []
                parts = []
                for ex in exps[:3]:
                    if isinstance(ex, dict):
                        parts.append(f"{ex.get('title', '')} at {ex.get('company', '')} ({ex.get('duration', '')})")
                    elif isinstance(ex, str):
                        parts.append(ex)
                experience_summary = '; '.join(parts)
        except Exception as e:
            logger.warning(f'[CoverLetter] Resume fetch failed: {e}')

        try:
            result = generate_cover_letter(
                job_title=job_title,
                company_name=company_name,
                jd_text=jd_text,
                candidate_name=candidate_name,
                candidate_skills=candidate_skills,
                experience_summary=experience_summary,
                tone=tone,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[CoverLetter] Failed: {e}')
            return Response({'error': f'Cover letter generation failed: {str(e)}'}, status=500)


class JobMatchAnalyzerView(APIView):
    """POST /auth/job-match/ — Deep-analyze any external JD against candidate profile."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import analyze_job_match
        from resumes.models import Resume

        if request.user.role != 'candidate':
            return Response({'error': 'Candidate access required.'}, status=403)

        jd_text     = request.data.get('jd_text', '').strip()
        target_role = request.data.get('target_role', '').strip()

        if not jd_text or len(jd_text) < 30:
            return Response({'error': 'Please paste the full job description (minimum 30 characters).'}, status=400)

        # Pull candidate data from resume/profile
        candidate_skills    = []
        experience_summary  = ''
        education           = ''
        try:
            resume = Resume.objects.filter(candidate_id=str(request.user.id), is_active=True).first()
            if resume:
                raw_skills = getattr(resume, 'skills', []) or []
                candidate_skills = [s.strip() for s in raw_skills if isinstance(s, str) and s.strip()]
                exps = getattr(resume, 'experience', []) or []
                parts = []
                for ex in exps[:4]:
                    if isinstance(ex, dict):
                        parts.append(f"{ex.get('title', '')} at {ex.get('company', '')} ({ex.get('duration', '')}): {ex.get('description', '')}")
                    elif isinstance(ex, str):
                        parts.append(ex)
                experience_summary = ' | '.join(parts)
                edus = getattr(resume, 'education', []) or []
                edu_parts = []
                for ed in edus[:2]:
                    if isinstance(ed, dict):
                        edu_parts.append(f"{ed.get('degree', '')} from {ed.get('institution', '')}")
                    elif isinstance(ed, str):
                        edu_parts.append(ed)
                education = ', '.join(edu_parts)
        except Exception as e:
            logger.warning(f'[JobMatch] Resume fetch failed: {e}')

        # Allow overrides from request (user can manually provide if no resume)
        if request.data.get('candidate_skills'):
            raw = request.data.get('candidate_skills')
            if isinstance(raw, list):
                candidate_skills = raw
            else:
                candidate_skills = [s.strip() for s in str(raw).split(',') if s.strip()]

        if request.data.get('experience_summary'):
            experience_summary = request.data.get('experience_summary')
        if request.data.get('education'):
            education = request.data.get('education')

        try:
            result = analyze_job_match(
                jd_text=jd_text,
                candidate_skills=candidate_skills,
                experience_summary=experience_summary,
                education=education,
                target_role=target_role,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[JobMatch] Failed: {e}')
            return Response({'error': f'Job match analysis failed: {str(e)}'}, status=500)


class SelfIntroCoachView(APIView):
    """POST /auth/self-intro/ — Generate 3 versions of Tell Me About Yourself."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import generate_self_intro
        from resumes.models import Resume

        if request.user.role != 'candidate':
            return Response({'error': 'Candidate access required.'}, status=403)

        target_role     = request.data.get('target_role', '').strip()
        key_achievement = request.data.get('key_achievement', '').strip()

        if not target_role:
            return Response({'error': 'target_role is required.'}, status=400)

        # Pull candidate details from profile & resume
        candidate_name  = getattr(request.user, 'name', 'Candidate')
        current_role    = ''
        experience_years = 0
        key_skills      = []
        try:
            resume = Resume.objects.filter(candidate_id=str(request.user.id), is_active=True).first()
            if resume:
                raw_skills = getattr(resume, 'skills', []) or []
                key_skills = [s.strip() for s in raw_skills[:8] if isinstance(s, str) and s.strip()]
                exps = getattr(resume, 'experience', []) or []
                if exps:
                    first_exp = exps[0]
                    if isinstance(first_exp, dict):
                        current_role = first_exp.get('title', '')
                    experience_years = len(exps)
        except Exception as e:
            logger.warning(f'[SelfIntro] Resume fetch failed: {e}')

        # Allow manual overrides
        if request.data.get('current_role'):
            current_role = request.data.get('current_role').strip()
        if request.data.get('experience_years') is not None:
            try:
                experience_years = int(float(request.data.get('experience_years') or 0))
            except:
                experience_years = 0

        if request.data.get('key_skills'):
            raw = request.data.get('key_skills')
            if isinstance(raw, list):
                key_skills = raw
            else:
                key_skills = [s.strip() for s in str(raw).split(',') if s.strip()]

        try:
            result = generate_self_intro(
                candidate_name=candidate_name,
                current_role=current_role or target_role,
                target_role=target_role,
                experience_years=experience_years,
                key_skills=key_skills,
                key_achievement=key_achievement,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[SelfIntro] Failed: {e}')
            return Response({'error': f'Intro generation failed: {str(e)}'}, status=500)


class PortfolioSuggesterView(APIView):
    """POST /auth/portfolio-advisor/ — AI-powered portfolio project suggestions."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.openai_client import suggest_portfolio_projects
        from resumes.models import Resume

        if request.user.role != 'candidate':
            return Response({'error': 'Candidate access required.'}, status=403)

        target_role      = request.data.get('target_role', '').strip()
        experience_level = request.data.get('experience_level', 'Mid-level').strip()
        industry         = request.data.get('industry', 'Technology').strip()

        if not target_role:
            return Response({'error': 'target_role is required.'}, status=400)

        # Auto-pull skills from resume
        current_skills = []
        try:
            resume = Resume.objects.filter(candidate_id=str(request.user.id), is_active=True).first()
            if resume:
                raw_skills = getattr(resume, 'skills', []) or []
                current_skills = [s.strip() for s in raw_skills if isinstance(s, str) and s.strip()]
        except Exception as e:
            logger.warning(f'[PortfolioSuggester] Resume fetch failed: {e}')

        # Allow manual skills override
        if request.data.get('current_skills'):
            raw = request.data.get('current_skills')
            if isinstance(raw, list):
                current_skills = raw
            else:
                current_skills = [s.strip() for s in str(raw).split(',') if s.strip()]

        try:
            result = suggest_portfolio_projects(
                target_role=target_role,
                current_skills=current_skills,
                experience_level=experience_level,
                industry=industry,
                user_id=str(request.user.id),
            )
            return Response(result)
        except Exception as e:
            logger.error(f'[PortfolioSuggester] Failed: {e}')
            return Response({'error': f'Portfolio suggestion failed: {str(e)}'}, status=500)
