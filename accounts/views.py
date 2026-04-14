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
                user = User(
                    name=name,
                    email=email,
                    role=role,
                    profile_pic=profile_pic,
                    password=bcrypt.hashpw(str(uuid.uuid4()).encode(), bcrypt.gensalt()).decode()
                )
                user.save()
            
            if not user.is_active:
                return Response({'error': 'Account is disabled.'}, status=403)

            # Audit Log
            log_audit(user, 'google_login', status='success', request=request)

            access_token = generate_token(user)
            return Response({'token': access_token, 'user': user.to_dict()})

        except ValueError as e:
            error_msg = str(e).upper()
            if 'TOKEN USED TOO EARLY' in error_msg or 'CLOCK' in error_msg:
                return Response({
                    'error': 'Clock synchronization issue detected. Please sync your system clock and try again.',
                    'details': str(e)
                }, status=400)
            return Response({'error': f'Invalid Google token: {str(e)}'}, status=400)
        except Exception as e:
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
