"""
MFA (Multi-Factor Authentication) Views
Setup, enable, disable, and verify MFA
"""
import bcrypt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from accounts.models import User
from core.mfa_service import (
    generate_mfa_secret,
    generate_qr_code,
    verify_mfa_token,
    generate_backup_codes
)
from core.audit_logger import log_audit


class MFASetupView(APIView):
    """Generate MFA secret and QR code for user to scan."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        
        if user.mfa_enabled:
            return Response({'error': 'MFA is already enabled.'}, status=400)
        
        # Generate new secret
        secret = generate_mfa_secret()
        qr_code = generate_qr_code(user.email, secret)
        
        # Store secret temporarily (not enabled yet)
        user.mfa_secret = secret
        user.save()
        
        log_audit(user, 'mfa_setup_initiated', request=request)
        
        return Response({
            'secret': secret,
            'qr_code': qr_code,
            'message': 'Scan the QR code with your authenticator app (Google Authenticator, Authy, etc.)'
        })


class MFAEnableView(APIView):
    """Enable MFA after verifying the first token."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        token = request.data.get('token', '')
        
        if not user.mfa_secret:
            return Response({'error': 'MFA setup not initiated. Call /mfa/setup first.'}, status=400)
        
        if user.mfa_enabled:
            return Response({'error': 'MFA is already enabled.'}, status=400)
        
        # Verify token
        if not verify_mfa_token(user.mfa_secret, token):
            return Response({'error': 'Invalid MFA token.'}, status=400)
        
        # Generate backup codes
        backup_codes = generate_backup_codes()
        
        # Hash backup codes before storing
        hashed_codes = [
            bcrypt.hashpw(code.encode(), bcrypt.gensalt()).decode()
            for code in backup_codes
        ]
        
        user.mfa_enabled = True
        user.mfa_backup_codes = hashed_codes
        user.save()
        
        log_audit(user, 'mfa_enabled', status='success', request=request)
        
        return Response({
            'message': 'MFA enabled successfully!',
            'backup_codes': backup_codes,
            'warning': 'Save these backup codes in a secure location. They can be used if you lose access to your authenticator app.'
        })


class MFADisableView(APIView):
    """Disable MFA (requires password confirmation)."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        password = request.data.get('password', '')
        
        if not user.mfa_enabled:
            return Response({'error': 'MFA is not enabled.'}, status=400)
        
        # Verify password
        try:
            if not bcrypt.checkpw(password.encode(), user.password.encode()):
                return Response({'error': 'Incorrect password.'}, status=400)
        except Exception:
            return Response({'error': 'Password verification failed.'}, status=400)
        
        # Disable MFA
        user.mfa_enabled = False
        user.mfa_secret = ''
        user.mfa_backup_codes = []
        user.save()
        
        log_audit(user, 'mfa_disabled', request=request)
        
        return Response({'message': 'MFA disabled successfully.'})


class MFAVerifyView(APIView):
    """Verify MFA token during login (called from login flow)."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        token = request.data.get('token', '')
        use_backup = request.data.get('use_backup', False)
        
        if not user.mfa_enabled:
            return Response({'error': 'MFA is not enabled for this account.'}, status=400)
        
        if use_backup:
            # Verify backup code
            for idx, hashed_code in enumerate(user.mfa_backup_codes):
                try:
                    if bcrypt.checkpw(token.encode(), hashed_code.encode()):
                        # Remove used backup code
                        user.mfa_backup_codes.pop(idx)
                        user.save()
                        
                        log_audit(user, 'mfa_verified_backup', request=request)
                        
                        return Response({
                            'verified': True,
                            'message': 'Backup code verified. This code has been consumed.',
                            'remaining_codes': len(user.mfa_backup_codes)
                        })
                except Exception:
                    continue
            
            return Response({'error': 'Invalid backup code.'}, status=400)
        else:
            # Verify TOTP token
            if verify_mfa_token(user.mfa_secret, token):
                log_audit(user, 'mfa_verified', request=request)
                return Response({'verified': True})
            else:
                log_audit(user, 'mfa_verification_failed', status='failure', request=request)
                return Response({'error': 'Invalid MFA token.'}, status=400)


class MFAStatusView(APIView):
    """Get MFA status for current user."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        return Response({
            'mfa_enabled': user.mfa_enabled,
            'backup_codes_remaining': len(user.mfa_backup_codes) if user.mfa_enabled else 0,
        })
