"""
Multi-Factor Authentication (MFA) Service
Supports TOTP (Time-based One-Time Password) via Google Authenticator, Authy, etc.
"""
import pyotp
import qrcode
import io
import base64
from datetime import datetime


def generate_mfa_secret():
    """Generate a new MFA secret for a user."""
    return pyotp.random_base32()


def generate_qr_code(user_email: str, secret: str, issuer_name: str = "InnovAIte"):
    """
    Generate QR code for MFA setup.
    Returns base64-encoded PNG image.
    """
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user_email,
        issuer_name=issuer_name
    )
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_base64}"


def verify_mfa_token(secret: str, token: str) -> bool:
    """
    Verify a 6-digit MFA token.
    
    Args:
        secret: User's MFA secret
        token: 6-digit code from authenticator app
    
    Returns:
        True if valid, False otherwise
    """
    if not secret or not token:
        return False
    
    try:
        totp = pyotp.TOTP(secret)
        # Allow 1 time step before/after for clock drift
        return totp.verify(token, valid_window=1)
    except Exception:
        return False


def generate_backup_codes(count: int = 8) -> list:
    """
    Generate backup codes for MFA recovery.
    Returns list of 8-character alphanumeric codes.
    """
    import secrets
    import string
    
    codes = []
    alphabet = string.ascii_uppercase + string.digits
    for _ in range(count):
        code = ''.join(secrets.choice(alphabet) for _ in range(8))
        # Format as XXXX-XXXX for readability
        formatted = f"{code[:4]}-{code[4:]}"
        codes.append(formatted)
    
    return codes
