"""
Email verification system for user registration
"""
import secrets
from datetime import datetime, timedelta
from django.conf import settings
from core.email_service import send_email


def generate_verification_token():
    """Generate a secure verification token"""
    return secrets.token_urlsafe(32)


def send_verification_email(user_email, user_name, token):
    """
    Send email verification link to user
    
    Args:
        user_email: User's email address
        user_name: User's name
        token: Verification token
    """
    verification_link = f"{settings.FRONTEND_URL}/verify-email?token={token}"
    
    subject = "Verify Your InnovAIte Account"
    body = f"""
    Hi {user_name},
    
    Welcome to InnovAIte Interview Guardian! 
    
    Please verify your email address by clicking the link below:
    
    {verification_link}
    
    This link will expire in 24 hours.
    
    If you didn't create an account, please ignore this email.
    
    Best regards,
    InnovAIte Team
    """
    
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #6366f1;">Welcome to InnovAIte! 🚀</h2>
            <p>Hi {user_name},</p>
            <p>Thank you for signing up! Please verify your email address to activate your account.</p>
            <div style="text-align: center; margin: 30px 0;">
                <a href="{verification_link}" 
                   style="background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); 
                          color: white; 
                          padding: 12px 30px; 
                          text-decoration: none; 
                          border-radius: 8px; 
                          display: inline-block;
                          font-weight: bold;">
                    Verify Email Address
                </a>
            </div>
            <p style="color: #666; font-size: 14px;">
                This link will expire in 24 hours. If you didn't create an account, please ignore this email.
            </p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <p style="color: #999; font-size: 12px;">
                InnovAIte Interview Guardian<br>
                AI-Powered Interview Platform
            </p>
        </div>
    </body>
    </html>
    """
    
    send_email(
        to_email=user_email,
        subject=subject,
        body=body,
        html_body=html_body
    )


def verify_token_expiry(created_at, hours=24):
    """
    Check if verification token has expired
    
    Args:
        created_at: Token creation datetime
        hours: Expiry time in hours
    
    Returns:
        bool: True if expired, False otherwise
    """
    expiry_time = created_at + timedelta(hours=hours)
    return datetime.utcnow() > expiry_time
