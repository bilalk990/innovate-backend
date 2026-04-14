"""
Email Service — Send notifications via SMTP
Supports: Gmail, SendGrid, AWS SES, or any SMTP provider
"""
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from django.conf import settings
from datetime import datetime

logger = logging.getLogger('innovaite')


def send_email(to_email: str, subject: str, html_content: str, text_content: str = None):
    """
    Send an email using configured SMTP settings.
    Falls back to console logging if SMTP is not configured.
    """
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = settings.EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject

        # Add text and HTML parts
        if text_content:
            msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))

        # Send via SMTP
        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            if settings.EMAIL_USE_TLS:
                server.starttls()
            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"[Email] Sent to {to_email}: {subject}")
        return True
    except Exception as e:
        logger.warning(f"[Email] Failed to send to {to_email}: {str(e)} | Subject: {subject}")
        return False


def send_interview_scheduled_email(candidate_email: str, candidate_name: str, interview_data: dict):
    """Send interview scheduled notification to candidate."""
    subject = f"Interview Scheduled: {interview_data['title']}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; background: #6366f1; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
            .details {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .footer {{ text-align: center; color: #6b7280; font-size: 12px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Interview Scheduled</h1>
            </div>
            <div class="content">
                <p>Hi <strong>{candidate_name}</strong>,</p>
                <p>Your interview has been scheduled on <strong>InnovAIte Interview Guardian</strong>.</p>
                
                <div class="details">
                    <h3>📋 Interview Details</h3>
                    <p><strong>Title:</strong> {interview_data['title']}</p>
                    <p><strong>Job Role:</strong> {interview_data.get('job_title', 'Not specified')}</p>
                    <p><strong>Scheduled At:</strong> {interview_data['scheduled_at']}</p>
                    <p><strong>Duration:</strong> {interview_data['duration_minutes']} minutes</p>
                    <p><strong>Room ID:</strong> {interview_data['room_id']}</p>
                </div>

                <p>Please join the interview room 5 minutes before the scheduled time.</p>
                
                <a href="{settings.FRONTEND_URL}/interview/room/{interview_data['room_id']}" class="button">
                    🎙️ Join Interview Room
                </a>

                <p><strong>Tips for Success:</strong></p>
                <ul>
                    <li>Test your camera and microphone beforehand</li>
                    <li>Ensure stable internet connection</li>
                    <li>Stay in the interview tab (AI proctoring is active)</li>
                    <li>Be clear and specific in your answers</li>
                </ul>

                <p>Good luck! 🚀</p>
            </div>
            <div class="footer">
                <p>© 2024 InnovAIte Interview Guardian | AI-Powered Recruitment Platform</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_content = f"""
    Hi {candidate_name},
    
    Your interview has been scheduled on InnovAIte Interview Guardian.
    
    Interview Details:
    - Title: {interview_data['title']}
    - Job Role: {interview_data.get('job_title', 'Not specified')}
    - Scheduled At: {interview_data['scheduled_at']}
    - Duration: {interview_data['duration_minutes']} minutes
    - Room ID: {interview_data['room_id']}
    
    Join Link: {settings.FRONTEND_URL}/interview/room/{interview_data['room_id']}
    
    Good luck!
    
    © 2024 InnovAIte Interview Guardian
    """
    
    return send_email(candidate_email, subject, html_content, text_content)


def send_interview_reminder_email(candidate_email: str, candidate_name: str, interview_data: dict):
    """Send interview reminder 1 hour before scheduled time."""
    subject = f"Reminder: Interview Starting Soon - {interview_data['title']}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%); color: white; padding: 30px; text-align: center; border-radius: 10px;">
                <h1>⏰ Interview Starting Soon!</h1>
            </div>
            <div style="background: #f9fafb; padding: 30px; border-radius: 10px; margin-top: 20px;">
                <p>Hi <strong>{candidate_name}</strong>,</p>
                <p>This is a reminder that your interview is starting in <strong>1 hour</strong>.</p>
                
                <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h3>📋 Interview: {interview_data['title']}</h3>
                    <p><strong>Time:</strong> {interview_data['scheduled_at']}</p>
                    <p><strong>Duration:</strong> {interview_data['duration_minutes']} minutes</p>
                </div>

                <a href="{settings.FRONTEND_URL}/interview/room/{interview_data['room_id']}" 
                   style="display: inline-block; background: #ef4444; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0;">
                    🎙️ Join Now
                </a>

                <p><strong>Pre-Interview Checklist:</strong></p>
                <ul>
                    <li>✅ Camera and microphone working</li>
                    <li>✅ Stable internet connection</li>
                    <li>✅ Quiet environment</li>
                    <li>✅ Resume reviewed</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(candidate_email, subject, html_content)


def send_evaluation_ready_email(candidate_email: str, candidate_name: str, evaluation_data: dict):
    """Notify candidate that their evaluation is ready."""
    subject = "Your Interview Evaluation is Ready"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 30px; text-align: center; border-radius: 10px;">
                <h1>🧠 Evaluation Complete</h1>
            </div>
            <div style="background: #f9fafb; padding: 30px; border-radius: 10px; margin-top: 20px;">
                <p>Hi <strong>{candidate_name}</strong>,</p>
                <p>Your interview evaluation has been completed by our AI-powered XAI engine.</p>
                
                <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
                    <h2 style="color: #6366f1; font-size: 3rem; margin: 0;">{evaluation_data['overall_score']}/100</h2>
                    <p style="color: #6b7280;">Overall Score</p>
                    <p style="background: #10b981; color: white; display: inline-block; padding: 8px 20px; border-radius: 20px; margin-top: 10px;">
                        {evaluation_data['recommendation'].replace('_', ' ').upper()}
                    </p>
                </div>

                <a href="{settings.FRONTEND_URL}/candidate/evaluations/{evaluation_data['id']}" 
                   style="display: inline-block; background: #6366f1; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0;">
                    📊 View Full Report
                </a>

                <p>The report includes detailed XAI explanations, skill analysis, and personalized feedback.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(candidate_email, subject, html_content)


def send_ai_quota_alert_email(admin_email: str, admin_name: str, alert_type: str, alert: dict):
    """Send AI quota warning or exhaustion alert to an admin."""
    is_exhausted = alert_type == 'AI_QUOTA_EXHAUSTED'
    color = '#dc2626' if is_exhausted else '#d97706'
    icon = '🔴' if is_exhausted else '⚠️'
    action_text = (
        'AI features are currently <strong>unavailable</strong> until quota resets at midnight UTC. '
        'Visit <a href="https://aistudio.google.com" style="color:#6366f1;">Google AI Studio</a> to upgrade your plan.'
        if is_exhausted else
        'AI features are still working but <strong>approaching the daily limit</strong>. '
        'Consider monitoring usage at <a href="https://aistudio.google.com" style="color:#6366f1;">Google AI Studio</a>.'
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: {color}; color: white; padding: 28px 30px; border-radius: 10px 10px 0 0; text-align: center;">
                <div style="font-size: 40px; margin-bottom: 8px;">{icon}</div>
                <h1 style="margin: 0; font-size: 22px;">{alert['title']}</h1>
                <p style="margin: 6px 0 0; opacity: 0.85; font-size: 13px;">InnovAIte Interview Guardian — System Alert</p>
            </div>
            <div style="background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; border: 1px solid #e5e7eb; border-top: none;">
                <p>Hi <strong>{admin_name}</strong>,</p>
                <p>{alert['message']}</p>
                <div style="background: white; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 6px; margin: 20px 0;">
                    <strong>Action Required:</strong><br>
                    <span style="font-size: 13px; color: #555;">{action_text}</span>
                </div>
                <a href="{settings.FRONTEND_URL}/admin/dashboard"
                   style="display: inline-block; background: #111; color: white; padding: 12px 28px; text-decoration: none; border-radius: 6px; margin-top: 10px; font-weight: bold;">
                    Go to Admin Dashboard
                </a>
                <p style="margin-top: 28px; font-size: 11px; color: #9ca3af;">
                    This is an automated system alert from InnovAIte Interview Guardian.<br>
                    © 2024 InnovAIte Interview Guardian
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return send_email(admin_email, alert['email_subject'], html_content)


def send_application_status_email(candidate_email: str, candidate_name: str, job_title: str, new_status: str, message: str):
    """Notify candidate that their application status has changed."""
    STATUS_CONFIG = {
        'shortlisted':   ('#6366f1', '🎉 You\'ve Been Shortlisted!'),
        'offer_sent':    ('#10b981', '🎊 Offer Letter Sent!'),
        'hired':         ('#059669', '🚀 Welcome Aboard!'),
        'rejected':      ('#6b7280', 'Application Update'),
        'reviewed':      ('#3b82f6', 'Application Reviewed'),
        'interview_scheduled': ('#f59e0b', '📅 Interview Scheduled'),
    }
    color, heading = STATUS_CONFIG.get(new_status, ('#6366f1', 'Application Update'))
    subject = f'{heading} — {job_title}'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: {color}; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="margin: 0; font-size: 22px;">{heading}</h1>
                <p style="margin: 8px 0 0; opacity: 0.85; font-size: 13px;">{job_title}</p>
            </div>
            <div style="background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; border: 1px solid #e5e7eb; border-top: none;">
                <p>Hi <strong>{candidate_name}</strong>,</p>
                <p>{message}</p>
                <div style="background: white; border-left: 4px solid {color}; padding: 16px 20px; border-radius: 6px; margin: 20px 0;">
                    <strong>Position:</strong> {job_title}<br>
                    <strong>New Status:</strong> <span style="color: {color}; font-weight: bold;">{new_status.replace('_', ' ').upper()}</span>
                </div>
                <a href="{settings.FRONTEND_URL}/candidate/applications"
                   style="display: inline-block; background: {color}; color: white; padding: 12px 28px; text-decoration: none; border-radius: 6px; margin-top: 10px; font-weight: bold;">
                    View My Applications
                </a>
                <p style="margin-top: 28px; font-size: 11px; color: #9ca3af;">
                    © 2024 InnovAIte Interview Guardian
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return send_email(candidate_email, subject, html_content)


def send_recruiter_notification_email(recruiter_email: str, recruiter_name: str, message: str, link: str = None):
    """Send generic notification to recruiter."""
    subject = "InnovAIte Notification"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #6366f1; color: white; padding: 20px; text-align: center; border-radius: 10px;">
                <h2>🎯 InnovAIte Notification</h2>
            </div>
            <div style="background: #f9fafb; padding: 30px; border-radius: 10px; margin-top: 20px;">
                <p>Hi <strong>{recruiter_name}</strong>,</p>
                <p>{message}</p>
                {f'<a href="{link}" style="display: inline-block; background: #6366f1; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0;">View Details</a>' if link else ''}
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email(recruiter_email, subject, html_content)
