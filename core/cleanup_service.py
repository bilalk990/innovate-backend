"""
Cleanup service for orphaned files and expired data
"""
import os
import logging
from datetime import datetime, timedelta
from django.conf import settings

logger = logging.getLogger('innovaite')


def cleanup_orphaned_resumes():
    """
    Clean up resume files for deleted users
    Should be run periodically (e.g., daily cron job)
    """
    from resumes.models import Resume
    from accounts.models import User
    
    deleted_count = 0
    
    try:
        # Get all resumes
        resumes = Resume.objects.all()
        
        for resume in resumes:
            # Check if user still exists
            user_exists = User.objects(id=resume.candidate_id).first()
            
            if not user_exists:
                # Delete file
                if os.path.exists(resume.file_path):
                    try:
                        os.remove(resume.file_path)
                        logger.info(f'[Cleanup] Deleted orphaned resume file: {resume.file_path}')
                    except Exception as e:
                        logger.error(f'[Cleanup] Failed to delete file {resume.file_path}: {e}')
                
                # Delete database record
                resume.delete()
                deleted_count += 1
        
        logger.info(f'[Cleanup] Cleaned up {deleted_count} orphaned resumes')
        return deleted_count
        
    except Exception as e:
        logger.error(f'[Cleanup] Resume cleanup failed: {e}')
        return 0


def cleanup_expired_tokens():
    """
    Clean up expired verification tokens and interview tokens
    """
    from accounts.models import User
    from interviews.models import Interview
    
    cleaned_count = 0
    
    try:
        # Clean expired email verification tokens (older than 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        users = User.objects(
            email_verified=False,
            verification_token_created__lt=cutoff
        )
        
        for user in users:
            user.verification_token = ''
            user.verification_token_created = None
            user.save()
            cleaned_count += 1
        
        logger.info(f'[Cleanup] Cleaned {cleaned_count} expired verification tokens')
        return cleaned_count
        
    except Exception as e:
        logger.error(f'[Cleanup] Token cleanup failed: {e}')
        return 0


def cleanup_old_interviews():
    """
    Archive or clean up very old completed interviews (older than 1 year)
    """
    from interviews.models import Interview
    
    try:
        cutoff = datetime.utcnow() - timedelta(days=365)
        old_interviews = Interview.objects(
            status='completed',
            scheduled_at__lt=cutoff
        )
        
        count = old_interviews.count()
        logger.info(f'[Cleanup] Found {count} old interviews (1+ year old)')
        
        # For now, just log. In production, you might want to:
        # - Archive to cold storage
        # - Delete after confirmation
        # - Keep metadata but remove responses
        
        return count
        
    except Exception as e:
        logger.error(f'[Cleanup] Interview cleanup failed: {e}')
        return 0


def run_all_cleanup_tasks():
    """Run all cleanup tasks"""
    logger.info('[Cleanup] Starting cleanup tasks...')
    
    resume_count = cleanup_orphaned_resumes()
    token_count = cleanup_expired_tokens()
    interview_count = cleanup_old_interviews()
    
    logger.info(f'[Cleanup] Completed: {resume_count} resumes, {token_count} tokens, {interview_count} old interviews')
    
    return {
        'resumes_cleaned': resume_count,
        'tokens_cleaned': token_count,
        'old_interviews': interview_count
    }
