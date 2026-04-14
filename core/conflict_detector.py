"""
Interview Scheduling Conflict Detection
Prevents double-booking and validates time slots
"""
from datetime import datetime, timedelta
from django.utils import timezone as django_timezone
from interviews.models import Interview


def check_scheduling_conflicts(recruiter_id: str, candidate_id: str, 
                               scheduled_at: datetime, duration_minutes: int,
                               exclude_interview_id: str = None):
    """
    Check for scheduling conflicts for both recruiter and candidate.
    
    Args:
        exclude_interview_id: Optional interview ID to exclude from conflict check (for editing existing interviews)
    
    Returns:
        dict: {
            'has_conflict': bool,
            'conflicts': list of conflicting interviews,
            'message': str
        }
    """
    conflicts = []

    # CRITICAL FIX: Use timezone-aware datetimes consistently
    # Convert to timezone-aware if naive
    if scheduled_at.tzinfo is None:
        scheduled_at = django_timezone.make_aware(scheduled_at, django_timezone.utc)
    
    # Normalize to UTC
    scheduled_at = scheduled_at.astimezone(django_timezone.utc)

    # Calculate time window
    start_time = scheduled_at
    end_time = scheduled_at + timedelta(minutes=duration_minutes)
    
    # Check recruiter conflicts
    recruiter_interviews = Interview.objects(
        recruiter_id=recruiter_id,
        status__in=['scheduled', 'active']
    )
    
    # Exclude current interview if editing
    if exclude_interview_id:
        recruiter_interviews = recruiter_interviews.filter(id__ne=exclude_interview_id)
    
    for interview in recruiter_interviews:
        iv_start = interview.scheduled_at
        # Ensure timezone-aware
        if iv_start.tzinfo is None:
            iv_start = django_timezone.make_aware(iv_start, django_timezone.utc)
        iv_start = iv_start.astimezone(django_timezone.utc)
        iv_end = iv_start + timedelta(minutes=interview.duration_minutes)
        
        # Check for overlap
        if (start_time < iv_end and end_time > iv_start):
            conflicts.append({
                'type': 'recruiter',
                'interview_id': str(interview.id),
                'title': interview.title,
                'scheduled_at': interview.scheduled_at.isoformat(),
            })
    
    # Check candidate conflicts (if candidate is assigned)
    if candidate_id:
        candidate_interviews = Interview.objects(
            candidate_id=candidate_id,
            status__in=['scheduled', 'active']
        )
        
        # Exclude current interview if editing
        if exclude_interview_id:
            candidate_interviews = candidate_interviews.filter(id__ne=exclude_interview_id)
        
        for interview in candidate_interviews:
            iv_start = interview.scheduled_at
            # Ensure timezone-aware
            if iv_start.tzinfo is None:
                iv_start = django_timezone.make_aware(iv_start, django_timezone.utc)
            iv_start = iv_start.astimezone(django_timezone.utc)
            iv_end = iv_start + timedelta(minutes=interview.duration_minutes)
            
            if (start_time < iv_end and end_time > iv_start):
                conflicts.append({
                    'type': 'candidate',
                    'interview_id': str(interview.id),
                    'title': interview.title,
                    'scheduled_at': interview.scheduled_at.isoformat(),
                })
    
    has_conflict = len(conflicts) > 0
    
    message = ''
    if has_conflict:
        conflict_types = set(c['type'] for c in conflicts)
        if 'recruiter' in conflict_types and 'candidate' in conflict_types:
            message = 'Both recruiter and candidate have conflicting interviews at this time.'
        elif 'recruiter' in conflict_types:
            message = 'Recruiter has a conflicting interview at this time.'
        else:
            message = 'Candidate has a conflicting interview at this time.'
    
    return {
        'has_conflict': has_conflict,
        'conflicts': conflicts,
        'message': message,
        'scheduled_at': scheduled_at  # Return normalized aware object
    }


def validate_interview_time(scheduled_at: datetime, duration_minutes: int):
    """
    Validate interview timing constraints.

    Returns:
        dict: {'valid': bool, 'message': str, 'scheduled_at': datetime}
    """
    # CRITICAL FIX: Use timezone-aware datetime consistently
    from django.utils import timezone as django_timezone
    now = django_timezone.now()

    # Convert to timezone-aware if naive
    if scheduled_at.tzinfo is None:
        scheduled_at = django_timezone.make_aware(scheduled_at, django_timezone.utc)
    
    # Normalize to UTC for comparison
    scheduled_at = scheduled_at.astimezone(django_timezone.utc)
    now = now.astimezone(django_timezone.utc)

    # Check if scheduled time is in the past
    if scheduled_at < now:
        return {
            'valid': False,
            'message': 'Interview cannot be scheduled in the past.',
            'scheduled_at': scheduled_at
        }
    
    # Check if scheduled too soon (minimum 30 minutes notice)
    min_notice = now + timedelta(minutes=30)
    if scheduled_at < min_notice:
        return {
            'valid': False,
            'message': 'Interview must be scheduled at least 30 minutes in advance.',
            'scheduled_at': scheduled_at
        }
    
    # Check duration constraints
    if duration_minutes < 15:
        return {
            'valid': False,
            'message': 'Interview duration must be at least 15 minutes.',
            'scheduled_at': scheduled_at
        }
    
    if duration_minutes > 180:
        return {
            'valid': False,
            'message': 'Interview duration cannot exceed 3 hours.',
            'scheduled_at': scheduled_at
        }
    
    # Allow interviews to be scheduled at any time (24/7 flexibility)
    return {
        'valid': True,
        'message': 'Interview time is valid.',
        'scheduled_at': scheduled_at
    }
