import os
import sys
import django

# Setup Django
sys.path.append(os.path.join(os.getcwd(), 'Backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend.settings')
django.setup()

from interviews.models import Interview

room_id = '759f7b96d5da4d0e'
try:
    interview = Interview.objects.get(room_id=room_id)
    print(f"Interview Found: {interview.title}")
    print(f"Status: {interview.status}")
    print(f"Questions Count: {len(interview.questions)}")
    print(f"Recruiter ID: {interview.recruiter_id}")
    print(f"Candidate ID: {interview.candidate_id}")
    print(f"Token Expires At: {interview.token_expires_at}")
    print(f"Questions List: {[q.text for q in interview.questions]}")
except Exception as e:
    print(f"Error finding interview: {e}")
