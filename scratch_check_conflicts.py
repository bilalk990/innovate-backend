import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from interviews.models import Interview
from accounts.models import User

# Check for specific candidate
candidate_id = '69d9f7d3c54a4917496e8e308'
recruiter = User.objects(role='recruiter').first()

print(f"Checking candidate: {candidate_id}")
if recruiter:
    print(f"Checking recruiter: {recruiter.email} (ID: {recruiter.id})")
    
ivs = Interview.objects(status__in=['scheduled', 'active'])
print(f"Total Active/Scheduled Interviews: {len(ivs)}")

for iv in ivs:
    print(f"\n--- Interview ---")
    print(f"ID: {iv.id}")
    print(f"Title: {iv.title}")
    print(f"Recruiter ID: {iv.recruiter_id}")
    print(f"Candidate ID: {iv.candidate_id}")
    print(f"Scheduled At: {iv.scheduled_at}")
    print(f"Status: {iv.status}")
