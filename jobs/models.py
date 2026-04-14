import mongoengine as me
from datetime import datetime

class Job(me.Document):
    title = me.StringField(required=True, max_length=200)
    company_name = me.StringField(required=True)
    location = me.StringField(default='Remote')
    job_type = me.StringField(choices=['full-time', 'part-time', 'contract', 'freelance'], default='full-time')
    salary_range = me.StringField(default='Not disclosed')
    description = me.StringField(required=True)
    requirements = me.ListField(me.StringField(), default=[])
    
    posted_by = me.StringField(required=True)  # Recruiter UserID
    is_active = me.BooleanField(default=True)
    created_at = me.DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'jobs',
        'indexes': ['posted_by', 'is_active', 'title'],
        'ordering': ['-created_at'],
    }
    
    def to_dict(self):
        return {
            'id': str(self.id),
            'title': self.title,
            'company_name': self.company_name,
            'location': self.location,
            'job_type': self.job_type,
            'salary_range': self.salary_range,
            'description': self.description,
            'requirements': self.requirements,
            'posted_by': self.posted_by,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
        }

class Application(me.Document):
    job_id = me.StringField(required=True)
    candidate_id = me.StringField(required=True)
    recruiter_id = me.StringField(required=True) # For direct HR dashboard visibility
    
    status = me.StringField(
        choices=['pending', 'reviewed', 'shortlisted', 'rejected', 'interview_scheduled', 'offer_sent', 'hired'],
        default='pending'
    )
    
    # Store a snapshot of candidate summary/headline at time of application
    candidate_name = me.StringField()
    candidate_headline = me.StringField()
    
    applied_at = me.DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'applications',
        'indexes': ['job_id', 'candidate_id', 'recruiter_id', 'status'],
        'ordering': ['-applied_at'],
    }
    
    def to_dict(self):
        d = {
            'id': str(self.id),
            'job_id': self.job_id,
            'candidate_id': self.candidate_id,
            'recruiter_id': self.recruiter_id,
            'status': self.status,
            'candidate_name': self.candidate_name,
            'candidate_headline': self.candidate_headline,
            'applied_at': self.applied_at.isoformat(),
        }
        # Enrich with job snapshot for candidate-facing views
        try:
            job = Job.objects.get(id=self.job_id)
            d['job_title'] = job.title
            d['company_name'] = job.company_name
            d['salary_range'] = job.salary_range
            d['location'] = job.location
            d['job_type'] = job.job_type
        except Exception:
            d['job_title'] = None
            d['company_name'] = None
            d['salary_range'] = 'N/A'
            d['location'] = 'N/A'
            d['job_type'] = None
        return d
