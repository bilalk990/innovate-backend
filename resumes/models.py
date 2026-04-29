"""
Resumes app models — MongoDB Documents via mongoengine
"""
import mongoengine as me
from datetime import datetime


class Resume(me.Document):
    candidate_id = me.StringField(required=True)  # User ObjectId
    file_path = me.StringField(required=True)
    original_filename = me.StringField(default='')
    file_size = me.IntField(default=0)       # bytes

    # Parsed data extracted from resume
    parsed_data = me.DictField(default={})
    # Expected structure:
    # {
    #   "name": str,
    #   "email": str,
    #   "phone": str,
    #   "skills": [str],
    #   "education": [{degree, institution, year}],
    #   "experience": [{title, company, years, description}],
    #   "certifications": [str],
    #   "languages": [str],
    #   "summary": str,
    # }

    parse_status = me.StringField(
        choices=['pending', 'processing', 'parsed', 'completed', 'failed'],
        default='pending'
    )
    parsed_by_ai = me.BooleanField(default=False)  # True if AI (OpenAI GPT) was used
    is_active = me.BooleanField(default=True)  # latest resume flag
    uploaded_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'resumes',
        'indexes': ['candidate_id', 'parse_status'],
        'ordering': ['-uploaded_at'],
    }

    def to_dict(self):
        return {
            'id': str(self.id),
            'candidate_id': self.candidate_id,
            'file_path': self.file_path,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'parsed_data': self.parsed_data,
            'parse_status': self.parse_status,
            'parsed_by_ai': self.parsed_by_ai,
            'is_active': self.is_active,
            'uploaded_at': self.uploaded_at.isoformat(),
        }
