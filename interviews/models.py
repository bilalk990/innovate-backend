"""
Interviews app models — MongoDB Documents via mongoengine
"""
import mongoengine as me
from datetime import datetime


class Question(me.EmbeddedDocument):
    text = me.StringField(required=True)
    category = me.StringField(default='general')  # technical, behavioral, general
    expected_keywords = me.ListField(me.StringField())
    ideal_answer = me.StringField(default='')   # AI-generated reference answer
    difficulty = me.StringField(default='medium')  # easy, medium, hard
    time_estimate_minutes = me.IntField(default=3)
    tags = me.ListField(me.StringField())


class QuestionBank(me.Document):
    """Reusable question library for recruiters — Feature 8"""
    name = me.StringField(required=True)
    recruiter_id = me.StringField(required=True, index=True)
    job_title = me.StringField(default='')
    description = me.StringField(default='')
    questions = me.ListField(me.EmbeddedDocumentField(Question))
    is_public = me.BooleanField(default=False)  # share across org
    created_at = me.DateTimeField(default=datetime.utcnow)
    updated_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'question_banks',
        'indexes': ['recruiter_id', 'job_title'],
        'ordering': ['-created_at'],
    }

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'recruiter_id': self.recruiter_id,
            'job_title': self.job_title,
            'description': self.description,
            'questions': [
                {
                    'text': q.text,
                    'category': q.category,
                    'expected_keywords': q.expected_keywords,
                    'ideal_answer': q.ideal_answer,
                    'difficulty': q.difficulty,
                    'time_estimate_minutes': q.time_estimate_minutes,
                    'tags': q.tags,
                }
                for q in self.questions
            ],
            'question_count': len(self.questions),
            'is_public': self.is_public,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class Interview(me.Document):
    title = me.StringField(required=True)
    recruiter_id = me.StringField(required=True, index=True)  # User ObjectId
    candidate_id = me.StringField(default=None, index=True)   # assigned after scheduling
    room_id = me.StringField(required=True, unique=True, index=True)  # WebRTC room
    room_token = me.StringField(default='')  # Cryptographically secure token
    evaluation_id = me.StringField(default='')  # Set after XAI evaluation completes
    token_expires_at = me.DateTimeField(default=None)  # Token expiry time
    scheduled_at = me.DateTimeField(required=True)
    duration_minutes = me.IntField(default=45)
    status = me.StringField(
        choices=['pending', 'scheduled', 'active', 'completed', 'cancelled'],
        default='scheduled'
    )
    job_id = me.StringField(default=None, index=True)  # Reference to Job posting
    job_title = me.StringField(default='')
    job_description = me.StringField(default='')
    meet_link = me.StringField(default='')  # External link (Google Meet, Zoom, etc.)
    questions = me.ListField(me.EmbeddedDocumentField(Question))
    candidate_responses = me.DictField(default={})  # question_index -> response
    tab_switch_count = me.IntField(default=0) # Proctoring violations
    chat_history = me.ListField(me.DictField(), default=[]) # Persistent conversation
    full_transcript = me.StringField(default="") # Entire session transcript for audit
    recording_url = me.StringField(default='')  # #56 — video recording URL
    notes = me.StringField(default='')
    created_at = me.DateTimeField(default=datetime.utcnow)
    updated_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'interviews',
        'indexes': ['recruiter_id', 'candidate_id', 'status', 'room_id'],
        'ordering': ['-scheduled_at'],
    }

    def to_dict(self):
        return {
            'id': str(self.id),
            'title': self.title,
            'recruiter_id': self.recruiter_id,
            'candidate_id': self.candidate_id,
            'room_id': self.room_id,
            'meet_link': self.meet_link,
            'scheduled_at': self.scheduled_at.isoformat(),
            'duration_minutes': self.duration_minutes,
            'status': self.status,
            'job_id': self.job_id,
            'job_title': self.job_title,
            'job_description': self.job_description,
            'questions': [
                {
                    'text': q.text,
                    'category': q.category,
                    'expected_keywords': q.expected_keywords,
                    'ideal_answer': q.ideal_answer,
                    'difficulty': q.difficulty,
                    'time_estimate_minutes': q.time_estimate_minutes,
                    'tags': q.tags,
                }
                for q in self.questions
            ],
            'candidate_responses': self.candidate_responses,
            'tab_switch_count': self.tab_switch_count,
            'evaluation_id': self.evaluation_id or '',
            'recording_url': self.recording_url,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
        }
