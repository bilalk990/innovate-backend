"""
Evaluations app models — MongoDB Documents via mongoengine
Stores XAI rule-based evaluation results per interview
"""
import mongoengine as me
from datetime import datetime


class MockInterviewSession(me.Document):
    """Stores a single AI mock interview session for a candidate."""
    user_id = me.StringField(required=True)
    role = me.StringField(required=True)        # e.g. "Software Engineer"
    level = me.StringField(default='mid')       # junior | mid | senior
    history = me.ListField(me.DictField())      # [{question, question_type, tip, answer, score, grade, feedback, ...}]
    current_question = me.IntField(default=0)
    total_questions = me.IntField(default=5)
    status = me.StringField(default='active', choices=['active', 'completed', 'abandoned'])
    final_report = me.DictField(default=None)
    created_at = me.DateTimeField(default=datetime.utcnow)
    completed_at = me.DateTimeField()

    meta = {
        'collection': 'mock_interview_sessions',
        'indexes': ['user_id', 'status', '-created_at'],
        'ordering': ['-created_at'],
    }

    def to_dict(self):
        return {
            'session_id': str(self.id),
            'user_id': self.user_id,
            'role': self.role,
            'level': self.level,
            'history': self.history or [],
            'current_question': self.current_question,
            'total_questions': self.total_questions,
            'status': self.status,
            'final_report': self.final_report,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class CriterionResult(me.EmbeddedDocument):
    """Score for a single evaluation criterion with XAI explanation"""
    criterion = me.StringField(required=True)   # clarity, consistency, alignment, etc.
    score = me.FloatField(min_value=0, max_value=10)
    max_score = me.FloatField(default=10.0)
    weight = me.FloatField(default=1.0)         # relative importance
    explanation = me.StringField(default='')    # human-readable reason
    rules_applied = me.ListField(me.StringField())   # list of rule names triggered
    evidence = me.ListField(me.StringField())        # textual evidence snippets


class Evaluation(me.Document):
    interview_id = me.StringField(required=True)
    candidate_id = me.StringField(required=True)
    recruiter_id = me.StringField(required=True)

    # Criterion-by-criterion XAI breakdown
    criterion_results = me.ListField(me.EmbeddedDocumentField(CriterionResult))

    overall_score = me.FloatField(min_value=0, max_value=100, default=0)
    recommendation = me.StringField(
        choices=['strong_yes', 'yes', 'maybe', 'no', 'strong_no'],
        default='maybe'
    )
    summary = me.StringField(default='')        # Overall AI-generated summary
    strengths = me.ListField(me.StringField())
    weaknesses = me.ListField(me.StringField())
    resume_alignment_score = me.FloatField(default=0)  # 0-100, resume vs answers

    # Behavioral & Integrity Metrics (Enterprise)
    confidence_score = me.FloatField(default=0)        # 0-100
    fluency_score = me.FloatField(default=0)           # 0-100
    behavioral_summary = me.StringField(default='')
    proctoring_score = me.FloatField(default=100)      # Starts at 100
    integrity_notes = me.StringField(default='')
    tab_switch_count = me.IntField(default=0)          # Number of proctoring violations
    culture_fit_score = me.FloatField(default=0)       # 0-100 analysis vs company values

    # Detailed analysis fields
    question_analysis = me.DictField(default=dict)
    emotion_timeline = me.DictField(default=dict)
    recommendation_reason = me.StringField(default='')
    performance_stats = me.DictField(default=dict)

    status = me.StringField(
        choices=['pending', 'processing', 'complete'],
        default='pending'
    )
    reviewed_by_hr = me.BooleanField(default=False)
    hr_notes = me.StringField(default='')
    candidate_visible = me.BooleanField(default=False)  # #57 — recruiter can share with candidate
    ai_summary_used = me.BooleanField(default=False)    # #52 — flag if AI generated summary

    created_at = me.DateTimeField(default=datetime.utcnow)
    updated_at = me.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'evaluations',
        'indexes': ['interview_id', 'candidate_id', 'recruiter_id', 'status'],
        'ordering': ['-created_at'],
    }

    def to_dict(self):
        # Resolve candidate name for frontend display
        candidate_name = ''
        try:
            from accounts.models import User
            candidate = User.objects(id=self.candidate_id).first()
            candidate_name = candidate.name if candidate else ''
        except Exception:
            pass

        # Resolve job info from Interview
        job_info = {
            'job_title': '',
            'job_description': '',
            'salary_range': 'N/A',
            'location': 'Remote'
        }
        try:
            from interviews.models import Interview
            from jobs.models import Job
            interview = Interview.objects(id=self.interview_id).first()
            if interview:
                job_info['job_title'] = interview.job_title
                job_info['job_description'] = interview.job_description
                if interview.job_id:
                    job = Job.objects(id=interview.job_id).first()
                    if job:
                        job_info['salary_range'] = job.salary_range
                        job_info['location'] = job.location
        except Exception:
            pass

        return {
            'id': str(self.id),
            'interview_id': self.interview_id,
            'candidate_id': self.candidate_id,
            'candidate_name': candidate_name,
            'recruiter_id': self.recruiter_id,
            **job_info,
            'criterion_results': [
                {
                    'criterion': cr.criterion,
                    'score': cr.score,
                    'max_score': cr.max_score,
                    'weight': cr.weight,
                    'explanation': cr.explanation,
                    'rules_applied': cr.rules_applied,
                    'evidence': cr.evidence,
                }
                for cr in self.criterion_results
            ],
            'overall_score': self.overall_score,
            'recommendation': self.recommendation,
            'summary': self.summary,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'resume_alignment_score': self.resume_alignment_score,
            'status': self.status,
            'reviewed_by_hr': self.reviewed_by_hr,
            'hr_notes': self.hr_notes,
            'candidate_visible': self.candidate_visible,
            'ai_summary_used': self.ai_summary_used,
            'confidence_score': self.confidence_score,
            'fluency_score': self.fluency_score,
            'behavioral_summary': self.behavioral_summary,
            'proctoring_score': self.proctoring_score,
            'integrity_notes': self.integrity_notes,
            'tab_switch_count': self.tab_switch_count,
            'culture_fit_score': self.culture_fit_score,
            'question_analysis': self.question_analysis or {},
            'emotion_timeline': self.emotion_timeline or {},
            'recommendation_reason': self.recommendation_reason,
            'performance_stats': self.performance_stats or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
