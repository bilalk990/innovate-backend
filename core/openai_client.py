"""
core/openai_client.py — OpenAI GPT Client Utility
All OpenAI API calls go through this module for consistency.
Replaces Gemini with GPT-4o-mini (GPT-5.4 mini)
"""
import json
import logging
from datetime import date
from openai import OpenAI
from django.conf import settings
from core.rate_limiter import ai_rate_limiter

logger = logging.getLogger('innovaite')

# Configure OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Use GPT-4o-mini (GPT-5.4 mini) — fast and cost-effective
MODEL_NAME = "gpt-4o-mini"

# ── Daily AI Usage Tracker (in-memory, resets at midnight) ──────────────────
# For $10 OpenAI credit: ~2,200 total calls possible
# Daily limits set for 30-day usage
_AI_WARNING_THRESHOLD = getattr(settings, 'AI_WARNING_THRESHOLD', 50)  # 70% of daily limit
_AI_DAILY_SOFT_LIMIT  = getattr(settings, 'AI_DAILY_SOFT_LIMIT', 73)   # ~73 calls/day for $10/month

_daily_stats = {
    'date': None,
    'total_calls': 0,
    'warning_sent': False,
    'exhausted_notified': False,
}


def get_ai_usage_stats() -> dict:
    """Return today's AI call stats — exposed to admin dashboard."""
    _reset_if_new_day()
    return {
        'date': str(_daily_stats['date']),
        'total_calls': _daily_stats['total_calls'],
        'warning_threshold': _AI_WARNING_THRESHOLD,
        'daily_soft_limit': _AI_DAILY_SOFT_LIMIT,
        'usage_pct': round((_daily_stats['total_calls'] / _AI_DAILY_SOFT_LIMIT) * 100, 1),
        'warning_sent': _daily_stats['warning_sent'],
        'exhausted_notified': _daily_stats['exhausted_notified'],
    }


def _reset_if_new_day():
    today = date.today()
    if _daily_stats['date'] != today:
        _daily_stats['date'] = today
        _daily_stats['total_calls'] = 0
        _daily_stats['warning_sent'] = False
        _daily_stats['exhausted_notified'] = False


def _increment_and_check_quota():
    """Increment daily counter and return alert type if threshold crossed."""
    _reset_if_new_day()
    _daily_stats['total_calls'] += 1
    count = _daily_stats['total_calls']

    if count >= _AI_WARNING_THRESHOLD and not _daily_stats['warning_sent']:
        _daily_stats['warning_sent'] = True
        logger.warning(f'[AI] Daily call count hit warning threshold: {count}/{_AI_DAILY_SOFT_LIMIT}')
        return 'AI_QUOTA_WARNING'

    return None


def _call(prompt: str, user_id: str = None, response_format: str = "text") -> str:
    """
    Make an OpenAI API call with rate limiting and proper error handling.
    """
    # Check rate limit if user_id provided
    if user_id:
        allowed, remaining, reset_time = ai_rate_limiter.check_limit(user_id, limit=20, window_minutes=60)
        if not allowed:
            raise Exception(f'AI rate limit exceeded. Try again after {reset_time.strftime("%H:%M:%S")}')
        logger.info(f'[AI] User {user_id} - {remaining} calls remaining')

    # Increment daily counter
    alert_type = _increment_and_check_quota()
    if alert_type:
        try:
            from core.ai_notifications import notify_admins_async
            notify_admins_async(alert_type)
        except Exception as notify_err:
            logger.warning(f'[AI] Could not send quota warning notification: {notify_err}')

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for HR and recruitment tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()

        # Quota exhausted
        if 'quota' in error_str or 'rate_limit' in error_str or '429' in error_str:
            logger.error(f'[AI] OpenAI quota exhausted: {e}')
            if not _daily_stats['exhausted_notified']:
                _daily_stats['exhausted_notified'] = True
                try:
                    from core.ai_notifications import notify_admins_async
                    notify_admins_async('AI_QUOTA_EXHAUSTED')
                except Exception as notify_err:
                    logger.warning(f'[AI] Could not send exhausted notification: {notify_err}')
            raise Exception('AI_QUOTA_EXHAUSTED')

        # Invalid API key
        if 'api_key' in error_str or 'invalid' in error_str or 'authentication' in error_str:
            logger.error(f'[AI] OpenAI API key invalid: {e}')
            raise Exception('AI_KEY_INVALID')

        # Generic failure
        logger.error(f'[AI] OpenAI API call failed: {e}')
        raise Exception(f'AI_ERROR: {str(e)}')


def _strip_json(text: str) -> str:
    """Strip markdown code fences from AI responses before JSON parsing."""
    text = text.strip()
    if text.startswith('```'):
        text = text[3:]
        if text.lower().startswith('json'):
            text = text[4:]
        if text.endswith('```'):
            text = text[:-3]
    return text.strip()


def check_ai_health() -> dict:
    """
    Check if OpenAI API is operational.
    Returns status dict with 'status', 'message', and 'error_type'.
    """
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in ('', 'your-openai-api-key'):
        return {
            'status': 'error',
            'error_type': 'AI_KEY_MISSING',
            'message': 'OpenAI API key is not configured.'
        }

    try:
        _call('Reply with the single word: OK', user_id=None)
        return {
            'status': 'ok',
            'error_type': None,
            'message': 'AI service is operational.'
        }
    except Exception as e:
        error_type = str(e)
        messages = {
            'AI_QUOTA_EXHAUSTED': 'OpenAI API quota exhausted. Please add more credits to your account.',
            'AI_KEY_INVALID': 'OpenAI API key is invalid. Please update OPENAI_API_KEY in your .env file.',
        }
        return {
            'status': 'error',
            'error_type': error_type,
            'message': messages.get(error_type, f'AI service error: {error_type}')
        }


def parse_resume_with_ai(raw_text: str, user_id: str = None) -> dict:
    """Use GPT to intelligently parse resume text into structured data."""
    if not raw_text or len(raw_text.strip()) < 20:
        return {}

    text_chunk = raw_text[:8000]

    prompt = f"""You are an expert HR resume parser. Analyze the resume text below and extract ALL structured information.

Return ONLY a valid JSON object with these exact keys (no markdown, no explanation):
{{
  "name": "full name of candidate",
  "email": "email address or empty string",
  "phone": "phone number or empty string",
  "linkedin": "LinkedIn URL or username, or empty string",
  "github": "GitHub URL or username, or empty string",
  "summary": "professional summary or objective in 2-3 sentences",
  "total_experience_years": 0,
  "skills": ["skill1", "skill2", "skill3"],
  "education": [
    {{"degree": "degree name", "institution": "university or school name", "year": "graduation year or expected year"}}
  ],
  "experience": [
    {{"title": "job title", "company": "company name", "duration": "time period e.g. Jan 2020 - Dec 2022", "years": 2.0, "description": "brief description of responsibilities"}}
  ],
  "certifications": ["cert1", "cert2"],
  "languages": ["English", "Urdu"]
}}

RESUME TEXT:
{text_chunk}
"""
    try:
        result_text = _call(prompt)
        stripped = _strip_json(result_text)
        parsed = json.loads(stripped)

        if not parsed.get('skills') and not parsed.get('name'):
            logger.warning('[Resume AI] GPT returned empty name+skills — falling back to rule-based.')
            return {}

        # Set defaults
        parsed.setdefault('name', '')
        parsed.setdefault('email', '')
        parsed.setdefault('phone', '')
        parsed.setdefault('linkedin', '')
        parsed.setdefault('github', '')
        parsed.setdefault('summary', '')
        parsed.setdefault('total_experience_years', 0)
        parsed.setdefault('skills', [])
        parsed.setdefault('education', [])
        parsed.setdefault('experience', [])
        parsed.setdefault('certifications', [])
        parsed.setdefault('languages', [])

        parsed['skills'] = [str(s).strip() for s in parsed['skills'] if s]
        parsed['raw_text'] = raw_text[:3000]
        logger.info(f'[Resume AI] Parsed {len(parsed["skills"])} skills via GPT.')
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f'[Resume AI] JSON decode failed: {e}')
        return {}
    except Exception as e:
        logger.error(f'[Resume AI] GPT parse_resume failed: {e}')
        return {}


def generate_interview_questions(
    job_title: str,
    job_description: str,
    num_questions: int = 8,
    categories: list = None,
    resume_data: dict = None,
    user_id: str = None
) -> list:
    """Use GPT to auto-generate smart, tailored interview questions."""
    if not job_title:
        return []

    categories_str = ', '.join(categories) if categories else 'general, technical, behavioral'
    
    context_str = f"Job Title: {job_title}\nJob Description: {job_description or 'Not provided'}"
    if resume_data:
        context_str += f"\n\nCandidate Resume Data: {json.dumps(resume_data)}"

    prompt = f"""
You are an expert HR interviewer. Generate {num_questions} high-quality interview questions.

CONTEXT:
{context_str}

Return ONLY a valid JSON array:
[
  {{
    "text": "Question text",
    "category": "technical" or "behavioral" or "general",
    "expected_keywords": ["keyword1", "keyword2"],
    "ideal_answer": "What a perfect candidate would say",
    "difficulty": "easy" or "medium" or "hard"
  }}
]
"""
    try:
        result_text = _call(prompt, user_id=user_id)
        stripped = _strip_json(result_text)
        questions = json.loads(stripped)
        return questions[:num_questions] if isinstance(questions, list) else []
    except Exception as e:
        logger.error(f'[GPT] Question generation failed: {e}')
        return []


def generate_candidate_hints(question: str, category: str = 'general', user_id: str = None) -> str:
    """Generate helpful hints for the candidate."""
    prompt = f"""
Provide 3 concise bullet points strategy hints for this interview question. Do NOT give the answer directly.
Category: {category}
Question: {question}
Return ONLY the 3 bullet points.
"""
    try:
        return _call(prompt, user_id=user_id)
    except Exception as e:
        if 'rate limit' in str(e).lower():
            raise
        return "Focus on clarity and provide specific examples from your experience."


def analyze_emotion_confidence(face_snapshots: list) -> dict:
    """Analyze face metric snapshots for emotion and confidence scoring."""
    if not face_snapshots:
        return {
            'emotion_score': 50,
            'confidence_level': 'neutral',
            'eye_contact_pct': 0,
            'stability_pct': 0,
            'dominant_emotion': 'unknown',
            'coaching_tip': 'No face data captured.',
        }

    prompt = f"""
Analyze {len(face_snapshots)} face metric snapshots from an interview.

Data: {json.dumps(face_snapshots[:20], indent=2)}

Return JSON:
{{
  "emotion_score": 0-100,
  "confidence_level": "high" | "medium" | "low",
  "eye_contact_pct": 0-100,
  "stability_pct": 0-100,
  "dominant_emotion": "confident" | "nervous" | "neutral",
  "emotion_trend": "improving" | "stable" | "declining",
  "coaching_tip": "1 sentence"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[GPT] Emotion analysis failed: {e}')
        return {
            'emotion_score': 60,
            'confidence_level': 'medium',
            'eye_contact_pct': 60.0,
            'stability_pct': 70.0,
            'dominant_emotion': 'neutral',
            'emotion_trend': 'stable',
            'coaching_tip': 'Maintain eye contact and speak clearly.',
        }


def analyze_live_transcript_chunk(
    transcript: str,
    question: str,
    job_title: str = '',
    user_id: str = None
) -> dict:
    """Perform real-time semantic analysis on transcript chunk."""
    if not transcript or len(transcript.strip()) < 15:
        return {'relevance': 0, 'keywords_detected': [], 'signal': 'too_short', 'live_tip': ''}

    prompt = f"""
Analyze this live interview transcript.

Job: {job_title}
Question: {question}
Transcript: "{transcript}"

Return JSON:
{{
  "relevance": 0-100,
  "keywords_detected": ["keyword1"],
  "signal": "on_track" | "off_topic" | "needs_examples" | "strong",
  "sentiment": "positive" | "neutral" | "negative",
  "filler_words": 0,
  "live_tip": "1 tip"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Live transcript analysis failed: {e}')
        return {
            'relevance': 50,
            'keywords_detected': [],
            'signal': 'on_track',
            'sentiment': 'neutral',
            'filler_words': 0,
            'live_tip': '',
        }


def suggest_adaptive_question(
    current_question: str,
    candidate_response: str,
    current_difficulty: str,
    job_title: str,
    category: str = 'technical',
    user_id: str = None,
) -> dict:
    """Suggest next question with adjusted difficulty based on response quality."""
    prompt = f"""
Adaptive interview for: {job_title}

Current Question ({current_difficulty}, {category}): "{current_question}"
Candidate Response: "{candidate_response}"

Return JSON:
{{
  "response_quality_score": 0-10,
  "response_assessment": "1 sentence",
  "next_difficulty": "easy" | "medium" | "hard",
  "next_question": "full question text",
  "next_category": "technical" | "behavioral" | "general",
  "expected_keywords": ["key1"],
  "ideal_answer": "brief answer"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Adaptive question failed: {e}')
        return {
            'response_quality_score': 5.0,
            'response_assessment': 'Unable to assess.',
            'next_difficulty': current_difficulty,
            'next_question': 'Can you elaborate with a specific example?',
            'next_category': category,
            'expected_keywords': [],
            'ideal_answer': '',
        }


def analyze_response_semantics(question: str, ideal_answer: str, candidate_response: str, user_id: str = None) -> dict:
    """Deep semantic comparison of candidate response vs ideal answer."""
    if not candidate_response or len(candidate_response.strip()) < 10:
        return {"score": 0, "explanation": "Response too short.", "missing_points": []}

    prompt = f"""
Compare candidate response to ideal answer.

Question: {question}
Ideal: {ideal_answer}
Candidate: {candidate_response}

Return JSON:
{{
  "score": 0-10,
  "explanation": "2-3 sentences",
  "missing_points": ["point1"]
}}
"""
    try:
        result_text = _call(prompt, user_id=user_id)
        return json.loads(_strip_json(result_text))
    except Exception:
        return {"score": 5.0, "explanation": "AI analysis unavailable.", "missing_points": []}


def enhance_evaluation_summary(
    overall_score: float,
    recommendation: str,
    criterion_results: list,
    job_title: str = '',
    user_id: str = None
) -> str:
    """Generate rich HR evaluation summary."""
    criteria_text = '\n'.join([
        f"- {cr['criterion']}: {cr['score']}/10 — {cr['explanation']}"
        for cr in criterion_results
    ])

    prompt = f"""
Write a professional evaluation summary.

Role: {job_title}
Score: {overall_score}/100
Recommendation: {recommendation}

Criteria:
{criteria_text}

Write 3-4 sentences focusing on strengths, weaknesses, and recommendation.
"""
    try:
        return _call(prompt, user_id=user_id)
    except Exception:
        return f"Candidate scored {overall_score}/100. Recommendation: {recommendation}."


# Additional helper functions for compatibility
def analyze_behavioral_traits(transcript, user_id: str = None):
    """Analyze transcript for behavioral traits."""
    prompt = f"""
Analyze interview transcript for behavioral traits:
{transcript}

Return JSON:
{{
  "confidence_score": 0-100,
  "fluency_score": 0-100,
  "filler_count": 0,
  "summary": "2 sentences"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception:
        return {"confidence_score": 70, "fluency_score": 70, "filler_count": 0, "summary": "Analysis unavailable."}


def check_integrity_plagiarism(responses, user_id: str = None):
    """Check for plagiarism or AI-generated content."""
    prompt = f"""
Check these responses for plagiarism/AI content:
{responses}

Return JSON:
{{
  "integrity_score": 0-100,
  "notes": "any red flags"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception:
        return {"integrity_score": 90, "notes": "Check skipped."}


def analyze_resume_jd_gap(resume_data: dict, job_description: str, job_title: str, requirements: list = None) -> dict:
    """Gap analysis comparing resume vs job description."""
    resume_summary = {
        'skills': resume_data.get('skills', []),
        'experience': resume_data.get('experience', []),
        'total_years': resume_data.get('total_experience_years', 0),
    }

    prompt = f"""
Gap Analysis:

Job: {job_title}
JD: {job_description[:2000]}
Resume: {json.dumps(resume_summary)}

Return JSON:
{{
  "match_percentage": 0-100,
  "matched_skills": ["skill1"],
  "missing_skills": ["skill1"],
  "strengths": ["strength1"],
  "gaps": ["gap1"],
  "summary": "2 sentences"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception:
        return {
            'match_percentage': 50,
            'matched_skills': [],
            'missing_skills': [],
            'strengths': [],
            'gaps': [],
            'summary': 'Analysis unavailable.',
        }



def generate_question_bank_suggestions(
    job_title: str,
    category: str = 'general',
    num_questions: int = 10,
    job_description: str = '',
    categories: list = None,
    count: int = None,
    user_id: str = None,
) -> list:
    """Generate question bank suggestions for a specific job and category."""
    # Support both old (category) and new (categories list) calling conventions
    if count is not None:
        num_questions = count
    categories_str = ', '.join(categories) if categories else category

    prompt = f"""
Generate {num_questions} interview questions for a Question Bank.

Job Title: {job_title}
{f'Job Description: {job_description[:500]}' if job_description else ''}
Categories: {categories_str}

Return a JSON array (no extra text):
[
  {{
    "text": "Question text",
    "category": "technical" | "behavioral" | "general",
    "difficulty": "easy" | "medium" | "hard",
    "expected_keywords": ["keyword1"],
    "ideal_answer": "Brief ideal answer"
  }}
]
"""
    try:
        result_text = _call(prompt, user_id=user_id)
        stripped = _strip_json(result_text)
        questions = json.loads(stripped)
        return questions[:num_questions] if isinstance(questions, list) else []
    except Exception as e:
        logger.error(f'[GPT] Question bank generation failed: {e}')
        return []


def suggest_next_question(
    transcript: str,
    questions_asked: list,
    job_title: str,
    current_question: str = '',
    user_id: str = None,
) -> dict:
    """Suggest the next best question based on conversation flow."""
    prompt = f"""
Based on the interview transcript, suggest the next best question.

Job: {job_title}
Current Question: {current_question or 'N/A'}
Transcript: {transcript[:2000]}
Questions Already Asked: {json.dumps(questions_asked)}

Return JSON:
{{
  "suggested_question": "Full question text",
  "category": "technical" | "behavioral" | "general",
  "difficulty": "easy" | "medium" | "hard",
  "reasoning": "Why this question is appropriate now",
  "expected_keywords": ["keyword1"]
}}
"""
    try:
        result = _call(prompt, user_id=user_id)
        stripped = _strip_json(result)
        return json.loads(stripped)
    except Exception as e:
        logger.warning(f'[GPT] Next question suggestion failed: {e}')
        # Return fallback question instead of raising
        return {
            "suggested_question": "Can you tell me more about your experience with this role?",
            "category": "general",
            "difficulty": "medium",
            "reasoning": "General follow-up question to continue the conversation",
            "expected_keywords": ["experience", "role", "skills"]
        }


def suggest_interview_slots(
    job_title: str,
    duration_minutes: int,
    recruiter_timezone: str = 'UTC',
    candidate_timezone: str = 'UTC',
    existing_interviews: list = None,
    preferred_days: list = None
) -> dict:
    """AI-powered suggestion of optimal interview time slots."""
    existing_count = len(existing_interviews) if existing_interviews else 0

    prompt = f"""
Suggest 5 optimal interview time slots.

Job: {job_title}
Duration: {duration_minutes} minutes
Recruiter TZ: {recruiter_timezone}
Candidate TZ: {candidate_timezone}
Existing interviews: {existing_count}
Preferred days: {preferred_days or ['Monday', 'Tuesday', 'Wednesday', 'Thursday']}

Return JSON:
{{
  "suggested_slots": [
    {{
      "datetime_utc": "2026-04-12T09:00:00",
      "day_label": "Friday, Apr 12",
      "time_recruiter": "2:00 PM",
      "time_candidate": "9:00 AM",
      "quality_score": 95,
      "reason": "Peak performance window"
    }}
  ],
  "scheduling_tip": "1 sentence",
  "optimal_slot_index": 0
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[GPT] Slot suggestion failed: {e}')
        return {
            'suggested_slots': [],
            'scheduling_tip': 'Schedule between 10am-12pm or 2pm-4pm for best results.',
            'optimal_slot_index': 0,
        }



def analyze_job_fitment(resume_data, job_description):
    """Deep semantic matching of resume vs job requirements."""
    prompt = f"""
Job Fitment Analysis:
Resume: {json.dumps(resume_data)[:2000]}
Job: {job_description[:2000]}

Return JSON:
{{
  "fitment_score": 0-100,
  "matched_dimensions": ["match1"],
  "missing_relevance": ["gap1"],
  "suggestion": "1 sentence"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception:
        return {"fitment_score": 60, "matched_dimensions": [], "missing_relevance": [], "suggestion": "Manual review recommended."}


def analyze_culture_fit(transcript, company_values):
    """Analyze candidate alignment with company values."""
    prompt = f"""
Culture Fit Analysis:
Values: {company_values}
Transcript: {transcript[:2000]}

Return JSON:
{{
  "culture_score": 0-100,
  "aligned_values": ["value1"],
  "red_flags": ["flag1"]
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception:
        return {"culture_score": 70, "aligned_values": [], "red_flags": []}



def generate_offer_letter(candidate_name, job_title, evaluation_score):
    """Generate a professional offer letter draft."""
    prompt = f"""
Generate a professional Job Offer Letter draft for:
Candidate: {candidate_name}
Role: {job_title}
AI Evaluation Score: {evaluation_score}/100

The tone should be enthusiastic. Keep placeholders for salary and start date using [BRACKETS].
"""
    try:
        return _call(prompt)
    except Exception as e:
        logger.warning(f"[GPT] Offer letter generation failed: {str(e)}")
        return "Offer letter generation failed."


def rank_candidates_for_job(job_title, job_description, candidates_data=None, candidates=None, user_id: str = None):
    """Rank candidates for a specific job. Accepts both 'candidates' and 'candidates_data' param names."""
    # Support both old and new calling conventions
    data = candidates_data if candidates_data is not None else (candidates or [])

    prompt = f"""
Rank these candidates for the job role.

Job: {job_title}
Description: {job_description[:800] if job_description else 'Not provided'}
Candidates: {json.dumps(data)[:2500]}

Return JSON:
{{
  "ranked": [
    {{
      "candidate_id": "id",
      "name": "candidate name",
      "rank": 1,
      "match_score": 0-100,
      "reasoning": "1 sentence why they rank here",
      "strengths": ["strength1"],
      "concerns": ["concern1"]
    }}
  ],
  "ranking_rationale": "1 sentence explaining the overall ranking methodology",
  "top_recommendation": "Name of top candidate and why"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Candidate ranking failed: {e}')
        # Fallback: sort by overall_score
        sorted_data = sorted(data, key=lambda x: x.get('overall_score', 0), reverse=True)
        return {
            "ranked": [
                {
                    "candidate_id": c.get('candidate_id', ''),
                    "name": c.get('name', 'Unknown'),
                    "rank": i + 1,
                    "match_score": c.get('overall_score', 0),
                    "reasoning": f"Ranked by overall evaluation score: {c.get('overall_score', 0)}/100",
                    "strengths": [],
                    "concerns": []
                }
                for i, c in enumerate(sorted_data)
            ],
            "ranking_rationale": "Candidates ranked by overall evaluation score (AI unavailable).",
            "top_recommendation": sorted_data[0].get('name', 'Top candidate') if sorted_data else ''
        }


def generate_interview_debrief(evaluation, interview, candidate_name: str = '', user_id: str = None) -> dict:
    """Generate structured interview debrief summary. Returns dict (not string)."""
    prompt = f"""
Generate a detailed, personalized interview debrief for {candidate_name or 'the candidate'}.

Evaluation Data: {json.dumps(evaluation)[:2000]}
Interview Data: {json.dumps(interview)[:800]}

Return JSON (not plain text):
{{
  "overall_impression": "2-3 sentence overall impression",
  "performance_highlights": ["highlight1", "highlight2", "highlight3"],
  "areas_for_improvement": ["area1", "area2"],
  "communication_feedback": "1-2 sentences on communication style",
  "technical_feedback": "1-2 sentences on technical responses",
  "recommendation_summary": "1 sentence final recommendation",
  "coaching_tips": ["tip1", "tip2", "tip3"]
}}
"""
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        return result
    except Exception as e:
        logger.warning(f'[GPT] Debrief generation failed: {e}')
        score = evaluation.get('overall_score', 50) if isinstance(evaluation, dict) else 50
        rec = evaluation.get('recommendation', 'maybe') if isinstance(evaluation, dict) else 'maybe'
        return {
            "overall_impression": f"{candidate_name or 'The candidate'} completed the interview with an overall score of {score}/100.",
            "performance_highlights": evaluation.get('strengths', ['Completed all questions']) if isinstance(evaluation, dict) else ['Completed the interview'],
            "areas_for_improvement": evaluation.get('weaknesses', ['Review feedback from recruiter']) if isinstance(evaluation, dict) else [],
            "communication_feedback": "Review the interview feedback for details on communication style.",
            "technical_feedback": "Technical performance has been evaluated. See detailed scores for breakdown.",
            "recommendation_summary": f"Recruiter recommendation: {rec}.",
            "coaching_tips": ["Practice STAR method for behavioral questions", "Prepare specific examples from your experience", "Review technical fundamentals for your role"]
        }


def predict_hire_probability(
    overall_score,
    confidence_score,
    proctoring_score=100,
    fluency_score=70,
    culture_fit_score=0,
    violations=0,
    recommendation='maybe',
    job_title='',
    criterion_results=None,
    # Legacy params kept for backward compat
    behavioral_score=None,
    technical_score=None,
    user_id=None,
):
    """Predict probability of successful hire based on full evaluation data."""
    # Build criterion summary
    criteria_text = ''
    if criterion_results:
        criteria_text = '\n'.join([
            f"- {cr.get('criterion','')}: {cr.get('score',0)}/{cr.get('max_score',10)}"
            for cr in (criterion_results or [])[:8]
        ])

    prompt = f"""
Predict the probability of a successful hire based on the following interview evaluation:

Role: {job_title or 'Not specified'}
Overall Score: {overall_score}/100
Confidence Score: {confidence_score}/100
Proctoring Score: {proctoring_score}/100
Fluency Score: {fluency_score}/100
Culture Fit Score: {culture_fit_score}/100
Tab Violations: {violations}
Current Recommendation: {recommendation}

Detailed Criteria:
{criteria_text or 'Not available'}

Return JSON:
{{
  "hire_probability": 0-100,
  "confidence_level": "high" | "medium" | "low",
  "key_factors": ["factor1", "factor2"],
  "risk_factors": ["risk1"],
  "recommendation": "1 sentence hiring recommendation"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception:
        # Deterministic fallback based on score
        prob = min(95, max(5, int(overall_score * 0.85 + confidence_score * 0.1 + proctoring_score * 0.05)))
        return {
            "hire_probability": prob,
            "confidence_level": "high" if prob >= 70 else "medium" if prob >= 45 else "low",
            "key_factors": ["Overall interview performance", "Confidence and communication"],
            "risk_factors": ["Limited data for full assessment"] if violations > 3 else [],
            "recommendation": "Strong hire" if prob >= 70 else "Consider for interview" if prob >= 45 else "Not recommended at this time"
        }


def generate_resume_content(name, email, phone, headline='', bio='', skills=None, work_history=None, education_history=None, location='', job_target=''):
    """Generate polished, complete professional resume with ALL sections."""
    # Safety: Explicitly cast to list to avoid MongoEngine proxy object issues
    try:
        skills_clean = list(skills) if skills else []
        work_clean = list(work_history) if work_history else []
        edu_clean = list(education_history) if education_history else []
    except Exception as e:
        logger.warning(f"[GPT] List data cast failed in resume generation: {e}")
        skills_clean, work_clean, edu_clean = [], [], []

    # Build comprehensive context
    context = f"""
Create a COMPLETE professional resume for:

PERSONAL INFO:
- Name: {name}
- Email: {email}
- Phone: {phone or 'Not provided'}
- Location: {location or 'Not specified'}
- Headline: {headline or 'Software Engineer'}

TARGET ROLE: {job_target or 'Software Engineer / Developer'}

CURRENT BIO: {bio or 'Motivated professional seeking opportunities'}

SKILLS ({len(skills_clean)} provided): {json.dumps(skills_clean) if skills_clean else 'NONE - Generate 10-12 relevant skills'}

WORK HISTORY ({len(work_clean)} entries): {json.dumps(work_clean)[:1200] if work_clean else 'NONE - Create 2-3 project experiences for fresher'}

EDUCATION ({len(edu_clean)} entries): {json.dumps(edu_clean)[:600] if edu_clean else 'NONE - Create realistic education entry'}
"""

    prompt = f"""
{context}

Generate a COMPLETE, ATS-optimized professional resume with ALL sections.

CRITICAL REQUIREMENTS:
1. If skills list is EMPTY (0 items), generate 10-12 relevant technical skills for the target role
2. If work history is EMPTY, create 2-3 relevant PROJECT experiences (not jobs) for a fresher
3. If education is EMPTY, create a realistic Bachelor's degree entry
4. Professional summary MUST be 3-4 sentences highlighting strengths and goals
5. Include ALL sections: Personal Info, Summary, Skills, Experience/Projects, Education, Certifications, Languages, Achievements

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "name": "{name}",
  "email": "{email}",
  "phone": "{phone or '+92-XXX-XXXXXXX'}",
  "location": "{location or 'Pakistan'}",
  "headline": "Professional headline based on target role (e.g., 'Full Stack Developer | MERN Stack Specialist')",
  "summary": "3-4 sentence professional summary highlighting technical expertise, passion for technology, problem-solving abilities, and career aspirations. Make it compelling and achievement-focused.",
  "skills": [
    "Technical Skill 1", "Technical Skill 2", "Technical Skill 3",
    "Technical Skill 4", "Technical Skill 5", "Technical Skill 6",
    "Soft Skill 1", "Soft Skill 2", "Tool 1", "Tool 2", "Framework 1", "Framework 2"
  ],
  "experience": [
    {{
      "title": "Project Name or Job Title",
      "company": "Company Name or 'Personal Project' or 'Academic Project'",
      "duration": "Jan 2024 - Present",
      "description": "• Developed feature X using technologies Y and Z\\n• Achieved result A improving metric B by C%\\n• Collaborated with team to deliver D"
    }},
    {{
      "title": "Another Project/Role",
      "company": "Company or Project Type",
      "duration": "Jun 2023 - Dec 2023",
      "description": "• Built application using tech stack\\n• Implemented features and solved problems\\n• Learned and applied new technologies"
    }}
  ],
  "education": [
    {{
      "degree": "Bachelor of Science in Computer Science",
      "institution": "University Name",
      "year": "2020 - 2024",
      "details": "CGPA: 3.5/4.0 | Relevant Coursework: Data Structures, Algorithms, Web Development"
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "description": "Brief 1-2 line description of what the project does",
      "technologies": ["React", "Node.js", "MongoDB"],
      "link": "github.com/username/project"
    }}
  ],
  "certifications": [
    "Relevant Certification 1 (Platform, Year)",
    "Relevant Certification 2 (Platform, Year)"
  ],
  "languages": ["English - Fluent", "Urdu - Native"],
  "achievements": [
    "Won hackathon or competition",
    "Published article or contributed to open source",
    "Led team project or initiative"
  ]
}}

CRITICAL: Skills array MUST have 10-12 items minimum. Experience MUST have 2-3 entries. Make it professional and realistic.
"""

    try:
        result_text = _call(prompt)
        stripped = _strip_json(result_text)
        result = json.loads(stripped)

        # Validation and fallbacks
        if not result.get('skills') or len(result.get('skills', [])) < 5:
            logger.warning('[GPT] Skills missing or too few, adding fallback skills')
            result['skills'] = [
                'Programming & Development', 'Problem Solving', 'Team Collaboration',
                'Communication Skills', 'Time Management', 'Critical Thinking',
                'Adaptability', 'Leadership', 'Project Management', 'Technical Writing',
                'Git & Version Control', 'Agile Methodologies'
            ]

        if not result.get('experience') or len(result.get('experience', [])) == 0:
            logger.warning('[GPT] Experience missing, adding fallback')
            result['experience'] = [{
                'title': 'Fresher - Seeking First Opportunity',
                'company': 'Recent Graduate',
                'duration': '2024',
                'description': '• Completed comprehensive education in computer science\n• Developed multiple academic and personal projects\n• Ready to contribute technical skills to a dynamic team'
            }]

        if not result.get('education') or len(result.get('education', [])) == 0:
            logger.warning('[GPT] Education missing, adding fallback')
            result['education'] = [{
                'degree': "Bachelor's Degree in Computer Science",
                'institution': 'University',
                'year': '2024',
                'details': 'Completed degree program with focus on software development'
            }]

        # Ensure all fields exist
        result.setdefault('name', name)
        result.setdefault('email', email)
        result.setdefault('phone', phone or '')
        result.setdefault('location', location or '')
        result.setdefault('headline', headline or 'Software Engineer')
        result.setdefault('summary', bio or 'Motivated professional with strong technical skills and passion for technology')
        result.setdefault('projects', [])
        result.setdefault('certifications', [])
        result.setdefault('languages', ['English', 'Urdu'])
        result.setdefault('achievements', [])

        logger.info(f'[GPT] Resume generated: {len(result.get("skills", []))} skills, {len(result.get("experience", []))} experiences, {len(result.get("education", []))} education entries')
        return result

    except Exception as e:
        logger.error(f'[GPT] Resume generation failed: {e}')
        # Return comprehensive fallback
        return {
            'name': name,
            'email': email,
            'phone': phone or '+92-XXX-XXXXXXX',
            'location': location or 'Pakistan',
            'headline': headline or 'Software Engineer | Developer',
            'summary': bio or 'Motivated and detail-oriented software professional with strong foundation in programming and software development. Passionate about creating efficient solutions and continuously learning new technologies. Seeking opportunities to contribute technical skills and grow in a dynamic environment.',
            'skills': [
                'Programming & Development', 'Problem Solving', 'Team Collaboration',
                'Communication Skills', 'Time Management', 'Critical Thinking',
                'Adaptability', 'Leadership', 'Project Management', 'Technical Writing',
                'Git & Version Control', 'Agile Methodologies'
            ],
            'experience': work_clean if work_clean else [{
                'title': 'Fresher - Recent Graduate',
                'company': 'Seeking First Opportunity',
                'duration': '2024',
                'description': '• Completed comprehensive education in computer science\n• Developed multiple academic and personal projects\n• Strong foundation in programming and software development'
            }],
            'education': edu_clean if edu_clean else [{
                'degree': "Bachelor's Degree in Computer Science",
                'institution': 'University',
                'year': '2024',
                'details': 'Completed degree with focus on software development and programming'
            }],
            'projects': [],
            'certifications': [],
            'languages': ['English - Fluent', 'Urdu - Native'],
            'achievements': []
        }


