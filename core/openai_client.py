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
    """Increment daily counter, enforce soft limit, return alert type if threshold crossed."""
    _reset_if_new_day()

    # Hard-block if daily soft limit already exceeded
    if _daily_stats['total_calls'] >= _AI_DAILY_SOFT_LIMIT:
        logger.error(f'[AI] Daily soft limit reached: {_daily_stats["total_calls"]}/{_AI_DAILY_SOFT_LIMIT}. Blocking call.')
        if not _daily_stats['exhausted_notified']:
            _daily_stats['exhausted_notified'] = True
            try:
                from core.ai_notifications import notify_admins_async
                notify_admins_async('AI_QUOTA_EXHAUSTED')
            except Exception:
                pass
        raise Exception('AI_QUOTA_EXHAUSTED')

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


def analyze_emotion_confidence(face_snapshots: list, user_id: str = None) -> dict:
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

    # Cap to 15 snapshots max to prevent token abuse
    safe_snapshots = face_snapshots[:15]
    # Strip any unexpected large fields from each snapshot
    sanitized = []
    for snap in safe_snapshots:
        if isinstance(snap, dict):
            sanitized.append({k: v for k, v in snap.items() if k in (
                'eye_contact', 'x', 'y', 'confidence', 'emotion',
                'head_pose', 'blink_rate', 'timestamp', 'stability'
            )})

    prompt = f"""
Analyze {len(sanitized)} face metric snapshots from an interview.

Data: {json.dumps(sanitized, indent=2)}

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
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
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
    """Generate structured interview debrief summary. Returns dict matching frontend schema."""
    score = evaluation.get('overall_score', 50) if isinstance(evaluation, dict) else 50
    rec = evaluation.get('recommendation', 'maybe') if isinstance(evaluation, dict) else 'maybe'

    prompt = f"""
Generate a detailed, personalized interview debrief for {candidate_name or 'the candidate'}.

Evaluation Data: {json.dumps(evaluation)[:2000]}
Interview Data: {json.dumps(interview)[:800]}

Return JSON (not plain text):
{{
  "headline": "Short punchy headline summarizing performance (e.g. 'Strong Technical Communicator')",
  "performance_tier": "One of: Exceptional, Strong, Developing, Needs Work",
  "executive_summary": "3-4 sentence executive summary of overall interview performance",
  "skill_scores": [
    {{"skill": "Technical Knowledge", "score": 0-100, "feedback": "1 sentence"}},
    {{"skill": "Communication", "score": 0-100, "feedback": "1 sentence"}},
    {{"skill": "Problem Solving", "score": 0-100, "feedback": "1 sentence"}},
    {{"skill": "Confidence", "score": 0-100, "feedback": "1 sentence"}}
  ],
  "top_strengths": [
    {{"title": "Strength Title", "detail": "Brief explanation"}},
    {{"title": "Strength Title", "detail": "Brief explanation"}},
    {{"title": "Strength Title", "detail": "Brief explanation"}}
  ],
  "improvement_areas": [
    {{"title": "Area Title", "action": "Actionable improvement step"}},
    {{"title": "Area Title", "action": "Actionable improvement step"}},
    {{"title": "Area Title", "action": "Actionable improvement step"}}
  ],
  "next_steps": ["Step 1 as a full sentence", "Step 2", "Step 3"],
  "recommended_resources": ["Resource or course name 1", "Resource or course name 2", "Resource or course name 3"],
  "motivational_note": "1-2 inspiring, personalized motivational sentences for the candidate"
}}
"""
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        # Ensure candidate_name is available to view layer
        result.setdefault('candidate_name', candidate_name)
        return result
    except Exception as e:
        logger.warning(f'[GPT] Debrief generation failed: {e}')
        tier = 'Strong' if score >= 70 else 'Developing' if score >= 50 else 'Needs Work'
        return {
            "candidate_name": candidate_name,
            "headline": f"Interview Complete — {score}/100 Overall Score",
            "performance_tier": tier,
            "executive_summary": (
                f"{candidate_name or 'The candidate'} completed the interview with an overall score of {score}/100. "
                f"The recruiter recommendation is: {rec}. "
                "See the detailed breakdown below for specific feedback on each area."
            ),
            "skill_scores": [
                {"skill": "Technical Knowledge", "score": score, "feedback": "See detailed evaluation for breakdown."},
                {"skill": "Communication", "score": score, "feedback": "Evaluated based on interview responses."},
                {"skill": "Problem Solving", "score": score, "feedback": "Based on question responses."},
                {"skill": "Confidence", "score": score, "feedback": "Assessed during the live session."},
            ],
            "top_strengths": [
                {"title": "Interview Completion", "detail": "Successfully completed all interview stages."},
                {"title": "Preparation", "detail": "Demonstrated readiness for the role."},
            ],
            "improvement_areas": [
                {"title": "Review Feedback", "action": "Carefully read the recruiter's comments for detailed improvement suggestions."},
                {"title": "Practice STAR Method", "action": "Structure answers using Situation, Task, Action, Result framework."},
            ],
            "next_steps": [
                "Review the full evaluation report shared by the recruiter.",
                "Practice answering behavioral questions using the STAR method.",
                "Strengthen technical fundamentals relevant to this role.",
            ],
            "recommended_resources": [
                "STAR Method Interview Guide",
                "LeetCode / HackerRank for technical practice",
                "LinkedIn Learning courses for skill development",
            ],
            "motivational_note": (
                f"Every interview is a step forward. Keep refining your skills and the right opportunity will come. "
                f"You scored {score}/100 — use this feedback as fuel for your next attempt!"
            ),
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


# ═══════════════════════════════════════════════════════════════════════════════
# NEW AI FEATURES — Features 1-8
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_voice_tone(audio_metrics: dict, user_id: str = None) -> dict:
    """
    Feature 1: Analyze audio metrics (pitch, energy, pace, pauses) for stress & confidence.
    Frontend extracts metrics via Web Audio API and sends here.
    """
    if not audio_metrics:
        return {
            'tone_score': 50, 'stress_level': 'medium',
            'confidence_from_voice': 50, 'pacing': 'normal',
            'coaching_tip': 'No audio data captured.', 'voice_trend': 'stable'
        }

    prompt = f"""
You are an expert voice & behavioral analyst. Analyze these audio metrics from a live interview.

Audio Metrics:
{json.dumps(audio_metrics, indent=2)}

Return ONLY valid JSON:
{{
  "tone_score": 0-100,
  "stress_level": "low" | "medium" | "high",
  "confidence_from_voice": 0-100,
  "pacing": "too_fast" | "normal" | "too_slow",
  "volume_consistency": "steady" | "erratic",
  "voice_trend": "improving" | "stable" | "deteriorating",
  "filler_word_rate": "low" | "medium" | "high",
  "coaching_tip": "One concrete tip to improve voice delivery",
  "recruiter_insight": "One sentence insight for the recruiter about candidate's vocal confidence"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Voice tone analysis failed: {e}')
        return {
            'tone_score': 65, 'stress_level': 'medium',
            'confidence_from_voice': 65, 'pacing': 'normal',
            'volume_consistency': 'steady', 'voice_trend': 'stable',
            'filler_word_rate': 'low',
            'coaching_tip': 'Speak at a steady, clear pace and project confidence.',
            'recruiter_insight': 'Candidate shows moderate vocal confidence.'
        }


def analyze_realtime_quality(
    transcript_chunk: str,
    question: str,
    job_title: str = '',
    elapsed_seconds: int = 0,
    user_id: str = None
) -> dict:
    """
    Feature 2: Real-time answer quality meter — called every 10s during interview.
    Returns a 0-100 live quality score + color + coaching message for recruiter.
    """
    if not transcript_chunk or len(transcript_chunk.strip()) < 10:
        return {
            'quality_score': 0, 'completeness': 0, 'depth': 0,
            'on_track': False, 'bar_color': 'gray',
            'coach_message': 'Candidate has not spoken yet.',
            'keyword_hits': [], 'estimated_completion': 'early'
        }

    prompt = f"""
Analyze this LIVE interview transcript chunk in real-time.

Job: {job_title}
Question: "{question}"
Answer so far ({elapsed_seconds}s elapsed): "{transcript_chunk}"

Return ONLY valid JSON:
{{
  "quality_score": 0-100,
  "completeness": 0-100,
  "depth": 0-100,
  "on_track": true | false,
  "bar_color": "red" | "orange" | "yellow" | "green",
  "keyword_hits": ["keyword found in answer"],
  "missing_key_points": ["important point not yet mentioned"],
  "coach_message": "Short message for recruiter (e.g. 'Strong start, push for examples')",
  "estimated_completion": "early" | "mid" | "complete" | "over_explaining"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Live quality meter failed: {e}')
        return {
            'quality_score': 50, 'completeness': 40, 'depth': 50,
            'on_track': True, 'bar_color': 'yellow',
            'keyword_hits': [], 'missing_key_points': [],
            'coach_message': 'Monitoring response quality.',
            'estimated_completion': 'mid'
        }


def transcribe_audio_whisper(audio_bytes: bytes, filename: str = 'audio.webm', user_id: str = None) -> str:
    """
    Feature 3: Transcribe audio using OpenAI Whisper API (much more accurate than Web Speech).
    Accepts raw audio bytes (webm/mp4/wav).
    """
    import io
    if not audio_bytes or len(audio_bytes) < 100:
        return ''

    # Check per-user rate limit before consuming quota
    if user_id:
        allowed, remaining, reset_time = ai_rate_limiter.check_limit(user_id, limit=20, window_minutes=60)
        if not allowed:
            raise Exception(f'AI rate limit exceeded. Try again after {reset_time.strftime("%H:%M:%S")}')

    # Increment quota INSIDE try so a quota-already-exceeded check fires first
    _increment_and_check_quota()

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        transcript = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            language='en'
        )
        logger.info(f'[Whisper] Transcribed {len(audio_bytes)} bytes successfully.')
        return transcript.text.strip()
    except Exception as e:
        logger.error(f'[Whisper] Transcription failed: {e}')
        return ''


def summarize_question_response(question: str, transcript: str, job_title: str = '', user_id: str = None) -> dict:
    """
    Feature 3: Generate a structured per-question summary after each answer.
    """
    if not transcript or len(transcript.strip()) < 15:
        return {'summary': '', 'key_points': [], 'score_estimate': 0, 'verdict': 'no_response'}

    prompt = f"""
Summarize this interview answer for the recruiter's notes.

Job: {job_title}
Question: "{question}"
Candidate Answer: "{transcript[:3000]}"

Return ONLY valid JSON:
{{
  "summary": "2-3 sentence summary of what candidate said",
  "key_points": ["main point 1", "main point 2", "main point 3"],
  "score_estimate": 0-10,
  "verdict": "excellent" | "good" | "average" | "weak" | "no_response",
  "missed_aspects": ["what candidate should have mentioned"],
  "standout_moment": "Best thing candidate said, or empty string"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Response summary failed: {e}')
        return {
            'summary': transcript[:200],
            'key_points': [], 'score_estimate': 5,
            'verdict': 'average', 'missed_aspects': [], 'standout_moment': ''
        }


def detect_resume_inconsistencies(
    resume_data: dict,
    live_responses: list,
    job_title: str = '',
    user_id: str = None
) -> dict:
    """
    Feature 4: Lie detection — compare resume claims vs what candidate said live.
    live_responses: list of {'question': '...', 'response': '...'}
    """
    if not resume_data or not live_responses:
        return {'inconsistencies': [], 'risk_level': 'low', 'flagged_claims': [], 'integrity_note': ''}

    resume_summary = {
        'skills': resume_data.get('skills', [])[:20],
        'experience': [
            {'title': e.get('title', ''), 'company': e.get('company', ''), 'duration': e.get('duration', '')}
            for e in (resume_data.get('experience') or [])[:5]
        ],
        'total_years': resume_data.get('total_experience_years', 0),
        'certifications': resume_data.get('certifications', [])[:5],
    }

    responses_text = '\n'.join([
        f"Q: {r.get('question','')}\nA: {r.get('response','')[:500]}"
        for r in live_responses[:5]
    ])

    prompt = f"""
You are an expert HR analyst detecting inconsistencies between a candidate's resume and their live interview responses.

Job Role: {job_title}

RESUME CLAIMS:
{json.dumps(resume_summary, indent=2)}

LIVE INTERVIEW RESPONSES:
{responses_text}

Identify any contradictions, exaggerations, or red flags.

Return ONLY valid JSON:
{{
  "inconsistencies": [
    {{
      "type": "skill_claim" | "experience_gap" | "timeline_mismatch" | "knowledge_gap" | "exaggeration",
      "resume_claim": "What the resume says",
      "interview_evidence": "What the candidate said that contradicts it",
      "severity": "low" | "medium" | "high",
      "flag": "Short flag label e.g. 'Python expertise mismatch'"
    }}
  ],
  "risk_level": "low" | "medium" | "high",
  "flagged_claims": ["claim1", "claim2"],
  "integrity_note": "1-2 sentence overall integrity assessment",
  "verification_questions": ["Follow-up question to verify claim"]
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Inconsistency detection failed: {e}')
        return {
            'inconsistencies': [], 'risk_level': 'low',
            'flagged_claims': [], 'integrity_note': 'Analysis unavailable.',
            'verification_questions': []
        }


def generate_recruiter_coaching(
    transcript_chunk: str,
    current_question: str,
    candidate_performance: dict,
    job_title: str = '',
    user_id: str = None
) -> dict:
    """
    Feature 5: Real-time AI coaching tips for the recruiter during the interview.
    candidate_performance: {'quality_score': 0-100, 'depth': 0-100, 'on_track': bool}
    """
    if not transcript_chunk:
        return {
            'coaching_action': 'wait', 'urgency': 'low',
            'suggestion': 'Continue with the current question.',
            'followup_question': '', 'observation': ''
        }

    perf_summary = json.dumps(candidate_performance) if candidate_performance else '{}'

    prompt = f"""
You are an expert interview coach helping a recruiter in real-time.

Job: {job_title}
Current Question: "{current_question}"
Candidate Performance So Far: {perf_summary}
Latest Transcript: "{transcript_chunk[:1500]}"

Analyze the situation and coach the recruiter on what to do next.

Return ONLY valid JSON:
{{
  "coaching_action": "probe_deeper" | "move_on" | "challenge" | "clarify" | "encourage" | "redirect" | "wait",
  "urgency": "low" | "medium" | "high",
  "observation": "What you notice about the candidate's response (1 sentence)",
  "suggestion": "Specific advice for the recruiter (1-2 sentences)",
  "followup_question": "Exact follow-up question to ask (or empty string if not needed)",
  "tone_advice": "How recruiter should adjust their tone/approach"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Recruiter coaching failed: {e}')
        return {
            'coaching_action': 'wait', 'urgency': 'low',
            'observation': 'Candidate is responding.',
            'suggestion': 'Listen carefully and note key points.',
            'followup_question': '', 'tone_advice': 'Remain neutral and encouraging.'
        }


def generate_followup_email(
    candidate_name: str,
    job_title: str,
    decision: str,
    evaluation_data: dict,
    company_name: str = 'InnovAIte',
    user_id: str = None
) -> dict:
    """
    Feature 6: Generate personalized post-interview follow-up email.
    decision: 'selected' | 'rejected' | 'hold' | 'next_round'
    """
    score = evaluation_data.get('overall_score', 0)
    strengths = evaluation_data.get('strengths', [])[:3]
    summary = evaluation_data.get('summary', '')

    decision_labels = {
        'selected': 'OFFER EXTENDED',
        'rejected': 'NOT MOVING FORWARD',
        'hold': 'ON HOLD',
        'next_round': 'ADVANCING TO NEXT ROUND'
    }

    prompt = f"""
Generate a professional, warm, and personalized post-interview follow-up email.

Candidate: {candidate_name}
Role: {job_title}
Company: {company_name}
Decision: {decision_labels.get(decision, 'UNDER REVIEW')}
Interview Score: {score}/100
Key Strengths: {json.dumps(strengths)}
Interview Summary: {summary[:500] if summary else 'Not provided'}

Guidelines:
- For 'selected': Enthusiastic, clear next steps, include [SALARY] and [START_DATE] placeholders
- For 'rejected': Empathetic, constructive, keep door open for future
- For 'hold': Transparent, positive framing, timeline for decision
- For 'next_round': Clear, exciting, specify what next round involves

Return ONLY valid JSON:
{{
  "subject": "Email subject line",
  "greeting": "Dear {candidate_name},",
  "body": "Full email body (3-5 paragraphs, professional yet warm)",
  "closing": "Sincerely,\\n[Recruiter Name]\\n{company_name}",
  "tone": "enthusiastic" | "empathetic" | "professional" | "encouraging",
  "key_message": "1 sentence summary of the email's main message"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Follow-up email generation failed: {e}')
        return {
            'subject': f'Regarding Your Interview for {job_title}',
            'greeting': f'Dear {candidate_name},',
            'body': f'Thank you for interviewing for the {job_title} position. We will be in touch soon with our decision.',
            'closing': f'Sincerely,\nThe {company_name} Team',
            'tone': 'professional',
            'key_message': 'Follow-up regarding interview outcome.'
        }


def analyze_job_description(
    job_title: str,
    job_description: str,
    user_id: str = None
) -> dict:
    """
    Feature 7: AI-powered JD analyzer — attractiveness, bias detection, clarity, improvements.
    """
    if not job_description or len(job_description.strip()) < 50:
        return {
            'attractiveness_score': 0, 'clarity_score': 0, 'bias_score': 100,
            'bias_flags': [], 'improvements': [], 'readability': 'poor',
            'summary': 'Job description too short to analyze.'
        }

    prompt = f"""
You are an expert HR content analyst. Analyze this job description for quality, attractiveness, and bias.

Job Title: {job_title}
Job Description:
{job_description[:3000]}

Return ONLY valid JSON:
{{
  "attractiveness_score": 0-100,
  "clarity_score": 0-100,
  "bias_score": 0-100,
  "readability": "excellent" | "good" | "average" | "poor",
  "bias_flags": [
    {{
      "text": "exact phrase from JD",
      "type": "gender_coded" | "age_bias" | "exclusionary" | "vague_requirement",
      "suggestion": "Better alternative phrasing"
    }}
  ],
  "missing_sections": ["e.g. salary range", "growth opportunities", "team culture"],
  "improvements": ["Specific improvement 1", "Specific improvement 2"],
  "strengths": ["What the JD does well"],
  "candidate_appeal": "high" | "medium" | "low",
  "estimated_applicant_quality": "senior" | "mid" | "junior" | "mixed",
  "summary": "2-3 sentence overall assessment"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] JD analysis failed: {e}')
        return {
            'attractiveness_score': 60, 'clarity_score': 60, 'bias_score': 80,
            'readability': 'average', 'bias_flags': [],
            'missing_sections': [], 'improvements': [],
            'strengths': [], 'candidate_appeal': 'medium',
            'estimated_applicant_quality': 'mixed',
            'summary': 'JD analysis unavailable. Please review manually.'
        }


def calibrate_interview_difficulty(
    resume_data: dict,
    job_title: str,
    job_description: str = '',
    user_id: str = None
) -> dict:
    """
    Feature 8: Auto-calibrate interview difficulty based on candidate's resume level.
    Returns recommended difficulty + question distribution.
    """
    resume_summary = {
        'skills_count': len(resume_data.get('skills', [])),
        'total_years': resume_data.get('total_experience_years', 0),
        'experience_count': len(resume_data.get('experience') or []),
        'skills_sample': (resume_data.get('skills') or [])[:10],
        'certifications': (resume_data.get('certifications') or [])[:5],
        'education': [
            e.get('degree', '') for e in (resume_data.get('education') or [])[:2]
        ]
    }

    prompt = f"""
Calibrate the difficulty level for an interview based on the candidate's resume.

Job: {job_title}
Job Description: {job_description[:500] if job_description else 'Not provided'}
Resume Summary: {json.dumps(resume_summary, indent=2)}

Return ONLY valid JSON:
{{
  "experience_level": "entry" | "junior" | "mid" | "senior" | "lead",
  "recommended_difficulty": "easy" | "medium" | "hard",
  "rationale": "1-2 sentences explaining why this difficulty is appropriate",
  "question_distribution": {{
    "easy": 0-10,
    "medium": 0-10,
    "hard": 0-10
  }},
  "focus_areas": ["Area to probe", "Another area"],
  "avoid_topics": ["Topics candidate clearly knows too well or not at all"],
  "estimated_years_experience": 0,
  "seniority_confidence": "high" | "medium" | "low"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Difficulty calibration failed: {e}')
        return {
            'experience_level': 'mid',
            'recommended_difficulty': 'medium',
            'rationale': 'Default medium difficulty applied.',
            'question_distribution': {'easy': 3, 'medium': 4, 'hard': 3},
            'focus_areas': ['Technical skills', 'Problem solving'],
            'avoid_topics': [],
            'estimated_years_experience': 0,
            'seniority_confidence': 'low'
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



# ═══════════════════════════════════════════════════════════════════════════════
# MISSING 15 AI FEATURES - CANDIDATE SIDE (4 Features)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_application_status(resume_data: dict, job_data: dict, user_id: str = None) -> dict:
    """
    Feature 1: Predict likelihood of getting interview based on profile match.
    Returns probability score and reasoning.
    """
    resume_summary = {
        'skills': (resume_data.get('skills') or [])[:15],
        'experience_years': resume_data.get('total_experience_years', 0),
        'education': [e.get('degree', '') for e in (resume_data.get('education') or [])[:2]],
    }
    
    job_summary = {
        'title': job_data.get('title', ''),
        'requirements': (job_data.get('requirements') or [])[:10],
        'description': (job_data.get('description', ''))[:500],
    }

    prompt = f"""
Predict the likelihood of this candidate getting an interview for this job.

CANDIDATE PROFILE:
{json.dumps(resume_summary, indent=2)}

JOB POSTING:
{json.dumps(job_summary, indent=2)}

Return ONLY valid JSON:
{{
  "interview_probability": 0-100,
  "confidence": "high" | "medium" | "low",
  "match_level": "excellent" | "good" | "fair" | "weak",
  "key_strengths": ["strength1", "strength2"],
  "key_gaps": ["gap1", "gap2"],
  "recommendation": "1-2 sentence advice for candidate",
  "estimated_response_time": "1-3 days" | "3-7 days" | "7-14 days" | "unlikely"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Application status prediction failed: {e}')
        return {
            'interview_probability': 50,
            'confidence': 'low',
            'match_level': 'fair',
            'key_strengths': ['Profile submitted'],
            'key_gaps': ['Analysis unavailable'],
            'recommendation': 'Keep applying to similar roles.',
            'estimated_response_time': '7-14 days'
        }


def suggest_profile_improvements(current_profile: dict, target_role: str = '', user_id: str = None) -> dict:
    """
    Feature 2: AI suggests what to add to profile based on role and experience level.
    """
    profile_summary = {
        'name': current_profile.get('name', ''),
        'headline': current_profile.get('headline', ''),
        'bio': current_profile.get('bio', ''),
        'skills_count': len(current_profile.get('detailed_skills') or []),
        'work_history_count': len(current_profile.get('work_history') or []),
        'education_count': len(current_profile.get('education_history') or []),
        'has_resume': current_profile.get('has_resume', False),
        'profile_completion': current_profile.get('profile_completion', 0),
    }

    prompt = f"""
Analyze this candidate profile and suggest improvements for {target_role or 'their career goals'}.

CURRENT PROFILE:
{json.dumps(profile_summary, indent=2)}

Return ONLY valid JSON:
{{
  "completion_score": 0-100,
  "priority_improvements": [
    {{
      "section": "headline" | "bio" | "skills" | "experience" | "education" | "resume",
      "current_status": "missing" | "incomplete" | "weak",
      "suggestion": "Specific actionable suggestion",
      "impact": "high" | "medium" | "low",
      "example": "Example of what to add"
    }}
  ],
  "quick_wins": ["Quick improvement 1", "Quick improvement 2"],
  "long_term_goals": ["Goal 1", "Goal 2"],
  "estimated_time_to_complete": "30 minutes" | "1 hour" | "2-3 hours"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Profile improvement suggestions failed: {e}')
        return {
            'completion_score': profile_summary.get('profile_completion', 50),
            'priority_improvements': [
                {
                    'section': 'skills',
                    'current_status': 'incomplete',
                    'suggestion': 'Add more technical skills relevant to your target role',
                    'impact': 'high',
                    'example': 'Python, JavaScript, React, Node.js'
                }
            ],
            'quick_wins': ['Update your headline', 'Add a professional bio'],
            'long_term_goals': ['Complete work history', 'Upload resume'],
            'estimated_time_to_complete': '1 hour'
        }


def calculate_readiness_score(profile_data: dict, practice_history: list = None, user_id: str = None) -> dict:
    """Feature 3: Calculate interview readiness level (0-100%)."""
    history_summary = []
    if practice_history:
        for h in (practice_history or [])[:5]:
            history_summary.append({
                'score': h.get('overall_score', 0),
                'recommendation': h.get('recommendation', 'MAYBE'),
                'job_title': h.get('job_title', ''),
            })
    _skills = profile_data.get('skills', [])
    _n_exp = len(profile_data.get('work_history', []))
    _bio = str(profile_data.get('bio', ''))[:200]
    _resume = bool(profile_data.get('resume_uploaded', False))
    _hist = json.dumps(history_summary) if history_summary else 'None yet.'
    prompt = (
        'Analyze this candidate interview readiness.\n\n'
        'Profile:\n'
        f'- Skills: {_skills}\n'
        f'- Experience positions: {_n_exp}\n'
        f'- Bio: {_bio}\n'
        f'- Has Resume: {_resume}\n\n'
        f'Past Interviews: {_hist}\n\n'
        'Return JSON ONLY:\n'
        '{\n'
        '  "readiness_score": 0-100,\n'
        '  "readiness_level": "Beginner|Developing|Intermediate|Advanced|Interview-Ready",\n'
        '  "readiness_summary": "2-3 sentence assessment",\n'
        '  "strengths": ["s1", "s2", "s3"],\n'
        '  "gaps": ["g1", "g2"],\n'
        '  "recommended_actions": ["a1", "a2", "a3"],\n'
        '  "estimated_prep_time": "e.g. 2-3 weeks",\n'
        '  "confidence_trend": "improving|declining|stable|no data"\n'
        '}'
    )
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        return result
    except Exception as e:
        logger.warning(f"[GPT] Readiness score failed: {e}")
        avg = sum(h.get("overall_score", 0) for h in history_summary) / max(len(history_summary), 1) if history_summary else 40
        has_skills = bool(profile_data.get("skills"))
        has_exp = bool(profile_data.get("work_history"))
        score = min(int(avg * 0.6 + (20 if has_skills else 0) + (20 if has_exp else 0)), 100)
        return {
            "readiness_score": score,
            "readiness_level": "Intermediate" if score >= 60 else "Developing" if score >= 40 else "Beginner",
            "readiness_summary": f"Based on {len(history_summary)} interview(s), readiness is {score}/100.",
            "strengths": ["Profile data available"] if has_skills else ["Getting started"],
            "gaps": ["Complete your profile"] if not has_skills else ["Practice more interviews"],
            "recommended_actions": ["Upload resume", "Take a practice interview", "Complete profile"],
            "estimated_prep_time": "2-3 weeks",
            "confidence_trend": "no data" if not history_summary else "stable",
        }
