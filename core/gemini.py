"""
core/gemini.py — Gemini AI Client Utility
All Gemini API calls go through this module for consistency.
"""
import json
import logging
from datetime import date
import google.generativeai as genai
from django.conf import settings
from core.rate_limiter import ai_rate_limiter

logger = logging.getLogger('innovaite')

# Configure Gemini once at import
genai.configure(api_key=settings.GEMINI_API_KEY)

# Use Gemini 1.5 Flash — fast and cost-effective
MODEL = genai.GenerativeModel('gemini-1.5-flash-latest')

# ── Daily AI Usage Tracker (in-memory, resets at midnight) ──────────────────
# WARNING_THRESHOLD: notify admins when this many calls made today
# DAILY_SOFT_LIMIT:  treat as exhausted for safety before Google cuts us off
_AI_WARNING_THRESHOLD = getattr(settings, 'AI_WARNING_THRESHOLD', 400)
_AI_DAILY_SOFT_LIMIT  = getattr(settings, 'AI_DAILY_SOFT_LIMIT', 480)

_daily_stats = {
    'date': None,         # date object — resets counter when day changes
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


def _strip_json(text: str) -> str:
    """Strip markdown code fences from Gemini responses before JSON parsing."""
    text = text.strip()
    if text.startswith('```'):
        # Remove opening fence and optional language tag
        text = text[3:]
        if text.lower().startswith('json'):
            text = text[4:]
        # Remove closing fence
        if text.endswith('```'):
            text = text[:-3]
    return text.strip()


def _call(prompt: str, user_id: str = None) -> str:
    """
    Make a Gemini API call with rate limiting and proper error handling.
    Distinguishes between quota exhaustion, invalid key, and transient errors.
    """
    # Check rate limit if user_id provided
    if user_id:
        allowed, remaining, reset_time = ai_rate_limiter.check_limit(user_id, limit=20, window_minutes=60)
        if not allowed:
            raise Exception(f'AI rate limit exceeded. Try again after {reset_time.strftime("%H:%M:%S")}')
        logger.info(f'[AI] User {user_id} - {remaining} calls remaining')

    # Increment daily counter — fire warning notification if threshold crossed
    alert_type = _increment_and_check_quota()
    if alert_type:
        try:
            from core.ai_notifications import notify_admins_async
            notify_admins_async(alert_type)
        except Exception as notify_err:
            logger.warning(f'[AI] Could not send quota warning notification: {notify_err}')

    try:
        response = MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        error_str = str(e).lower()

        # Google API quota exhausted (free tier daily/minute limit)
        if 'quota' in error_str or 'resource_exhausted' in error_str or '429' in error_str:
            logger.error(f'[AI] Gemini quota exhausted: {e}')
            # Notify admins once per day that quota is exhausted
            if not _daily_stats['exhausted_notified']:
                _daily_stats['exhausted_notified'] = True
                try:
                    from core.ai_notifications import notify_admins_async
                    notify_admins_async('AI_QUOTA_EXHAUSTED')
                except Exception as notify_err:
                    logger.warning(f'[AI] Could not send exhausted notification: {notify_err}')
            raise Exception('AI_QUOTA_EXHAUSTED')

        # Invalid or missing API key
        if 'api_key' in error_str or 'invalid' in error_str or 'api key' in error_str or '400' in error_str:
            logger.error(f'[AI] Gemini API key invalid: {e}')
            raise Exception('AI_KEY_INVALID')

        # Billing issue
        if 'billing' in error_str or 'payment' in error_str:
            logger.error(f'[AI] Gemini billing issue: {e}')
            raise Exception('AI_BILLING_ISSUE')

        # Generic failure
        logger.error(f'[AI] Gemini API call failed: {e}')
        raise Exception(f'AI_ERROR: {str(e)}')


def parse_resume_with_ai(raw_text: str, user_id: str = None) -> dict:
    """
    Use Gemini to intelligently parse resume text into structured data.
    Returns empty dict on failure so the caller falls back to rule-based parsing.
    """
    if not raw_text or len(raw_text.strip()) < 20:
        return {}

    # Send up to 8000 chars for better accuracy on long resumes
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

IMPORTANT:
- For total_experience_years: sum all work experience durations, return a number.
- For skills: extract ALL technical and soft skills mentioned anywhere in the resume.
- Be thorough and accurate. Do not hallucinate data not present in the resume.

RESUME TEXT:
{text_chunk}
"""
    try:
        result_text = _call(prompt)

        # Robustly strip all Gemini markdown wrappers
        stripped = result_text.strip()
        if stripped.startswith('```'):
            # Remove opening fence
            stripped = stripped[3:]
            # Remove optional language tag (e.g. 'json')
            if stripped.lower().startswith('json'):
                stripped = stripped[4:]
            # Remove closing fence
            if stripped.endswith('```'):
                stripped = stripped[:-3]
        stripped = stripped.strip()

        parsed = json.loads(stripped)

        # Validate minimum useful data
        if not parsed.get('skills') and not parsed.get('name'):
            logger.warning('[Resume AI] Gemini returned empty name+skills — falling back to rule-based.')
            return {}

        # Ensure all expected keys exist with defaults
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

        # Normalise skills to strings
        parsed['skills'] = [str(s).strip() for s in parsed['skills'] if s]

        parsed['raw_text'] = raw_text[:3000]
        logger.info(f'[Resume AI] Parsed {len(parsed["skills"])} skills, {len(parsed["experience"])} roles via Gemini.')
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f'[Resume AI] JSON decode failed: {e} — falling back to rule-based.')
        return {}
    except Exception as e:
        logger.error(f'[Resume AI] Gemini parse_resume failed: {e}')
        return {}



def generate_interview_questions(
    job_title: str,
    job_description: str,
    num_questions: int = 8,
    categories: list = None,
    resume_data: dict = None,
    user_id: str = None
) -> list:
    """
    Use Gemini to auto-generate smart, tailored interview questions.
    If resume_data is provided, questions will be customized to the candidate's background.
    """
    if not job_title:
        return []

    categories_str = ', '.join(categories) if categories else 'general, technical, behavioral'
    
    context_str = f"Job Title: {job_title}\nJob Description: {job_description or 'Not provided'}"
    if resume_data:
        context_str += f"\n\nCandidate Resume Data: {json.dumps(resume_data)}"

    tailoring_instruction = ""
    if resume_data:
        tailoring_instruction = "IMPORTANT: Since resume data is provided, generate at least 4 questions that specifically reference the candidate's past projects, company experiences, or specific technologies mentioned in their resume to verify their depth of knowledge."

    prompt = f"""
You are an expert HR interviewer for a top-tier tech firm. Generate {num_questions} high-quality interview questions based on the context below.

CONTEXT:
{context_str}

{tailoring_instruction}

Return ONLY a valid JSON array of objects with these exact keys:
[
  {{
    "text": "Question text",
    "category": "technical", "behavioral", or "general",
    "expected_keywords": ["keyword1", "keyword2"],
    "ideal_answer": "What a perfect senior candidate would say",
    "difficulty": "easy", "medium", or "hard"
  }}
]

Generate exactly {num_questions} questions:
"""
    try:
        result_text = _call(prompt, user_id=user_id)
        # Robust cleanup
        stripped = result_text.strip()
        if stripped.startswith('```'):
            stripped = stripped[3:]
            if stripped.lower().startswith('json'):
                stripped = stripped[4:]
            if stripped.endswith('```'):
                stripped = stripped[:-3]
        
        questions = json.loads(stripped.strip())
        return questions[:num_questions] if isinstance(questions, list) else []
    except Exception as e:
        logger.error(f'[Gemini] Question generation failed: {e}')
        return []


def analyze_behavioral_traits(transcript, user_id: str = None):
    """Analyze transcript for confidence, fluency, and filler words."""
    prompt = f"""
    Analyze the following interview transcript for behavioral traits:
    Transcript: {transcript}
    
    Provide a JSON response with:
    - confidence_score (0-100)
    - fluency_score (0-100)
    - filler_count (estimate number of 'um', 'uh', 'like')
    - summary (2 sentence behavioral overview)
    """
    try:
        response = _call(prompt, user_id=user_id)
        return json.loads(_strip_json(response))
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[Gemini] Behavioral analysis failed: {str(e)}")
        return {"confidence_score": 70, "fluency_score": 70, "filler_count": 0, "summary": "Behavioral analysis unavailable."}

def check_integrity_plagiarism(responses, user_id: str = None):
    """Check if responses are suspiciously similar to LLM outputs or show signs of external help."""
    prompt = f"""
    Review these interview responses for signs of plagiarism or AI-generated content (ChatGPT-style):
    Responses: {responses}
    
    Provide a JSON response with:
    - integrity_score (0-100, 100 is high integrity)
    - notes (any red flags or 'Suspiciously similar to AI' notes)
    """
    try:
        response = _call(prompt, user_id=user_id)
        return json.loads(_strip_json(response))
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[Gemini] Integrity check failed: {str(e)}")
        return {"integrity_score": 90, "notes": "Integrity check skipped."}

def generate_offer_letter(candidate_name, job_title, evaluation_score):
    """Generate a professional, high-conversion offer letter draft."""
    prompt = f"""
    Generate a professional and welcoming Job Offer Letter draft for:
    Candidate: {candidate_name}
    Role: {job_title}
    AI Evaluation Score: {evaluation_score}/100
    
    The tone should be enthusiastic and elite. Keep placeholders for salary and start date using [BRACKETS].
    """
    try:
        return _call(prompt)
    except Exception as e:
        logger.warning(f"[Gemini] Offer letter generation failed: {str(e)}")
        return "Offer letter generation failed."

def generate_executive_summary(interview_data, evaluation_results):
    """Generate a 3-paragraph executive summary for HR decision makers."""
    prompt = f"""
    Write a 3-paragraph executive summary for this candidate's interview:
    Role: {interview_data.get('job_title')}
    Scores: {evaluation_results}
    
    Paragraph 1: Technical fit and key achievements.
    Paragraph 2: Behavioral traits and cultural alignment.
    Paragraph 3: Final hiring recommendation and 'Red Flags' if any.
    Tone: Professional, direct, and data-driven.
    """
    try:
        return _call(prompt)
    except Exception as e:
        logger.warning(f"[Gemini] Executive summary generation failed: {str(e)}")
        return "Executive summary generation failed."

def analyze_job_fitment(resume_data, job_description):
    """Deep semantic matching of resume experience vs job requirements."""
    prompt = f"""
    Perform a Deep Job-Fitment Analysis:
    Resume: {resume_data}
    Job Description: {job_description}
    
    Don't just match keywords. Analyze 'Experience Relevance'.
    Return JSON:
    - fitment_score (0-100)
    - matched_dimensions (array of things that match well)
    - missing_relevance (array of gaps in experience depth)
    - suggestion (1 sentence)
    """
    try:
        response = _call(prompt)
        return json.loads(_strip_json(response))
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[Gemini] Job fitment analysis failed: {str(e)}")
        return {"fitment_score": 60, "matched_dimensions": [], "missing_relevance": [], "suggestion": "Manual review recommended."}

def analyze_culture_fit(transcript, company_values):
    """Analyze if candidate's language and attitude align with company values."""
    prompt = f"""
    Analyze if this candidate aligns with these Company Values:
    Values: {company_values}
    Transcript: {transcript}
    
    Return JSON:
    - culture_score (0-100)
    - aligned_values (array of values they demonstrated)
    - red_flags (array of potential cultural misalignments)
    """
    try:
        response = _call(prompt)
        return json.loads(_strip_json(response))
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"[Gemini] Culture fit analysis failed: {str(e)}")
        return {"culture_score": 70, "aligned_values": [], "red_flags": []}


def enhance_evaluation_summary(
    overall_score: float,
    recommendation: str,
    criterion_results: list,
    job_title: str = '',
    user_id: str = None
) -> str:
    """
    Use Gemini to generate a rich, human-readable HR evaluation summary.
    """
    criteria_text = '\n'.join([
        f"- {cr['criterion']}: {cr['score']}/10 — {cr['explanation']}"
        for cr in criterion_results
    ])

    prompt = f"""
You are an expert HR evaluation assistant. Write a professional evaluation summary.

Role: {job_title or 'Not specified'}
Overall Score: {overall_score}/100
Recommendation: {recommendation.replace('_', ' ').title()}

Criterion Scores:
{criteria_text}

Write a 3-4 sentence professional summary focusing on strengths, weaknesses, and hiring recommendation.
Return ONLY the summary text.
"""
    try:
        return _call(prompt, user_id=user_id)
    except Exception:
        return f"Candidate scored {overall_score}/100. Recommendation: {recommendation.replace('_', ' ').title()}."


def analyze_response_semantics(question: str, ideal_answer: str, candidate_response: str, user_id: str = None) -> dict:
    """
    Use Gemini to perform a deep semantic comparison.
    """
    if not candidate_response or len(candidate_response.strip()) < 10:
        return {"score": 0, "explanation": "Response too short for semantic analysis.", "missing_points": []}

    prompt = f"""
Compare the candidate's response to the ideal answer.

Question: {question}
Ideal Answer: {ideal_answer}
Candidate Response: {candidate_response}

Analyze based on accuracy, relevance, and completeness.
Return ONLY a JSON:
{{
  "score": float (0-10),
  "explanation": "2-3 sentences",
  "missing_points": ["point1", "point2"]
}}
"""
    try:
        result_text = _call(prompt, user_id=user_id)
        return json.loads(_strip_json(result_text))
    except Exception:
        return {"score": 5.0, "explanation": "AI semantic analysis unavailable.", "missing_points": []}


def generate_candidate_hints(question: str, category: str = 'general', user_id: str = None) -> str:
    """
    Generate helpful hints for the candidate.
    user_id is used for per-user rate limiting.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Real-Time Emotion & Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────

def analyze_emotion_confidence(face_snapshots: list) -> dict:
    """
    Analyze a list of face metric snapshots captured during the interview.
    Each snapshot: {expression: str, eye_contact: bool, head_stable: bool, timestamp: float}
    Returns aggregated proctoring intelligence.
    """
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
You are an expert behavioral psychologist analyzing video interview data.
You have received {len(face_snapshots)} face metric snapshots from a candidate during an interview.

Face Snapshot Data (sampled):
{json.dumps(face_snapshots[:20], indent=2)}

Analyze this data and return a JSON object:
{{
  "emotion_score": integer 0-100 (higher = more positive/confident emotions),
  "confidence_level": "high" | "medium" | "low",
  "eye_contact_pct": float 0-100 (% of time maintaining eye contact),
  "stability_pct": float 0-100 (% of time head was stable/composed),
  "dominant_emotion": "confident" | "nervous" | "neutral" | "engaged" | "distracted",
  "emotion_trend": "improving" | "stable" | "declining",
  "coaching_tip": "1 actionable sentence to improve interview body language"
}}

Return ONLY the JSON. No markdown.
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Emotion analysis failed: {e}')
        return {
            'emotion_score': 60,
            'confidence_level': 'medium',
            'eye_contact_pct': 60.0,
            'stability_pct': 70.0,
            'dominant_emotion': 'neutral',
            'emotion_trend': 'stable',
            'coaching_tip': 'Maintain eye contact and speak clearly.',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Live Transcript Semantic Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_live_transcript_chunk(
    transcript: str,
    question: str,
    job_title: str = '',
    user_id: str = None
) -> dict:
    """
    Perform real-time semantic analysis on a transcript chunk during interview.
    Returns live coaching signals for the recruiter panel.
    """
    if not transcript or len(transcript.strip()) < 15:
        return {'relevance': 0, 'keywords_detected': [], 'signal': 'too_short', 'live_tip': ''}

    prompt = f"""
You are a real-time interview coach analyzing a candidate's LIVE spoken answer.

Job Title: {job_title or 'Software Role'}
Interview Question: {question}
Transcript So Far: "{transcript}"

Analyze and return JSON:
{{
  "relevance": integer 0-100 (how relevant is the answer to the question),
  "keywords_detected": ["keyword1", "keyword2"],
  "signal": "on_track" | "off_topic" | "needs_examples" | "too_brief" | "strong",
  "sentiment": "positive" | "neutral" | "negative",
  "filler_words": integer (count of um/uh/like in transcript),
  "live_tip": "1 short actionable tip for the recruiter to guide the candidate RIGHT NOW"
}}

Return ONLY the JSON.
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[Gemini] Live transcript analysis failed: {e}')
        return {
            'relevance': 50,
            'keywords_detected': [],
            'signal': 'on_track',
            'sentiment': 'neutral',
            'filler_words': 0,
            'live_tip': '',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Adaptive Question Difficulty Engine
# ─────────────────────────────────────────────────────────────────────────────

def suggest_adaptive_question(
    current_question: str,
    candidate_response: str,
    current_difficulty: str,
    job_title: str,
    category: str = 'technical'
) -> dict:
    """
    Based on how well the candidate answered the current question,
    suggest the next question with adjusted difficulty.
    """
    prompt = f"""
You are an expert interviewer running an adaptive interview for: {job_title}

Current Question ({current_difficulty} difficulty, {category}):
"{current_question}"

Candidate Response:
"{candidate_response}"

Evaluate the response quality and suggest the NEXT question.
Rules:
- If response quality is HIGH (score 7-10): suggest a HARDER follow-up
- If response quality is MEDIUM (score 4-6): suggest a SIMILAR difficulty question
- If response quality is LOW (score 0-3): suggest an EASIER clarifying question

Return ONLY JSON:
{{
  "response_quality_score": float 0-10,
  "response_assessment": "1 sentence on what was good/missing",
  "next_difficulty": "easy" | "medium" | "hard",
  "next_question": "The full next interview question text",
  "next_category": "technical" | "behavioral" | "general",
  "expected_keywords": ["key1", "key2", "key3"],
  "ideal_answer": "Brief ideal answer for scoring reference"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Adaptive question failed: {e}')
        return {
            'response_quality_score': 5.0,
            'response_assessment': 'Unable to assess response.',
            'next_difficulty': current_difficulty,
            'next_question': 'Can you elaborate on your previous answer with a specific example?',
            'next_category': category,
            'expected_keywords': [],
            'ideal_answer': '',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Resume vs JD Gap Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def analyze_resume_jd_gap(
    resume_data: dict,
    job_description: str,
    job_title: str,
    requirements: list = None
) -> dict:
    """
    Deep gap analysis comparing a candidate's resume against a specific job posting.
    Returns actionable skill gaps and match percentage.
    """
    resume_summary = {
        'skills': resume_data.get('skills', []),
        'experience': resume_data.get('experience', []),
        'total_years': resume_data.get('total_experience_years', 0),
        'education': resume_data.get('education', []),
        'certifications': resume_data.get('certifications', []),
    }

    prompt = f"""
You are a senior talent acquisition specialist performing a Gap Analysis.

Job Title: {job_title}
Job Description: {job_description[:3000]}
Job Requirements: {json.dumps(requirements or [])}

Candidate Resume Summary:
{json.dumps(resume_summary, indent=2)}

Perform a comprehensive gap analysis. Return ONLY JSON:
{{
  "match_percentage": integer 0-100,
  "match_tier": "Excellent" | "Good" | "Fair" | "Weak",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "experience_gap": "e.g. Needs 2 more years in cloud architecture",
  "education_match": true | false,
  "strengths": ["strength1", "strength2", "strength3"],
  "gaps": ["gap1", "gap2", "gap3"],
  "recommended_actions": [
    "Action 1 to close the gap",
    "Action 2 to close the gap"
  ],
  "interview_readiness": "Ready" | "Nearly Ready" | "Needs Prep",
  "summary": "2 sentence overall assessment"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Gap analysis failed: {e}')
        return {
            'match_percentage': 50,
            'match_tier': 'Fair',
            'matched_skills': [],
            'missing_skills': [],
            'experience_gap': 'Analysis unavailable.',
            'education_match': True,
            'strengths': [],
            'gaps': [],
            'recommended_actions': [],
            'interview_readiness': 'Needs Prep',
            'summary': 'Gap analysis could not be completed at this time.',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Smart Interview Slot Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def suggest_interview_slots(
    job_title: str,
    duration_minutes: int,
    recruiter_timezone: str = 'UTC',
    candidate_timezone: str = 'UTC',
    existing_interviews: list = None,
    preferred_days: list = None
) -> dict:
    """
    AI-powered suggestion of optimal interview time slots.
    Avoids conflicts and respects timezones and cognitive load research.
    """
    existing_count = len(existing_interviews) if existing_interviews else 0

    prompt = f"""
You are an expert scheduling AI for HR departments.

Context:
- Job: {job_title}
- Interview Duration: {duration_minutes} minutes
- Recruiter Timezone: {recruiter_timezone}
- Candidate Timezone: {candidate_timezone}
- Existing interviews this week: {existing_count}
- Preferred days: {preferred_days or ['Monday', 'Tuesday', 'Wednesday', 'Thursday']}

Based on cognitive load research and HR best practices, suggest 5 optimal interview slots for the NEXT 5 business days from today (2026-04-09).

Rules:
- Avoid Monday mornings and Friday afternoons
- Prefer 10am-12pm and 2pm-4pm slots (recruiter's timezone)
- Space interviews at least 2 hours apart
- Consider candidate's timezone for reasonable hours

Return ONLY JSON:
{{
  "suggested_slots": [
    {{
      "datetime_utc": "2026-04-10T09:00:00",
      "day_label": "Thursday, Apr 10",
      "time_recruiter": "2:00 PM UTC+5",
      "time_candidate": "9:00 AM UTC",
      "quality_score": 95,
      "reason": "Peak cognitive performance window"
    }}
  ],
  "scheduling_tip": "1 sentence general scheduling recommendation",
  "optimal_slot_index": 0
}}

Generate exactly 5 slots.
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Slot suggestion failed: {e}')
        return {
            'suggested_slots': [],
            'scheduling_tip': 'Schedule interviews between 10am-12pm or 2pm-4pm for best results.',
            'optimal_slot_index': 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Candidate Ranking AI Engine
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates_for_job(
    job_title: str,
    job_description: str,
    candidates: list
) -> dict:
    """
    AI-powered ranking of multiple candidates for a single job.
    candidates: list of dicts with {candidate_id, name, overall_score, skills, experience_years, recommendation}
    """
    if not candidates:
        return {'ranked': [], 'ranking_rationale': 'No candidates to rank.'}

    prompt = f"""
You are a senior AI recruitment consultant performing final candidate ranking.

Job: {job_title}
Description: {job_description[:2000]}

Candidates to rank:
{json.dumps(candidates, indent=2)}

Rank these candidates from best to worst fit for the role.
Consider: evaluation scores, skills match, experience relevance, and recommendation.

Return ONLY JSON:
{{
  "ranked": [
    {{
      "rank": 1,
      "candidate_id": "id_here",
      "name": "Candidate Name",
      "composite_score": float 0-100,
      "rank_reason": "2 sentence explanation of why this rank",
      "hire_signal": "Strong Hire" | "Hire" | "Maybe" | "No Hire",
      "key_strengths": ["strength1", "strength2"],
      "key_gaps": ["gap1"]
    }}
  ],
  "ranking_rationale": "2 sentence overview of the ranking methodology used",
  "top_recommendation": "1 sentence on the clear top candidate and why"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Candidate ranking failed: {e}')
        # Return sorted by score as fallback
        sorted_candidates = sorted(candidates, key=lambda x: x.get('overall_score', 0), reverse=True)
        return {
            'ranked': [
                {
                    'rank': i + 1,
                    'candidate_id': c.get('candidate_id', ''),
                    'name': c.get('name', ''),
                    'composite_score': c.get('overall_score', 0),
                    'rank_reason': 'Ranked by evaluation score.',
                    'hire_signal': 'Maybe',
                    'key_strengths': [],
                    'key_gaps': [],
                }
                for i, c in enumerate(sorted_candidates)
            ],
            'ranking_rationale': 'Ranked by overall evaluation score (AI unavailable).',
            'top_recommendation': 'Review top-ranked candidate manually.',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — AI Interview Debrief Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_interview_debrief(
    evaluation: dict,
    interview: dict,
    candidate_name: str = ''
) -> dict:
    """
    Generate a comprehensive post-interview coaching debrief for the candidate.
    """
    prompt = f"""
You are an expert interview coach creating a personalized post-interview debrief report.

Candidate: {candidate_name or 'Candidate'}
Job Title: {interview.get('job_title', 'Unknown Role')}
Overall Score: {evaluation.get('overall_score', 0)}/100
Recommendation: {evaluation.get('recommendation', 'maybe')}
Strengths: {evaluation.get('strengths', [])}
Weaknesses: {evaluation.get('weaknesses', [])}
Confidence Score: {evaluation.get('confidence_score', 0)}/100
Behavioral Summary: {evaluation.get('behavioral_summary', '')}
Criterion Results: {json.dumps(evaluation.get('criterion_results', [])[:5])}

Create a comprehensive, encouraging yet honest debrief. Return ONLY JSON:
{{
  "headline": "Personalized 1-line debrief headline",
  "performance_tier": "Exceptional" | "Strong" | "Developing" | "Needs Work",
  "executive_summary": "3-4 sentence honest summary of the interview performance",
  "top_strengths": [
    {{"title": "Strength Name", "detail": "Specific evidence from interview"}}
  ],
  "improvement_areas": [
    {{"title": "Area Name", "detail": "What was observed", "action": "Specific action to improve"}}
  ],
  "skill_scores": [
    {{"skill": "Communication", "score": integer 0-100, "feedback": "brief feedback"}}
  ],
  "next_steps": ["Step 1", "Step 2", "Step 3"],
  "motivational_note": "1-2 sentence encouraging closing message",
  "recommended_resources": ["Resource or practice tip 1", "Resource or practice tip 2"]
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Debrief generation failed: {e}')
        score = evaluation.get('overall_score', 50)
        return {
            'headline': f'Interview Debrief — {interview.get("job_title", "Role")}',
            'performance_tier': 'Developing' if score < 60 else 'Strong',
            'executive_summary': evaluation.get('summary', 'Interview completed successfully.'),
            'top_strengths': [{'title': s, 'detail': ''} for s in evaluation.get('strengths', [])[:3]],
            'improvement_areas': [{'title': w, 'detail': '', 'action': 'Practice this area.'} for w in evaluation.get('weaknesses', [])[:3]],
            'skill_scores': [],
            'next_steps': ['Review your responses', 'Practice weak areas', 'Apply to similar roles'],
            'motivational_note': 'Keep improving — every interview is a learning opportunity.',
            'recommended_resources': ['Mock interview practice', 'Technical documentation review'],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8 — Question Bank Suggestions
# ─────────────────────────────────────────────────────────────────────────────

def generate_question_bank_suggestions(
    job_title: str,
    job_description: str = '',
    categories: list = None,
    count: int = 15
) -> list:
    """
    Generate a rich question bank for a given role, suitable for saving and reuse.
    """
    cats = categories or ['technical', 'behavioral', 'general', 'situational', 'culture']
    prompt = f"""
You are a senior HR director building a professional interview question bank for: {job_title}

Job Context: {job_description[:1500] if job_description else 'General role'}
Required Categories: {cats}

Generate exactly {count} high-quality, reusable interview questions.
Mix difficulties: 30% easy, 50% medium, 20% hard.

Return ONLY a JSON array:
[
  {{
    "text": "Full question text",
    "category": "technical" | "behavioral" | "general" | "situational" | "culture",
    "difficulty": "easy" | "medium" | "hard",
    "expected_keywords": ["kw1", "kw2", "kw3"],
    "ideal_answer": "What a strong answer should include",
    "time_estimate_minutes": integer,
    "tags": ["tag1", "tag2"]
  }}
]
"""
    try:
        result = _call(prompt)
        questions = json.loads(_strip_json(result))
        return questions if isinstance(questions, list) else []
    except Exception as e:
        logger.warning(f'[Gemini] Question bank generation failed: {e}')
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Feature 9 — Live Question Recommender (Recruiter side)
# ─────────────────────────────────────────────────────────────────────────────

def suggest_next_question(
    transcript: str,
    job_title: str,
    current_question: str,
    questions_asked: list = None,
    user_id: str = None
) -> dict:
    """
    Based on candidate's live transcript, suggest the best next follow-up questions
    for the recruiter to ask in real time.
    """
    asked_str = '\n'.join([f'- {q}' for q in (questions_asked or [])]) or 'None yet'

    prompt = f"""
You are a real-time AI interview coach helping a recruiter conduct a live interview.

Job Title: {job_title}
Current Question Asked: "{current_question}"
Candidate's Live Answer So Far: "{transcript[:1500]}"

Questions Already Asked:
{asked_str}

Based on the candidate's answer, suggest 3 smart follow-up questions the recruiter should consider asking next.
Mix depths: one clarifying, one deeper technical/behavioral, one wildcard.

Return ONLY JSON:
{{
  "suggestions": [
    {{
      "question": "Full question text",
      "type": "clarifying" | "deeper" | "wildcard",
      "why": "1 sentence — why this question is valuable right now",
      "difficulty": "easy" | "medium" | "hard"
    }}
  ],
  "signal": "positive" | "needs_probing" | "off_track",
  "coaching_note": "1 sentence tip for the recruiter on how candidate is performing"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[Gemini] Next question suggestion failed: {e}')
        return {
            'suggestions': [
                {'question': 'Can you walk me through a specific example of that?', 'type': 'clarifying', 'why': 'Get concrete evidence', 'difficulty': 'easy'},
                {'question': 'How did you measure the success of that approach?', 'type': 'deeper', 'why': 'Assess analytical thinking', 'difficulty': 'medium'},
                {'question': 'What would you do differently if you faced this again?', 'type': 'wildcard', 'why': 'Test self-awareness', 'difficulty': 'medium'},
            ],
            'signal': 'positive',
            'coaching_note': 'Keep the conversation flowing naturally.',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 10 — Predictive Hiring Score
# ─────────────────────────────────────────────────────────────────────────────

def predict_hire_probability(
    overall_score: float,
    confidence_score: float,
    proctoring_score: float,
    fluency_score: float,
    culture_fit_score: float,
    violations: int,
    recommendation: str,
    job_title: str,
    criterion_results: list = None
) -> dict:
    """
    Use Gemini to predict the probability of a successful hire based on all evaluation signals.
    Returns hire_probability (0-100), tier, reasoning, and risk factors.
    """
    criteria_text = ''
    if criterion_results:
        criteria_text = '\n'.join([f"- {cr.get('criterion','')}: {cr.get('score','')}/10" for cr in criterion_results[:8]])

    prompt = f"""
You are a senior HR data scientist predicting hiring success probability.

Role: {job_title}
Overall AI Score: {overall_score}/100
Confidence Score: {confidence_score}/100
Proctoring/Integrity Score: {proctoring_score}/100
Communication/Fluency Score: {fluency_score}/100
Culture Fit Score: {culture_fit_score}/100
Integrity Violations: {violations}
AI Recommendation: {recommendation}

Criterion Breakdown:
{criteria_text or 'Not available'}

Predict the probability this candidate will succeed if hired. Consider ALL signals holistically.

Return ONLY JSON:
{{
  "hire_probability": integer 0-100,
  "tier": "Exceptional Hire" | "Strong Hire" | "Potential Hire" | "Risky Hire" | "Do Not Hire",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "3-4 sentence explanation of the prediction",
  "risk_factors": ["risk1", "risk2"],
  "green_flags": ["green1", "green2"],
  "recommended_action": "1 concrete hiring action for the recruiter"
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Hire probability prediction failed: {e}')
        # Fallback: simple weighted formula
        score = round(
            overall_score * 0.4 + confidence_score * 0.2 +
            proctoring_score * 0.2 + fluency_score * 0.1 + culture_fit_score * 0.1
        )
        score = max(0, min(100, score - violations * 5))
        tier = (
            'Exceptional Hire' if score >= 85 else
            'Strong Hire' if score >= 70 else
            'Potential Hire' if score >= 55 else
            'Risky Hire' if score >= 40 else
            'Do Not Hire'
        )
        return {
            'hire_probability': score,
            'tier': tier,
            'confidence': 'Low',
            'reasoning': f'Predicted based on weighted evaluation scores. Overall: {overall_score}/100.',
            'risk_factors': [f'{violations} integrity violation(s) detected'] if violations else [],
            'green_flags': ['Completed full interview'],
            'recommended_action': 'Review full evaluation report before deciding.',
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 11 — AI Resume Content Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_resume_content(
    name: str,
    email: str,
    phone: str,
    headline: str,
    bio: str,
    skills: list,
    work_history: list,
    education_history: list,
    location: str = '',
    job_target: str = ''
) -> dict:
    """
    Generate a polished, ATS-optimized resume from the candidate's profile data.
    Returns structured resume content ready for frontend template rendering.
    """
    prompt = f"""
You are a professional resume writer at a top executive recruitment firm.
Create a polished, ATS-optimized resume from the candidate data below.

Candidate Data:
- Name: {name}
- Email: {email}
- Phone: {phone}
- Location: {location or 'Not provided'}
- Headline: {headline or 'Professional'}
- Summary/Bio: {bio or 'Not provided'}
- Target Role: {job_target or 'Not specified'}
- Skills: {json.dumps(skills)}
- Work History: {json.dumps(work_history[:5])}
- Education: {json.dumps(education_history[:3])}

Generate a complete, professional resume. Polish the language, fix grammar, and make it impressive.

Return ONLY JSON:
{{
  "name": "candidate full name",
  "contact": {{
    "email": "email",
    "phone": "phone",
    "location": "city, country",
    "linkedin": ""
  }},
  "headline": "Polished professional headline (max 10 words)",
  "summary": "3-4 sentence powerful professional summary optimized for ATS",
  "skills": {{
    "technical": ["skill1", "skill2", "skill3"],
    "soft": ["skill1", "skill2", "skill3"]
  }},
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "duration": "Jan 2022 – Dec 2023",
      "location": "City, Country",
      "bullets": ["Achievement 1 with metrics", "Achievement 2", "Achievement 3"]
    }}
  ],
  "education": [
    {{
      "degree": "Degree Name",
      "institution": "University Name",
      "year": "2020",
      "gpa": ""
    }}
  ],
  "certifications": [],
  "languages": ["English"],
  "ats_keywords": ["keyword1", "keyword2", "keyword3"]
}}
"""
    try:
        return json.loads(_strip_json(_call(prompt)))
    except Exception as e:
        logger.warning(f'[Gemini] Resume generation failed: {e}')
        return {
            'name': name,
            'contact': {'email': email, 'phone': phone, 'location': location, 'linkedin': ''},
            'headline': headline or 'Professional',
            'summary': bio or f'Experienced professional with expertise in {", ".join(skills[:3]) if skills else "various domains"}.',
            'skills': {'technical': skills[:10], 'soft': ['Communication', 'Teamwork', 'Problem Solving']},
            'experience': [{'title': w.get('title',''), 'company': w.get('company',''), 'duration': w.get('duration',''), 'location': '', 'bullets': [w.get('desc','')]} for w in work_history[:3]],
            'education': [{'degree': e.get('degree',''), 'institution': e.get('institution',''), 'year': e.get('year',''), 'gpa': ''} for e in education_history[:2]],
            'certifications': [],
            'languages': ['English'],
            'ats_keywords': skills[:5],
        }


# ─────────────────────────────────────────────────────────────────────────────
# AI Health Check
# ─────────────────────────────────────────────────────────────────────────────

def check_ai_health() -> dict:
    """
    Check if Gemini AI is operational.
    Returns status dict with 'status', 'message', and 'error_type'.
    """
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY in ('', 'your-gemini-api-key'):
        return {
            'status': 'error',
            'error_type': 'AI_KEY_MISSING',
            'message': 'Gemini API key is not configured.'
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
            'AI_QUOTA_EXHAUSTED': 'Gemini API quota exhausted. Free tier daily limit reached. Wait 24h or upgrade your plan.',
            'AI_KEY_INVALID': 'Gemini API key is invalid. Please update GEMINI_API_KEY in your .env file.',
            'AI_BILLING_ISSUE': 'Gemini billing issue. Please check your Google Cloud billing account.',
        }
        return {
            'status': 'error',
            'error_type': error_type,
            'message': messages.get(error_type, f'AI service error: {error_type}')
        }
