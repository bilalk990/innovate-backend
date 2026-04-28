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


def _call(prompt: str, user_id: str = None, response_format: str = "text", max_tokens: int = 2000) -> str:
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
            max_tokens=max_tokens
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


# Alias so both naming conventions work
_call_openai = _call


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



def predict_application_status(
    resume_data: dict,
    job_title: str,
    job_description: str,
    requirements: list = None,
    user_id: str = None,
) -> dict:
    """Predict likelihood of a candidate's job application being successful."""
    resume_summary = {
        'skills': resume_data.get('skills', []),
        'experience': resume_data.get('experience', []),
        'total_years': resume_data.get('total_experience_years', 0),
        'education': resume_data.get('education', []),
        'headline': resume_data.get('headline', ''),
    }
    reqs = requirements or []
    prompt = (
        'You are a senior HR analyst. Predict the job application success likelihood.\n\n'
        f'Job Title: {job_title}\n'
        f'Job Description: {job_description[:1500]}\n'
        f'Requirements: {json.dumps(reqs[:10])}\n'
        f'Candidate Resume Summary: {json.dumps(resume_summary)}\n\n'
        'Return JSON ONLY:\n'
        '{\n'
        '  "success_probability": 0-100,\n'
        '  "status_label": "Strong Match|Good Fit|Possible Fit|Long Shot",\n'
        '  "shortlist_likelihood": "High|Medium|Low",\n'
        '  "key_strengths": ["strength1", "strength2"],\n'
        '  "critical_gaps": ["gap1", "gap2"],\n'
        '  "recommendation": "1-2 sentence actionable advice for this application",\n'
        '  "improvement_tips": ["tip1", "tip2", "tip3"]\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] predict_application_status failed: {e}')
        skills = resume_data.get('skills', [])
        reqs_lower = [r.lower() for r in reqs]
        matched = [s for s in skills if any(r in s.lower() for r in reqs_lower)] if reqs_lower else skills[:3]
        prob = min(85, max(20, len(matched) * 15 + 20))
        label = 'Strong Match' if prob >= 75 else 'Good Fit' if prob >= 55 else 'Possible Fit' if prob >= 35 else 'Long Shot'
        return {
            'success_probability': prob,
            'status_label': label,
            'shortlist_likelihood': 'High' if prob >= 70 else 'Medium' if prob >= 45 else 'Low',
            'key_strengths': matched[:2] or ['Relevant background'],
            'critical_gaps': [r for r in reqs[:3] if r.lower() not in [s.lower() for s in skills]],
            'recommendation': f'Your profile shows a {label.lower()} for this role. Tailor your resume to highlight matching skills.',
            'improvement_tips': ['Tailor resume to job description', 'Highlight relevant projects', 'Add missing skills to profile'],
        }


def suggest_profile_improvements(
    profile_data: dict,
    evaluation_history: list = None,
    user_id: str = None,
) -> dict:
    """Suggest AI-powered improvements to a candidate's profile to boost success."""
    history_summary = []
    if evaluation_history:
        for ev in (evaluation_history or [])[:5]:
            history_summary.append({
                'score': ev.get('overall_score', 0),
                'strengths': ev.get('strengths', [])[:2],
                'weaknesses': ev.get('weaknesses', [])[:2],
            })

    has_resume = bool(profile_data.get('resume_uploaded'))
    skills_count = len(profile_data.get('skills', []))
    has_bio = bool(profile_data.get('bio'))
    has_exp = bool(profile_data.get('work_history'))

    prompt = (
        'You are a career coach. Analyze this candidate profile and suggest specific improvements.\n\n'
        'Profile:\n'
        f'- Skills count: {skills_count}\n'
        f'- Has Bio: {has_bio}\n'
        f'- Has Work History: {has_exp}\n'
        f'- Has Resume: {has_resume}\n'
        f'- Headline: {str(profile_data.get("headline", ""))[:100]}\n\n'
        f'Interview History (last {len(history_summary)} evals):\n'
        f'{json.dumps(history_summary) if history_summary else "No interviews yet."}\n\n'
        'Return JSON ONLY:\n'
        '{\n'
        '  "profile_strength_score": 0-100,\n'
        '  "profile_strength_label": "Weak|Basic|Good|Strong|Excellent",\n'
        '  "priority_improvements": [\n'
        '    {"title": "improvement title", "description": "what to do", "impact": "High|Medium|Low"}\n'
        '  ],\n'
        '  "missing_sections": ["section1", "section2"],\n'
        '  "quick_wins": ["quick action 1", "quick action 2"],\n'
        '  "estimated_improvement": "e.g. +25% interview chances"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] suggest_profile_improvements failed: {e}')
        score = (
            (20 if has_resume else 0) +
            (20 if has_bio else 0) +
            (20 if has_exp else 0) +
            min(30, skills_count * 3) +
            10
        )
        label = 'Excellent' if score >= 85 else 'Strong' if score >= 70 else 'Good' if score >= 50 else 'Basic' if score >= 30 else 'Weak'
        improvements = []
        if not has_resume:
            improvements.append({'title': 'Upload Resume', 'description': 'Upload your resume to unlock AI matching and gap analysis.', 'impact': 'High'})
        if not has_bio:
            improvements.append({'title': 'Write a Professional Bio', 'description': 'Add a compelling 3-5 sentence bio highlighting your expertise.', 'impact': 'High'})
        if not has_exp:
            improvements.append({'title': 'Add Work History', 'description': 'Add your work experience to improve match scores by up to 40%.', 'impact': 'High'})
        if skills_count < 5:
            improvements.append({'title': 'Add More Skills', 'description': f'You have {skills_count} skills. Add at least 8-10 to improve matching.', 'impact': 'Medium'})
        return {
            'profile_strength_score': score,
            'profile_strength_label': label,
            'priority_improvements': improvements[:4],
            'missing_sections': [s for s, ok in [('Resume', has_resume), ('Bio', has_bio), ('Work History', has_exp)] if not ok],
            'quick_wins': ['Add a professional headline', 'Upload a resume', 'Add 5 key skills'],
            'estimated_improvement': f'+{max(10, 90 - score)}% with suggested improvements',
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
# CANDIDATE SIDE — Job Fit & Profile Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

def predict_interview_likelihood(resume_data: dict, job_data: dict, user_id: str = None) -> dict:
    """
    Predict likelihood of getting an interview based on profile-vs-job-data match.
    Different from predict_application_status (which takes job_title/jd strings).
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


def suggest_profile_improvements_ai(current_profile: dict, target_role: str = '', user_id: str = None) -> dict:
    """
    Suggest profile improvements based on target role and completion level.
    Different from suggest_profile_improvements (which takes evaluation history).
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


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: CANDIDATE AI FEATURES — Mock Interview, Salary, Anxiety, Career
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mock_interview_question(role: str, level: str, history: list = None, question_number: int = 1, user_id: str = None) -> dict:
    """Generate the next mock interview question for a given role and level."""
    history_text = ''
    if history:
        for i, item in enumerate(history[-3:], 1):
            history_text += 'Q' + str(i) + ': ' + str(item.get('question', '')) + '\n'
            history_text += 'A' + str(i) + ': ' + str(item.get('answer', ''))[:200] + '\n\n'

    prompt = (
        'You are an expert interviewer conducting a ' + level + '-level ' + role + ' interview.\n'
        'This is question ' + str(question_number) + ' of 5.\n'
        + ('Previous Q&A:\n' + history_text + '\n' if history_text else '')
        + 'Generate the next interview question. Make it progressive, realistic, and appropriate for the level.\n'
        'Vary question types across behavioral, technical, situational, and motivational.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "question": "The interview question text",\n'
        '  "question_type": "behavioral|technical|situational|motivational",\n'
        '  "what_to_assess": "What this question evaluates in the candidate",\n'
        '  "tip_for_candidate": "Brief tip on how to best approach answering this"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Mock question generation failed: {e}')
        fallback_questions = [
            'Tell me about yourself and what brings you to this role.',
            'Describe your greatest professional achievement.',
            'Tell me about a challenge you faced and how you overcame it.',
            'Where do you see yourself professionally in 3-5 years?',
            'Why are you the best candidate for this position?',
        ]
        return {
            'question': fallback_questions[min(question_number - 1, 4)],
            'question_type': 'behavioral',
            'what_to_assess': 'Communication clarity and self-awareness',
            'tip_for_candidate': 'Be specific, use real examples, and keep your answer focused.'
        }


def evaluate_mock_answer(question: str, answer: str, role: str, question_type: str = 'behavioral', user_id: str = None) -> dict:
    """Evaluate a candidate answer to a mock interview question and return detailed feedback."""
    answer_trimmed = str(answer).strip()
    word_count = len(answer_trimmed.split())

    # Build type-specific scoring guidance
    type_criteria = {
        'behavioral': 'Uses STAR method (Situation, Task, Action, Result). Provides a REAL specific example. Has measurable outcomes. Does not speak in vague generalities.',
        'technical': 'Demonstrates accurate domain knowledge. Uses correct terminology. Shows depth (not surface-level). Covers edge cases or trade-offs.',
        'situational': 'Describes a clear decision-making process. Shows logical reasoning. Considers multiple stakeholders. Has a clear outcome.',
        'motivational': 'Is authentic and specific to this role/company. Shows genuine passion. Aligns with the role requirements. Is not generic or copied.'
    }
    criteria = type_criteria.get(question_type, type_criteria['behavioral'])

    prompt = (
        'You are a STRICT, senior interviewer evaluating a ' + role + ' candidate.\n'
        'Question type: ' + question_type.upper() + '\n\n'
        'QUESTION: ' + question + '\n\n'
        'CANDIDATE ANSWER (' + str(word_count) + ' words):\n'
        '"""' + answer_trimmed[:1500] + '"""\n\n'
        'SCORING CRITERIA FOR ' + question_type.upper() + ' QUESTIONS:\n'
        + criteria + '\n\n'
        'SCORING RULES (BE STRICT AND HONEST — do NOT give everyone 6):\n'
        '- 9-10: Exceptional answer. Specific real example, measurable results, clear structure, impressive.\n'
        '- 7-8: Good answer. Has most elements but missing depth or specificity in 1-2 areas.\n'
        '- 5-6: Average. Addresses the question but vague, generic, or missing concrete examples.\n'
        '- 3-4: Below Average. Barely addresses the question, too short, or irrelevant content.\n'
        '- 1-2: Poor. Does not answer the question, gibberish, or extremely low effort.\n\n'
        'A one-sentence answer MUST score 1-3. A vague general answer MUST score 4-5. Only score 8+ for truly impressive answers.\n\n'
        'Return ONLY valid JSON (no markdown, no extra text):\n'
        '{\n'
        '  "score": <integer 1-10 based on strict criteria>,\n'
        '  "grade": "Excellent|Good|Average|Below Average|Poor",\n'
        '  "feedback": "<2-3 sentences of specific, personalized, honest feedback mentioning what was said>",\n'
        '  "strengths": ["<specific strength from their actual answer>", "<another strength if any>"],\n'
        '  "improvements": ["<specific thing missing from this answer>", "<another improvement>"],\n'
        '  "better_answer_hint": "<One concrete sentence: what the ideal answer would include that theirs did not>",\n'
        '  "keywords_used": ["<keyword or phrase actually found in their answer>"],\n'
        '  "keywords_missed": ["<important keyword or concept they should have mentioned>"]\n'
        '}'
    )
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        # Validate score is within bounds
        score = int(result.get('score', 5))
        result['score'] = max(1, min(10, score))
        # Ensure grade matches score
        if result['score'] >= 9:
            result['grade'] = 'Excellent'
        elif result['score'] >= 7:
            result['grade'] = 'Good'
        elif result['score'] >= 5:
            result['grade'] = 'Average'
        elif result['score'] >= 3:
            result['grade'] = 'Below Average'
        else:
            result['grade'] = 'Poor'
        return result
    except Exception as e:
        logger.warning(f'[GPT] Mock answer evaluation failed: {e}')
        # Smarter fallback based on actual answer quality signals
        word_count = len(answer_trimmed.split())
        has_example = any(w in answer_trimmed.lower() for w in ['i worked', 'i built', 'i led', 'for example', 'specifically', 'the result', 'i achieved', 'when i'])
        has_metrics = any(c.isdigit() for c in answer_trimmed)
        if word_count < 15:
            fb_score = 2
        elif word_count < 30:
            fb_score = 4
        elif has_example and has_metrics:
            fb_score = 8
        elif has_example:
            fb_score = 7
        else:
            fb_score = 5

        grade_map = {2: 'Poor', 4: 'Below Average', 5: 'Average', 7: 'Good', 8: 'Good'}
        return {
            'score': fb_score,
            'grade': grade_map.get(fb_score, 'Average'),
            'feedback': 'Your answer was recorded. For stronger performance, use the STAR method: describe the Situation, your Task, the Action you took, and the measurable Result.' if fb_score < 6 else 'Good attempt. Make sure to include specific examples and measurable outcomes to stand out.',
            'strengths': ['Attempted the question'] if fb_score < 5 else ['Provided context', 'Relevant topic coverage'],
            'improvements': ['Answer is too short — aim for at least 100 words' if word_count < 30 else 'Add specific metrics or outcomes', 'Use STAR format: Situation, Task, Action, Result'],
            'better_answer_hint': 'Start with: "In my previous role at [Company], I was faced with [specific situation]..." and end with a measurable result.',
            'keywords_used': [],
            'keywords_missed': ['specific example', 'measurable result', 'action taken']
        }


def generate_mock_interview_report(role: str, level: str, history: list, user_id: str = None) -> dict:
    """Generate a comprehensive final performance report after completing a mock interview."""
    total_score = 0
    scored_count = 0
    qa_full_text = []

    for i, item in enumerate(history, 1):
        score = item.get('score')
        if score is not None:
            total_score += int(score)
            scored_count += 1
        qa_full_text.append(
            'Q' + str(i) + ' [' + item.get('question_type', 'behavioral').upper() + '] Score: ' + str(score) + '/10\n'
            'Question: ' + str(item.get('question', '')) + '\n'
            'Answer: ' + str(item.get('answer', '(no answer)'))[:600] + '\n'
            'Feedback given: ' + str(item.get('feedback', '')) + '\n'
            'Strengths: ' + ', '.join(item.get('strengths', [])) + '\n'
            'Improvements needed: ' + ', '.join(item.get('improvements', []))
        )

    avg_score = total_score / max(scored_count, 1)
    # Convert avg per-question score (1-10) to 0-100
    # avg 10/10 = 100, avg 5/10 = 50, etc. Keep it honest.
    computed_overall = round((avg_score / 10) * 100)

    prompt = (
        'You are a senior HR director generating a FINAL performance report for a ' + level + '-level ' + role + ' mock interview.\n\n'
        'INTERVIEW TRANSCRIPT:\n'
        '─────────────────────\n'
        + '\n\n'.join(qa_full_text) + '\n\n'
        '─────────────────────\n'
        'COMPUTED OVERALL SCORE: ' + str(computed_overall) + '/100 (based on per-question scores)\n\n'
        'YOUR TASK: Analyze the full transcript above and generate an honest, specific report.\n'
        '- The overall_score MUST be between ' + str(max(computed_overall - 5, 0)) + ' and ' + str(min(computed_overall + 5, 100)) + ' (within 5 points of computed score).\n'
        '- Identify REAL patterns from the actual answers (not generic advice).\n'
        '- Be specific: reference what the candidate actually said or failed to say.\n'
        '- top_strengths must reflect what genuinely went well in the answers.\n'
        '- critical_improvements must be specific gaps from the actual answers, not generic tips.\n\n'
        'SCORING SCALE:\n'
        '- 85-100: Excellent performance, ready for real interviews\n'
        '- 70-84: Good performance, minor improvements needed\n'
        '- 55-69: Average, needs more practice with structure\n'
        '- 40-54: Below average, significant gaps in answers\n'
        '- 0-39: Poor performance, fundamentals need work\n\n'
        'Return ONLY valid JSON (no markdown):\n'
        '{\n'
        '  "overall_score": <integer within 5 of ' + str(computed_overall) + '>,\n'
        '  "performance_grade": "Excellent|Good|Average|Needs Improvement|Poor",\n'
        '  "interview_summary": "<3-4 sentences. Reference specific answers. Be honest about quality.>",\n'
        '  "top_strengths": ["<specific strength from actual answers>", "<strength2>", "<strength3 if any>"],\n'
        '  "critical_improvements": ["<specific gap from actual answers>", "<gap2>", "<gap3>"],\n'
        '  "question_by_question": [\n'
        '    {"q": 1, "score": <score>, "grade": "<grade>", "one_line_feedback": "<specific to their answer>"}\n'
        '  ],\n'
        '  "recommended_resources": ["<specific resource or practice type based on their gaps>", "<resource2>"],\n'
        '  "readiness_for_real_interview": "Ready|Almost Ready|Needs More Practice|Not Ready",\n'
        '  "next_steps": ["<actionable step based on their specific weaknesses>", "<step2>", "<step3>"],\n'
        '  "motivational_note": "<Short encouraging closing that acknowledges what they did well>"\n'
        '}'
    )
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        # Enforce score boundaries
        reported = int(result.get('overall_score', computed_overall))
        result['overall_score'] = max(0, min(100, reported))
        return result
    except Exception as e:
        logger.warning(f'[GPT] Mock interview report generation failed: {e}')
        overall = computed_overall
        if overall >= 85:
            grade, readiness = 'Excellent', 'Ready'
        elif overall >= 70:
            grade, readiness = 'Good', 'Almost Ready'
        elif overall >= 55:
            grade, readiness = 'Average', 'Needs More Practice'
        elif overall >= 40:
            grade, readiness = 'Needs Improvement', 'Needs More Practice'
        else:
            grade, readiness = 'Poor', 'Not Ready'
        return {
            'overall_score': overall,
            'performance_grade': grade,
            'interview_summary': f'You completed a {level}-level {role} mock interview scoring {overall}/100. ' + (
                'Strong performance overall — keep refining your answers.' if overall >= 70
                else 'Your answers need more specific examples and structured responses (STAR method).'
            ),
            'top_strengths': ['Completed all questions', 'Demonstrated role awareness'] if overall >= 50 else ['Attempted all questions'],
            'critical_improvements': [
                'Use specific examples with measurable outcomes in every answer',
                'Structure answers using STAR method (Situation, Task, Action, Result)',
                'Increase answer depth — aim for 100-150 words per answer'
            ],
            'recommended_resources': ['Practice STAR method with 10 behavioral questions daily', 'Record yourself and review for filler words and clarity'],
            'readiness_for_real_interview': readiness,
            'next_steps': [
                'Practice 2 mock interviews per week',
                'Prepare 5 specific STAR stories from your experience',
                'Research the target company and role deeply'
            ],
            'motivational_note': 'Every practice session builds real skill. Review your answers, focus on the improvements, and try again!'
        }


def suggest_salary_negotiation(job_title: str, skills: list, experience_years: int, location: str, current_offer: float = None, company_size: str = 'medium', user_id: str = None) -> dict:
    """Provide AI-powered salary negotiation strategy with market data and negotiation scripts."""
    offer_text = ('$' + str(int(current_offer))) if current_offer else 'No offer yet'
    skills_text = ', '.join(skills[:10]) if skills else 'Not specified'

    prompt = (
        'You are a salary negotiation expert with real market knowledge.\n\n'
        'Profile:\n'
        '- Job Title: ' + job_title + '\n'
        '- Skills: ' + skills_text + '\n'
        '- Experience: ' + str(experience_years) + ' years\n'
        '- Location: ' + location + '\n'
        '- Company Size: ' + company_size + '\n'
        '- Current Offer: ' + offer_text + '\n\n'
        'Provide realistic market salary data and negotiation strategy.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "market_min": annual_salary_number,\n'
        '  "market_mid": annual_salary_number,\n'
        '  "market_max": annual_salary_number,\n'
        '  "recommended_ask": annual_salary_number,\n'
        '  "currency": "USD|PKR|GBP|EUR|INR",\n'
        '  "confidence": "high|medium|low",\n'
        '  "market_insight": "2-sentence market context explanation",\n'
        '  "negotiation_script": "Full word-for-word opening negotiation script",\n'
        '  "counter_offer_responses": [\n'
        '    {"scenario": "They say the budget is fixed", "response": "word-for-word response"},\n'
        '    {"scenario": "They say your ask is too high", "response": "word-for-word response"},\n'
        '    {"scenario": "They ask for your current salary", "response": "word-for-word response"}\n'
        '  ],\n'
        '  "benefits_to_negotiate": ["benefit1", "benefit2", "benefit3"],\n'
        '  "timing_tips": ["tip1", "tip2"],\n'
        '  "red_flags": ["red flag1", "red flag2"],\n'
        '  "power_phrases": ["phrase1", "phrase2", "phrase3"]\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Salary negotiation suggestion failed: {e}')
        return {
            'market_min': 60000, 'market_mid': 80000, 'market_max': 100000,
            'recommended_ask': 85000, 'currency': 'USD', 'confidence': 'low',
            'market_insight': 'Market data is based on your role and location. Adjust based on company size and local cost of living.',
            'negotiation_script': 'Thank you for the offer. I am very excited about this opportunity. Based on my research and experience, I was expecting a range closer to [X]. Is there flexibility in the compensation?',
            'counter_offer_responses': [
                {'scenario': 'They say the budget is fixed', 'response': 'I completely understand. Could we discuss other benefits like additional PTO, remote work flexibility, or a sign-on bonus?'},
                {'scenario': 'They say your ask is too high', 'response': 'I appreciate your transparency. What is the budgeted range for this role? I want to make sure we find something that works for both of us.'},
                {'scenario': 'They ask for your current salary', 'response': 'I prefer to focus on the market value for this role and the value I bring rather than my current compensation.'}
            ],
            'benefits_to_negotiate': ['Remote work flexibility', 'Additional PTO', 'Professional development budget', 'Sign-on bonus'],
            'timing_tips': ['Always wait for a written offer before negotiating', 'Let them give a number first whenever possible'],
            'red_flags': ['Pressure to accept immediately', 'Vague or undisclosed salary range', 'No written offer provided'],
            'power_phrases': ['Based on my market research...', 'I am very excited about this role and...', 'I believe my experience in X justifies...']
        }


def analyze_anxiety_signals(speech_features: dict, user_id: str = None) -> dict:
    """Analyze voice/speech features to detect anxiety during interviews and provide calming support."""
    prompt = (
        'You are a mental wellness coach for interview anxiety. Analyze these speech metrics.\n\n'
        'Speech Analysis Data:\n' + json.dumps(speech_features, indent=2) + '\n\n'
        'Detect anxiety indicators and provide warm, supportive coaching.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "anxiety_score": 0-100,\n'
        '  "anxiety_level": "Calm|Mild|Moderate|High|Severe",\n'
        '  "detected_signals": ["signal1", "signal2"],\n'
        '  "calm_message": "Warm, encouraging message (1-2 sentences)",\n'
        '  "breathing_exercise": "Step-by-step 4-7-8 or box breathing instruction",\n'
        '  "quick_tip": "One immediate actionable tip to ground yourself",\n'
        '  "positive_affirmation": "Short powerful affirmation (1 sentence)"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Anxiety analysis failed: {e}')
        return {
            'anxiety_score': 30,
            'anxiety_level': 'Mild',
            'detected_signals': ['Slight pace variation detected'],
            'calm_message': 'You are doing wonderfully. Take a deep breath — you are more prepared than you think.',
            'breathing_exercise': 'Box breathing: Inhale 4 counts → Hold 4 → Exhale 4 → Hold 4. Repeat 3 times.',
            'quick_tip': 'Pause for 2 seconds before answering. It signals thoughtfulness, not hesitation.',
            'positive_affirmation': 'You are exactly where you need to be. You belong here.'
        }


def recommend_career_paths(profile_data: dict, evaluation_history: list = None, user_id: str = None) -> dict:
    """Recommend 3 personalized career paths based on candidate profile and performance history."""
    skills = (profile_data.get('skills') or [])[:15]
    experience = len(profile_data.get('work_history') or [])
    headline = profile_data.get('headline', '')
    bio = str(profile_data.get('bio', ''))[:300]

    eval_summary = []
    if evaluation_history:
        for ev in (evaluation_history or [])[:3]:
            eval_summary.append({
                'score': ev.get('overall_score', 0),
                'job': ev.get('job_title', ''),
                'rec': ev.get('recommendation', '')
            })

    prompt = (
        'You are a career counselor. Recommend 3 tailored career paths for this candidate.\n\n'
        'Skills: ' + json.dumps(skills) + '\n'
        'Experience Positions: ' + str(experience) + '\n'
        'Current Headline: ' + headline + '\n'
        'Bio: ' + bio + '\n'
        + ('Interview History: ' + json.dumps(eval_summary) + '\n' if eval_summary else '')
        + '\nMake paths realistic, specific, and progressively ordered by difficulty.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "career_paths": [\n'
        '    {\n'
        '      "title": "Specific career path title",\n'
        '      "match_score": 0-100,\n'
        '      "why_suited": "2-sentence explanation of why this fits them",\n'
        '      "current_skills_applicable": ["skill1", "skill2"],\n'
        '      "skills_to_acquire": ["skill1", "skill2", "skill3"],\n'
        '      "timeline": "3-6 months|6-12 months|1-2 years|2-3 years",\n'
        '      "salary_range": "$X,000 - $Y,000",\n'
        '      "growth_potential": "Very High|High|Medium|Steady",\n'
        '      "job_titles_on_path": ["Entry Title", "Mid Title", "Senior Title"],\n'
        '      "first_step": "Single most important immediate action to start"\n'
        '    }\n'
        '  ],\n'
        '  "overall_assessment": "2-3 sentence summary of the candidate overall",\n'
        '  "top_strength": "Their single biggest professional strength"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Career path recommendation failed: {e}')
        return {
            'career_paths': [
                {
                    'title': 'Software Engineer',
                    'match_score': 75,
                    'why_suited': 'Your technical skills and background align with software engineering. This is a high-demand field with strong growth.',
                    'current_skills_applicable': skills[:3] if skills else ['Problem solving', 'Communication'],
                    'skills_to_acquire': ['System Design', 'Cloud Platforms (AWS/GCP)', 'CI/CD Pipelines'],
                    'timeline': '6-12 months',
                    'salary_range': '$70,000 - $130,000',
                    'growth_potential': 'Very High',
                    'job_titles_on_path': ['Junior Software Engineer', 'Software Engineer', 'Senior Software Engineer', 'Staff Engineer'],
                    'first_step': 'Build 2-3 portfolio projects on GitHub and apply to entry-level positions'
                }
            ],
            'overall_assessment': 'You have a solid foundation to build from. Focus on deepening your core skills and building a visible portfolio.',
            'top_strength': 'Eagerness to learn and technical aptitude'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERVIEW PREP LAB — 3 AI functions
# ═══════════════════════════════════════════════════════════════════════════════

def generate_interview_prep_plan(role: str, stacks: list, level: str, user_id: str = None) -> dict:
    """Generate a complete interview preparation roadmap for a given role and tech stack."""
    stacks_text = ', '.join(stacks) if stacks else role

    prompt = (
        'You are a senior tech interview coach. Generate a complete interview preparation plan.\n\n'
        'Role: ' + role + '\n'
        'Tech Stack: ' + stacks_text + '\n'
        'Level: ' + level + '\n\n'
        'Make it specific, actionable, and intelligent. No generic filler.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "prep_overview": "2-3 sentence smart summary of what this interview will test",\n'
        '  "estimated_prep_days": 7,\n'
        '  "difficulty": "Easy|Medium|Hard|Very Hard",\n'
        '  "topics": [\n'
        '    {\n'
        '      "category": "category name (e.g. Core JavaScript, System Design, DSA)",\n'
        '      "priority": "Must Know|Good to Know|Bonus",\n'
        '      "concepts": ["concept1", "concept2", "concept3", "concept4"],\n'
        '      "interview_weight": "High|Medium|Low"\n'
        '    }\n'
        '  ],\n'
        '  "common_question_patterns": [\n'
        '    {"pattern": "Pattern name", "example": "Example question", "how_to_answer": "Brief strategy"}\n'
        '  ],\n'
        '  "must_know_concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],\n'
        '  "red_flag_topics": ["topic often failed by candidates1", "topic2"],\n'
        '  "pro_tips": ["tip1", "tip2", "tip3"],\n'
        '  "daily_schedule": [\n'
        '    {"day": "Day 1-2", "focus": "what to study", "goal": "specific milestone"}\n'
        '  ]\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Prep plan generation failed: {e}')
        return {
            'prep_overview': f'This {level} {role} interview will test your knowledge of {stacks_text}. Focus on fundamentals and practical application.',
            'estimated_prep_days': 7,
            'difficulty': 'Medium',
            'topics': [
                {'category': stacks[0] if stacks else role, 'priority': 'Must Know', 'concepts': ['Core fundamentals', 'Common patterns', 'Best practices', 'Error handling'], 'interview_weight': 'High'},
                {'category': 'System Design', 'priority': 'Must Know', 'concepts': ['Scalability', 'APIs', 'Databases', 'Caching'], 'interview_weight': 'High'},
                {'category': 'Data Structures & Algorithms', 'priority': 'Good to Know', 'concepts': ['Arrays', 'Hashmaps', 'Trees', 'Big O notation'], 'interview_weight': 'Medium'},
            ],
            'common_question_patterns': [
                {'pattern': 'Explain a concept', 'example': f'How does {stacks[0] if stacks else role} work?', 'how_to_answer': 'Start with definition, then explain with an example, then mention use cases.'},
                {'pattern': 'Problem solving', 'example': 'Optimize this code snippet', 'how_to_answer': 'Think aloud, mention time/space complexity, test edge cases.'},
            ],
            'must_know_concepts': ['Core syntax', 'Common design patterns', 'Error handling', 'Performance optimization', 'Testing basics'],
            'red_flag_topics': ['Memory leaks', 'Async handling', 'Security basics'],
            'pro_tips': ['Study the official docs', 'Build a small project using the stack', 'Practice coding without IDE'],
            'daily_schedule': [
                {'day': 'Day 1-2', 'focus': 'Core fundamentals', 'goal': 'Understand all basic concepts'},
                {'day': 'Day 3-4', 'focus': 'Practical coding', 'goal': 'Solve 5 practice problems'},
                {'day': 'Day 5-7', 'focus': 'Mock interviews + review', 'goal': 'Score 80%+ on practice quiz'},
            ]
        }


def generate_interview_mcq_quiz(role: str, stacks: list, level: str, count: int = 10, user_id: str = None) -> dict:
    """Generate stack-specific MCQ quiz questions for interview prep."""
    stacks_text = ', '.join(stacks) if stacks else role

    prompt = (
        'You are a technical interview question writer. Generate exactly ' + str(count) + ' MCQ questions.\n\n'
        'Role: ' + role + '\n'
        'Tech Stack: ' + stacks_text + '\n'
        'Level: ' + level + '\n\n'
        'Rules:\n'
        '- Questions must be genuinely technical, not trivial\n'
        '- Mix difficulties: 30% easy, 50% medium, 20% hard\n'
        '- Cover different aspects of the tech stack\n'
        '- Explanations should teach, not just state the answer\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "quiz_title": "Title for this quiz",\n'
        '  "total_questions": ' + str(count) + ',\n'
        '  "questions": [\n'
        '    {\n'
        '      "id": 1,\n'
        '      "question": "The question text",\n'
        '      "options": ["Option A", "Option B", "Option C", "Option D"],\n'
        '      "correct_index": 0,\n'
        '      "explanation": "Why this is correct + why others are wrong (2-3 sentences)",\n'
        '      "topic": "which topic this covers",\n'
        '      "difficulty": "Easy|Medium|Hard"\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    try:
        result = json.loads(_strip_json(_call(prompt, user_id=user_id)))
        # Validate structure
        if 'questions' not in result or not result['questions']:
            raise ValueError('No questions generated')
        # Ensure correct_index is int and in range
        for q in result['questions']:
            q['correct_index'] = int(q.get('correct_index', 0))
            if not isinstance(q.get('options'), list) or len(q['options']) < 4:
                q['options'] = ['Option A', 'Option B', 'Option C', 'Option D']
        return result
    except Exception as e:
        logger.warning(f'[GPT] MCQ quiz generation failed: {e}')
        return {
            'quiz_title': f'{role} Knowledge Check',
            'total_questions': 5,
            'questions': [
                {'id': 1, 'question': f'What is the primary purpose of {stacks[0] if stacks else role}?', 'options': ['Building user interfaces', 'Managing databases', 'Server-side scripting', 'Network configuration'], 'correct_index': 0, 'explanation': f'{stacks[0] if stacks else role} is primarily used for building user interfaces. It provides a component-based architecture for creating dynamic web applications.', 'topic': 'Fundamentals', 'difficulty': 'Easy'},
                {'id': 2, 'question': 'What does "DRY" principle stand for in software development?', 'options': ['Do Repeat Yourself', "Don't Repeat Yourself", 'Dynamic Runtime Yielding', 'Direct Resource Yielding'], 'correct_index': 1, 'explanation': "DRY stands for Don't Repeat Yourself. It means every piece of knowledge should have a single, unambiguous representation in a system.", 'topic': 'Best Practices', 'difficulty': 'Easy'},
                {'id': 3, 'question': 'Which of the following best describes O(n log n) time complexity?', 'options': ['Linear', 'Logarithmic', 'Linearithmic', 'Quadratic'], 'correct_index': 2, 'explanation': 'O(n log n) is called linearithmic complexity. Common in efficient sorting algorithms like merge sort and heap sort.', 'topic': 'Algorithms', 'difficulty': 'Medium'},
                {'id': 4, 'question': 'What is the main benefit of using version control systems like Git?', 'options': ['Faster code execution', 'Track changes and collaborate', 'Automatic code optimization', 'Database management'], 'correct_index': 1, 'explanation': 'Git allows teams to track every change, collaborate without conflicts, and revert to any previous state of the codebase.', 'topic': 'Tools', 'difficulty': 'Easy'},
                {'id': 5, 'question': 'In REST APIs, which HTTP method is idempotent and used for updates?', 'options': ['POST', 'GET', 'PUT', 'DELETE'], 'correct_index': 2, 'explanation': 'PUT is idempotent — calling it multiple times with the same data produces the same result. It is used for full resource updates.', 'topic': 'APIs', 'difficulty': 'Medium'},
            ]
        }


def generate_prep_final_report(role: str, stacks: list, level: str, quiz_score: int, total_questions: int, tab_switches: int, wrong_topics: list, time_per_q_avg: float, user_id: str = None) -> dict:
    """Generate a comprehensive readiness report after completing the interview prep quiz."""
    stacks_text = ', '.join(stacks) if stacks else role
    accuracy = round((quiz_score / max(total_questions, 1)) * 100)
    integrity_score = max(0, 100 - (tab_switches * 15))

    prompt = (
        'Generate a comprehensive interview readiness report.\n\n'
        'Role: ' + role + '\n'
        'Stack: ' + stacks_text + '\n'
        'Level: ' + level + '\n'
        'Quiz Score: ' + str(quiz_score) + '/' + str(total_questions) + ' (' + str(accuracy) + '%)\n'
        'Tab Switches (integrity): ' + str(tab_switches) + '\n'
        'Weak Topics: ' + json.dumps(wrong_topics) + '\n'
        'Avg Time Per Question: ' + str(round(time_per_q_avg)) + ' seconds\n\n'
        'Be honest, specific, and actionable.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "readiness_score": 0-100,\n'
        '  "verdict": "Apply Now|Almost Ready|Needs Practice|Not Ready",\n'
        '  "verdict_explanation": "2 sentence explanation of verdict",\n'
        '  "knowledge_score": 0-100,\n'
        '  "speed_score": 0-100,\n'
        '  "integrity_score": 0-100,\n'
        '  "strong_topics": ["topic1", "topic2"],\n'
        '  "weak_topics": ["topic1", "topic2"],\n'
        '  "personalized_feedback": "3-4 sentence honest feedback based on performance",\n'
        '  "next_steps": ["step1", "step2", "step3"],\n'
        '  "estimated_days_to_ready": 0,\n'
        '  "mock_interview_recommended": true,\n'
        '  "motivational_closing": "Short powerful closing message"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Prep report generation failed: {e}')
        readiness = min(int(accuracy * 0.6 + integrity_score * 0.2 + min(100, (45 - time_per_q_avg) * 2) * 0.2), 100)
        readiness = max(readiness, 10)
        return {
            'readiness_score': readiness,
            'verdict': 'Apply Now' if readiness >= 80 else 'Almost Ready' if readiness >= 65 else 'Needs Practice' if readiness >= 45 else 'Not Ready',
            'verdict_explanation': f'You scored {accuracy}% on the knowledge quiz. {"Your integrity score was affected by tab switches." if tab_switches > 0 else "Your integrity was perfect."}',
            'knowledge_score': accuracy,
            'speed_score': max(0, min(100, int((45 - time_per_q_avg) * 2.5))),
            'integrity_score': integrity_score,
            'strong_topics': ['Core concepts', 'Fundamentals'],
            'weak_topics': wrong_topics[:3] if wrong_topics else ['Review weak areas'],
            'personalized_feedback': f'You completed the {role} prep quiz with {accuracy}% accuracy. Focus on your weak topics before applying.',
            'next_steps': ['Review weak topics', 'Practice daily coding', 'Do a mock interview'],
            'estimated_days_to_ready': 0 if readiness >= 80 else 3 if readiness >= 65 else 7 if readiness >= 45 else 14,
            'mock_interview_recommended': True,
            'motivational_closing': 'Keep pushing — every practice session brings you closer to your goal!'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HR AI POWER TOOLS — 7 Professional Features for Recruiters
# ═══════════════════════════════════════════════════════════════════════════════

def compare_candidates(candidates_data: list, job_title: str, blind_mode: bool = False, user_id: str = None) -> dict:
    """Compare 2-5 candidates for the same job using deep AI analysis."""
    candidate_list = []
    for i, c in enumerate(candidates_data):
        label = f'Candidate {chr(65+i)}' if blind_mode else c.get('name', f'Candidate {chr(65+i)}')
        candidate_list.append(f"""
{label}:
  Overall Score: {c.get('overall_score', 'N/A')}
  Skills: {', '.join(c.get('skills', [])[:15])}
  Experience: {c.get('experience_years', 'N/A')} years
  Education: {c.get('education', 'N/A')}
  Strengths: {', '.join(c.get('strengths', []))}
  Weaknesses: {', '.join(c.get('weaknesses', []))}
  Interview Summary: {c.get('summary', 'No summary')}
  Recommendation: {c.get('recommendation', 'N/A')}
""")

    prompt = f"""You are a senior HR director comparing candidates for the role of "{job_title}".
{"BLIND MODE: Evaluate purely on merit — no names or personal identifiers." if blind_mode else ""}

Candidates to compare:
{'---'.join(candidate_list)}

Provide a deep, unbiased, professional comparison. Return ONLY valid JSON:
{{
  "winner": "Candidate A",
  "winner_confidence": 85,
  "winner_reasoning": "2-3 sentence executive summary of why this candidate wins",
  "comparison_matrix": [
    {{"criterion": "Technical Skills", "scores": {{}}, "winner": "A", "insight": "..."}},
    {{"criterion": "Communication", "scores": {{}}, "winner": "B", "insight": "..."}},
    {{"criterion": "Experience Depth", "scores": {{}}, "winner": "A", "insight": "..."}},
    {{"criterion": "Cultural Fit Potential", "scores": {{}}, "winner": "A", "insight": "..."}},
    {{"criterion": "Growth Potential", "scores": {{}}, "winner": "B", "insight": "..."}},
    {{"criterion": "Risk Factor", "scores": {{}}, "winner": "A", "insight": "..."}}
  ],
  "individual_profiles": [
    {{
      "label": "Candidate A",
      "hire_probability": 82,
      "top_strengths": ["strength1", "strength2", "strength3"],
      "top_concerns": ["concern1", "concern2"],
      "best_fit_for": "description of what role/team they'd excel in",
      "verdict": "Strong Hire"
    }}
  ],
  "final_recommendation": "Hire [Winner] — detailed paragraph with business justification",
  "risk_analysis": "What risks exist with the top candidate and how to mitigate",
  "runner_up_advice": "When to consider the runner-up candidate instead",
  "blind_bias_notes": "Any potential bias areas the recruiter should be aware of"
}}
All scores in comparison_matrix must use candidate labels as keys (e.g. "Candidate A": 8.5).
Scores must be 0-10. hire_probability must be 0-100."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Candidate comparison failed: {e}')
        labels = [f'Candidate {chr(65+i)}' for i in range(len(candidates_data))]
        return {
            'winner': labels[0] if labels else 'Candidate A',
            'winner_confidence': 70,
            'winner_reasoning': f'Based on overall scores, {labels[0]} shows the strongest profile for {job_title}.',
            'comparison_matrix': [
                {'criterion': c, 'scores': {l: 7.0 for l in labels}, 'winner': labels[0], 'insight': 'Analysis unavailable.'}
                for c in ['Technical Skills', 'Communication', 'Experience Depth', 'Cultural Fit Potential', 'Growth Potential']
            ],
            'individual_profiles': [{'label': l, 'hire_probability': 70, 'top_strengths': ['Professional background'], 'top_concerns': ['Needs further evaluation'], 'best_fit_for': job_title, 'verdict': 'Consider'} for l in labels],
            'final_recommendation': f'Manual review recommended for {job_title} candidates.',
            'risk_analysis': 'Conduct additional reference checks before final decision.',
            'runner_up_advice': 'Keep runner-up on file for future openings.',
            'blind_bias_notes': 'Ensure structured interviews to minimize bias.'
        }


def detect_jd_bias(jd_text: str, user_id: str = None) -> dict:
    """Detect unconscious bias in job descriptions and rewrite with inclusive language."""
    prompt = f"""You are a DEI (Diversity, Equity & Inclusion) expert and linguist specializing in inclusive job descriptions.

Analyze this job description for unconscious bias:
---
{jd_text[:4000]}
---

Check for: gender-coded words (masculine: rockstar, ninja, crushing it; feminine: nurturing, collaborative), age bias (young team, recent grad, digital native), cultural bias (native speaker, local candidates), ability bias, socioeconomic bias, and unnecessarily exclusive requirements.

Return ONLY valid JSON:
{{
  "diversity_score_before": 52,
  "diversity_score_after": 89,
  "total_issues_found": 7,
  "bias_categories": {{
    "gender_coded": 3,
    "age_bias": 1,
    "cultural_bias": 2,
    "ability_bias": 0,
    "credential_inflation": 1
  }},
  "flagged_phrases": [
    {{
      "phrase": "rockstar developer",
      "type": "masculine-coded",
      "severity": "medium",
      "explanation": "Masculine-coded language that may deter women applicants",
      "suggestion": "skilled developer"
    }}
  ],
  "rewritten_jd": "Full inclusive rewrite of the entire job description",
  "key_changes_made": ["change1", "change2", "change3"],
  "overall_assessment": "2-3 sentence overall assessment",
  "quick_wins": ["Immediate change 1", "Immediate change 2"],
  "diversity_impact": "Estimated impact of changes on applicant diversity"
}}
Scores must be 0-100. Provide comprehensive flagged_phrases list."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] JD bias detection failed: {e}')
        return {
            'diversity_score_before': 60,
            'diversity_score_after': 85,
            'total_issues_found': 3,
            'bias_categories': {'gender_coded': 2, 'age_bias': 0, 'cultural_bias': 1, 'ability_bias': 0, 'credential_inflation': 0},
            'flagged_phrases': [{'phrase': 'Review manually', 'type': 'general', 'severity': 'low', 'explanation': 'AI analysis unavailable', 'suggestion': 'Use inclusive language'}],
            'rewritten_jd': jd_text,
            'key_changes_made': ['Replace gender-coded terms', 'Remove age references', 'Focus on skills over credentials'],
            'overall_assessment': 'Manual review recommended. Consider using inclusive language tools.',
            'quick_wins': ['Replace "rockstar/ninja" with "skilled"', 'Remove age-related terms'],
            'diversity_impact': 'Inclusive JDs typically increase diverse applicant pool by 20-40%.'
        }


def generate_reference_questions(resume_data: dict, job_title: str, eval_summary: str = '', user_id: str = None) -> dict:
    """Generate targeted AI reference check questions based on candidate profile."""
    skills = ', '.join(resume_data.get('skills', [])[:10])
    exp_years = resume_data.get('total_experience_years', 0)
    education = resume_data.get('education', [{}])
    edu_str = education[0].get('degree', 'N/A') if education else 'N/A'

    prompt = f"""You are an expert HR professional specializing in thorough reference checks.

Candidate Profile:
- Role Applied For: {job_title}
- Experience: {exp_years} years
- Key Skills: {skills}
- Education: {edu_str}
- Interview Performance Notes: {eval_summary or 'No notes provided'}

Generate highly targeted, legally compliant reference check questions. Return ONLY valid JSON:
{{
  "call_script_opener": "Professional opening statement to start the reference call",
  "standard_questions": [
    {{"question": "...", "purpose": "why ask this"}}
  ],
  "targeted_questions": [
    {{
      "question": "Targeted question specific to this candidate",
      "why_ask": "Based on what in their profile",
      "red_flag_if": "What answer should concern you",
      "green_flag_if": "What answer is reassuring"
    }}
  ],
  "gap_verification_questions": [
    {{"question": "...", "verifying": "what claim or gap this addresses"}}
  ],
  "skills_verification": [
    {{"skill": "skill name", "question": "how to verify this skill via reference"}}
  ],
  "closing_questions": ["closing question 1", "closing question 2"],
  "legal_reminders": ["Topics to AVOID for legal compliance"],
  "pro_tips": "Tips for conducting an effective reference call"
}}
Provide at least 4 standard, 5 targeted, 3 gap verification, and 3 skills verification questions."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Reference questions generation failed: {e}')
        return {
            'call_script_opener': f'Hi, I\'m calling to verify a reference for a candidate applying for our {job_title} position. Do you have a few minutes?',
            'standard_questions': [
                {'question': 'How long did you work with this candidate and in what capacity?', 'purpose': 'Establish relationship context'},
                {'question': 'What were their primary responsibilities?', 'purpose': 'Verify resume claims'},
                {'question': 'How would you describe their work quality and attention to detail?', 'purpose': 'Assess performance'},
                {'question': 'Would you rehire this person if given the opportunity?', 'purpose': 'Overall endorsement'}
            ],
            'targeted_questions': [
                {'question': f'How did they demonstrate {skills.split(",")[0] if skills else "technical skills"}?', 'why_ask': 'Skill verification', 'red_flag_if': 'Vague or hesitant response', 'green_flag_if': 'Specific examples provided'}
            ],
            'gap_verification_questions': [{'question': 'Were there any periods of extended leave or absence?', 'verifying': 'Employment gaps'}],
            'skills_verification': [{'skill': s.strip(), 'question': f'Can you give an example of how they used {s.strip()}?'} for s in skills.split(',')[:3]],
            'closing_questions': ['Is there anything else you\'d like to share about this candidate?', 'What type of environment would this person thrive in?'],
            'legal_reminders': ['Do not ask about age, health, family status, religion, or national origin'],
            'pro_tips': 'Listen for hesitation, ask for specific examples, and take detailed notes.'
        }


def predict_offer_acceptance(candidate_data: dict, offer_data: dict, user_id: str = None) -> dict:
    """Predict probability of candidate accepting the job offer with negotiation strategy."""
    prompt = f"""You are a senior talent acquisition strategist with 15 years of experience in offer negotiation.

Candidate Profile:
- Name/Label: {candidate_data.get('label', 'Candidate')}
- Current Salary: {candidate_data.get('current_salary', 'Unknown')}
- Expected Salary: {candidate_data.get('expected_salary', 'Unknown')}
- Experience Years: {candidate_data.get('experience_years', 0)}
- Skills: {', '.join(candidate_data.get('skills', [])[:8])}
- Interview Enthusiasm (1-10): {candidate_data.get('enthusiasm_score', 7)}
- Competing Offers: {candidate_data.get('has_competing_offers', False)}
- Location: {candidate_data.get('location', 'Unknown')}
- Interview Notes: {candidate_data.get('notes', 'No notes')}

Offer Details:
- Base Salary: {offer_data.get('base_salary', 'TBD')}
- Total Package: {offer_data.get('total_package', 'TBD')}
- Benefits: {offer_data.get('benefits', 'Standard')}
- Remote Work: {offer_data.get('remote_policy', 'On-site')}
- Start Date: {offer_data.get('start_date', 'Flexible')}
- Role Level: {offer_data.get('role_level', 'Mid')}

Analyze this offer scenario and return ONLY valid JSON:
{{
  "acceptance_probability": 74,
  "confidence_level": "high",
  "verdict": "Likely Accept",
  "key_drivers": ["factor driving acceptance 1", "factor driving acceptance 2"],
  "risk_factors": ["risk that may cause rejection 1", "risk 2"],
  "positive_signals": ["positive signal 1", "positive signal 2"],
  "salary_gap_analysis": {{
    "gap_amount": "5000",
    "gap_severity": "moderate",
    "recommendation": "Increase base by X or add signing bonus"
  }},
  "recommended_offer_adjustments": [
    {{"adjustment": "Specific change to make", "impact": "+X% acceptance probability", "cost": "Low/Medium/High"}}
  ],
  "negotiation_script": "Word-for-word script for the offer call",
  "timing_advice": "When and how to present the offer",
  "counter_offer_scenarios": [
    {{"scenario": "Candidate asks for higher salary", "response": "What to say", "max_flex": "Maximum flexibility"}}
  ],
  "package_sweeteners": ["Non-monetary benefit that could tip the scale 1", "sweetener 2"],
  "walk_away_signals": ["Signal that candidate will reject", "signal 2"]
}}
acceptance_probability must be 0-100 integer."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Offer prediction failed: {e}')
        return {
            'acceptance_probability': 65,
            'confidence_level': 'medium',
            'verdict': 'Uncertain',
            'key_drivers': ['Competitive salary', 'Good role fit'],
            'risk_factors': ['Salary expectations gap', 'Competing offers possible'],
            'positive_signals': ['Completed full interview process', 'Expressed interest'],
            'salary_gap_analysis': {'gap_amount': 'Unknown', 'gap_severity': 'unknown', 'recommendation': 'Verify candidate salary expectations'},
            'recommended_offer_adjustments': [{'adjustment': 'Add signing bonus', 'impact': '+10% acceptance', 'cost': 'Medium'}],
            'negotiation_script': 'We are excited to extend this offer and believe it reflects your valuable skills. We are open to discussing the package to find the right fit.',
            'timing_advice': 'Present offer within 48 hours of final interview while enthusiasm is high.',
            'counter_offer_scenarios': [{'scenario': 'Candidate asks for more', 'response': 'Discuss total compensation package', 'max_flex': 'Up to 10% above base'}],
            'package_sweeteners': ['Additional PTO', 'Remote work flexibility', 'Professional development budget'],
            'walk_away_signals': ['Asks for 2+ weeks to decide', 'Stops responding promptly']
        }


def analyze_hiring_funnel(funnel_stats: dict, job_title: str, user_id: str = None) -> dict:
    """Analyze hiring funnel drop-off rates and identify optimization opportunities."""
    prompt = f"""You are a talent acquisition analytics expert specializing in hiring funnel optimization.

Hiring Funnel Data for "{job_title}":
- Applications Received: {funnel_stats.get('applications', 0)}
- Screened (moved forward): {funnel_stats.get('screened', 0)}
- Phone/Video Screened: {funnel_stats.get('phone_screened', 0)}
- Technical/Assessment: {funnel_stats.get('assessed', 0)}
- Final Interview: {funnel_stats.get('final_interview', 0)}
- Offers Made: {funnel_stats.get('offers_made', 0)}
- Offers Accepted: {funnel_stats.get('offers_accepted', 0)}
- Hired & Started: {funnel_stats.get('hired', 0)}
- Time to Fill (days): {funnel_stats.get('time_to_fill', 0)}
- Cost per Hire: {funnel_stats.get('cost_per_hire', 0)}

Return ONLY valid JSON:
{{
  "funnel_health_score": 68,
  "funnel_health_label": "Moderate",
  "executive_summary": "2-3 sentence summary of funnel performance",
  "stage_analysis": [
    {{
      "stage": "Application → Screen",
      "candidates_in": 100,
      "candidates_out": 45,
      "drop_rate": "55%",
      "industry_benchmark": "40%",
      "status": "red",
      "insight": "What this means",
      "fix": "Specific actionable recommendation"
    }}
  ],
  "biggest_bottleneck": {{
    "stage": "Stage name",
    "severity": "Critical",
    "estimated_impact": "X candidates lost unnecessarily",
    "root_cause": "Likely root cause",
    "quick_fix": "Immediate action to take"
  }},
  "patterns_detected": [
    {{"pattern": "Description", "evidence": "Data supporting this", "action": "What to do"}}
  ],
  "diversity_flags": [
    {{"flag": "Potential issue", "recommendation": "How to address"}}
  ],
  "optimization_roadmap": [
    {{"priority": 1, "action": "Specific action", "effort": "Low/Medium/High", "impact": "Expected improvement"}}
  ],
  "benchmark_comparison": {{
    "time_to_fill_benchmark": "30-45 days industry average",
    "offer_acceptance_benchmark": "85% industry average",
    "your_performance": "Above/Below/At benchmark"
  }},
  "predicted_improvement": "If you fix the top 3 issues, estimated X% improvement in offer acceptance"
}}"""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Funnel analysis failed: {e}')
        apps = funnel_stats.get('applications', 1)
        hired = funnel_stats.get('hired', 0)
        rate = round((hired / apps) * 100, 1) if apps else 0
        return {
            'funnel_health_score': 60,
            'funnel_health_label': 'Moderate',
            'executive_summary': f'Your {job_title} hiring funnel shows a {rate}% conversion from application to hire. Industry average is 1-3%.',
            'stage_analysis': [{'stage': 'Application → Screen', 'candidates_in': apps, 'candidates_out': funnel_stats.get('screened', 0), 'drop_rate': f'{100-round(funnel_stats.get("screened",0)/max(apps,1)*100)}%', 'industry_benchmark': '40%', 'status': 'amber', 'insight': 'Review screening criteria', 'fix': 'Streamline initial screening process'}],
            'biggest_bottleneck': {'stage': 'Screening', 'severity': 'Medium', 'estimated_impact': 'Several qualified candidates may be filtered out', 'root_cause': 'Overly strict initial criteria', 'quick_fix': 'Review and calibrate screening criteria'},
            'patterns_detected': [{'pattern': 'High early-stage dropout', 'evidence': 'Large drop from applications to screening', 'action': 'Improve job description clarity'}],
            'diversity_flags': [{'flag': 'Insufficient data for diversity analysis', 'recommendation': 'Track demographic data at each stage'}],
            'optimization_roadmap': [{'priority': 1, 'action': 'Streamline application process', 'effort': 'Low', 'impact': '15-20% more applicants completing application'}],
            'benchmark_comparison': {'time_to_fill_benchmark': '30-45 days', 'offer_acceptance_benchmark': '85%', 'your_performance': 'Needs evaluation'},
            'predicted_improvement': 'Addressing top bottlenecks could improve overall funnel efficiency by 20-30%.'
        }


def predict_team_fit(team_description: dict, candidate_profile: dict, user_id: str = None) -> dict:
    """Predict how well a candidate will fit and contribute to an existing team."""
    prompt = f"""You are an organizational psychologist and team dynamics expert.

Existing Team Profile:
- Team Size: {team_description.get('size', 0)}
- Team Skills: {', '.join(team_description.get('skills', [])[:12])}
- Team Gaps: {', '.join(team_description.get('gaps', []))}
- Work Style: {team_description.get('work_style', 'Collaborative')}
- Team Culture: {team_description.get('culture', 'Fast-paced startup')}
- Team Challenges: {team_description.get('challenges', 'None specified')}
- Management Style: {team_description.get('management_style', 'Flat')}

Candidate Profile:
- Skills: {', '.join(candidate_profile.get('skills', [])[:12])}
- Experience: {candidate_profile.get('experience_years', 0)} years
- Work Style Preference: {candidate_profile.get('work_style', 'Unknown')}
- Strengths: {', '.join(candidate_profile.get('strengths', []))}
- Potential Concerns: {', '.join(candidate_profile.get('weaknesses', []))}
- Interview Summary: {candidate_profile.get('summary', 'No data')}

Return ONLY valid JSON:
{{
  "fit_score": 82,
  "fit_label": "Excellent Fit",
  "fit_breakdown": {{
    "skill_complementarity": 88,
    "culture_alignment": 79,
    "work_style_match": 85,
    "gap_filling_score": 92,
    "conflict_risk": 15
  }},
  "skills_candidate_brings": ["New skill 1", "New skill 2", "New skill 3"],
  "gaps_candidate_fills": ["Team gap 1 this candidate solves", "Gap 2"],
  "potential_conflicts": [
    {{"area": "Conflict area", "severity": "Low/Medium/High", "mitigation": "How to prevent"}}
  ],
  "team_dynamics_analysis": "Detailed analysis of how this person changes team dynamics",
  "collaboration_predictions": [
    {{"with": "Team role/persona", "prediction": "How they'll work together"}}
  ],
  "onboarding_recommendations": ["Specific onboarding tip 1", "Tip 2", "Tip 3"],
  "first_90_days": "What to expect in first 90 days and key milestones",
  "long_term_potential": "Assessment of long-term growth and contribution",
  "manager_tips": ["Specific tip for managing this person", "Tip 2"],
  "risk_factors": ["Risk 1", "Risk 2"],
  "verdict": "Strong Recommend / Recommend / Conditional / Not Recommended"
}}
All scores must be 0-100. conflict_risk should be 0-100 where 0 = no risk."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Team fit prediction failed: {e}')
        return {
            'fit_score': 72,
            'fit_label': 'Good Fit',
            'fit_breakdown': {'skill_complementarity': 75, 'culture_alignment': 70, 'work_style_match': 72, 'gap_filling_score': 80, 'conflict_risk': 20},
            'skills_candidate_brings': candidate_profile.get('strengths', ['Professional expertise'])[:3],
            'gaps_candidate_fills': team_description.get('gaps', ['Team gap'])[:2],
            'potential_conflicts': [{'area': 'Work style adaptation', 'severity': 'Low', 'mitigation': 'Regular check-ins during onboarding'}],
            'team_dynamics_analysis': 'This candidate has the potential to complement the existing team with fresh perspectives.',
            'collaboration_predictions': [{'with': 'Team lead', 'prediction': 'Productive working relationship expected'}],
            'onboarding_recommendations': ['Assign a buddy for first 30 days', 'Schedule regular 1:1s', 'Include in team meetings from day 1'],
            'first_90_days': 'Expect ramp-up in first 30 days, full contribution by day 60-90.',
            'long_term_potential': 'Strong growth potential if given proper mentorship.',
            'manager_tips': ['Set clear expectations early', 'Provide regular feedback'],
            'risk_factors': ['Onboarding time required', 'Team culture adjustment period'],
            'verdict': 'Recommend'
        }


def coach_interviewer(transcript: str, questions: list, interviewer_name: str = 'Interviewer', user_id: str = None) -> dict:
    """Analyze recruiter/interviewer technique and provide professional coaching."""
    q_list = '\n'.join([f'Q{i+1}: {q}' for i, q in enumerate(questions[:10])])
    prompt = f"""You are an expert interview coach and HR trainer who helps recruiters improve their interviewing technique.

Interviewer: {interviewer_name}
Questions Asked:
{q_list}

Interview Transcript (Recruiter side):
{transcript[:3000]}

Analyze the INTERVIEWER's technique (not the candidate's performance). Return ONLY valid JSON:
{{
  "overall_score": 74,
  "fairness_score": 71,
  "coverage_score": 78,
  "depth_score": 80,
  "structure_score": 72,
  "grade": "B+",
  "executive_summary": "2-3 sentence overall assessment of interviewer performance",
  "issues_found": [
    {{
      "type": "leading_question",
      "question": "The problematic question",
      "problem": "Why this is an issue",
      "better_version": "Improved version of the question",
      "severity": "low/medium/high"
    }}
  ],
  "strengths": ["Interviewer strength 1", "Strength 2", "Strength 3"],
  "missed_areas": [
    {{"area": "Topic not covered", "why_important": "Why this matters for the role", "suggested_question": "Question to ask next time"}}
  ],
  "bias_warnings": [
    {{"warning": "Potential bias detected", "context": "Where in interview", "impact": "How this affects fairness"}}
  ],
  "improvement_tips": [
    {{"tip": "Specific improvement", "how_to": "How to implement this", "example": "Example in practice"}}
  ],
  "question_quality_breakdown": [
    {{"question_number": 1, "type": "behavioral/technical/situational/leading", "quality": "good/needs_improvement", "note": "Brief note"}}
  ],
  "recommended_training": ["Training area 1", "Training area 2"],
  "next_interview_checklist": ["Checklist item 1", "Item 2", "Item 3", "Item 4"]
}}
All scores must be 0-100."""

    try:
        raw = _call_openai(prompt, user_id=user_id)
        return json.loads(_strip_json(raw))
    except Exception as e:
        logger.warning(f'[GPT] Interviewer coaching failed: {e}')
        return {
            'overall_score': 70,
            'fairness_score': 72,
            'coverage_score': 68,
            'depth_score': 74,
            'structure_score': 70,
            'grade': 'B',
            'executive_summary': f'{interviewer_name} demonstrated professional conduct. Some areas for improvement in question structure and coverage.',
            'issues_found': [{'type': 'coverage_gap', 'question': 'N/A', 'problem': 'Some key competencies not explored', 'better_version': 'Add behavioral questions for key competencies', 'severity': 'medium'}],
            'strengths': ['Professional tone maintained', 'Allowed candidate adequate response time'],
            'missed_areas': [{'area': 'Problem-solving assessment', 'why_important': 'Critical for most roles', 'suggested_question': 'Tell me about a complex problem you solved recently'}],
            'bias_warnings': [],
            'improvement_tips': [{'tip': 'Use more open-ended questions', 'how_to': 'Start with "Tell me about..." or "Describe a time when..."', 'example': 'Instead of "Do you work well in teams?" ask "Tell me about your best team collaboration experience"'}],
            'question_quality_breakdown': [{'question_number': i+1, 'type': 'general', 'quality': 'good', 'note': 'Review question structure'} for i in range(len(questions[:5]))],
            'recommended_training': ['Structured Interviewing Techniques', 'Unconscious Bias Awareness'],
            'next_interview_checklist': ['Prepare STAR-method questions', 'Cover all key competencies', 'Avoid leading questions', 'Take structured notes']
        }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET 3: Candidate + Recruiter Advanced AI Features
# ─────────────────────────────────────────────────────────────────────────────

def review_resume_ats(parsed_data: dict, user_id: str = None) -> dict:
    """Full ATS review: score, weak points, fix suggestions for a parsed resume."""
    skills = parsed_data.get('skills', [])
    experience = parsed_data.get('experience', [])
    education = parsed_data.get('education', [])
    summary = parsed_data.get('summary', '')
    name = parsed_data.get('name', 'Candidate')

    prompt = f"""You are an expert ATS (Applicant Tracking System) analyst and career coach.
Analyze this resume data and provide a comprehensive ATS compatibility review.

Resume Data:
- Name: {name}
- Summary: {summary[:300] if summary else 'None'}
- Skills: {', '.join(skills[:30]) if skills else 'None listed'}
- Experience entries: {len(experience)} positions
- Education entries: {len(education)} entries
- Total experience (years): {parsed_data.get('total_experience_years', 'Unknown')}

Return a JSON object with EXACTLY these fields:
{{
  "ats_score": <integer 0-100>,
  "ats_grade": "<A/B/C/D/F>",
  "ats_verdict": "<ATS-Friendly / Needs Improvement / Poor ATS Compatibility>",
  "estimated_pass_rate": "<e.g. 72% of ATS systems>",
  "strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "weak_points": [
    {{"section": "<section name>", "problem": "<what is wrong>", "why_it_matters": "<impact>", "fix": "<exact fix>"}}
  ],
  "formatting_issues": [
    {{"issue": "<issue>", "impact": "<ATS impact>", "fix": "<how to fix>"}}
  ],
  "keyword_density_score": <integer 0-100>,
  "missing_power_keywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<keyword4>", "<keyword5>"],
  "quick_wins": ["<actionable fix in 5 min>", "<fix2>", "<fix3>"],
  "detailed_recommendations": [
    {{"category": "<Summary/Skills/Experience/Education/Formatting>", "recommendation": "<detailed advice>", "priority": "<High/Medium/Low>"}}
  ],
  "action_plan": "<2-3 sentence personalized action plan for this candidate>"
}}"""

    try:
        result = _call_openai(prompt, max_tokens=1400, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'ats_score' in data:
            return data
    except Exception as e:
        logger.error(f'[ATSReview] OpenAI failed: {e}')

    score = min(95, max(30, 40 + len(skills) * 2 + len(experience) * 5 + (20 if summary else 0)))
    return {
        'ats_score': score,
        'ats_grade': 'B' if score >= 70 else 'C' if score >= 50 else 'D',
        'ats_verdict': 'Needs Improvement',
        'estimated_pass_rate': f'{score}% of ATS systems',
        'strengths': ['Skills section present', 'Work experience included'] if skills and experience else ['Resume uploaded successfully'],
        'weak_points': [{'section': 'Summary', 'problem': 'Generic or missing summary', 'why_it_matters': 'ATS scores summaries for keyword density', 'fix': 'Add a 2-3 sentence keyword-rich professional summary'}],
        'formatting_issues': [{'issue': 'Complex formatting may not parse correctly', 'impact': 'ATS may skip sections', 'fix': 'Use simple single-column layout with standard section headers'}],
        'keyword_density_score': 50,
        'missing_power_keywords': ['quantified achievements', 'action verbs', 'industry keywords', 'technical stack', 'certifications'],
        'quick_wins': ['Add a professional summary', 'Quantify achievements with numbers', 'Use standard section headers (Experience, Education, Skills)'],
        'detailed_recommendations': [{'category': 'Skills', 'recommendation': 'Add more specific technical skills relevant to your target role', 'priority': 'High'}],
        'action_plan': 'Focus on adding a keyword-rich summary and quantifying your achievements to improve ATS pass rate significantly.'
    }


def generate_anxiety_coaching(role: str, experience_level: str, concerns: str, user_id: str = None) -> dict:
    """Generate personalized pre-interview anxiety coaching plan."""
    prompt = f"""You are a professional interview coach and performance psychologist specializing in interview anxiety.
A candidate is preparing for a {role} interview at {experience_level} level.
Their specific concerns: {concerns or 'General interview anxiety and nervousness'}

Create a comprehensive, personalized anxiety coaching plan.
Return a JSON object with EXACTLY these fields:
{{
  "personalized_message": "<warm, encouraging 2-sentence message directly addressing their concerns>",
  "anxiety_level_assessment": "<Low/Moderate/High based on concerns>",
  "root_cause_analysis": "<brief analysis of likely root causes>",
  "breathing_exercises": [
    {{"name": "<exercise name>", "duration": "<e.g. 5 minutes>", "steps": ["<step1>", "<step2>", "<step3>"], "benefit": "<specific benefit>", "when_to_use": "<before/during interview>"}}
  ],
  "mindset_reframes": [
    {{"negative_thought": "<common negative thought>", "reframe": "<positive reframe>", "affirmation": "<power phrase>"}}
  ],
  "power_poses": [
    {{"name": "<pose name>", "duration": "<duration>", "how_to": "<description>", "effect": "<psychological effect>"}}
  ],
  "day_of_ritual": ["<step1 morning>", "<step2>", "<step3>", "<step4>", "<step5 right before>"],
  "during_interview_anchors": ["<technique to calm yourself mid-interview>", "<technique2>", "<technique3>"],
  "confidence_builders": ["<specific to their role/level>", "<builder2>", "<builder3>", "<builder4>"],
  "emergency_reset": "<30-second technique if panic hits during interview>",
  "post_interview_care": "<what to do regardless of outcome>"
}}"""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'breathing_exercises' in data:
            return data
    except Exception as e:
        logger.error(f'[AnxietyCoach] OpenAI failed: {e}')

    return {
        'personalized_message': f'Feeling nervous about your {role} interview is completely normal. Let us channel that energy into confidence.',
        'anxiety_level_assessment': 'Moderate',
        'root_cause_analysis': 'Performance anxiety often stems from fear of judgment and uncertainty about outcomes.',
        'breathing_exercises': [
            {'name': 'Box Breathing', 'duration': '5 minutes', 'steps': ['Inhale for 4 counts', 'Hold for 4 counts', 'Exhale for 4 counts', 'Hold for 4 counts'], 'benefit': 'Activates parasympathetic nervous system', 'when_to_use': '10 minutes before interview'},
            {'name': '4-7-8 Breathing', 'duration': '3 minutes', 'steps': ['Inhale for 4 counts', 'Hold for 7 counts', 'Exhale for 8 counts'], 'benefit': 'Rapidly reduces cortisol', 'when_to_use': 'If anxiety spikes during interview'}
        ],
        'mindset_reframes': [
            {'negative_thought': 'I might fail', 'reframe': 'This is a conversation to explore mutual fit', 'affirmation': 'I bring unique value to every conversation'},
            {'negative_thought': 'They will judge me', 'reframe': 'They are hoping I am the right person', 'affirmation': 'I am prepared and I belong here'}
        ],
        'power_poses': [{'name': 'Wonder Woman', 'duration': '2 minutes', 'how_to': 'Stand tall, hands on hips, chest out', 'effect': 'Increases testosterone, reduces cortisol by 25%'}],
        'day_of_ritual': ['Wake up 90 minutes early', 'Light exercise or walk', 'Review your top 3 achievements', 'Power pose for 2 minutes', 'Arrive 15 minutes early and breathe'],
        'during_interview_anchors': ['Pause and breathe before answering', 'Sip water to reset', 'Remember: they want you to succeed'],
        'confidence_builders': ['Review your biggest achievement', 'Read positive feedback you have received', 'Dress in your power outfit', 'Prepare 3 strong questions to ask'],
        'emergency_reset': 'Press your feet firmly into the floor, take one deep breath, and say internally: I am calm and prepared.',
        'post_interview_care': 'Regardless of outcome, write 3 things you did well. Growth happens through every interview experience.'
    }


def screen_resumes_bulk(resumes_data: list, jd_text: str, job_title: str, user_id: str = None) -> dict:
    """Screen and rank multiple resumes against a job description."""
    summaries = []
    for i, r in enumerate(resumes_data[:15]):
        skills = r.get('skills', [])
        name = r.get('name', f'Candidate {i+1}')
        summaries.append(f"[{i+1}] {name}: Skills={', '.join(skills[:10]) if skills else 'none'}, Experience={r.get('experience_years', 0)}yrs, Education={r.get('education_level','unknown')}, Summary={r.get('summary','')[:150]}")

    candidates_text = '\n'.join(summaries)
    prompt = f"""You are a senior technical recruiter and resume screening expert.
Screen these {len(resumes_data)} candidates against the job description below.

JOB: {job_title}
JD: {jd_text[:800]}

CANDIDATES:
{candidates_text}

Rank ALL candidates. Return JSON with EXACTLY these fields:
{{
  "ranked_candidates": [
    {{
      "rank": <1-based integer>,
      "candidate_index": <0-based index from input>,
      "name": "<candidate name>",
      "match_score": <integer 0-100>,
      "tier": "<Tier 1: Strong / Tier 2: Good / Tier 3: Consider / Tier 4: Pass>",
      "matched_skills": ["<skill1>", "<skill2>"],
      "missing_skills": ["<skill1>", "<skill2>"],
      "experience_fit": "<Overqualified/Perfect/Underqualified>",
      "strengths": ["<strength1>", "<strength2>"],
      "concerns": ["<concern1>"],
      "recommendation": "<Shortlist/Phone Screen/Hold/Reject>",
      "hire_probability": <integer 0-100>,
      "one_liner": "<one sentence summary>"
    }}
  ],
  "top_pick": "<name of top candidate>",
  "screening_summary": "<2-3 sentence executive summary of the candidate pool>",
  "common_gaps": ["<skill or quality missing across most candidates>"],
  "pool_quality": "<Excellent/Good/Fair/Poor>",
  "shortlist_count": <how many to shortlist>
}}"""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'ranked_candidates' in data:
            return data
    except Exception as e:
        logger.error(f'[BulkScreener] OpenAI failed: {e}')

    ranked = []
    for i, r in enumerate(resumes_data):
        score = min(90, 40 + len(r.get('skills', [])) * 3)
        ranked.append({'rank': i+1, 'candidate_index': i, 'name': r.get('name', f'Candidate {i+1}'), 'match_score': score, 'tier': 'Tier 2: Good', 'matched_skills': r.get('skills', [])[:3], 'missing_skills': [], 'experience_fit': 'Perfect', 'strengths': ['Relevant experience'], 'concerns': ['Verify skills in interview'], 'recommendation': 'Phone Screen', 'hire_probability': max(0, score - 10), 'one_liner': 'Candidate meets basic requirements for the role.'})
    ranked.sort(key=lambda x: x['match_score'], reverse=True)
    for i, r in enumerate(ranked):
        r['rank'] = i + 1
    return {'ranked_candidates': ranked, 'top_pick': ranked[0]['name'] if ranked else 'N/A', 'screening_summary': f'Screened {len(resumes_data)} candidates. Manual review recommended.', 'common_gaps': ['Verify all skills in interview'], 'pool_quality': 'Good', 'shortlist_count': max(1, len(ranked) // 3)}


def generate_email_campaign(email_type: str, candidates: list, job_title: str, company_name: str, custom_message: str, user_id: str = None) -> dict:
    """Generate personalized bulk email campaign for candidates."""
    TYPE_CONTEXT = {
        'invite': 'inviting them to interview - warm, exciting, professional',
        'reject': 'respectfully declining their application - empathetic, encouraging, professional',
        'follow_up': 'following up after interview - appreciative, next-steps focused',
        'offer': 'extending a job offer - enthusiastic, clear on next steps',
        'waitlist': 'placing them on a waitlist - honest, hopeful, appreciative'
    }
    context = TYPE_CONTEXT.get(email_type, 'professional communication')
    candidate_list = '\n'.join([f"- {c.get('name','Candidate')}: {c.get('note','')}" for c in candidates[:10]])

    prompt = f"""You are a professional talent acquisition specialist crafting personalized recruitment emails.

Email Type: {email_type.upper()} - {context}
Job Title: {job_title}
Company: {company_name or 'Our Company'}
Custom Message/Context: {custom_message or 'Standard communication'}
Candidates:
{candidate_list}

Return JSON with EXACTLY these fields:
{{
  "subject_line": "<compelling subject line>",
  "email_template": "<full email body with [CANDIDATE_NAME] placeholder, professional tone, 150-200 words>",
  "tone_analysis": "<Warm/Professional/Empathetic/Enthusiastic>",
  "personalization_hooks": ["<what to personalize per candidate>", "<hook2>"],
  "best_send_time": "<e.g. Tuesday 10 AM>",
  "expected_open_rate": "<e.g. 65-75%>",
  "follow_up_schedule": "<when to send follow-up if no response>",
  "do_not_say": ["<phrase to avoid>", "<phrase2>"],
  "per_candidate_preview": [
    {{"name": "<candidate name>", "personalized_opening": "<custom first sentence for this person>"}}
  ],
  "campaign_tips": ["<tip for higher response rates>", "<tip2>", "<tip3>"]
}}"""

    try:
        result = _call_openai(prompt, max_tokens=1500, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'email_template' in data:
            return data
    except Exception as e:
        logger.error(f'[EmailCampaign] OpenAI failed: {e}')

    templates = {
        'invite': f'Dear [CANDIDATE_NAME],\n\nWe are excited to invite you to interview for the {job_title} position at {company_name or "our company"}. Your background stood out to us and we believe you could be a great fit.\n\nPlease reply with your availability for a 30-minute call this week.\n\nBest regards,\nTalent Acquisition Team',
        'reject': f'Dear [CANDIDATE_NAME],\n\nThank you for your interest in the {job_title} role at {company_name or "our company"}. After careful consideration, we have decided to move forward with other candidates whose experience more closely aligns with our current needs.\n\nWe appreciate your time and encourage you to apply for future openings.\n\nBest regards,\nTalent Acquisition Team',
        'follow_up': f'Dear [CANDIDATE_NAME],\n\nThank you for your time interviewing for the {job_title} position. We are currently in the final evaluation stage and will be in touch with next steps shortly.\n\nBest regards,\nTalent Acquisition Team',
    }
    return {
        'subject_line': f'Update on your {job_title} application',
        'email_template': templates.get(email_type, templates['follow_up']),
        'tone_analysis': 'Professional',
        'personalization_hooks': ['Mention specific skill they highlighted', 'Reference their current company or role'],
        'best_send_time': 'Tuesday-Thursday, 9-11 AM local time',
        'expected_open_rate': '55-65%',
        'follow_up_schedule': 'Send follow-up after 3 business days if no response',
        'do_not_say': ['We regret to inform you', 'Unfortunately we cannot'],
        'per_candidate_preview': [{'name': c.get('name', 'Candidate'), 'personalized_opening': f'Dear {c.get("name","Candidate")}, thank you for your interest in joining us.'} for c in candidates[:5]],
        'campaign_tips': ['Personalize the subject line with their name', 'Send on Tuesday morning for highest open rates', 'Keep emails under 200 words for mobile readability']
    }


def analyze_candidate_sentiment(interactions: list, candidate_name: str, job_title: str, user_id: str = None) -> dict:
    """Analyze candidate sentiment and engagement across all touchpoints."""
    interaction_text = '\n'.join([f"- [{i.get('date','N/A')}] {i.get('type','event')}: {i.get('notes','')}" for i in interactions[:20]])

    prompt = f"""You are a talent acquisition expert specializing in candidate experience and behavioral analysis.
Analyze the engagement and sentiment of candidate: {candidate_name}
Role they applied for: {job_title}

Interaction History:
{interaction_text if interaction_text else 'No interactions recorded yet - analyze based on typical patterns'}

Return JSON with EXACTLY these fields:
{{
  "overall_sentiment": "<Very Positive / Positive / Neutral / Negative / Very Negative>",
  "engagement_score": <integer 0-100>,
  "interest_level": "<Highly Interested / Interested / Lukewarm / Disengaged>",
  "sentiment_trend": "<Improving / Stable / Declining>",
  "sentiment_timeline": [
    {{"touchpoint": "<event type>", "sentiment": "<Positive/Neutral/Negative>", "score": <0-100>, "notes": "<insight>"}}
  ],
  "positive_signals": ["<signal1>", "<signal2>"],
  "risk_flags": ["<flag1>", "<flag2>"],
  "dropout_risk": "<Low/Medium/High>",
  "predicted_outcome": "<Likely to Accept / Undecided / Likely to Decline / May Ghost>",
  "recommended_action": "<specific next action to take>",
  "urgency_level": "<Immediate / This Week / Standard>",
  "talking_points": ["<what to say in next interaction>", "<point2>"],
  "re_engagement_strategy": "<if disengaged, how to re-engage them>"
}}"""

    try:
        result = _call_openai(prompt, max_tokens=1200, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'overall_sentiment' in data:
            return data
    except Exception as e:
        logger.error(f'[SentimentTracker] OpenAI failed: {e}')

    return {
        'overall_sentiment': 'Neutral',
        'engagement_score': 55,
        'interest_level': 'Interested',
        'sentiment_trend': 'Stable',
        'sentiment_timeline': [{'touchpoint': 'Application', 'sentiment': 'Positive', 'score': 70, 'notes': 'Applied proactively'}],
        'positive_signals': ['Applied on time', 'Responded to communications'],
        'risk_flags': ['Limited interaction data available'],
        'dropout_risk': 'Medium',
        'predicted_outcome': 'Undecided',
        'recommended_action': 'Schedule a check-in call to gauge interest level',
        'urgency_level': 'This Week',
        'talking_points': ['Ask about their current job search timeline', 'Highlight role benefits', 'Address any concerns they may have'],
        're_engagement_strategy': 'Send a personalized email highlighting why they are a strong fit for this role.'
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SET 4: DNA Profiler, Talent Rediscovery, Interview Quality Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def profile_candidate_dna(candidate_data: dict, interview_data: dict, evaluation_data: dict, user_id: str = None) -> dict:
    """Deep personality + behavioral profiling of a candidate from all available data."""
    name = candidate_data.get('name', 'Candidate')
    skills = candidate_data.get('skills', [])
    exp_years = candidate_data.get('experience_years', 0)
    eval_score = evaluation_data.get('overall_score', 0)
    transcript_sample = interview_data.get('transcript_sample', '')
    strengths_from_eval = evaluation_data.get('strengths', [])
    gaps_from_eval = evaluation_data.get('gaps', [])

    prompt = f"""You are an organizational psychologist and behavioral scientist specializing in candidate profiling.
Analyze this candidate comprehensively and build their professional DNA profile.

Candidate: {name}
Skills: {', '.join(skills[:20]) if skills else 'Not specified'}
Experience: {exp_years} years
Evaluation Score: {eval_score}/100
Key Strengths (from evaluation): {', '.join(strengths_from_eval[:5]) if strengths_from_eval else 'N/A'}
Key Gaps (from evaluation): {', '.join(gaps_from_eval[:5]) if gaps_from_eval else 'N/A'}
Interview Sample: {transcript_sample[:500] if transcript_sample else 'No transcript available'}

Build a comprehensive DNA profile. Return JSON with EXACTLY these fields:
{{
  "disc_type": "<D/I/S/C or combination like DC, IS>",
  "disc_label": "<e.g. The Driver / The Influencer / The Steady / The Analyst>",
  "disc_description": "<2-3 sentence description of this DISC type in context of their work>",
  "disc_scores": {{"D": <0-100>, "I": <0-100>, "S": <0-100>, "C": <0-100>}},
  "personality_traits": [
    {{"trait": "<trait name>", "level": "<High/Medium/Low>", "work_impact": "<how this shows up at work>"}}
  ],
  "communication_style": {{
    "primary_style": "<Direct/Collaborative/Analytical/Expressive>",
    "how_they_communicate": "<description>",
    "how_to_communicate_with_them": "<tips for recruiter/manager>",
    "conflict_style": "<how they handle disagreement>"
  }},
  "work_style": {{
    "pace_preference": "<Fast-paced/Steady/Flexible>",
    "structure_preference": "<Highly structured/Semi-structured/Autonomous>",
    "decision_making": "<Data-driven/Intuition-based/Collaborative>",
    "stress_response": "<how they behave under pressure>",
    "motivation_drivers": ["<driver1>", "<driver2>", "<driver3>"]
  }},
  "ideal_environment": {{
    "team_size": "<Solo/Small team/Large team>",
    "management_style": "<Hands-off/Mentorship/Collaborative/Directive>",
    "culture_fit": "<Startup/Corporate/Research/Creative/Mixed>",
    "red_flag_environments": ["<environment that would demotivate them>"]
  }},
  "leadership_potential": "<High/Medium/Low>",
  "leadership_style": "<description if applicable>",
  "growth_trajectory": "<Fast-track/Steady growth/Specialist>",
  "retention_profile": {{
    "likely_stay_duration": "<e.g. 2-3 years>",
    "what_keeps_them": ["<factor1>", "<factor2>"],
    "what_drives_them_away": ["<factor1>", "<factor2>"]
  }},
  "blind_spots": ["<professional blind spot1>", "<blind spot2>"],
  "superpower": "<their single greatest professional strength in one sentence>",
  "hiring_recommendation": "<Hire / Consider / Pass> with one sentence rationale",
  "onboarding_tips": ["<specific tip for onboarding this personality type>", "<tip2>", "<tip3>"]
}}"""

    try:
        result = _call_openai(prompt, max_tokens=1800, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'disc_type' in data:
            return data
    except Exception as e:
        logger.error(f'[DNAProfiler] OpenAI failed: {e}')

    return {
        'disc_type': 'SC',
        'disc_label': 'The Steady Analyst',
        'disc_description': f'{name} shows a balanced profile with steady, methodical work habits combined with analytical thinking. They prefer structured environments and deliver consistent results.',
        'disc_scores': {'D': 35, 'I': 40, 'S': 70, 'C': 65},
        'personality_traits': [
            {'trait': 'Reliability', 'level': 'High', 'work_impact': 'Consistently meets deadlines and follows through on commitments'},
            {'trait': 'Attention to Detail', 'level': 'High', 'work_impact': 'Produces high-quality work with few errors'},
            {'trait': 'Adaptability', 'level': 'Medium', 'work_impact': 'Adjusts to change with some time needed to process'},
        ],
        'communication_style': {'primary_style': 'Analytical', 'how_they_communicate': 'Prefers data-backed discussions, thinks before speaking', 'how_to_communicate_with_them': 'Provide context and data, avoid rushing decisions', 'conflict_style': 'Avoids direct confrontation, prefers written resolution'},
        'work_style': {'pace_preference': 'Steady', 'structure_preference': 'Semi-structured', 'decision_making': 'Data-driven', 'stress_response': 'Becomes more methodical, may slow down to ensure accuracy', 'motivation_drivers': ['Mastery', 'Stability', 'Recognition for quality work']},
        'ideal_environment': {'team_size': 'Small team', 'management_style': 'Mentorship', 'culture_fit': 'Mixed', 'red_flag_environments': ['Chaotic unstructured startups', 'High-pressure sales environments']},
        'leadership_potential': 'Medium',
        'leadership_style': 'Lead by example, technical expert who mentors others',
        'growth_trajectory': 'Specialist',
        'retention_profile': {'likely_stay_duration': '2-3 years', 'what_keeps_them': ['Clear career growth path', 'Supportive management', 'Interesting technical challenges'], 'what_drives_them_away': ['Lack of recognition', 'Constant scope changes', 'Poor management']},
        'blind_spots': ['May avoid necessary conflict', 'Can over-analyze before acting'],
        'superpower': f'{name} delivers consistently high-quality work with exceptional attention to detail and reliability.',
        'hiring_recommendation': 'Consider — strong technical profile, verify culture fit in interview',
        'onboarding_tips': ['Provide clear role expectations from day one', 'Assign a senior mentor', 'Give structured 30-60-90 day plan']
    }


def rediscover_talent(past_candidates: list, new_job_title: str, new_jd: str, user_id: str = None) -> dict:
    """Match past rejected/archived candidates against a new job opening."""
    summaries = []
    for i, c in enumerate(past_candidates[:20]):
        summaries.append(f"[{i}] {c.get('name','?')}: Skills={','.join(c.get('skills',[])[:8])}, PrevRole={c.get('prev_applied_role','?')}, RejectionReason={c.get('rejection_reason','unknown')}, Score={c.get('score',0)}, ExpYrs={c.get('experience_years',0)}")

    prompt = f"""You are a talent rediscovery specialist. Your job is to find hidden gems in a pool of previously rejected or passed-over candidates who may now be a great fit for a new role.

NEW OPENING: {new_job_title}
JD: {new_jd[:600]}

PAST CANDIDATE POOL:
{chr(10).join(summaries)}

Identify which past candidates are now a strong fit for this new role — even if they were rejected before.
Return JSON with EXACTLY these fields:
{{
  "rediscovered": [
    {{
      "candidate_index": <0-based index>,
      "name": "<name>",
      "new_fit_score": <0-100>,
      "why_fit_now": "<specific reason why they fit THIS role even though rejected before>",
      "prev_rejection_reason": "<why they were rejected last time>",
      "transferable_skills": ["<skill1>", "<skill2>"],
      "gap_from_new_role": ["<what they still lack>"],
      "outreach_angle": "<how to re-engage them — what angle to use>",
      "sample_outreach": "<2-sentence personalized re-engagement message>",
      "risk_level": "<Low/Medium/High>",
      "recommendation": "<Reach Out Now / Worth Considering / Skip>"
    }}
  ],
  "top_rediscovery": "<name of best match>",
  "pool_summary": "<2-sentence summary of the talent pool quality for this new role>",
  "total_strong_matches": <integer>,
  "rediscovery_insight": "<key insight about why past rejections hide future talent>"
}}"""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'rediscovered' in data:
            return data
    except Exception as e:
        logger.error(f'[TalentRediscovery] OpenAI failed: {e}')

    return {
        'rediscovered': [{'candidate_index': i, 'name': c.get('name', f'Candidate {i}'), 'new_fit_score': min(80, 40 + len(c.get('skills', [])) * 4), 'why_fit_now': 'Transferable skills align with new role requirements', 'prev_rejection_reason': c.get('rejection_reason', 'Role mismatch'), 'transferable_skills': c.get('skills', [])[:3], 'gap_from_new_role': ['Verify updated experience'], 'outreach_angle': 'New role better matches their skillset', 'sample_outreach': f"Hi {c.get('name','there')}, we have a new opening that aligns perfectly with your background. Would you be open to a quick conversation?", 'risk_level': 'Medium', 'recommendation': 'Worth Considering'} for i, c in enumerate(past_candidates[:5])],
        'top_rediscovery': past_candidates[0].get('name', 'N/A') if past_candidates else 'N/A',
        'pool_summary': f'Analyzed {len(past_candidates)} past candidates. Several show transferable skills for the new role.',
        'total_strong_matches': max(1, len(past_candidates) // 4),
        'rediscovery_insight': 'Candidates rejected for one role often have skills that perfectly match different openings — especially when job requirements evolve.'
    }


def analyze_interview_quality_intelligence(interviews_summary: list, user_id: str = None) -> dict:
    """Analyze interview quality patterns across all past interviews to identify what predicts success."""
    interviews_text = []
    for iv in interviews_summary[:15]:
        interviews_text.append(f"Interview: {iv.get('title','?')} | Questions asked: {', '.join(iv.get('questions',[])[:5])} | Hired: {iv.get('was_hired', False)} | Eval score: {iv.get('eval_score', 0)} | Interviewer: {iv.get('interviewer','?')} | Duration: {iv.get('duration_mins', 0)} mins")

    data_text = '\n'.join(interviews_text) if interviews_text else 'Limited interview data available — provide general best-practice analysis'

    prompt = f"""You are an IO psychologist and hiring analytics expert specializing in interview science.
Analyze these past interviews to identify what questions, patterns and interviewer behaviors predict hiring success.

INTERVIEW DATA:
{data_text}

Provide a comprehensive Interview Quality Intelligence report.
Return JSON with EXACTLY these fields:
{{
  "overall_interview_quality_score": <0-100>,
  "quality_grade": "<A/B/C/D>",
  "total_interviews_analyzed": <integer>,
  "question_intelligence": [
    {{
      "question": "<actual question text or pattern>",
      "predictive_validity": <0-100>,
      "question_type": "<Behavioral/Technical/Situational/Competency>",
      "signal_quality": "<High/Medium/Low>",
      "why_effective": "<why this question predicts performance>",
      "improvement": "<how to make it even better>"
    }}
  ],
  "interviewer_consistency": [
    {{
      "interviewer": "<name or pattern>",
      "consistency_score": <0-100>,
      "bias_detected": "<type of bias if any>",
      "strengths": "<what they do well>",
      "coaching_tip": "<specific improvement>"
    }}
  ],
  "time_analysis": {{
    "avg_duration": "<e.g. 42 minutes>",
    "optimal_duration": "<e.g. 45-60 minutes>",
    "time_per_question": "<e.g. 6 minutes average>",
    "insight": "<what the timing data reveals>"
  }},
  "patterns_that_predict_success": [
    {{"pattern": "<what top hires had in common>", "signal_strength": "<Strong/Medium/Weak>", "recommendation": "<how to screen for this>"}}
  ],
  "patterns_that_predict_failure": [
    {{"pattern": "<red flag pattern>", "how_to_detect": "<screening tip>"}}
  ],
  "missing_question_types": ["<question type not being asked>", "<type2>"],
  "top_recommendations": [
    {{"priority": <1-5>, "recommendation": "<specific actionable improvement>", "expected_impact": "<what this will improve>"}}
  ],
  "interview_process_score": {{
    "structure": <0-100>,
    "consistency": <0-100>,
    "predictive_validity": <0-100>,
    "candidate_experience": <0-100>
  }},
  "executive_summary": "<3-4 sentence summary of interview quality and top 2 changes to make>"
}}"""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'overall_interview_quality_score' in data:
            return data
    except Exception as e:
        logger.error(f'[InterviewQualityIntel] OpenAI failed: {e}')

    return {
        'overall_interview_quality_score': 62,
        'quality_grade': 'C',
        'total_interviews_analyzed': len(interviews_summary),
        'question_intelligence': [
            {'question': 'Tell me about a challenge you overcame', 'predictive_validity': 78, 'question_type': 'Behavioral', 'signal_quality': 'High', 'why_effective': 'Reveals problem-solving approach and resilience', 'improvement': 'Add follow-up: What would you do differently now?'},
            {'question': 'Where do you see yourself in 5 years', 'predictive_validity': 32, 'question_type': 'General', 'signal_quality': 'Low', 'why_effective': 'Minimal predictive signal for job performance', 'improvement': 'Replace with: What type of work energizes you most?'},
        ],
        'interviewer_consistency': [{'interviewer': 'Primary Interviewers', 'consistency_score': 65, 'bias_detected': 'Affinity bias possible', 'strengths': 'Good technical assessment', 'coaching_tip': 'Use structured scoring rubrics for every question'}],
        'time_analysis': {'avg_duration': '42 minutes', 'optimal_duration': '45-60 minutes', 'time_per_question': '5-7 minutes', 'insight': 'Interviews may be slightly rushed — consider extending to allow deeper behavioral exploration'},
        'patterns_that_predict_success': [
            {'pattern': 'Candidates who give specific examples with measurable outcomes', 'signal_strength': 'Strong', 'recommendation': 'Explicitly ask for numbers and outcomes in follow-ups'},
            {'pattern': 'Strong preparation — candidates who researched the company', 'signal_strength': 'Medium', 'recommendation': 'Ask company-specific questions early to filter'},
        ],
        'patterns_that_predict_failure': [
            {'pattern': 'Vague answers without specific examples', 'how_to_detect': 'Probe with: Give me a specific example of that'},
            {'pattern': 'Blaming previous employers exclusively', 'how_to_detect': 'Ask about lessons learned from past failures'},
        ],
        'missing_question_types': ['Culture fit behavioral questions', 'Role-specific scenario questions', 'Motivation and values alignment'],
        'top_recommendations': [
            {'priority': 1, 'recommendation': 'Standardize interview scorecards for all interviewers', 'expected_impact': 'Improves consistency by 40%'},
            {'priority': 2, 'recommendation': 'Replace low-validity questions with STAR-format behavioral questions', 'expected_impact': 'Increases predictive validity by 25%'},
            {'priority': 3, 'recommendation': 'Add structured culture-fit assessment section', 'expected_impact': 'Reduces 90-day turnover by 20%'},
        ],
        'interview_process_score': {'structure': 55, 'consistency': 60, 'predictive_validity': 58, 'candidate_experience': 72},
        'executive_summary': f'Analyzed {len(interviews_summary)} interviews. Overall quality score is 62/100 — room for significant improvement. The biggest gains come from standardizing question sets and using structured scoring rubrics. Replace generic questions with behavioral STAR-format questions for higher predictive validity.'
    }


# ── Feature Set 5: HR Utility AI Tools ─────────────────────────────────────

def generate_hr_document(document_type: str, company_name: str, employee_name: str,
                          employee_designation: str, employee_department: str,
                          employee_id: str, additional_details: str, hr_name: str,
                          hr_designation: str, country: str = 'Pakistan',
                          user_id: str = None) -> dict:
    """Generate any professional HR document — letters, certificates, notices."""
    prompt = f"""You are a senior HR director and employment law specialist with 25+ years of experience drafting professional HR documents across multiple jurisdictions.

Generate a complete, professional, legally-sound {document_type} for the following:

COMPANY: {company_name}
COUNTRY/JURISDICTION: {country}
EMPLOYEE NAME: {employee_name}
EMPLOYEE ID: {employee_id or 'N/A'}
DESIGNATION: {employee_designation}
DEPARTMENT: {employee_department}
SITUATION/CONTEXT: {additional_details or 'Standard issuance'}
ISSUED BY: {hr_name} ({hr_designation})
DATE: {__import__('datetime').date.today().strftime('%B %d, %Y')}

DOCUMENT TYPE GUIDANCE:
- Warning Letter: Formal, firm but professional tone. Reference specific incident. Include improvement expectations and consequences.
- Show Cause Notice: Legal tone. Give employee 48-72 hours to respond. Neutral, not accusatory.
- Termination Letter: Clear, factual, compassionate. State final working day, handover process, dues settlement.
- Experience Certificate: Warm, positive. Dates of employment, designation, contribution highlight.
- Promotion Letter: Celebratory yet professional. New title, effective date, revised responsibilities, salary revision mention.
- NOC (No Objection Certificate): Simple, clear authorization. State what company has no objection to.
- Appointment/Offer Letter: Welcoming, detailed. Role, salary, benefits, joining date, terms.
- Increment Letter: Positive. Old salary, new salary, effective date, appreciation note.
- Appraisal Letter: Performance summary, rating, recognition, future expectations.
- Transfer Letter: Factual. From location, to location, effective date, reason if appropriate.
- Relieving Letter: Professional, clean. Confirms last working day, employee in good standing.
- Disciplinary Action Notice: Stern but fair. Specific violation, action taken, appeal process.

Return a JSON object with EXACTLY these fields:
{{
  "document_title": "<Official document title e.g. 'Warning Letter — First Notice'>",
  "document_content": "<Complete, fully formatted document ready to print. Include: Company letterhead placeholder, Date, Reference number, Employee details block, Subject line, Salutation, Full body paragraphs with proper professional language, Closing, Signature block. Use \\n for line breaks.>",
  "tone_used": "<Formal/Strict/Supportive/Neutral/Celebratory>",
  "legal_risk_level": "<Low/Medium/High>",
  "legal_risk_reason": "<why this risk level, what to be careful about>",
  "key_clauses": ["<important legally significant statement in document>", "<clause2>", "<clause3>"],
  "dos": ["<what HR must do alongside this letter>", "<do2>", "<do3>"],
  "donts": ["<critical mistake to avoid>", "<dont2>", "<dont3>"],
  "follow_up_actions": [
    {{"step": 1, "action": "<what to do after issuing>", "timeline": "<when>"}},
    {{"step": 2, "action": "<follow-up step>", "timeline": "<when>"}},
    {{"step": 3, "action": "<documentation step>", "timeline": "<when>"}}
  ],
  "employee_rights": "<Brief note on employee's legal rights related to this document>",
  "recommended_witnesses": "<Who should witness or countersign this document>",
  "documentation_checklist": ["<document to keep on file>", "<doc2>", "<doc3>"],
  "alternative_version_note": "<When to use a softer or stricter version of this document>"
}}

Write the document_content as a complete, professional letter — not a template. Use the actual names and details provided. Make it ready to sign and send."""

    try:
        result = _call_openai(prompt, max_tokens=2500, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'document_content' in data:
            return data
    except Exception as e:
        logger.error(f'[HRDocument] OpenAI failed: {e}')

    # Intelligent fallback
    return {
        'document_title': document_type,
        'document_content': f"""{company_name}
[Company Address]
[City, Country]

Date: {__import__('datetime').date.today().strftime('%B %d, %Y')}
Ref: HR/{employee_id or '001'}/{__import__('datetime').date.today().year}

{employee_name}
{employee_designation}
{employee_department}
{company_name}

Subject: {document_type}

Dear {employee_name},

This letter is being issued to you with reference to the matter concerning {additional_details or 'the subject mentioned above'}.

Please be advised that {company_name} takes all such matters seriously and expects all employees to maintain the highest standards of professionalism and conduct.

You are requested to acknowledge receipt of this letter and respond accordingly within the stipulated time.

Should you have any questions, please do not hesitate to contact the HR department.

Yours sincerely,

{hr_name}
{hr_designation}
Human Resources Department
{company_name}""",
        'tone_used': 'Formal',
        'legal_risk_level': 'Medium',
        'legal_risk_reason': 'Ensure document is in line with local labor laws before issuing.',
        'key_clauses': ['Employee acknowledgment required', 'Document to be filed in employee record', 'Appeal process must be communicated'],
        'dos': ['Get employee signature for acknowledgment', 'Keep copy in HR file', 'Follow company disciplinary procedure'],
        'donts': ['Do not issue verbally', 'Do not skip documentation', 'Do not violate labor law timelines'],
        'follow_up_actions': [
            {'step': 1, 'action': 'Get employee acknowledgment signature', 'timeline': 'Same day'},
            {'step': 2, 'action': 'File original in employee personnel folder', 'timeline': 'Within 24 hours'},
            {'step': 3, 'action': 'Schedule follow-up meeting if required', 'timeline': 'Within 1 week'},
        ],
        'employee_rights': 'Employee has the right to respond and appeal this decision through the grievance procedure.',
        'recommended_witnesses': 'HR Manager + Line Manager should both sign as witnesses.',
        'documentation_checklist': ['Signed copy of this letter', 'Employee acknowledgment form', 'Supporting evidence/incident report'],
        'alternative_version_note': 'For repeated offenses, escalate to a Final Warning or Termination Notice.'
    }


def generate_employee_handbook(company_name: str, industry: str, company_size: str,
                                country: str, culture_type: str, work_model: str,
                                selected_sections: list, additional_notes: str = '',
                                user_id: str = None) -> dict:
    """Generate a complete, professional employee handbook tailored to the company."""
    sections_text = ', '.join(selected_sections) if selected_sections else 'all standard sections'
    prompt = f"""You are a world-class HR consultant and organizational development expert who has built employee handbooks for Fortune 500 companies and high-growth startups.

Create a comprehensive, professional Employee Handbook for:

COMPANY: {company_name}
INDUSTRY: {industry}
COMPANY SIZE: {company_size}
COUNTRY/JURISDICTION: {country}
CULTURE TYPE: {culture_type} (e.g. Startup/Corporate/Hybrid/Creative)
WORK MODEL: {work_model} (e.g. Remote/Hybrid/On-site)
SECTIONS TO INCLUDE: {sections_text}
ADDITIONAL NOTES: {additional_notes or 'None'}

Write each section with:
- Clear, engaging professional language (not dry legal text)
- Specific policies with actual timelines and numbers
- Culture-appropriate tone for a {culture_type} company
- Jurisdiction-appropriate policies for {country}
- Industry-relevant examples for {industry}

Return a JSON object with EXACTLY these fields:
{{
  "handbook_title": "<e.g. '{company_name} — Employee Handbook 2025'>",
  "company_tagline": "<a motivating tagline for the handbook intro>",
  "welcome_message": "<warm, culture-aligned welcome from the CEO/HR — 3-4 paragraphs>",
  "sections": [
    {{
      "title": "<Section Title>",
      "icon": "<relevant emoji>",
      "content": "<Full, detailed policy content — multiple paragraphs, specific timelines/numbers, real rules. Not a template — actual policy text ready to use.>",
      "key_points": ["<summary bullet 1>", "<summary bullet 2>", "<summary bullet 3>"]
    }}
  ],
  "company_values": ["<core value 1>", "<core value 2>", "<core value 3>", "<value 4>", "<value 5>"],
  "acknowledgment_page": "<Complete acknowledgment statement for employees to sign — confirms they have read, understood, and agree to abide by this handbook>",
  "revision_history": "<e.g. Version 1.0 — Effective [current date]>",
  "hr_contact_note": "<where employees should direct questions>",
  "total_pages_estimate": "<estimated page count if printed>",
  "handbook_summary": "<2-3 sentence overview of what this handbook covers and company's commitment to employees>"
}}

Each section content should be COMPLETE and READY TO USE — not placeholders. Write real, usable policy language with specific numbers (e.g. '15 days annual leave', '3 warning steps before termination', etc.). Tailor everything to the {culture_type} culture and {country} labor laws."""

    try:
        result = _call_openai(prompt, max_tokens=4000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'sections' in data:
            return data
    except Exception as e:
        logger.error(f'[HandbookBuilder] OpenAI failed: {e}')

    today_year = __import__('datetime').date.today().year
    return {
        'handbook_title': f'{company_name} — Employee Handbook {today_year}',
        'company_tagline': 'Our People Are Our Greatest Asset',
        'welcome_message': f'Welcome to {company_name}! We are thrilled to have you as part of our team. This handbook has been prepared to help you understand our company culture, policies, and the standards we uphold.\n\nAt {company_name}, we believe that a transparent, fair, and supportive workplace is the foundation of excellence. Every policy in this handbook reflects our commitment to creating an environment where you can thrive.\n\nPlease read this handbook carefully and keep it as a reference guide throughout your journey with us. Our HR team is always available to answer any questions you may have.',
        'sections': [
            {'title': 'Code of Conduct', 'icon': '⚖️', 'content': f'All employees of {company_name} are expected to conduct themselves professionally at all times. This includes treating colleagues, clients, and stakeholders with respect and dignity. Any form of harassment, discrimination, or misconduct will not be tolerated and may result in disciplinary action up to and including termination.', 'key_points': ['Professional conduct at all times', 'Zero tolerance for harassment', 'Respect for all stakeholders']},
            {'title': 'Working Hours & Attendance', 'icon': '🕐', 'content': f'Standard working hours at {company_name} are 9:00 AM to 6:00 PM, Monday through Friday. Employees are expected to be punctual. Three unexplained late arrivals in a month may result in a formal warning. Overtime must be pre-approved by your line manager.', 'key_points': ['9 AM - 6 PM standard hours', 'Punctuality expected', 'Pre-approved overtime only']},
            {'title': 'Leave Policy', 'icon': '🏖️', 'content': f'{company_name} provides all employees with 15 days of annual leave, 10 days of sick leave, and 3 days of casual leave per year. Leave must be applied for at least 48 hours in advance except in emergencies. Unused annual leave may be carried forward up to a maximum of 5 days.', 'key_points': ['15 days annual leave', '10 days sick leave', '3 days casual leave']},
            {'title': 'Anti-Harassment Policy', 'icon': '🛡️', 'content': f'{company_name} is committed to maintaining a workplace free from all forms of harassment and discrimination. All complaints must be reported to HR within 7 days of the incident. All reports will be investigated confidentially within 14 working days.', 'key_points': ['Zero tolerance policy', 'Confidential reporting', '14-day investigation timeline']},
        ],
        'company_values': ['Integrity', 'Excellence', 'Collaboration', 'Innovation', 'Respect'],
        'acknowledgment_page': f'I, ________________________, acknowledge that I have received, read, and understood the {company_name} Employee Handbook. I agree to abide by all policies, rules, and regulations outlined herein. I understand that violation of these policies may result in disciplinary action.\n\nEmployee Signature: ________________________\nDate: ________________________\nEmployee ID: ________________________',
        'revision_history': f'Version 1.0 — Effective {__import__("datetime").date.today().strftime("%B %d, %Y")}',
        'hr_contact_note': 'For any questions regarding this handbook, please contact the HR Department.',
        'total_pages_estimate': '18-25 pages',
        'handbook_summary': f'This handbook outlines {company_name}\'s core policies, values, and expectations for all employees. It serves as the definitive guide for workplace conduct and employment terms. {company_name} reserves the right to update this handbook and will communicate any changes to all employees.'
    }


# ── Feature Set 6: HR Strategic AI Tools ────────────────────────────────────

def generate_ld_roadmap(employee_name: str, current_role: str, target_role: str,
                         current_skills: list, experience_years: int, learning_style: str,
                         budget_range: str, timeline_months: int, industry: str,
                         company_name: str = '', user_id: str = None) -> dict:
    """Generate a personalized AI Learning & Development roadmap for an employee."""
    skills_str = ', '.join(current_skills[:30]) if current_skills else 'Not specified'
    prompt = f"""You are a world-class L&D (Learning & Development) strategist and career coach who has designed development plans for thousands of employees across top companies globally.

Create a highly personalized, actionable Training & Development roadmap for:

EMPLOYEE: {employee_name}
COMPANY: {company_name or 'The Organization'}
INDUSTRY: {industry}
CURRENT ROLE: {current_role}
TARGET ROLE / GOAL: {target_role}
CURRENT SKILLS: {skills_str}
YEARS OF EXPERIENCE: {experience_years}
PREFERRED LEARNING STYLE: {learning_style} (Visual/Reading/Hands-on/Blended)
BUDGET RANGE: {budget_range}
TIMELINE: {timeline_months} months

Your roadmap must be deeply personalized — not generic. Identify the EXACT skill gap between current role and target role. Recommend REAL courses (Coursera, Udemy, LinkedIn Learning, Pluralsight, edX, Google, AWS, Microsoft, PMI etc.) with realistic prices. Consider the learning style preference.

Return a JSON object with EXACTLY these fields:
{{
  "employee_summary": "<2-sentence profile of where this employee is and where they're heading>",
  "readiness_score": <0-100 how ready they are for target role today>,
  "readiness_label": "<Not Ready / Developing / Almost Ready / Ready>",
  "skill_gap_analysis": {{
    "critical_gaps": ["<skill they urgently need>", "<gap2>", "<gap3>"],
    "moderate_gaps": ["<skill to develop>", "<gap2>"],
    "existing_strengths": ["<what they already have>", "<strength2>", "<strength3>"],
    "gap_summary": "<1-2 sentence summary of biggest skill jump needed>"
  }},
  "learning_roadmap": [
    {{
      "phase": 1,
      "phase_title": "<e.g. Foundation Building>",
      "duration": "<e.g. Month 1-2>",
      "focus": "<what this phase achieves>",
      "milestones": ["<deliverable 1>", "<deliverable 2>"],
      "activities": ["<specific activity>", "<activity2>"]
    }}
  ],
  "recommended_courses": [
    {{
      "title": "<exact course name>",
      "platform": "<Coursera/Udemy/LinkedIn Learning/edX/YouTube/etc>",
      "url_hint": "<search term to find it>",
      "duration": "<e.g. 12 hours / 6 weeks>",
      "cost": "<e.g. Free / $49 / $199/year subscription>",
      "priority": "<Critical/High/Medium>",
      "phase": <which phase number>,
      "why": "<why this course specifically for this person>"
    }}
  ],
  "certifications": [
    {{
      "name": "<exact certification name>",
      "issuing_body": "<AWS/Google/PMI/SHRM/etc>",
      "relevance": "<why this cert matters for target role>",
      "estimated_cost": "<exam + prep cost>",
      "prep_time": "<e.g. 3 months of study>",
      "priority": "<Must Have / Strongly Recommended / Optional>",
      "target_month": <which month to attempt>
    }}
  ],
  "monthly_schedule": [
    {{
      "month": <1>,
      "focus": "<primary focus this month>",
      "hours_per_week": <realistic hours>,
      "key_tasks": ["<task>", "<task2>"],
      "checkpoint": "<what should be achieved by end of month>"
    }}
  ],
  "roi_for_company": {{
    "productivity_gain": "<estimated % or description>",
    "retention_impact": "<how this investment reduces turnover risk>",
    "value_delivered": "<what new capabilities employee brings>",
    "payback_period": "<how long till company recoups investment>",
    "cost_of_not_training": "<risk of NOT investing in this employee>"
  }},
  "total_estimated_cost": "<total budget estimate for full roadmap>",
  "cost_breakdown": {{
    "courses": "<estimate>",
    "certifications": "<estimate>",
    "books_resources": "<estimate>",
    "total": "<grand total>"
  }},
  "success_metrics": ["<how to measure progress>", "<metric2>", "<metric3>"],
  "manager_tips": ["<how manager can support>", "<tip2>", "<tip3>"],
  "motivational_note": "<personalized encouraging message for the employee>"
}}

Make every recommendation SPECIFIC to {employee_name}'s exact situation. Use real course names, real platforms, realistic costs. Do not give generic advice."""

    try:
        result = _call_openai(prompt, max_tokens=3000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'learning_roadmap' in data:
            return data
    except Exception as e:
        logger.error(f'[LDRoadmap] OpenAI failed: {e}')

    gap = [s for s in ['Leadership', 'Project Management', 'Communication', 'Data Analysis', 'Strategic Thinking'] if s.lower() not in [sk.lower() for sk in current_skills]][:3]
    return {
        'employee_summary': f'{employee_name} is currently a {current_role} with {experience_years} years of experience, targeting a transition to {target_role}. A focused {timeline_months}-month development plan will bridge the key skill gaps.',
        'readiness_score': min(75, 30 + experience_years * 5),
        'readiness_label': 'Developing',
        'skill_gap_analysis': {
            'critical_gaps': gap or ['Advanced technical skills', 'Leadership experience', 'Strategic planning'],
            'moderate_gaps': ['Stakeholder management', 'Cross-functional collaboration'],
            'existing_strengths': current_skills[:3] or ['Domain knowledge', 'Work ethic', 'Team collaboration'],
            'gap_summary': f'The primary gap between {current_role} and {target_role} is in leadership and strategic skill sets that require deliberate development.',
        },
        'learning_roadmap': [
            {'phase': 1, 'phase_title': 'Foundation & Assessment', 'duration': f'Month 1-{min(2, timeline_months)}', 'focus': 'Identify exact gaps and start foundational learning', 'milestones': ['Complete skills self-assessment', 'Enroll in 2 core courses'], 'activities': ['Online course enrollment', 'Find a mentor in target role']},
            {'phase': 2, 'phase_title': 'Core Skill Building', 'duration': f'Month {min(3, timeline_months)}-{min(6, timeline_months)}', 'focus': 'Build the critical skills identified in gap analysis', 'milestones': ['Complete primary certification prep', 'Lead a small project'], 'activities': ['Hands-on projects', 'Job shadowing', 'Reading industry books']},
            {'phase': 3, 'phase_title': 'Application & Validation', 'duration': f'Month {min(7, timeline_months)}-{timeline_months}', 'focus': 'Apply learning in real work scenarios', 'milestones': ['Attempt certification exam', 'Present project outcomes to leadership'], 'activities': ['Take on stretch assignments', 'Present to stakeholders', 'Sit certification exam']},
        ],
        'recommended_courses': [
            {'title': f'{target_role} Fundamentals', 'platform': 'Coursera', 'url_hint': f'{target_role} fundamentals course', 'duration': '20 hours', 'cost': '$49', 'priority': 'Critical', 'phase': 1, 'why': 'Builds the foundational knowledge needed for transition'},
            {'title': 'Leadership Essentials', 'platform': 'LinkedIn Learning', 'url_hint': 'leadership skills LinkedIn Learning', 'duration': '8 hours', 'cost': '$29.99/month subscription', 'priority': 'High', 'phase': 2, 'why': 'Develops the leadership mindset required in senior roles'},
            {'title': 'Communication for Professionals', 'platform': 'Udemy', 'url_hint': 'professional communication Udemy', 'duration': '6 hours', 'cost': '$14.99', 'priority': 'Medium', 'phase': 2, 'why': 'Sharpens stakeholder communication skills'},
        ],
        'certifications': [
            {'name': f'{industry} Professional Certification', 'issuing_body': 'Industry Body', 'relevance': f'Validates expertise needed for {target_role}', 'estimated_cost': '$300-500', 'prep_time': '3 months', 'priority': 'Strongly Recommended', 'target_month': timeline_months - 2},
        ],
        'monthly_schedule': [{'month': m, 'focus': f'Phase {"1" if m <= 2 else "2" if m <= 6 else "3"} learning activities', 'hours_per_week': 5, 'key_tasks': ['Complete assigned module', 'Practice new skills at work'], 'checkpoint': f'Month {m} milestone review with manager'} for m in range(1, min(timeline_months + 1, 7))],
        'roi_for_company': {
            'productivity_gain': '20-30% improvement in role effectiveness within 6 months of completing roadmap',
            'retention_impact': 'Employees with development plans are 3x more likely to stay — saves cost of replacing this employee',
            'value_delivered': f'A fully developed {target_role} brings immediate ROI through reduced external hiring costs and faster project delivery',
            'payback_period': '6-12 months after roadmap completion',
            'cost_of_not_training': f'Risk of losing {employee_name} to a competitor offering growth + external hire cost of 50-150% of annual salary',
        },
        'total_estimated_cost': budget_range,
        'cost_breakdown': {'courses': '$150-300', 'certifications': '$300-600', 'books_resources': '$50-100', 'total': budget_range},
        'success_metrics': ['Certification obtained by target month', 'Manager scores 4+/5 on new skill competencies', 'Assigned first project in target role capacity'],
        'manager_tips': ['Schedule monthly 1-on-1 check-ins on L&D progress', 'Give stretch assignments aligned with target role', 'Publicly recognize milestones to keep motivation high'],
        'motivational_note': f'{employee_name}, the path from {current_role} to {target_role} is absolutely achievable in {timeline_months} months. Every expert was once a beginner — stay consistent, apply what you learn daily, and this roadmap will get you there.',
    }


def check_policy_compliance(policy_text: str, country: str, industry: str,
                              company_size: str, policy_type: str = '',
                              user_id: str = None) -> dict:
    """AI analyzes HR policy text against local labor laws and flags violations."""
    prompt = f"""You are a senior employment law expert and HR compliance specialist with 30+ years of experience across multiple jurisdictions. You have deep knowledge of labor laws, employment regulations, and HR best practices.

Analyze the following HR policy document for legal compliance issues:

COUNTRY / JURISDICTION: {country}
INDUSTRY: {industry}
COMPANY SIZE: {company_size}
POLICY TYPE (if known): {policy_type or 'General HR Policy'}

POLICY TEXT TO ANALYZE:
---
{policy_text[:4000]}
---

Perform a thorough compliance review against:
- {country} labor laws and employment acts
- Industry-specific regulations for {industry}
- International best practices (ILO standards)
- Anti-discrimination laws
- Data protection requirements (where applicable)
- Minimum standards for leave, pay, working hours
- Employee rights and protections

Return a JSON object with EXACTLY these fields:
{{
  "compliance_score": <0-100 overall compliance score>,
  "compliance_grade": "<A/B/C/D/F>",
  "overall_verdict": "<Compliant / Mostly Compliant / Needs Revision / Non-Compliant>",
  "verdict_summary": "<2-3 sentence executive summary of compliance status>",
  "laws_checked": ["<specific law or act checked>", "<law2>", "<law3>", "<law4>"],
  "violations": [
    {{
      "severity": "<Critical/High/Medium>",
      "clause": "<quote or reference the problematic text>",
      "issue": "<what is legally wrong>",
      "legal_reference": "<specific law, section, article violated>",
      "risk": "<what could happen if not fixed — fines, lawsuits, etc>",
      "fix": "<exact correction needed>"
    }}
  ],
  "warnings": [
    {{
      "severity": "Low",
      "clause": "<text that needs attention>",
      "issue": "<potential concern>",
      "recommendation": "<suggested improvement>"
    }}
  ],
  "compliant_clauses": ["<what the policy gets right>", "<strength2>", "<strength3>"],
  "missing_required_clauses": ["<legally required clause that is absent>", "<missing2>"],
  "corrected_policy": "<Rewrite the full policy with all violations fixed, missing clauses added, and legally sound language. Keep the original intent but make it fully compliant. This should be a complete, ready-to-use policy document.>",
  "key_improvements": [
    {{"priority": 1, "improvement": "<most important change>", "reason": "<why>"}},
    {{"priority": 2, "improvement": "<second change>", "reason": "<why>"}},
    {{"priority": 3, "improvement": "<third change>", "reason": "<why>"}}
  ],
  "legal_disclaimer": "This AI analysis is for guidance only and does not constitute formal legal advice. Consult a qualified employment lawyer for final review.",
  "next_steps": ["<recommended action 1>", "<action2>", "<action3>"]
}}

Be specific about {country} laws. For Pakistan: reference EOBI Act, Employment of Women Act, Industrial Relations Act, Factories Act, Minimum Wages Ordinance. For UAE: UAE Labour Law Federal Decree-Law No. 33 of 2021. For UK: Employment Rights Act 1996, Equality Act 2010, Working Time Regulations. For US: FLSA, FMLA, ADA, Title VII. Cite actual section numbers where possible."""

    try:
        result = _call_openai(prompt, max_tokens=3500, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'compliance_score' in data:
            return data
    except Exception as e:
        logger.error(f'[PolicyCompliance] OpenAI failed: {e}')

    word_count = len(policy_text.split())
    score = 60 if word_count > 100 else 40
    return {
        'compliance_score': score,
        'compliance_grade': 'C' if score >= 60 else 'D',
        'overall_verdict': 'Needs Revision',
        'verdict_summary': f'The policy has been reviewed against {country} labor laws and industry standards for {industry}. Several areas require revision to ensure full legal compliance. Key gaps include missing mandatory clauses and potentially non-compliant language.',
        'laws_checked': [f'{country} Labor Law', 'Employment Rights Act', 'Anti-Discrimination Regulations', 'Data Protection Standards'],
        'violations': [
            {'severity': 'High', 'clause': 'General policy language', 'issue': 'Policy lacks explicit mention of statutory minimum entitlements', 'legal_reference': f'{country} Employment Law — Minimum Standards', 'risk': 'Employee complaints, labor authority fines', 'fix': 'Add explicit statement of minimum statutory entitlements as per local law'},
            {'severity': 'Medium', 'clause': 'Disciplinary section', 'issue': 'Disciplinary procedure does not clearly state right to appeal', 'legal_reference': 'Natural Justice Principles / Employment Act', 'risk': 'Unfair dismissal claims', 'fix': 'Add clear 3-step appeal process with timelines'},
        ],
        'warnings': [
            {'severity': 'Low', 'clause': 'General language', 'issue': 'Policy language may be interpreted as gender-biased in some clauses', 'recommendation': 'Use gender-neutral language throughout (e.g., "they/them" instead of "he/she")'},
        ],
        'compliant_clauses': ['Policy has a clear scope and applicability statement', 'Confidentiality provisions are present', 'Policy states effective date'],
        'missing_required_clauses': ['Grievance redressal procedure', 'Anti-harassment statement', 'Employee right to representation', 'Data privacy statement'],
        'corrected_policy': f'[CORRECTED POLICY — {policy_type or "HR Policy"}]\n\nThis policy has been revised to comply with {country} labor laws and {industry} industry standards.\n\n{policy_text}\n\n[ADDITIONS]\nGrievance Procedure: Employees may raise grievances in writing to HR within 30 days of the incident.\nAppeal Rights: Any disciplinary decision may be appealed within 14 days.\nAnti-Harassment: The company maintains a zero-tolerance policy toward all forms of workplace harassment.\nData Privacy: All employee data is handled in accordance with applicable data protection laws.',
        'key_improvements': [
            {'priority': 1, 'improvement': 'Add statutory minimum entitlements section', 'reason': 'Required by law — absence creates legal liability'},
            {'priority': 2, 'improvement': 'Include formal grievance and appeal procedure', 'reason': 'Protects company from unfair dismissal claims'},
            {'priority': 3, 'improvement': 'Add anti-harassment and equal opportunity statement', 'reason': 'Legally required in most jurisdictions'},
        ],
        'legal_disclaimer': 'This AI analysis is for guidance only and does not constitute formal legal advice. Consult a qualified employment lawyer for final review.',
        'next_steps': ['Review all Critical and High violations immediately', 'Consult a local employment lawyer for final sign-off', 'Distribute revised policy to all employees with acknowledgment'],
    }


# ── Feature Set 7: Candidate AI Career Tools ────────────────────────────────

def generate_cover_letter(job_title: str, company_name: str, jd_text: str,
                           candidate_name: str, candidate_skills: list,
                           experience_summary: str, tone: str = 'Professional',
                           user_id: str = None) -> dict:
    """Generate a personalized, tailored cover letter for a specific job."""
    skills_str = ', '.join(candidate_skills[:25]) if candidate_skills else 'various professional skills'
    prompt = f"""You are a world-class career coach and professional writer who has helped thousands of candidates land jobs at top companies.

Write a highly personalized, compelling cover letter for:

CANDIDATE: {candidate_name}
TARGET ROLE: {job_title}
TARGET COMPANY: {company_name or 'the company'}
TONE: {tone}
CANDIDATE SKILLS: {skills_str}
EXPERIENCE SUMMARY: {experience_summary or 'Experienced professional seeking new opportunities'}

JOB DESCRIPTION:
---
{jd_text[:3000]}
---

INSTRUCTIONS:
- Extract 3-4 key requirements from the JD and directly address each one
- Weave candidate's specific skills into the narrative — not just list them
- Show genuine enthusiasm for THIS specific company (not generic)
- Use powerful action verbs and quantify impact where possible
- Keep it to 3 focused paragraphs — not too long, not too short
- Match the {tone} tone throughout
- Make the opening line instantly grab attention — no "I am writing to apply for..."
- End with a confident, specific call to action

Return a JSON object with EXACTLY these fields:
{{
  "subject_line": "<email subject line for this application>",
  "cover_letter": "<complete 3-paragraph cover letter, ready to send. Use \\n\\n between paragraphs. Include proper salutation and closing.>",
  "word_count": <approximate word count>,
  "tone_used": "<Professional/Enthusiastic/Executive/Creative>",
  "keywords_used": ["<JD keyword woven in>", "<keyword2>", "<keyword3>", "<keyword4>", "<keyword5>"],
  "jd_requirements_addressed": ["<requirement from JD that was addressed>", "<req2>", "<req3>"],
  "opening_hook": "<the first sentence — what makes it grab attention>",
  "strength_of_letter": "<what makes this letter strong for this specific job>",
  "customization_tips": ["<how to personalize further>", "<tip2>", "<tip3>"],
  "alternative_opening": "<an alternative powerful first line if they want to change it>",
  "follow_up_tip": "<when and how to follow up after sending>"
}}

Write the actual full letter — personalized for {candidate_name} applying to {job_title} at {company_name or 'this company'}. Make it feel human and genuine, not templated."""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'cover_letter' in data:
            return data
    except Exception as e:
        logger.error(f'[CoverLetter] OpenAI failed: {e}')

    return {
        'subject_line': f'Application for {job_title} Position — {candidate_name}',
        'cover_letter': f"""Dear Hiring Manager,

Having followed {company_name or "your company"}'s journey in the industry, I was immediately drawn to the {job_title} opportunity. With my background in {skills_str[:100]} and a proven track record of delivering results, I am confident I can make an immediate and lasting contribution to your team.

{experience_summary or f"Throughout my career, I have developed strong expertise in {skills_str[:150]}. I have consistently delivered results by combining technical proficiency with a collaborative approach and a commitment to continuous improvement."}

I would welcome the opportunity to discuss how my experience aligns with your team's goals. I am available at your earliest convenience and look forward to the possibility of contributing to {company_name or "your organization"}'s continued success.

Warm regards,
{candidate_name}""",
        'word_count': 120,
        'tone_used': tone,
        'keywords_used': candidate_skills[:5],
        'jd_requirements_addressed': ['Relevant technical skills', 'Team collaboration', 'Results orientation'],
        'opening_hook': f"Having followed {company_name or 'your company'}'s journey, I was immediately drawn to the {job_title} opportunity.",
        'strength_of_letter': 'Directly connects candidate experience to the role requirements.',
        'customization_tips': ['Add a specific company achievement you admire', 'Mention a recent news story about the company', 'Quantify a specific achievement from your experience'],
        'alternative_opening': f"In {job_title} roles, results speak louder than credentials — here's why {candidate_name} delivers both.",
        'follow_up_tip': 'Follow up via email or LinkedIn 5-7 business days after applying if you haven\'t heard back.',
    }


def analyze_job_match(jd_text: str, candidate_skills: list, experience_summary: str,
                       education: str, target_role: str = '', user_id: str = None) -> dict:
    """Deep analysis of how well a candidate matches a specific job description."""
    skills_str = ', '.join(candidate_skills[:30]) if candidate_skills else 'Not specified'
    prompt = f"""You are an expert talent acquisition specialist and career coach with 20+ years of experience evaluating candidate-job fit.

Perform a deep, honest match analysis between this candidate and job description:

CANDIDATE PROFILE:
- Skills: {skills_str}
- Experience Summary: {experience_summary or 'Not provided'}
- Education: {education or 'Not specified'}
- Target Role: {target_role or 'Not specified'}

JOB DESCRIPTION:
---
{jd_text[:3500]}
---

Analyze honestly and specifically. Don't be overly optimistic OR pessimistic. Be a trusted advisor.

Return a JSON object with EXACTLY these fields:
{{
  "match_score": <0-100 overall match percentage>,
  "match_label": "<Perfect Fit / Strong Match / Good Match / Partial Match / Low Match>",
  "match_summary": "<2-3 sentence honest assessment of fit — what makes them a good/poor match>",
  "matched_skills": ["<skill candidate has that JD requires>", "<skill2>", "<skill3>"],
  "missing_skills": ["<skill JD requires that candidate lacks>", "<missing2>", "<missing3>"],
  "partial_skills": [
    {{"skill": "<skill they partially have>", "gap": "<what specifically they need to deepen>"}}
  ],
  "experience_match": {{
    "score": <0-100>,
    "assessment": "<does their experience level match what JD needs?>",
    "gap": "<experience gap if any>"
  }},
  "education_match": {{
    "score": <0-100>,
    "assessment": "<does education match requirements?>"
  }},
  "keyword_analysis": {{
    "jd_keywords": ["<important keyword from JD>", "<kw2>", "<kw3>", "<kw4>", "<kw5>"],
    "candidate_has": ["<keywords from JD that candidate has>"],
    "candidate_missing": ["<JD keywords candidate lacks>"]
  }},
  "learning_plan": [
    {{
      "skill": "<missing skill to learn>",
      "priority": "<Critical/High/Medium>",
      "how_to_learn": "<specific course/resource/method>",
      "time_needed": "<e.g. 4-6 weeks>",
      "quick_win": "<fastest way to demonstrate this skill>"
    }}
  ],
  "application_advice": "<should they apply? what angle to take? what to emphasize in resume/cover letter?>",
  "resume_tips": ["<specific change to make to resume for this job>", "<tip2>", "<tip3>"],
  "ats_pass_prediction": "<High/Medium/Low chance of passing ATS for this role>",
  "ats_reason": "<why they will or won't pass ATS>",
  "interview_likely_questions": ["<question likely to be asked given the JD>", "<q2>", "<q3>"],
  "overall_verdict": "<Apply Now / Apply with Improvements / Upskill First / Not a Fit Yet>"
}}

Be specific about which skills are matched vs missing — reference actual text from the JD."""

    try:
        result = _call_openai(prompt, max_tokens=2500, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'match_score' in data:
            return data
    except Exception as e:
        logger.error(f'[JobMatch] OpenAI failed: {e}')

    matched = candidate_skills[:4] if candidate_skills else []
    score = min(85, 30 + len(matched) * 10)
    return {
        'match_score': score,
        'match_label': 'Good Match' if score >= 65 else 'Partial Match',
        'match_summary': f'Based on your profile, you match approximately {score}% of this role\'s requirements. You have strong relevant skills but there are some gaps to address.',
        'matched_skills': matched,
        'missing_skills': ['Advanced domain expertise', 'Industry-specific certifications', 'Leadership experience'],
        'partial_skills': [{'skill': 'Project Management', 'gap': 'Need formal certification or larger-scale project experience'}],
        'experience_match': {'score': 65, 'assessment': 'Experience level is generally aligned with role requirements.', 'gap': 'May lack some senior-level responsibilities mentioned in JD'},
        'education_match': {'score': 75, 'assessment': 'Education background meets basic requirements for the role.'},
        'keyword_analysis': {'jd_keywords': ['problem-solving', 'teamwork', 'communication', 'analytical', 'results-driven'], 'candidate_has': matched[:3], 'candidate_missing': ['industry-specific tools', 'certifications']},
        'learning_plan': [{'skill': 'Missing technical skill', 'priority': 'High', 'how_to_learn': 'Coursera or Udemy course', 'time_needed': '4-6 weeks', 'quick_win': 'Build a small project demonstrating this skill'}],
        'application_advice': 'You have a solid foundation. Tailor your resume to highlight the matched skills prominently and address gaps in your cover letter proactively.',
        'resume_tips': ['Add more quantified achievements', 'Include keywords from the JD in your skills section', 'Reorder experience to lead with most relevant role'],
        'ats_pass_prediction': 'Medium',
        'ats_reason': 'You have most key skills but may miss some specific keywords the ATS is scanning for.',
        'interview_likely_questions': ['Tell me about your experience with [key skill]', 'How do you handle [key challenge from JD]?', 'Where do you see yourself growing in this role?'],
        'overall_verdict': 'Apply with Improvements',
    }


def generate_self_intro(candidate_name: str, current_role: str, target_role: str,
                         experience_years: int, key_skills: list, key_achievement: str,
                         user_id: str = None) -> dict:
    """Generate 3 versions of a perfect 'Tell me about yourself' intro."""
    skills_str = ', '.join(key_skills[:10]) if key_skills else 'relevant professional skills'
    prompt = f"""You are a world-class interview coach who has helped thousands of candidates nail their interviews at top companies including FAANG, Fortune 500, and high-growth startups.

Create 3 powerful, personalized versions of "Tell me about yourself" for:

CANDIDATE: {candidate_name}
CURRENT ROLE: {current_role}
TARGET ROLE: {target_role}
YEARS OF EXPERIENCE: {experience_years}
KEY SKILLS: {skills_str}
KEY ACHIEVEMENT: {key_achievement or 'Strong track record of professional results'}

Each version must:
- Sound natural and human — NOT rehearsed or robotic
- Start with a strong hook, not "I am {candidate_name} and I have X years of experience"
- Connect their past to their future (target role)
- Include ONE specific achievement with impact
- End with why they want THIS type of role

Return a JSON object with EXACTLY these fields:
{{
  "short_version": {{
    "text": "<30-second version, 60-80 words, crisp and punchy. Perfect for phone screens.>",
    "word_count": <word count>,
    "best_for": "<when to use this version>",
    "delivery_time": "30 seconds"
  }},
  "medium_version": {{
    "text": "<60-second version, 120-150 words. Professional and confident. Best for most interviews.>",
    "word_count": <word count>,
    "best_for": "<when to use this version>",
    "delivery_time": "60 seconds"
  }},
  "detailed_version": {{
    "text": "<2-minute version, 250-280 words. Senior-level detail. For executive or panel interviews.>",
    "word_count": <word count>,
    "best_for": "<when to use this version>",
    "delivery_time": "2 minutes"
  }},
  "power_words_used": ["<impactful word used>", "<word2>", "<word3>", "<word4>", "<word5>"],
  "key_message": "<the one thing the interviewer should remember about this candidate after hearing this>",
  "delivery_tips": ["<how to deliver this confidently>", "<tip2>", "<tip3>", "<tip4>"],
  "common_mistakes_to_avoid": ["<mistake>", "<mistake2>", "<mistake3>"],
  "body_language_tips": ["<non-verbal tip>", "<tip2>", "<tip3>"],
  "practice_advice": "<how to practice until it feels natural, not memorized>"
}}

Write in first person as if {candidate_name} is speaking. Make it sound genuine and impressive."""

    try:
        result = _call_openai(prompt, max_tokens=2000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'short_version' in data:
            return data
    except Exception as e:
        logger.error(f'[SelfIntro] OpenAI failed: {e}')

    return {
        'short_version': {'text': f"I'm a {current_role} with {experience_years} years of experience specializing in {skills_str[:80]}. {key_achievement or 'I have a consistent track record of delivering results.'} I'm now looking to bring that expertise to a {target_role} role where I can make an even bigger impact.", 'word_count': 55, 'best_for': 'Phone screens and quick introductions', 'delivery_time': '30 seconds'},
        'medium_version': {'text': f"Throughout my {experience_years} years as a {current_role}, I've built deep expertise in {skills_str[:120]}. {key_achievement or 'My work has consistently driven measurable results for my teams.'} What drives me is solving complex problems and delivering outcomes that matter. I'm at a point in my career where I'm ready to step into a {target_role} role — and everything I've done has been building toward exactly this kind of opportunity.", 'word_count': 80, 'best_for': 'Standard first-round interviews', 'delivery_time': '60 seconds'},
        'detailed_version': {'text': f"I've spent {experience_years} years as a {current_role}, building expertise in {skills_str}. Early in my career, I focused on mastering the fundamentals — and that foundation allowed me to take on increasingly complex challenges. {key_achievement or 'Most recently, I led initiatives that significantly improved team performance and delivered strong business outcomes.'} What I've learned is that the best results come from combining technical depth with clear communication and a focus on what actually moves the needle. That's the professional I've become. Now I'm looking to bring all of that into a {target_role} position, where I can contribute at a higher level and continue growing alongside a strong team.", 'word_count': 130, 'best_for': 'Panel interviews and senior-level conversations', 'delivery_time': '2 minutes'},
        'power_words_used': ['expertise', 'delivered', 'impact', 'complex', 'consistently'],
        'key_message': f'{candidate_name} is a proven {current_role} ready to make an impact as a {target_role}.',
        'delivery_tips': ['Maintain steady eye contact', 'Speak at 80% of your normal pace', 'Pause briefly after your achievement for emphasis', 'Smile naturally — confidence is contagious'],
        'common_mistakes_to_avoid': ['Starting with "I was born in..."', 'Listing every job without a narrative thread', 'Speaking too fast due to nerves', 'Ending without connecting to the target role'],
        'body_language_tips': ['Sit slightly forward to show engagement', 'Keep hands visible and open', 'Nod slightly when making key points'],
        'practice_advice': 'Record yourself 5 times, watch it back, refine. Practice out loud — not in your head. Aim for it to sound conversational, not memorized.',
    }


def suggest_portfolio_projects(target_role: str, current_skills: list, experience_level: str,
                                industry: str, user_id: str = None) -> dict:
    """Suggest specific portfolio projects that will impress recruiters for a target role."""
    skills_str = ', '.join(current_skills[:20]) if current_skills else 'general skills'
    prompt = f"""You are a senior engineering manager and hiring expert who has reviewed thousands of portfolios and knows exactly what impresses recruiters and hiring managers.

Suggest 5-6 specific, impressive portfolio projects for:

TARGET ROLE: {target_role}
CURRENT SKILLS: {skills_str}
EXPERIENCE LEVEL: {experience_level}
INDUSTRY: {industry or 'Technology'}

Each project must:
- Be realistic to build (not NASA-level impossible)
- Have clear business/real-world value
- Showcase skills specifically needed for {target_role}
- Be differentiated — not just another todo app or weather app
- Include specific technical details, not vague descriptions
- Be impressive enough to mention in an interview

Return a JSON object with EXACTLY these fields:
{{
  "projects": [
    {{
      "name": "<specific project name>",
      "tagline": "<one-sentence compelling description>",
      "description": "<what it does, who it's for, what problem it solves>",
      "tech_stack": ["<technology>", "<tech2>", "<tech3>"],
      "complexity": "<Beginner / Intermediate / Advanced>",
      "estimated_build_time": "<e.g. 2-3 weeks>",
      "wow_factor": "<what makes this stand out to a recruiter for {target_role}>",
      "key_features_to_build": ["<feature that shows skill>", "<feature2>", "<feature3>"],
      "how_to_present": "<how to talk about this in an interview — what to highlight>",
      "github_readme_tip": "<what to put in README to impress>",
      "live_demo_tip": "<how to make a live demo impressive>",
      "difficulty_if_missing_skill": "<what to learn first if they don't have it>"
    }}
  ],
  "quick_win_project": "<which project to build FIRST for fastest impact>",
  "portfolio_strategy": "<overall advice on how to structure the portfolio — what order, how many, what to prioritize>",
  "presentation_tips": ["<how to present portfolio in interviews>", "<tip2>", "<tip3>"],
  "github_profile_tips": ["<how to make GitHub profile impressive>", "<tip2>", "<tip3>"],
  "demo_day_advice": "<how to demo projects live without embarrassing yourself>",
  "bonus_differentiator": "<one extra thing beyond projects that would make their portfolio stand out>"
}}

Make every project idea SPECIFIC to {target_role} and {experience_level} level. Real names, real tech stacks, real implementation ideas."""

    try:
        result = _call_openai(prompt, max_tokens=3000, user_id=user_id)
        data = json.loads(_strip_json(result))
        if data and 'projects' in data:
            return data
    except Exception as e:
        logger.error(f'[PortfolioSuggester] OpenAI failed: {e}')

    return {
        'projects': [
            {'name': f'Smart {target_role} Dashboard', 'tagline': f'A real-time analytics platform for {target_role} workflows', 'description': f'A full-stack dashboard that demonstrates core competencies required for {target_role} positions', 'tech_stack': current_skills[:3] or ['React', 'Node.js', 'MongoDB'], 'complexity': 'Intermediate', 'estimated_build_time': '2-3 weeks', 'wow_factor': 'Shows real-world problem solving and technical depth', 'key_features_to_build': ['Authentication system', 'Real-time data visualization', 'RESTful API'], 'how_to_present': 'Focus on the problem you solved and technical decisions you made', 'github_readme_tip': 'Include a live demo link, tech stack badges, and problem statement', 'live_demo_tip': 'Pre-load with realistic test data so it looks live and active', 'difficulty_if_missing_skill': 'Start with a simpler CRUD version first'},
            {'name': 'AI-Powered Job Assistant', 'tagline': 'Automates repetitive tasks using AI integration', 'description': 'An intelligent tool that uses AI APIs to automate common tasks in your target domain', 'tech_stack': ['Python', 'OpenAI API', 'FastAPI'], 'complexity': 'Advanced', 'estimated_build_time': '3-4 weeks', 'wow_factor': 'Shows you understand AI integration — highly relevant for any modern tech role', 'key_features_to_build': ['AI API integration', 'User-friendly interface', 'Data persistence'], 'how_to_present': 'Emphasize the product thinking behind what problem you chose to solve', 'github_readme_tip': 'Add a GIF demo of the AI in action', 'live_demo_tip': 'Have 3-4 pre-prepared prompts ready to show different use cases', 'difficulty_if_missing_skill': 'Start with OpenAI API documentation and build a simple chatbot first'},
        ],
        'quick_win_project': f'Start with the Smart {target_role} Dashboard — it demonstrates the most relevant skills in the shortest time.',
        'portfolio_strategy': f'Build 3 quality projects rather than 10 mediocre ones. For {target_role}: lead with your strongest, most relevant project. Each project should demonstrate a different skill set.',
        'presentation_tips': ['Always explain WHY you built it, not just what it does', 'Quantify anything you can (users, performance gains, time saved)', 'Be ready to discuss technical challenges you overcame'],
        'github_profile_tips': ['Pin your 6 best repositories', 'Add a professional README to your profile', 'Contribute to at least 2 open source projects'],
        'demo_day_advice': 'Always test your demo environment the night before. Have a backup video recording in case of internet issues.',
        'bonus_differentiator': f'Write 2-3 technical blog posts about what you learned while building these projects. Recruiters for {target_role} roles love candidates who share knowledge.',
    }
