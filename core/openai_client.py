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
    prompt = (
        'You are evaluating a ' + role + ' interview answer for a ' + question_type + ' question.\n\n'
        'QUESTION: ' + question + '\n\n'
        'CANDIDATE ANSWER: ' + str(answer)[:1000] + '\n\n'
        'Score and give actionable feedback. Be honest but constructive.\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "score": 1-10,\n'
        '  "grade": "Excellent|Good|Average|Below Average|Poor",\n'
        '  "feedback": "Detailed 2-3 sentence personalized feedback",\n'
        '  "strengths": ["strength1", "strength2"],\n'
        '  "improvements": ["specific improvement1", "specific improvement2"],\n'
        '  "better_answer_hint": "One-line hint for a stronger answer",\n'
        '  "keywords_used": ["keyword found in answer"],\n'
        '  "keywords_missed": ["important keyword missing"]\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Mock answer evaluation failed: {e}')
        return {
            'score': 6,
            'grade': 'Average',
            'feedback': 'Your answer addressed the question but could be more specific. Try using concrete examples.',
            'strengths': ['Attempted the question', 'Relevant topic coverage'],
            'improvements': ['Add specific examples with measurable outcomes', 'Use STAR format: Situation, Task, Action, Result'],
            'better_answer_hint': 'Ground your answer in a real situation from your experience.',
            'keywords_used': [],
            'keywords_missed': ['specific example', 'measurable result']
        }


def generate_mock_interview_report(role: str, level: str, history: list, user_id: str = None) -> dict:
    """Generate a comprehensive final performance report after completing a mock interview."""
    qa_summary = []
    total_score = 0
    for i, item in enumerate(history, 1):
        score = item.get('score', 5)
        total_score += score
        qa_summary.append({
            'q': i,
            'question_type': item.get('question_type', 'behavioral'),
            'score': score,
            'grade': item.get('grade', 'Average')
        })
    avg_score = total_score / max(len(history), 1)

    prompt = (
        'Generate a comprehensive final mock interview report.\n\n'
        'Role: ' + role + '\n'
        'Level: ' + level + '\n'
        'Questions Attempted: ' + str(len(history)) + '\n'
        'Average Score: ' + str(round(avg_score, 1)) + '/10\n'
        'Per-question breakdown:\n' + json.dumps(qa_summary, indent=2) + '\n\n'
        'Return ONLY valid JSON:\n'
        '{\n'
        '  "overall_score": 0-100,\n'
        '  "performance_grade": "Excellent|Good|Average|Needs Improvement|Poor",\n'
        '  "interview_summary": "3-4 sentence holistic assessment of performance",\n'
        '  "top_strengths": ["strength1", "strength2", "strength3"],\n'
        '  "critical_improvements": ["area1", "area2", "area3"],\n'
        '  "recommended_resources": ["resource or action 1", "resource or action 2"],\n'
        '  "readiness_for_real_interview": "Ready|Almost Ready|Needs More Practice|Not Ready",\n'
        '  "next_steps": ["actionable step1", "step2", "step3"],\n'
        '  "motivational_note": "Short encouraging closing message for the candidate"\n'
        '}'
    )
    try:
        return json.loads(_strip_json(_call(prompt, user_id=user_id)))
    except Exception as e:
        logger.warning(f'[GPT] Mock interview report generation failed: {e}')
        overall = min(int(avg_score * 10), 100)
        return {
            'overall_score': overall,
            'performance_grade': 'Good' if overall >= 70 else 'Average' if overall >= 50 else 'Needs Improvement',
            'interview_summary': f'You completed a {role} mock interview scoring {overall}/100. Keep practicing to build confidence.',
            'top_strengths': ['Completed the full interview', 'Demonstrated engagement', 'Showed relevant knowledge'],
            'critical_improvements': ['Use more specific examples', 'Practice concise answers', 'Research the role deeper'],
            'recommended_resources': ['Practice STAR method daily', 'Record and review your answers', 'Study role-specific topics'],
            'readiness_for_real_interview': 'Almost Ready' if overall >= 65 else 'Needs More Practice',
            'next_steps': ['Practice one mock interview daily', 'Record yourself and review', 'Research company-specific questions'],
            'motivational_note': 'Every practice session makes you stronger. You are on the right path!'
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
