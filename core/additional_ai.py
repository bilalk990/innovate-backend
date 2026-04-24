"""
Additional AI Features - 15 Missing Features Implementation
All functions use OpenAI GPT-4o-mini for real AI analysis
"""
import json
import logging
from openai import OpenAI
from django.conf import settings
from core.rate_limiter import ai_rate_limiter

logger = logging.getLogger('innovaite')
client = OpenAI(api_key=settings.OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"

def _call_ai(prompt: str, user_id: str = None) -> str:
    """Make OpenAI API call with rate limiting."""
    if user_id:
        allowed, remaining, reset_time = ai_rate_limiter.check_limit(user_id, limit=20, window_minutes=60)
        if not allowed:
            raise Exception(f'AI rate limit exceeded. Try again after {reset_time.strftime("%H:%M:%S")}')
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert HR and recruitment AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f'[AI] OpenAI call failed: {e}')
        raise Exception(f'AI_ERROR: {str(e)}')

def _strip_json(text: str) -> str:
    """Strip markdown from AI responses."""
    text = text.strip()
    if text.startswith('```'):
        text = text[3:]
        if text.lower().startswith('json'):
            text = text[4:]
        if text.endswith('```'):
            text = text[:-3]
    return text.strip()
