"""
Rate limiting for AI API calls to prevent abuse and cost overruns
"""
from datetime import datetime, timedelta
from collections import defaultdict
import threading


class AIRateLimiter:
    """
    Simple in-memory rate limiter for AI API calls
    For production, use Redis-based rate limiting
    """
    
    def __init__(self):
        self.calls = defaultdict(list)
        self.lock = threading.Lock()
    
    def check_limit(self, user_id, limit=20, window_minutes=60):
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User identifier
            limit: Maximum calls allowed
            window_minutes: Time window in minutes
        
        Returns:
            tuple: (allowed: bool, remaining: int, reset_time: datetime)
        """
        with self.lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=window_minutes)
            
            # Clean old entries
            self.calls[user_id] = [
                call_time for call_time in self.calls[user_id]
                if call_time > cutoff
            ]
            
            current_count = len(self.calls[user_id])
            remaining = max(0, limit - current_count)
            
            if current_count >= limit:
                # Find when oldest call will expire
                oldest = min(self.calls[user_id])
                reset_time = oldest + timedelta(minutes=window_minutes)
                return False, 0, reset_time
            
            # Record this call
            self.calls[user_id].append(now)
            reset_time = now + timedelta(minutes=window_minutes)
            
            return True, remaining - 1, reset_time
    
    def reset_user(self, user_id):
        """Reset rate limit for a user (admin function)"""
        with self.lock:
            if user_id in self.calls:
                del self.calls[user_id]


# Global rate limiter instance
ai_rate_limiter = AIRateLimiter()
