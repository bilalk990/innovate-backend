"""
Unit tests for authentication system
"""
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from django.conf import settings
import jwt
import bcrypt


class TestAuthentication(unittest.TestCase):
    """Test authentication flows"""
    
    def test_jwt_token_generation(self):
        """Test JWT token generation"""
        payload = {
            'user_id': 'test123',
            'email': 'test@example.com',
            'role': 'candidate',
            'token_version': 0,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow(),
        }
        
        token = jwt.encode(payload, 'test-secret', algorithm='HS256')
        self.assertIsNotNone(token)
        
        # Decode and verify
        decoded = jwt.decode(token, 'test-secret', algorithms=['HS256'])
        self.assertEqual(decoded['user_id'], 'test123')
        self.assertEqual(decoded['role'], 'candidate')
    
    def test_password_hashing(self):
        """Test bcrypt password hashing"""
        password = 'TestPassword123!'
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Verify correct password
        self.assertTrue(bcrypt.checkpw(password.encode(), hashed.encode()))
        
        # Verify incorrect password
        self.assertFalse(bcrypt.checkpw('WrongPassword'.encode(), hashed.encode()))
    
    def test_token_expiry(self):
        """Test expired token detection"""
        payload = {
            'user_id': 'test123',
            'exp': datetime.utcnow() - timedelta(hours=1),  # Expired 1 hour ago
            'iat': datetime.utcnow() - timedelta(hours=25),
        }
        
        token = jwt.encode(payload, 'test-secret', algorithm='HS256')
        
        with self.assertRaises(jwt.ExpiredSignatureError):
            jwt.decode(token, 'test-secret', algorithms=['HS256'])
    
    def test_token_version_invalidation(self):
        """Test token invalidation via version bump"""
        # User changes password, token_version increments
        old_token_payload = {
            'user_id': 'test123',
            'token_version': 0,
            'exp': datetime.utcnow() + timedelta(hours=24),
        }
        
        new_token_payload = {
            'user_id': 'test123',
            'token_version': 1,  # Incremented
            'exp': datetime.utcnow() + timedelta(hours=24),
        }
        
        old_token = jwt.encode(old_token_payload, 'test-secret', algorithm='HS256')
        new_token = jwt.encode(new_token_payload, 'test-secret', algorithm='HS256')
        
        old_decoded = jwt.decode(old_token, 'test-secret', algorithms=['HS256'])
        new_decoded = jwt.decode(new_token, 'test-secret', algorithms=['HS256'])
        
        # Old token has version 0, new has version 1
        self.assertEqual(old_decoded['token_version'], 0)
        self.assertEqual(new_decoded['token_version'], 1)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def test_ai_rate_limiter(self):
        """Test AI API rate limiting"""
        from core.rate_limiter import AIRateLimiter
        
        limiter = AIRateLimiter()
        user_id = 'test_user'
        
        # Should allow first 20 calls
        for i in range(20):
            allowed, remaining, _ = limiter.check_limit(user_id, limit=20, window_minutes=60)
            self.assertTrue(allowed, f'Call {i+1} should be allowed')
        
        # 21st call should be blocked
        allowed, remaining, reset_time = limiter.check_limit(user_id, limit=20, window_minutes=60)
        self.assertFalse(allowed, '21st call should be blocked')
        self.assertEqual(remaining, 0)
        self.assertIsNotNone(reset_time)


class TestXSSSanitization(unittest.TestCase):
    """Test XSS sanitization"""
    
    def test_script_tag_removal(self):
        """Test that script tags are removed"""
        import bleach
        
        malicious_input = '<script>alert("XSS")</script>Hello'
        sanitized = bleach.clean(malicious_input, tags=[], attributes={}, strip=True)
        
        self.assertNotIn('<script>', sanitized)
        self.assertNotIn('</script>', sanitized)
        self.assertEqual(sanitized, 'Hello')
    
    def test_html_tag_removal(self):
        """Test that HTML tags are removed"""
        import bleach
        
        html_input = '<div><b>Bold</b> text</div>'
        sanitized = bleach.clean(html_input, tags=[], attributes={}, strip=True)
        
        self.assertNotIn('<div>', sanitized)
        self.assertNotIn('<b>', sanitized)
        self.assertEqual(sanitized, 'Bold text')


if __name__ == '__main__':
    unittest.main()
