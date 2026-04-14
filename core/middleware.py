"""
core/middleware.py — Security headers middleware
Adds CSP, X-Frame-Options, and other security headers to all responses.
"""
from django.conf import settings


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Get frontend URL from settings
        frontend_url = getattr(settings, 'FRONTEND_URL', 'http://localhost:5173')
        
        # Content Security Policy — #49 - Allow frontend origin
        response['Content-Security-Policy'] = (
            f"default-src 'self' {frontend_url}; "
            f"script-src 'self' 'unsafe-inline' https://accounts.google.com https://accounts.google.com/gsi/client {frontend_url}; "
            f"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com {frontend_url}; "
            f"font-src 'self' https://fonts.gstatic.com {frontend_url}; "
            f"img-src 'self' data: https: {frontend_url}; "
            f"connect-src 'self' ws: wss: https://generativelanguage.googleapis.com {frontend_url}; "
            f"frame-src https://accounts.google.com {frontend_url};"
        )

        # Prevent clickjacking - but allow same origin and frontend
        response['X-Frame-Options'] = 'SAMEORIGIN'

        # Prevent MIME sniffing
        response['X-Content-Type-Options'] = 'nosniff'

        # Referrer policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions policy
        response['Permissions-Policy'] = 'camera=self, microphone=self, geolocation=()'

        return response
