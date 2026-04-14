"""
core/middleware.py — Security headers middleware
Adds CSP, X-Frame-Options, and other security headers to all responses.
"""


class SecurityHeadersMiddleware:
    """Add security headers to every HTTP response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Content Security Policy — #49
        response['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://accounts.google.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' ws: wss: https://generativelanguage.googleapis.com; "
            "frame-src https://accounts.google.com;"
        )

        # Prevent clickjacking
        response['X-Frame-Options'] = 'DENY'

        # Prevent MIME sniffing
        response['X-Content-Type-Options'] = 'nosniff'

        # Referrer policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions policy
        response['Permissions-Policy'] = 'camera=self, microphone=self, geolocation=()'

        return response
