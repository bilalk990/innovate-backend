"""
Request ID Middleware for distributed tracing
Adds unique request ID to each request for debugging
"""
import uuid
import logging


class RequestIDMiddleware:
    """Add unique request ID to each request for tracing"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger('innovaite')
    
    def __call__(self, request):
        # Generate or extract request ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.request_id = request_id
        
        # Add to logging context
        self.logger.info(f'[{request_id}] {request.method} {request.path}')
        
        # Process request
        response = self.get_response(request)
        
        # Add request ID to response headers
        response['X-Request-ID'] = request_id
        
        return response
