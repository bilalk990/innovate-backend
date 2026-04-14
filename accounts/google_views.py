"""
accounts/google_views.py — Google OAuth Handshake
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from core.google_service import get_google_auth_url, exchange_code_for_tokens
from accounts.models import User

class GoogleAuthURLView(APIView):
    """
    GET /api/accounts/google/url/
    Returns the URL for the recruiter to sync their Google account.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role != 'recruiter':
            return Response({'error': 'Only recruiters can sync Google Meet.'}, status=403)
        
        try:
            url = get_google_auth_url()
            return Response({'url': url})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

class GoogleCallbackView(APIView):
    """
    POST /api/accounts/google/callback/
    Exchanges the 'code' for tokens and saves them to the User document.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        code = request.data.get('code')
        if not code:
            return Response({'error': 'No code provided.'}, status=400)

        try:
            tokens = exchange_code_for_tokens(code)
            
            # Save tokens to the recruiter's user document
            user = User.objects(id=request.user.id).first()
            if user:
                user.set_google_tokens(tokens)
                user.save()
                return Response({'success': True, 'message': 'Google account synced successfully!'})
            
            return Response({'error': 'User not found.'}, status=404)
        except Exception as e:
            return Response({'error': str(e)}, status=500)
