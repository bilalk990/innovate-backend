"""
Standardized API response format for consistency
"""
from rest_framework.response import Response


def success_response(data=None, message=None, status=200):
    """
    Standard success response format
    
    Returns:
        {
            "success": true,
            "data": {...},
            "message": "Optional message"
        }
    """
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    if message:
        payload["message"] = message
    return Response(payload, status=status)


def error_response(error, message=None, status=400, details=None):
    """
    Standard error response format
    
    Returns:
        {
            "success": false,
            "error": "Error message",
            "details": {...}  # Optional
        }
    """
    payload = {
        "success": False,
        "error": error if isinstance(error, str) else str(error)
    }
    if message:
        payload["message"] = message
    if details:
        payload["details"] = details
    return Response(payload, status=status)


def paginated_response(results, total, limit, offset, status=200):
    """
    Standard paginated response format
    
    Returns:
        {
            "success": true,
            "data": {
                "results": [...],
                "pagination": {
                    "total": 100,
                    "limit": 50,
                    "offset": 0,
                    "has_more": true
                }
            }
        }
    """
    return Response({
        "success": True,
        "data": {
            "results": results,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
        }
    }, status=status)
