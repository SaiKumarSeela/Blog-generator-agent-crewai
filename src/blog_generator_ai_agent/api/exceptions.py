import logging
import traceback
from datetime import datetime
from typing import Dict, Any

from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class UserFriendlyError(Exception):
    """Custom exception for user-friendly error messages"""

    def __init__(self, message: str, technical_details: str | None = None, error_code: str | None = None):
        self.message = message
        self.technical_details = technical_details
        self.error_code = error_code
        super().__init__(self.message)


def extract_user_friendly_error(error: Exception) -> Dict[str, Any]:
    """Extract user-friendly error message from various exception types"""
    error_str = str(error).lower()
    # Capture full traceback for logs only
    # error_traceback = traceback.format_exc()

    if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
        return {
            "user_message": "We've hit our AI service limit for today. Please try again tomorrow or upgrade your plan.",
            "technical_details": "Rate limit exceeded - API quota reached",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "suggestions": [
                "Wait a few minutes and try again",
                "Check your AI service plan and billing",
                "Consider upgrading to a higher tier plan",
            ],
        }
    elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
        return {
            "user_message": "Authentication failed. Please check your API keys and credentials.",
            "technical_details": "Authentication/authorization error",
            "error_code": "AUTHENTICATION_FAILED",
            "suggestions": [
                "Verify your API keys are correct",
                "Check if your account has proper permissions",
                "Ensure your subscription is active",
            ],
        }
    elif "timeout" in error_str or "timed out" in error_str:
        return {
            "user_message": "The request took too long to complete. Please try again with a simpler request.",
            "technical_details": "Request timeout exceeded",
            "error_code": "REQUEST_TIMEOUT",
            "suggestions": [
                "Try breaking down your request into smaller parts",
                "Check your internet connection",
                "Wait a few minutes and try again",
            ],
        }
    elif "network" in error_str or "connection" in error_str:
        return {
            "user_message": "Network connection issue. Please check your internet connection and try again.",
            "technical_details": "Network connectivity problem",
            "error_code": "NETWORK_ERROR",
            "suggestions": [
                "Check your internet connection",
                "Try refreshing the page",
                "Check if the service is available",
            ],
        }
    elif "memory" in error_str or "out of memory" in error_str:
        return {
            "user_message": "The system is running low on memory. Please try a simpler request or contact support.",
            "technical_details": "Memory allocation error",
            "error_code": "MEMORY_ERROR",
            "suggestions": [
                "Try a simpler request",
                "Close other applications",
                "Contact support if the issue persists",
            ],
        }
    elif "validation" in error_str or "invalid" in error_str:
        return {
            "user_message": "The input data is not valid. Please check your input and try again.",
            "technical_details": "Input validation error",
            "error_code": "VALIDATION_ERROR",
            "suggestions": [
                "Review your input data",
                "Check required fields are filled",
                "Ensure data format is correct",
            ],
        }
    elif "litellm" in error_str or "vertexai" in error_str or "gemini" in error_str:
        if "quota" in error_str or "free_tier" in error_str:
            return {
                "user_message": "You've reached the free tier limit for AI services. Please upgrade your plan or try again tomorrow.",
                "technical_details": "AI service quota exceeded",
                "error_code": "AI_QUOTA_EXCEEDED",
                "suggestions": [
                    "Upgrade to a paid plan for higher limits",
                    "Wait until tomorrow when limits reset",
                    "Use a different AI service provider",
                ],
            }
        else:
            return {
                "user_message": "AI service is temporarily unavailable. Please try again in a few minutes.",
                "technical_details": "AI service error",
                "error_code": "AI_SERVICE_ERROR",
                "suggestions": [
                    "Wait a few minutes and try again",
                    "Check if the AI service is operational",
                    "Contact support if the issue persists",
                ],
            }
    else:
        return {
            "user_message": "Something went wrong. Please try again or contact support if the issue persists.",
            "technical_details": f"Unexpected error: {str(error)}",
            "error_code": "UNKNOWN_ERROR",
            "suggestions": [
                "Try refreshing the page",
                "Check your input data",
                "Contact support with error details",
            ],
        }


def create_error_response(error: Exception, session_id: str | None = None) -> Dict[str, Any]:
    """Create a standardized error response"""
    error_info = extract_user_friendly_error(error)

    logger.error(f"Error occurred: {str(error)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    response: Dict[str, Any] = {
        "status": "error",
        "error": error_info,
        "timestamp": datetime.now().isoformat(),
    }
    if session_id:
        response["session_id"] = session_id
    return response


def global_exception_handler(_: Any, exc: Exception):
    """Global exception handler for unhandled errors"""
    error_response = create_error_response(exc)
    return JSONResponse(status_code=500, content=error_response)


