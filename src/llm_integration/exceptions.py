"""
Custom exceptions for LLM integration module.
"""


class LLMAPIError(Exception):
    """Base exception for LLM API errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class LLMTimeoutError(LLMAPIError):
    """Exception raised when LLM API request times out."""
    pass


class LLMRateLimitError(LLMAPIError):
    """Exception raised when LLM API rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMAPIError):
    """Exception raised when LLM API authentication fails."""
    pass


class LLMInvalidRequestError(LLMAPIError):
    """Exception raised when LLM API request is invalid."""
    pass 