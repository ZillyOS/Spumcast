"""
LLM Integration Module

This module provides integration with OpenRouter and OpenAI APIs for conversational AI.
"""

from .openrouter_client import OpenRouterClient
from .openai_compatibility import OpenAICompatClient
from .base_client import BaseLLMClient, LLMMessage, LLMResponse
from .exceptions import LLMAPIError, LLMTimeoutError, LLMRateLimitError, LLMAuthenticationError

__all__ = [
    'OpenRouterClient',
    'OpenAICompatClient', 
    'BaseLLMClient',
    'LLMMessage',
    'LLMResponse',
    'LLMAPIError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'LLMAuthenticationError'
] 