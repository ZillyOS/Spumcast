"""
Base client class for LLM integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    response_id: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, base_url: str = None, timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
    
    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of conversation messages
            model: Model identifier to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Returns:
            LLMResponse containing the generated content
        """
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion for the given messages.
        
        Args:
            messages: List of conversation messages
            model: Model identifier to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters
            
        Yields:
            Chunks of generated content
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of available models with their metadata
        """
        pass
    
    @abstractmethod
    async def validate_api_key(self) -> bool:
        """
        Validate the API key.
        
        Returns:
            True if API key is valid, False otherwise
        """
        pass 