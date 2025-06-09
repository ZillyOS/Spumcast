"""
OpenAI compatibility layer for OpenRouter client.
"""

import os
from typing import Dict, List, Optional, Any, AsyncGenerator
import structlog

from .openrouter_client import OpenRouterClient
from .base_client import LLMMessage, LLMResponse

logger = structlog.get_logger(__name__)


class OpenAICompatClient(OpenRouterClient):
    """OpenAI-compatible client that uses OpenRouter as backend."""
    
    # Map OpenAI model names to OpenRouter model names
    OPENAI_TO_OPENROUTER_MODELS = {
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-3.5-turbo-1106": "openai/gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k": "openai/gpt-3.5-turbo-16k",
        "gpt-4": "openai/gpt-4",
        "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4-turbo-preview": "openai/gpt-4-turbo-preview",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
    }
    
    OPENROUTER_TO_OPENAI_MODELS = {v: k for k, v in OPENAI_TO_OPENROUTER_MODELS.items()}
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = 30.0,
        app_name: str = "Spumcast-OpenAI-Compat",
        app_url: str = "https://github.com/yourusername/spumcast"
    ):
        # Can use either OpenRouter or OpenAI API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY or OPENROUTER_API_KEY environment variable")
        
        # Determine if this is a real OpenAI API key or OpenRouter key
        self.is_openai_direct = api_key.startswith("sk-") and not api_key.startswith("sk-or-")
        
        if self.is_openai_direct:
            # Use OpenAI's API directly
            base_url = base_url or "https://api.openai.com/v1"
        
        super().__init__(api_key, base_url, timeout, app_name, app_url)
        logger.info("Initialized OpenAI compatibility client", 
                   direct_openai=self.is_openai_direct, 
                   base_url=self.base_url)
    
    def _convert_model_name(self, model: str, to_openrouter: bool = True) -> str:
        """Convert between OpenAI and OpenRouter model names."""
        if to_openrouter and not self.is_openai_direct:
            # Convert OpenAI model names to OpenRouter format
            if model in self.OPENAI_TO_OPENROUTER_MODELS:
                converted = self.OPENAI_TO_OPENROUTER_MODELS[model]
                logger.debug("Converted OpenAI model to OpenRouter", 
                           original=model, converted=converted)
                return converted
            # If it's already an OpenRouter model name, return as-is
            return model
        return model
    
    def _convert_response_model_name(self, model: str) -> str:
        """Convert OpenRouter model names back to OpenAI format in responses."""
        if not self.is_openai_direct and model in self.OPENROUTER_TO_OPENAI_MODELS:
            converted = self.OPENROUTER_TO_OPENAI_MODELS[model]
            logger.debug("Converted response model to OpenAI format", 
                       original=model, converted=converted)
            return converted
        return model
    
    async def complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion with OpenAI-compatible interface."""
        # Convert model name if needed
        actual_model = self._convert_model_name(model)
        
        logger.info("OpenAI-compatible completion request", 
                   requested_model=model, actual_model=actual_model,
                   direct_openai=self.is_openai_direct)
        
        # Call parent method
        response = await super().complete(messages, actual_model, max_tokens, temperature, **kwargs)
        
        # Convert response model name back to OpenAI format
        response.model = self._convert_response_model_name(response.model)
        
        return response
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion with OpenAI-compatible interface."""
        # Convert model name if needed
        actual_model = self._convert_model_name(model)
        
        logger.info("OpenAI-compatible streaming completion request", 
                   requested_model=model, actual_model=actual_model,
                   direct_openai=self.is_openai_direct)
        
        # Stream from parent method
        async for chunk in super().stream_complete(messages, actual_model, max_tokens, temperature, **kwargs):
            yield chunk
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models with OpenAI-compatible format."""
        if self.is_openai_direct:
            # For direct OpenAI API, return the models we support
            models = []
            for openai_model in self.OPENAI_TO_OPENROUTER_MODELS.keys():
                models.append({
                    "id": openai_model,
                    "object": "model",
                    "created": 1677610602,  # Static timestamp
                    "owned_by": "openai"
                })
            return models
        else:
            # For OpenRouter, get all models and convert names where possible
            openrouter_models = await super().list_models()
            
            # Add OpenAI-named versions for supported models
            openai_models = []
            for model in openrouter_models:
                model_id = model.get("id", "")
                
                # Add the original OpenRouter model
                openai_models.append(model)
                
                # If this OpenRouter model has an OpenAI equivalent, add it too
                if model_id in self.OPENROUTER_TO_OPENAI_MODELS:
                    openai_model = model.copy()
                    openai_model["id"] = self.OPENROUTER_TO_OPENAI_MODELS[model_id]
                    openai_model["openrouter_original"] = model_id
                    openai_models.append(openai_model)
            
            return openai_models
    
    # Convenience methods that match OpenAI SDK patterns
    class ChatCompletions:
        """OpenAI-style chat completions interface."""
        
        def __init__(self, client):
            self.client = client
        
        async def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stream: bool = False,
            **kwargs
        ):
            """Create a chat completion (OpenAI SDK compatible)."""
            # Convert message dicts to LLMMessage objects
            llm_messages = [LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages]
            
            if stream:
                return self.client.stream_complete(llm_messages, model, max_tokens, temperature, **kwargs)
            else:
                return await self.client.complete(llm_messages, model, max_tokens, temperature, **kwargs)
    
    @property
    def chat(self):
        """Access chat completions interface."""
        return type('Chat', (), {'completions': self.ChatCompletions(self)})()
    
    # Additional OpenAI-compatible methods
    async def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Create a chat completion (alternative interface)."""
        llm_messages = [LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages]
        return await self.complete(llm_messages, model, max_tokens, temperature, **kwargs) 