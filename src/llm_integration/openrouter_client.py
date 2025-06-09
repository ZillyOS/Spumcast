"""
OpenRouter API client implementation.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
import structlog
import backoff

from .base_client import BaseLLMClient, LLMMessage, LLMResponse
from .exceptions import (
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMInvalidRequestError
)

logger = structlog.get_logger(__name__)


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client for LLM completions."""
    
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = 30.0,
        app_name: str = "Spumcast",
        app_url: str = "https://github.com/yourusername/spumcast"
    ):
        # Load API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY environment variable")
        
        super().__init__(api_key, base_url or self.DEFAULT_BASE_URL, timeout)
        self.app_name = app_name
        self.app_url = app_url
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.app_url,
                "X-Title": self.app_name,
            }
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _handle_api_error(self, status_code: int, response_data: dict, message: str = None):
        """Handle API errors and raise appropriate exceptions."""
        error_message = message or response_data.get("error", {}).get("message", f"API error {status_code}")
        
        if status_code == 401:
            raise LLMAuthenticationError(error_message, status_code, response_data)
        elif status_code == 429:
            raise LLMRateLimitError(error_message, status_code, response_data)
        elif status_code == 400:
            raise LLMInvalidRequestError(error_message, status_code, response_data)
        else:
            raise LLMAPIError(error_message, status_code, response_data)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, LLMAPIError),
        max_tries=3,
        max_time=60,
        base=2
    )
    async def complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion using OpenRouter API."""
        session = await self._get_session()
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": [msg.to_dict() for msg in messages],
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional parameters
        payload.update(kwargs)
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            logger.info("Making OpenRouter API request", model=model, url=url)
            
            async with session.post(url, json=payload) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error("OpenRouter API error", status=response.status, response=response_data)
                    self._handle_api_error(response.status, response_data)
                
                # Extract response data
                choice = response_data["choices"][0]
                message = choice["message"]
                
                llm_response = LLMResponse(
                    content=message["content"],
                    model=response_data.get("model", model),
                    usage=response_data.get("usage"),
                    finish_reason=choice.get("finish_reason"),
                    response_id=response_data.get("id")
                )
                
                logger.info("Successfully completed OpenRouter request", 
                           model=llm_response.model, 
                           usage=llm_response.usage)
                
                return llm_response
                
        except asyncio.TimeoutError:
            logger.error("OpenRouter API timeout", model=model)
            raise LLMTimeoutError(f"Request to OpenRouter API timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            logger.error("OpenRouter API client error", error=str(e), model=model)
            raise LLMAPIError(f"OpenRouter API client error: {e}")
    
    async def stream_complete(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion using OpenRouter API."""
        session = await self._get_session()
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Add any additional parameters
        payload.update(kwargs)
        
        url = f"{self.base_url}/chat/completions"
        
        try:
            logger.info("Making streaming OpenRouter API request", model=model, url=url)
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    response_data = await response.json()
                    logger.error("OpenRouter streaming API error", status=response.status, response=response_data)
                    self._handle_api_error(response.status, response_data)
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
                            
        except asyncio.TimeoutError:
            logger.error("OpenRouter streaming API timeout", model=model)
            raise LLMTimeoutError(f"Streaming request to OpenRouter API timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            logger.error("OpenRouter streaming API client error", error=str(e), model=model)
            raise LLMAPIError(f"OpenRouter streaming API client error: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenRouter."""
        session = await self._get_session()
        url = f"{self.base_url}/models"
        
        try:
            logger.info("Fetching OpenRouter models", url=url)
            
            async with session.get(url) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    logger.error("OpenRouter models API error", status=response.status, response=response_data)
                    self._handle_api_error(response.status, response_data)
                
                models = response_data.get("data", [])
                logger.info("Successfully fetched OpenRouter models", count=len(models))
                
                return models
                
        except asyncio.TimeoutError:
            logger.error("OpenRouter models API timeout")
            raise LLMTimeoutError(f"Request to OpenRouter models API timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            logger.error("OpenRouter models API client error", error=str(e))
            raise LLMAPIError(f"OpenRouter models API client error: {e}")
    
    async def validate_api_key(self) -> bool:
        """Validate the OpenRouter API key."""
        try:
            # Try to list models as a way to validate the API key
            await self.list_models()
            return True
        except LLMAuthenticationError:
            return False
        except Exception as e:
            logger.error("Error validating OpenRouter API key", error=str(e))
            return False 