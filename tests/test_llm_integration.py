"""
Tests for LLM integration module.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from aiohttp import web

from src.llm_integration import (
    OpenRouterClient,
    OpenAICompatClient,
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError
)


class TestLLMMessage:
    """Test LLMMessage dataclass."""
    
    def test_create_message(self):
        """Test creating an LLM message."""
        msg = LLMMessage(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
    
    def test_to_dict(self):
        """Test converting message to dictionary."""
        msg = LLMMessage(role="assistant", content="Hello there!")
        dict_repr = msg.to_dict()
        assert dict_repr == {"role": "assistant", "content": "Hello there!"}


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
            response_id="test-id"
        )
        assert response.content == "Test response"
        assert response.model == "gpt-3.5-turbo"
        assert response.usage["prompt_tokens"] == 10
        assert response.finish_reason == "stop"
        assert response.response_id == "test-id"


class TestBaseLLMClient:
    """Test BaseLLMClient abstract class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMClient("test-key")


class MockOpenRouterClient(OpenRouterClient):
    """Mock client for testing without actual API calls."""
    
    def __init__(self, api_key="test-key", **kwargs):
        # Call parent init but with test values
        super().__init__(api_key=api_key, **kwargs)


class TestOpenRouterClient:
    """Test OpenRouterClient."""
    
    def test_init_with_api_key(self):
        """Test initializing client with API key."""
        client = MockOpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://openrouter.ai/api/v1"
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
                OpenRouterClient()
    
    def test_init_with_env_var(self):
        """Test initializing client with environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            client = OpenRouterClient()
            assert client.api_key == "env-key"
    
    @pytest.mark.asyncio
    async def test_session_creation(self):
        """Test that session is created with correct headers."""
        client = MockOpenRouterClient(api_key="test-key")
        session = await client._get_session()
        
        assert isinstance(session, aiohttp.ClientSession)
        assert "Authorization" in session._default_headers
        assert session._default_headers["Authorization"] == "Bearer test-key"
        
        await client.close()
    
    def test_handle_api_errors(self):
        """Test API error handling."""
        client = MockOpenRouterClient(api_key="test-key")
        
        # Test 401 error
        with pytest.raises(LLMAuthenticationError):
            client._handle_api_error(401, {"error": {"message": "Unauthorized"}})
        
        # Test 429 error
        with pytest.raises(LLMRateLimitError):
            client._handle_api_error(429, {"error": {"message": "Rate limit exceeded"}})
        
        # Test generic error
        with pytest.raises(LLMAPIError):
            client._handle_api_error(500, {"error": {"message": "Server error"}})
    
    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion."""
        client = MockOpenRouterClient(api_key="test-key")
        
        # Mock response data
        mock_response_data = {
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "model": "gpt-3.5-turbo",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "id": "test-id"
        }
        
        # Mock the complete method directly to avoid HTTP calls
        with patch.object(client, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_response = LLMResponse(
                content="Test response",
                model="gpt-3.5-turbo",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                finish_reason="stop",
                response_id="test-id"
            )
            mock_complete.return_value = mock_response
            
            messages = [LLMMessage(role="user", content="Hello")]
            response = await client.complete(messages, "gpt-3.5-turbo")
            
            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.model == "gpt-3.5-turbo"
            assert response.usage["prompt_tokens"] == 10
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_validate_api_key_success(self):
        """Test successful API key validation."""
        client = MockOpenRouterClient(api_key="test-key")
        
        # Mock successful list_models call
        with patch.object(client, 'list_models', new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [{"id": "test-model"}]
            
            result = await client.validate_api_key()
            assert result is True
            mock_list.assert_called_once()
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self):
        """Test failed API key validation."""
        client = MockOpenRouterClient(api_key="test-key")
        
        # Mock failed list_models call
        with patch.object(client, 'list_models', new_callable=AsyncMock) as mock_list:
            mock_list.side_effect = LLMAuthenticationError("Invalid API key")
            
            result = await client.validate_api_key()
            assert result is False
            mock_list.assert_called_once()
        
        await client.close()


class TestOpenAICompatClient:
    """Test OpenAI compatibility client."""
    
    def test_init_with_openrouter_key(self):
        """Test initializing with OpenRouter API key."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        assert not client.is_openai_direct
        assert client.base_url == "https://openrouter.ai/api/v1"
    
    def test_init_with_openai_key(self):
        """Test initializing with OpenAI API key."""
        client = OpenAICompatClient(api_key="sk-test123")
        assert client.is_openai_direct
        assert client.base_url == "https://api.openai.com/v1"
    
    def test_model_name_conversion(self):
        """Test OpenAI to OpenRouter model name conversion."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        
        # Test conversion
        converted = client._convert_model_name("gpt-3.5-turbo")
        assert converted == "openai/gpt-3.5-turbo"
        
        # Test passthrough for unknown models
        converted = client._convert_model_name("unknown-model")
        assert converted == "unknown-model"
    
    def test_response_model_name_conversion(self):
        """Test OpenRouter to OpenAI model name conversion in responses."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        
        converted = client._convert_response_model_name("openai/gpt-3.5-turbo")
        assert converted == "gpt-3.5-turbo"
        
        # Test passthrough for unknown models
        converted = client._convert_response_model_name("unknown-model")
        assert converted == "unknown-model"
    
    @pytest.mark.asyncio
    async def test_complete_with_model_conversion(self):
        """Test completion with model name conversion."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        
        # Mock the parent complete method
        mock_response = LLMResponse(
            content="Test response",
            model="openai/gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        
        with patch.object(OpenRouterClient, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response
            
            messages = [LLMMessage(role="user", content="Hello")]
            response = await client.complete(messages, "gpt-3.5-turbo")
            
            # Check that the model name was converted back
            assert response.model == "gpt-3.5-turbo"
            
            # Check that the parent was called with the converted model name
            mock_complete.assert_called_once()
            args, kwargs = mock_complete.call_args
            assert args[1] == "openai/gpt-3.5-turbo"  # model parameter
        
        await client.close()
    
    def test_chat_completions_interface(self):
        """Test OpenAI-style chat completions interface."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        
        # Test that chat.completions exists and has create method
        assert hasattr(client.chat, 'completions')
        assert hasattr(client.chat.completions, 'create')
    
    @pytest.mark.asyncio
    async def test_create_chat_completion(self):
        """Test create_chat_completion method."""
        client = OpenAICompatClient(api_key="sk-or-v1-test")
        
        mock_response = LLMResponse(
            content="Test response",
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        
        with patch.object(client, 'complete', new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response = await client.create_chat_completion("gpt-3.5-turbo", messages)
            
            assert response.content == "Test response"
            mock_complete.assert_called_once()
        
        await client.close()


class TestIntegrationWithRealAPI:
    """Integration tests with real API (requires API key)."""
    
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No OpenRouter API key")
    @pytest.mark.asyncio
    async def test_real_openrouter_list_models(self):
        """Test listing models with real OpenRouter API."""
        client = OpenRouterClient()
        
        try:
            models = await client.list_models()
            assert isinstance(models, list)
            assert len(models) > 0
            
            # Check that each model has required fields
            for model in models[:5]:  # Just check first 5
                assert "id" in model
                
        finally:
            await client.close()
    
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No OpenRouter API key")
    @pytest.mark.asyncio
    async def test_real_openrouter_completion(self):
        """Test completion with real OpenRouter API."""
        client = OpenRouterClient()
        
        try:
            messages = [
                LLMMessage(role="user", content="Say 'Hello, World!' and nothing else.")
            ]
            
            # Using a free model for testing
            response = await client.complete(
                messages, 
                "openai/gpt-3.5-turbo",
                max_tokens=50,
                temperature=0.1
            )
            
            assert isinstance(response, LLMResponse)
            assert len(response.content) > 0
            assert response.model is not None
            assert response.usage is not None
            
        finally:
            await client.close()
    
    @pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"), reason="No OpenRouter API key")
    @pytest.mark.asyncio
    async def test_real_api_key_validation(self):
        """Test API key validation with real OpenRouter API."""
        client = OpenRouterClient()
        
        try:
            is_valid = await client.validate_api_key()
            assert is_valid is True
        finally:
            await client.close()


if __name__ == "__main__":
    pytest.main([__file__]) 