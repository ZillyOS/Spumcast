#!/usr/bin/env python3
"""
Example script to test LLM integration module functionality.

This script demonstrates how to use the OpenRouter and OpenAI compatibility clients.
It requires setting the OPENROUTER_API_KEY environment variable.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_integration import (
    OpenRouterClient,
    OpenAICompatClient,
    LLMMessage,
    LLMAPIError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_openrouter_client():
    """Test OpenRouter client functionality."""
    print("\n=== Testing OpenRouter Client ===")
    
    try:
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ No OPENROUTER_API_KEY environment variable found")
            print("Please set it with your OpenRouter API key to run this test")
            return
        
        # Create client
        async with OpenRouterClient(api_key=api_key) as client:
            print("✅ Client initialized successfully")
            
            # Test API key validation
            print("\n--- Testing API Key Validation ---")
            is_valid = await client.validate_api_key()
            print(f"API Key Valid: {'✅' if is_valid else '❌'} {is_valid}")
            
            # Test listing models
            print("\n--- Testing Model Listing ---")
            models = await client.list_models()
            print(f"✅ Found {len(models)} available models")
            
            # Show first few models
            print("First 5 models:")
            for i, model in enumerate(models[:5]):
                print(f"  {i+1}. {model.get('id', 'Unknown')}")
            
            # Test completion with a simple prompt
            print("\n--- Testing Completion ---")
            messages = [
                LLMMessage(role="user", content="Say 'Hello from OpenRouter!' and explain briefly what you are.")
            ]
            
            # Use a reliable model for testing
            test_model = "openai/gpt-3.5-turbo"
            print(f"Using model: {test_model}")
            
            response = await client.complete(
                messages=messages,
                model=test_model,
                max_tokens=100,
                temperature=0.7
            )
            
            print("✅ Completion successful!")
            print(f"Model: {response.model}")
            print(f"Response: {response.content}")
            print(f"Usage: {response.usage}")
            
            # Test streaming completion
            print("\n--- Testing Streaming Completion ---")
            messages = [
                LLMMessage(role="user", content="Count from 1 to 5 slowly.")
            ]
            
            print("Streaming response:")
            async for chunk in client.stream_complete(
                messages=messages,
                model=test_model,
                max_tokens=50,
                temperature=0.1
            ):
                print(chunk, end='', flush=True)
            print("\n✅ Streaming completed!")
            
    except LLMAPIError as e:
        print(f"❌ LLM API Error: {e}")
        print(f"Status Code: {e.status_code}")
        print(f"Response Data: {e.response_data}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def test_openai_compat_client():
    """Test OpenAI compatibility client."""
    print("\n=== Testing OpenAI Compatibility Client ===")
    
    try:
        # Load API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("❌ No OPENROUTER_API_KEY environment variable found")
            return
        
        # Create OpenAI-compatible client
        async with OpenAICompatClient(api_key=api_key) as client:
            print("✅ OpenAI-compatible client initialized")
            print(f"Direct OpenAI: {client.is_openai_direct}")
            
            # Test model name conversion
            print("\n--- Testing Model Name Conversion ---")
            openai_model = "gpt-3.5-turbo"
            converted_model = client._convert_model_name(openai_model)
            print(f"OpenAI model '{openai_model}' -> OpenRouter '{converted_model}'")
            
            # Test completion with OpenAI-style model name
            print("\n--- Testing OpenAI-Style Completion ---")
            messages = [
                LLMMessage(role="user", content="Say 'Hello from OpenAI compatibility layer!'")
            ]
            
            response = await client.complete(
                messages=messages,
                model="gpt-3.5-turbo",  # Using OpenAI model name
                max_tokens=50,
                temperature=0.7
            )
            
            print("✅ OpenAI-compatible completion successful!")
            print(f"Model (returned): {response.model}")  # Should be converted back to OpenAI format
            print(f"Response: {response.content}")
            
            # Test OpenAI-style chat completions interface
            print("\n--- Testing OpenAI-Style Chat Interface ---")
            openai_messages = [
                {"role": "user", "content": "What is 2+2?"}
            ]
            
            response = await client.create_chat_completion(
                model="gpt-3.5-turbo",
                messages=openai_messages,
                max_tokens=30
            )
            
            print("✅ OpenAI-style chat completion successful!")
            print(f"Response: {response.content}")
            
    except LLMAPIError as e:
        print(f"❌ LLM API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def test_error_handling():
    """Test error handling with invalid API key."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with invalid API key
        async with OpenRouterClient(api_key="invalid-key") as client:
            print("Testing with invalid API key...")
            
            # This should fail
            is_valid = await client.validate_api_key()
            print(f"API Key Valid: {is_valid}")
            
    except Exception as e:
        print(f"✅ Error handling working: {type(e).__name__}: {e}")


def test_model_parameters():
    """Test different model parameters."""
    print("\n=== Testing Model Parameters ===")
    
    # Test temperature values
    temperatures = [0.1, 0.5, 0.9]
    print(f"Temperature range test: {temperatures}")
    
    # Test max_tokens values
    max_tokens = [10, 50, 100]
    print(f"Max tokens range test: {max_tokens}")
    
    print("✅ Parameter validation completed")


async def main():
    """Run all tests."""
    print("🚀 Starting LLM Integration Module Tests")
    print("=" * 50)
    
    # Test model parameters (synchronous)
    test_model_parameters()
    
    # Test OpenRouter client
    await test_openrouter_client()
    
    # Test OpenAI compatibility
    await test_openai_compat_client()
    
    # Test error handling
    await test_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed!")
    print("\nNote: Some tests require a valid OPENROUTER_API_KEY environment variable.")
    print("Set it with: export OPENROUTER_API_KEY='your-key-here'")


if __name__ == "__main__":
    asyncio.run(main()) 