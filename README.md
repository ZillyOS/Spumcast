# Spumcast - Conversational AI Telegram Bot

A sophisticated conversational AI Telegram bot leveraging OpenRouter LLM endpoints with OpenAI compatibility. Built with Python, featuring robust conversation management, persistent storage, and comprehensive monitoring capabilities.

## 🚀 Features

- **Multi-LLM Support**: Integration with OpenRouter API (320+ models available)
- **OpenAI Compatibility**: Seamless switching between OpenRouter and OpenAI APIs
- **Telegram Integration**: Full Telegram Bot API support with webhook handling
- **Persistent Conversations**: SQLite-based conversation history and context management
- **Real-time Streaming**: Support for streaming LLM responses
- **Robust Error Handling**: Comprehensive error handling with retry mechanisms
- **Dashboard Interface**: Web-based dashboard for testing and model selection
- **Monitoring & Analytics**: Structured logging and performance metrics
- **Docker Support**: Containerized deployment with Docker Compose

## 📋 Current Status

✅ **Phase 1.1: LLM Integration Module** - COMPLETED
- OpenRouter API client with authentication
- OpenAI compatibility layer
- Streaming completion support
- Model listing and selection
- Comprehensive error handling
- Async session management

🔄 **Phase 1.2: Basic Telegram Bot** - PENDING  
🔄 **Phase 1.3: Database Layer** - PENDING  
🔄 **Phase 2.1: Conversation Management** - PENDING  

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))
- Optional: OpenAI API key for direct OpenAI integration

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ZillyOS/Spumcast.git
   cd Spumcast
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-api-key"
   # Optional: export OPENAI_API_KEY="your-openai-api-key"
   ```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Integration Tests (requires API key)
```bash
OPENROUTER_API_KEY="your-key" pytest tests/ -v -m integration
```

### Run Example Script
```bash
OPENROUTER_API_KEY="your-key" python examples/test_llm_integration.py
```

## 🏗️ Architecture

### Core Components

1. **LLM Integration Layer** (`src/llm_integration/`)
   - `OpenRouterClient`: Direct OpenRouter API integration
   - `OpenAICompatClient`: OpenAI-compatible interface
   - `BaseLLMClient`: Abstract base for extensible LLM support

2. **Data Models**
   - `LLMMessage`: Standardized message format
   - `LLMResponse`: Unified response structure
   - Custom exception hierarchy for robust error handling

3. **Testing Suite**
   - Comprehensive unit tests with mocking
   - Integration tests with real API endpoints
   - Example implementations and demonstrations

## 📖 Usage Examples

### Basic OpenRouter Integration

```python
import asyncio
from src.llm_integration import OpenRouterClient, LLMMessage

async def chat_example():
    async with OpenRouterClient(api_key="your-key") as client:
        messages = [
            LLMMessage(role="user", content="Hello, how are you?")
        ]
        
        response = await client.complete(
            messages=messages,
            model="openai/gpt-3.5-turbo",
            max_tokens=100
        )
        
        print(f"Response: {response.content}")

asyncio.run(chat_example())
```

### OpenAI Compatibility

```python
from src.llm_integration import OpenAICompatClient

async def openai_style():
    async with OpenAICompatClient(api_key="your-key") as client:
        # Uses OpenAI-style interface but routes through OpenRouter
        response = await client.create_chat_completion(
            model="gpt-3.5-turbo",  # Automatically converted to openai/gpt-3.5-turbo
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content)
```

### Streaming Responses

```python
async def streaming_example():
    async with OpenRouterClient(api_key="your-key") as client:
        messages = [LLMMessage(role="user", content="Count to 10")]
        
        async for chunk in client.stream_complete(
            messages=messages,
            model="openai/gpt-3.5-turbo"
        ):
            print(chunk, end='', flush=True)
```

## 🔧 Development

### Project Structure
```
Spumcast/
├── src/
│   ├── llm_integration/          # LLM integration module
│   │   ├── __init__.py
│   │   ├── base_client.py        # Abstract base client
│   │   ├── openrouter_client.py  # OpenRouter implementation
│   │   ├── openai_compatibility.py # OpenAI compatibility layer
│   │   └── exceptions.py         # Custom exceptions
│   └── core/                     # Core application logic (planned)
├── tests/                        # Test suite
├── examples/                     # Example implementations
├── requirements.txt              # Dependencies
├── setup.py                      # Package configuration
└── pytest.ini                   # Test configuration
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📊 Supported Models

The system supports 320+ models through OpenRouter, including:

- **OpenAI Models**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **Google**: Gemini Pro, Gemini Flash
- **Meta**: Llama 3.1, Llama 3.2
- **Mistral**: Mixtral, Mistral Large
- **And many more...**

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Network Issues**: Automatic retries with exponential backoff
- **API Limits**: Rate limiting and quota management
- **Authentication**: Clear error messages for invalid API keys
- **Model Errors**: Graceful handling of model-specific issues

## 🔐 Security

- API keys are never exposed in the codebase
- Environment variable-based configuration
- Input sanitization and validation
- Secure session management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- 📧 Email: support@spumcast.com
- 🐛 Issues: [GitHub Issues](https://github.com/ZillyOS/Spumcast/issues)
- 📖 Documentation: [Project Wiki](https://github.com/ZillyOS/Spumcast/wiki)

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai) for providing access to multiple LLM APIs
- [Telegram Bot API](https://core.telegram.org/bots/api) for bot platform
- [LangChain](https://langchain.com) for LLM framework components 