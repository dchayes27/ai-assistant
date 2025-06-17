"""
Pytest configuration and shared fixtures for AI Assistant tests
"""

import os
import sys
import tempfile
import asyncio
import sqlite3
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
import json
import uuid
from datetime import datetime, timedelta
import wave
import numpy as np

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, patch
import httpx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import DatabaseManager, VectorStore
from core import SmartAssistant, ConversationMode, AssistantConfig
from mcp_server.ollama_client import OllamaClient
from mcp_server.models import QueryRequest, QueryResponse


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "audio: mark test as audio processing test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "audio" in str(item.fspath):
            item.add_marker(pytest.mark.audio)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# ==================== EVENT LOOP FIXTURE ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==================== DATABASE FIXTURES ====================

@pytest.fixture
def temp_db_path():
    """Create a temporary database file path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


@pytest.fixture
async def test_db_manager(temp_db_path):
    """Create a test database manager with isolated database"""
    db_manager = DatabaseManager(db_path=temp_db_path)
    await db_manager.initialize()
    yield db_manager
    await db_manager.close()


@pytest.fixture
async def populated_db_manager(test_db_manager):
    """Create a database manager with test data"""
    # Create test conversations
    conv_id1 = await test_db_manager.create_conversation(
        "test-conv-1", 
        "Test Conversation 1",
        {"mode": "chat"}
    )
    
    conv_id2 = await test_db_manager.create_conversation(
        "test-conv-2", 
        "Test Conversation 2", 
        {"mode": "project"}
    )
    
    # Add test messages
    await test_db_manager.add_message(
        conv_id1, "user", "Hello, how are you?", metadata={"test": True}
    )
    await test_db_manager.add_message(
        conv_id1, "assistant", "I'm doing well, thank you!", metadata={"test": True}
    )
    
    await test_db_manager.add_message(
        conv_id2, "user", "Let's discuss the project", metadata={"test": True}
    )
    
    # Add test knowledge
    await test_db_manager.add_knowledge(
        title="Test Knowledge",
        content="This is test knowledge content about machine learning",
        category="ai",
        tags=["test", "ml", "ai"],
        metadata={"test": True}
    )
    
    # Add test project
    await test_db_manager.create_project(
        name="Test Project",
        description="A test project for development",
        metadata={"status": "active", "test": True}
    )
    
    yield test_db_manager


@pytest.fixture
async def test_vector_store(test_db_manager):
    """Create a test vector store"""
    vector_store = VectorStore(test_db_manager)
    yield vector_store


# ==================== MOCK FIXTURES ====================

@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client"""
    client = Mock(spec=OllamaClient)
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.health_check = AsyncMock(return_value=True)
    client.available_models = ["llama3.2:3b", "mistral:7b"]
    client.is_model_available = Mock(return_value=True)
    
    # Mock generate method
    async def mock_generate(request, context_messages=None):
        return QueryResponse(
            response="This is a mock response",
            conversation_id=request.conversation_id or "test-conv",
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            model_used=request.model,
            tokens_used=50,
            response_time=0.5,
            context_used=context_messages or []
        )
    
    client.generate = AsyncMock(side_effect=mock_generate)
    
    # Mock stream generation
    async def mock_generate_stream(request, context_messages=None):
        from mcp_server.models import StreamChunk
        chunks = ["This ", "is ", "a ", "mock ", "streaming ", "response"]
        for i, chunk in enumerate(chunks):
            yield StreamChunk(
                chunk=chunk,
                is_final=i == len(chunks) - 1,
                conversation_id=request.conversation_id or "test-conv",
                message_id=f"msg_{uuid.uuid4().hex[:8]}"
            )
    
    client.generate_stream = AsyncMock(side_effect=mock_generate_stream)
    
    # Mock embedding
    client.embed = AsyncMock(return_value=[0.1] * 768)  # Mock 768-dim embedding
    
    return client


@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model"""
    model = Mock()
    model.transcribe = Mock(return_value={
        "text": "This is a mock transcription",
        "language": "en",
        "segments": []
    })
    return model


@pytest.fixture
def mock_tts_engine():
    """Create a mock TTS engine"""
    engine = Mock()
    engine.tts_to_file = Mock()
    engine.save_to_file = Mock()
    engine.runAndWait = Mock()
    return engine


@pytest.fixture
def mock_httpx_client():
    """Create a mock HTTP client for API testing"""
    client = Mock(spec=httpx.AsyncClient)
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.stream = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ==================== ASSISTANT FIXTURES ====================

@pytest.fixture
def test_assistant_config():
    """Create a test configuration for the assistant"""
    return AssistantConfig(
        whisper_model="tiny",
        ollama_model="llama3.2:3b",
        tts_model="pyttsx3",
        max_context_length=5,
        performance_logging=False,
        max_retries=1,
        retry_delay=0.1
    )


@pytest.fixture
async def test_assistant(test_assistant_config, mock_ollama_client, mock_whisper_model, mock_tts_engine):
    """Create a test assistant with mocked components"""
    assistant = SmartAssistant(test_assistant_config)
    
    # Mock the components
    assistant.ollama_client = mock_ollama_client
    assistant.whisper_model = mock_whisper_model
    assistant.tts_engine = mock_tts_engine
    
    # Use a real database manager for testing
    assistant.db_manager = DatabaseManager(db_path=":memory:")
    await assistant.db_manager.initialize()
    
    assistant.vector_store = VectorStore(assistant.db_manager)
    assistant._initialized = True
    
    yield assistant
    
    if assistant.db_manager:
        await assistant.db_manager.close()


# ==================== AUDIO FIXTURES ====================

@pytest.fixture
def sample_audio_data():
    """Create sample audio data for testing"""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    return audio_data.tobytes()


@pytest.fixture
def sample_wav_file(sample_audio_data):
    """Create a sample WAV file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(sample_audio_data)
        
        yield tmp.name
    
    # Cleanup
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


# ==================== HTTP CLIENT FIXTURES ====================

@pytest.fixture
async def test_http_client():
    """Create a test HTTP client"""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def mock_api_responses():
    """Create mock API responses for testing"""
    return {
        "health": {"status": "healthy"},
        "models": {"models": [{"name": "llama3.2:3b"}]},
        "generate": {
            "model": "llama3.2:3b",
            "message": {"role": "assistant", "content": "Test response"},
            "done": True,
            "eval_count": 50
        },
        "embed": {"embedding": [0.1] * 768}
    }


# ==================== TEST DATA FIXTURES ====================

@pytest.fixture
def sample_conversation_data():
    """Create sample conversation data for testing"""
    return [
        {
            "role": "user",
            "content": "Hello, how are you?",
            "timestamp": datetime.utcnow() - timedelta(minutes=5)
        },
        {
            "role": "assistant", 
            "content": "I'm doing well, thank you! How can I help you today?",
            "timestamp": datetime.utcnow() - timedelta(minutes=4)
        },
        {
            "role": "user",
            "content": "Can you help me with a Python project?",
            "timestamp": datetime.utcnow() - timedelta(minutes=3)
        }
    ]


@pytest.fixture
def sample_project_data():
    """Create sample project data for testing"""
    return {
        "name": "AI Assistant Test Project",
        "description": "A comprehensive test project for the AI Assistant",
        "status": "active",
        "tags": ["test", "development", "ai"],
        "metadata": {
            "priority": "high",
            "team": "development",
            "deadline": "2024-12-31"
        }
    }


@pytest.fixture
def sample_knowledge_data():
    """Create sample knowledge data for testing"""
    return [
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence...",
            "category": "ai",
            "tags": ["ml", "ai", "basics"],
            "metadata": {"difficulty": "beginner"}
        },
        {
            "title": "Python Best Practices",
            "content": "When writing Python code, follow these best practices...",
            "category": "programming",
            "tags": ["python", "coding", "best-practices"],
            "metadata": {"difficulty": "intermediate"}
        }
    ]


# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture
def performance_timer():
    """Create a performance timer for benchmarking"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def memory_profiler():
    """Create a memory profiler for testing"""
    import psutil
    import os
    
    class MemoryProfiler:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            self.peak_memory = None
        
        def start(self):
            self.initial_memory = self.process.memory_info().rss
            self.peak_memory = self.initial_memory
        
        def update(self):
            current_memory = self.process.memory_info().rss
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
        
        @property
        def memory_increase(self):
            if self.initial_memory and self.peak_memory:
                return self.peak_memory - self.initial_memory
            return None
    
    return MemoryProfiler()


# ==================== CLEANUP FIXTURES ====================

@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after tests"""
    temp_files = []
    
    def register_temp_file(filepath):
        temp_files.append(filepath)
    
    yield register_temp_file
    
    # Cleanup
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass  # Ignore cleanup errors


# ==================== ASYNC FIXTURES ====================

@pytest_asyncio.fixture
async def async_mock_context():
    """Create an async context for mocking async operations"""
    class AsyncMockContext:
        def __init__(self):
            self.calls = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def record_call(self, func_name, *args, **kwargs):
            self.calls.append((func_name, args, kwargs))
    
    return AsyncMockContext()


# ==================== HELPER FUNCTIONS ====================

def create_test_message(role: str = "user", content: str = "test message") -> Dict[str, Any]:
    """Create a test message dictionary"""
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow(),
        "metadata": {"test": True}
    }


def create_test_embedding(dimension: int = 768) -> list:
    """Create a test embedding vector"""
    return [0.1] * dimension


async def wait_for_async_operation(operation, timeout: float = 5.0):
    """Wait for an async operation with timeout"""
    try:
        return await asyncio.wait_for(operation, timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Async operation timed out after {timeout} seconds")


# ==================== SKIP CONDITIONS ====================

def skip_if_no_ollama():
    """Skip test if Ollama is not available"""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return response.status_code != 200
    except:
        return True


def skip_if_no_audio():
    """Skip test if audio libraries are not available"""
    try:
        import sounddevice
        import whisper
        return False
    except ImportError:
        return True