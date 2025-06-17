"""
Mock tests for LLM interactions
Tests the LLM client with mocked responses to ensure proper handling
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, AsyncGenerator
import httpx
from datetime import datetime

from mcp_server.ollama_client import OllamaClient
from mcp_server.models import QueryRequest, QueryResponse, StreamChunk
from core import SmartAssistant, ConversationMode, AssistantConfig


class TestOllamaClientMocks:
    """Test OllamaClient with mocked HTTP responses"""
    
    @pytest.fixture
    def mock_httpx_client(self):
        """Create a mock HTTP client"""
        client = AsyncMock(spec=httpx.AsyncClient)
        return client
    
    @pytest.fixture
    async def mocked_ollama_client(self, mock_httpx_client):
        """Create OllamaClient with mocked HTTP client"""
        client = OllamaClient()
        client.client = mock_httpx_client
        return client
    
    @pytest.mark.unit
    async def test_health_check_success(self, mocked_ollama_client, mock_httpx_client):
        """Test successful health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.get.return_value = mock_response
        
        result = await mocked_ollama_client.health_check()
        assert result is True
        mock_httpx_client.get.assert_called_once()
    
    @pytest.mark.unit
    async def test_health_check_failure(self, mocked_ollama_client, mock_httpx_client):
        """Test failed health check"""
        # Mock failed response
        mock_httpx_client.get.side_effect = httpx.RequestError("Connection failed")
        
        result = await mocked_ollama_client.health_check()
        assert result is False
    
    @pytest.mark.unit
    async def test_refresh_models(self, mocked_ollama_client, mock_httpx_client):
        """Test model list refresh"""
        # Mock models response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "mistral:7b"},
                {"name": "codellama:7b"}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.get.return_value = mock_response
        
        models = await mocked_ollama_client.refresh_models()
        
        assert len(models) == 3
        assert "llama3.2:3b" in models
        assert "mistral:7b" in models
        assert "codellama:7b" in models
        assert mocked_ollama_client.available_models == models
    
    @pytest.mark.unit
    async def test_generate_success(self, mocked_ollama_client, mock_httpx_client):
        """Test successful text generation"""
        # Mock generation response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "This is a test response"},
            "eval_count": 50,
            "model": "llama3.2:3b"
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        # Set up available models
        mocked_ollama_client.available_models = ["llama3.2:3b"]
        
        request = QueryRequest(
            message="Hello, how are you?",
            conversation_id="test-conv",
            model="llama3.2:3b"
        )
        
        response = await mocked_ollama_client.generate(request)
        
        assert isinstance(response, QueryResponse)
        assert response.response == "This is a test response"
        assert response.conversation_id == "test-conv"
        assert response.model_used == "llama3.2:3b"
        assert response.tokens_used == 50
    
    @pytest.mark.unit
    async def test_generate_with_context(self, mocked_ollama_client, mock_httpx_client):
        """Test generation with conversation context"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Response with context"},
            "eval_count": 75
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        mocked_ollama_client.available_models = ["llama3.2:3b"]
        
        request = QueryRequest(
            message="Continue our conversation",
            conversation_id="test-conv",
            model="llama3.2:3b"
        )
        
        context_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        response = await mocked_ollama_client.generate(request, context_messages)
        
        assert response.response == "Response with context"
        
        # Verify context was included in the request
        call_args = mock_httpx_client.post.call_args
        request_data = call_args[1]["json"]
        assert "messages" in request_data
        assert len(request_data["messages"]) == 3  # 2 context + 1 new
    
    @pytest.mark.unit
    async def test_generate_stream(self, mocked_ollama_client, mock_httpx_client):
        """Test streaming text generation"""
        # Mock streaming response
        stream_data = [
            '{"message": {"content": "Hello"}, "done": false}',
            '{"message": {"content": " world"}, "done": false}',
            '{"message": {"content": "!"}, "done": true}'
        ]
        
        async def mock_aiter_lines():
            for line in stream_data:
                yield line
        
        mock_stream_response = AsyncMock()
        mock_stream_response.aiter_lines.return_value = mock_aiter_lines()
        mock_stream_response.raise_for_status = Mock()
        
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_stream_response
        mock_stream_context.__aexit__.return_value = None
        
        mock_httpx_client.stream.return_value = mock_stream_context
        
        mocked_ollama_client.available_models = ["llama3.2:3b"]
        
        request = QueryRequest(
            message="Stream this",
            conversation_id="test-conv",
            model="llama3.2:3b"
        )
        
        chunks = []
        async for chunk in mocked_ollama_client.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0].chunk == "Hello"
        assert chunks[1].chunk == " world"
        assert chunks[2].chunk == "!"
        assert chunks[2].is_final is True
    
    @pytest.mark.unit
    async def test_embed_text(self, mocked_ollama_client, mock_httpx_client):
        """Test text embedding generation"""
        # Mock embedding response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_response.raise_for_status = Mock()
        mock_httpx_client.post.return_value = mock_response
        
        mocked_ollama_client.available_models = ["nomic-embed-text"]
        
        embedding = await mocked_ollama_client.embed("Test text for embedding")
        
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Verify correct endpoint was called
        call_args = mock_httpx_client.post.call_args
        assert "/api/embeddings" in call_args[0][0]
    
    @pytest.mark.unit
    async def test_model_not_available(self, mocked_ollama_client, mock_httpx_client):
        """Test handling of unavailable model"""
        # Mock pull model response
        stream_data = [
            '{"status": "pulling manifest"}',
            '{"status": "downloading"}',
            '{"status": "success"}'
        ]
        
        async def mock_aiter_lines():
            for line in stream_data:
                yield line
        
        mock_pull_response = AsyncMock()
        mock_pull_response.aiter_lines.return_value = mock_aiter_lines()
        mock_pull_response.raise_for_status = Mock()
        
        mock_pull_context = AsyncMock()
        mock_pull_context.__aenter__.return_value = mock_pull_response
        mock_pull_context.__aexit__.return_value = None
        
        # Mock refresh models after pull
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "new-model:latest"}]
        }
        mock_models_response.raise_for_status = Mock()
        
        # Set up mock responses
        mock_httpx_client.stream.return_value = mock_pull_context
        mock_httpx_client.get.return_value = mock_models_response
        
        # Initially no models available
        mocked_ollama_client.available_models = []
        
        # Try to pull model
        success = await mocked_ollama_client.pull_model("new-model:latest")
        
        assert success is True
        assert "new-model:latest" in mocked_ollama_client.available_models
    
    @pytest.mark.unit
    async def test_error_handling(self, mocked_ollama_client, mock_httpx_client):
        """Test error handling in LLM operations"""
        # Mock HTTP error
        mock_httpx_client.post.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=Mock(status_code=500)
        )
        
        mocked_ollama_client.available_models = ["llama3.2:3b"]
        
        request = QueryRequest(
            message="This will fail",
            model="llama3.2:3b"
        )
        
        with pytest.raises(ValueError):
            await mocked_ollama_client.generate(request)
    
    @pytest.mark.unit
    async def test_request_timeout(self, mocked_ollama_client, mock_httpx_client):
        """Test request timeout handling"""
        # Mock timeout
        mock_httpx_client.post.side_effect = httpx.TimeoutException("Request timed out")
        
        mocked_ollama_client.available_models = ["llama3.2:3b"]
        
        request = QueryRequest(
            message="This will timeout",
            model="llama3.2:3b"
        )
        
        with pytest.raises(Exception):
            await mocked_ollama_client.generate(request)


class TestSmartAssistantMocks:
    """Test SmartAssistant with mocked LLM components"""
    
    @pytest.mark.unit
    async def test_assistant_with_mocked_llm(self, test_assistant):
        """Test assistant functionality with mocked LLM"""
        # The test_assistant fixture already has mocked components
        response = await test_assistant.process_message(
            "Hello, how are you?",
            mode=ConversationMode.CHAT
        )
        
        assert response == "This is a mock response"
        assert test_assistant.ollama_client.generate.called
    
    @pytest.mark.unit
    async def test_conversation_with_context(self, test_assistant):
        """Test conversation with context using mocks"""
        # Create a conversation thread
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.CHAT,
            title="Test Conversation"
        )
        
        # Send first message
        response1 = await test_assistant.process_message(
            "Hello", thread_id=thread_id
        )
        assert response1 == "This is a mock response"
        
        # Send second message (should have context)
        response2 = await test_assistant.process_message(
            "How's the weather?", thread_id=thread_id
        )
        assert response2 == "This is a mock response"
        
        # Verify the mock was called twice
        assert test_assistant.ollama_client.generate.call_count == 2
    
    @pytest.mark.unit
    async def test_voice_processing_mocks(self, test_assistant, sample_audio_data):
        """Test voice processing with mocked components"""
        # Mock whisper transcription is already set up in fixture
        result = await test_assistant.process_voice_message(
            sample_audio_data,
            synthesize_response=True
        )
        
        assert "transcript" in result
        assert "response" in result
        assert result["transcript"] == "This is a mock transcription"
        assert result["response"] == "This is a mock response"
    
    @pytest.mark.unit
    async def test_embedding_generation_mock(self, test_assistant):
        """Test embedding generation with mocked LLM"""
        # Mock embedding is set up in fixture
        text = "Test text for embedding"
        embedding = await test_assistant.ollama_client.embed(text)
        
        assert embedding == [0.1] * 768
        test_assistant.ollama_client.embed.assert_called_with(text)
    
    @pytest.mark.unit
    async def test_streaming_response_mock(self, test_assistant):
        """Test streaming response with mocked LLM"""
        request = QueryRequest(
            message="Stream this response",
            conversation_id="test-conv",
            stream=True
        )
        
        chunks = []
        async for chunk in test_assistant.ollama_client.generate_stream(request):
            chunks.append(chunk.chunk)
        
        expected_chunks = ["This ", "is ", "a ", "mock ", "streaming ", "response"]
        assert chunks == expected_chunks
    
    @pytest.mark.unit
    async def test_error_handling_mocks(self, test_assistant):
        """Test error handling with mocked failures"""
        # Set up mock to raise an exception
        test_assistant.ollama_client.generate.side_effect = Exception("LLM service unavailable")
        
        with pytest.raises(Exception):
            await test_assistant.process_message("This will fail")
    
    @pytest.mark.unit
    async def test_retry_logic_with_mocks(self, test_assistant):
        """Test retry logic with mocked failures and recovery"""
        # Configure mock to fail twice then succeed
        call_count = 0
        
        async def mock_generate_with_retry(request, context=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Failure {call_count}")
            return QueryResponse(
                response="Success after retries",
                conversation_id=request.conversation_id or "test",
                message_id="msg-123",
                model_used=request.model,
                tokens_used=50,
                response_time=0.5,
                context_used=[]
            )
        
        test_assistant.ollama_client.generate.side_effect = mock_generate_with_retry
        
        # This should succeed after retries
        response = await test_assistant.process_message("Retry test")
        assert response == "Success after retries"
        assert call_count == 3  # Failed twice, succeeded on third try


class TestLLMResponseVariations:
    """Test different types of LLM responses"""
    
    @pytest.mark.unit
    async def test_empty_response(self, mock_ollama_client):
        """Test handling of empty LLM response"""
        async def mock_empty_generate(request, context=None):
            return QueryResponse(
                response="",
                conversation_id=request.conversation_id or "test",
                message_id="msg-empty",
                model_used=request.model,
                tokens_used=0,
                response_time=0.1,
                context_used=[]
            )
        
        mock_ollama_client.generate.side_effect = mock_empty_generate
        
        request = QueryRequest(message="Test", model="test-model")
        response = await mock_ollama_client.generate(request)
        
        assert response.response == ""
        assert response.tokens_used == 0
    
    @pytest.mark.unit
    async def test_long_response(self, mock_ollama_client):
        """Test handling of very long LLM response"""
        long_text = "This is a very long response. " * 1000  # ~30KB
        
        async def mock_long_generate(request, context=None):
            return QueryResponse(
                response=long_text,
                conversation_id=request.conversation_id or "test",
                message_id="msg-long",
                model_used=request.model,
                tokens_used=5000,
                response_time=2.5,
                context_used=[]
            )
        
        mock_ollama_client.generate.side_effect = mock_long_generate
        
        request = QueryRequest(message="Generate long text", model="test-model")
        response = await mock_ollama_client.generate(request)
        
        assert len(response.response) > 20000
        assert response.tokens_used == 5000
    
    @pytest.mark.unit
    async def test_special_characters_response(self, mock_ollama_client):
        """Test handling of response with special characters"""
        special_text = "Response with émojis 🤖, unicode ñoño, and symbols @#$%^&*()"
        
        async def mock_special_generate(request, context=None):
            return QueryResponse(
                response=special_text,
                conversation_id=request.conversation_id or "test",
                message_id="msg-special",
                model_used=request.model,
                tokens_used=25,
                response_time=0.3,
                context_used=[]
            )
        
        mock_ollama_client.generate.side_effect = mock_special_generate
        
        request = QueryRequest(message="Special chars", model="test-model")
        response = await mock_ollama_client.generate(request)
        
        assert response.response == special_text
        assert "🤖" in response.response
        assert "ñoño" in response.response
    
    @pytest.mark.unit
    async def test_json_in_response(self, mock_ollama_client):
        """Test handling of JSON content in LLM response"""
        json_response = json.dumps({
            "code": "print('Hello, World!')",
            "explanation": "This prints a greeting message",
            "language": "python"
        }, indent=2)
        
        async def mock_json_generate(request, context=None):
            return QueryResponse(
                response=json_response,
                conversation_id=request.conversation_id or "test",
                message_id="msg-json",
                model_used=request.model,
                tokens_used=100,
                response_time=0.7,
                context_used=[]
            )
        
        mock_ollama_client.generate.side_effect = mock_json_generate
        
        request = QueryRequest(message="Generate JSON", model="test-model")
        response = await mock_ollama_client.generate(request)
        
        # Verify we can parse the JSON
        parsed_json = json.loads(response.response)
        assert "code" in parsed_json
        assert "explanation" in parsed_json
        assert parsed_json["language"] == "python"


class TestMockConfiguration:
    """Test mock configuration and setup"""
    
    @pytest.mark.unit
    def test_mock_setup_consistency(self, mock_ollama_client):
        """Test that mock setup is consistent"""
        # Verify all expected methods are mocked
        assert hasattr(mock_ollama_client, 'generate')
        assert hasattr(mock_ollama_client, 'generate_stream')
        assert hasattr(mock_ollama_client, 'embed')
        assert hasattr(mock_ollama_client, 'health_check')
        
        # Verify they are async mocks
        assert asyncio.iscoroutinefunction(mock_ollama_client.generate)
        assert asyncio.iscoroutinefunction(mock_ollama_client.generate_stream)
        assert asyncio.iscoroutinefunction(mock_ollama_client.embed)
    
    @pytest.mark.unit
    async def test_mock_call_tracking(self, mock_ollama_client):
        """Test that mock calls are properly tracked"""
        request = QueryRequest(message="Test", model="test-model")
        
        # Make calls
        await mock_ollama_client.generate(request)
        await mock_ollama_client.embed("test text")
        await mock_ollama_client.health_check()
        
        # Verify call counts
        assert mock_ollama_client.generate.call_count == 1
        assert mock_ollama_client.embed.call_count == 1
        assert mock_ollama_client.health_check.call_count == 1
    
    @pytest.mark.unit
    async def test_mock_return_value_customization(self, mock_ollama_client):
        """Test customizing mock return values"""
        # Customize the mock response
        custom_response = QueryResponse(
            response="Custom mock response",
            conversation_id="custom-conv",
            message_id="custom-msg",
            model_used="custom-model",
            tokens_used=123,
            response_time=1.23,
            context_used=[]
        )
        
        mock_ollama_client.generate.return_value = custom_response
        
        request = QueryRequest(message="Test", model="test-model")
        response = await mock_ollama_client.generate(request)
        
        assert response.response == "Custom mock response"
        assert response.tokens_used == 123
        assert response.response_time == 1.23