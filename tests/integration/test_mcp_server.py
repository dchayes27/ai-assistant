"""
Integration tests for MCP server
Tests the FastAPI endpoints and WebSocket functionality
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from mcp_server.main import create_app
from mcp_server.models import QueryRequest, MemorySearchRequest, MessageRole


@pytest.fixture
def test_app():
    """Create test FastAPI app"""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing"""
    return {"X-API-Key": "dev-key-12345"}


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    @pytest.mark.integration
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "database_status" in data
        assert "ollama_status" in data
        assert "timestamp" in data
    
    @pytest.mark.integration
    def test_status_endpoint(self, test_client, auth_headers):
        """Test status endpoint"""
        response = test_client.get("/status", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "uptime" in data
        assert "timestamp" in data
    
    @pytest.mark.integration
    def test_status_unauthorized(self, test_client):
        """Test status endpoint without authentication"""
        response = test_client.get("/status")
        assert response.status_code == 401


class TestAuthenticationEndpoints:
    """Test authentication endpoints"""
    
    @pytest.mark.integration
    def test_login_with_api_key(self, test_client):
        """Test login with API key"""
        auth_data = {"api_key": "dev-key-12345"}
        
        response = test_client.post("/auth/login", json=auth_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.integration
    def test_login_invalid_api_key(self, test_client):
        """Test login with invalid API key"""
        auth_data = {"api_key": "invalid-key"}
        
        response = test_client.post("/auth/login", json=auth_data)
        assert response.status_code == 401
    
    @pytest.mark.integration
    def test_refresh_token(self, test_client, auth_headers):
        """Test token refresh"""
        response = test_client.post("/auth/refresh", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data


class TestAgentEndpoints:
    """Test agent interaction endpoints"""
    
    @pytest.mark.integration
    @patch('mcp_server.main.ollama_client')
    @patch('mcp_server.main.db_manager')
    def test_agent_query(self, mock_db, mock_ollama, test_client, auth_headers):
        """Test agent query endpoint"""
        # Mock the ollama client response
        mock_response = Mock()
        mock_response.response = "This is a test response"
        mock_response.conversation_id = "test-conv"
        mock_response.message_id = "msg-123"
        mock_response.model_used = "llama3.2:3b"
        mock_response.tokens_used = 50
        mock_response.response_time = 0.5
        mock_response.context_used = []
        
        mock_ollama.generate = AsyncMock(return_value=mock_response)
        mock_db.get_conversation_context = AsyncMock(return_value=[])
        mock_db.add_message = AsyncMock(return_value="msg-123")
        
        query_data = {
            "message": "Hello, how are you?",
            "conversation_id": "test-conv",
            "model": "llama3.2:3b"
        }
        
        response = test_client.post(
            "/agent/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "message_id" in data
        assert data["response"] == "This is a test response"
    
    @pytest.mark.integration
    @patch('mcp_server.main.ollama_client')
    def test_agent_query_unauthorized(self, mock_ollama, test_client):
        """Test agent query without authentication"""
        query_data = {
            "message": "Hello",
            "conversation_id": "test-conv"
        }
        
        response = test_client.post("/agent/query", json=query_data)
        assert response.status_code == 401
    
    @pytest.mark.integration
    @patch('mcp_server.main.ollama_client')
    def test_agent_stream(self, mock_ollama, test_client, auth_headers):
        """Test agent streaming endpoint"""
        # Mock streaming response
        async def mock_stream(request, context):
            from mcp_server.models import StreamChunk
            chunks = ["Hello", " world", "!"]
            for i, chunk in enumerate(chunks):
                yield StreamChunk(
                    chunk=chunk,
                    is_final=i == len(chunks) - 1,
                    conversation_id="test-conv",
                    message_id="msg-123"
                )
        
        mock_ollama.generate_stream = AsyncMock(side_effect=mock_stream)
        
        query_data = {
            "message": "Hello",
            "conversation_id": "test-conv",
            "stream": True
        }
        
        with test_client.stream(
            "POST",
            "/agent/stream",
            json=query_data,
            headers=auth_headers
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestMemoryEndpoints:
    """Test memory management endpoints"""
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_memory_search(self, mock_db, test_client, auth_headers):
        """Test memory search endpoint"""
        # Mock search results
        mock_results = [
            {
                "message_id": "msg-1",
                "conversation_id": "conv-1",
                "role": "user",
                "content": "Hello world",
                "timestamp": datetime.utcnow(),
                "metadata": {}
            }
        ]
        mock_db.search_messages = AsyncMock(return_value=mock_results)
        
        search_data = {
            "query": "hello",
            "search_type": "fts",
            "limit": 10
        }
        
        response = test_client.post(
            "/memory/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_count" in data
        assert "search_time" in data
        assert len(data["results"]) == 1
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_memory_save(self, mock_db, test_client, auth_headers):
        """Test memory save endpoint"""
        mock_db.add_message = AsyncMock(return_value="msg-123")
        
        save_data = {
            "conversation_id": "test-conv",
            "role": "user",
            "content": "Test message",
            "metadata": {"test": True}
        }
        
        response = test_client.post(
            "/memory/save",
            json=save_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message_id" in data
        assert "success" in data
        assert data["success"] is True
    
    @pytest.mark.integration
    @patch('mcp_server.main.vector_store')
    def test_memory_search_semantic(self, mock_vector, test_client, auth_headers):
        """Test semantic memory search"""
        # Mock vector search results
        mock_results = [
            {
                "message_id": "msg-1",
                "conversation_id": "conv-1",
                "content": "Machine learning is fascinating",
                "similarity_score": 0.85,
                "timestamp": datetime.utcnow(),
                "metadata": {}
            }
        ]
        mock_vector.search_similar = AsyncMock(return_value=mock_results)
        
        search_data = {
            "query": "AI and ML",
            "search_type": "semantic",
            "limit": 5
        }
        
        response = test_client.post(
            "/memory/search",
            json=search_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["similarity_score"] == 0.85


class TestProjectEndpoints:
    """Test project management endpoints"""
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_projects_list(self, mock_db, test_client, auth_headers):
        """Test projects list endpoint"""
        # Mock projects data
        mock_projects = [
            {
                "project_id": "proj-1",
                "name": "Test Project",
                "description": "A test project",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {"status": "active"}
            }
        ]
        mock_db.get_projects = AsyncMock(return_value=mock_projects)
        mock_db.get_projects_count = AsyncMock(return_value=1)
        
        response = test_client.get("/projects/list", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "total_count" in data
        assert len(data["projects"]) == 1
        assert data["projects"][0]["name"] == "Test Project"
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_create_project(self, mock_db, test_client, auth_headers):
        """Test project creation endpoint"""
        mock_db.create_project = AsyncMock(return_value="proj-123")
        
        project_data = {
            "name": "New Project",
            "description": "A new test project",
            "metadata": {"status": "planning"}
        }
        
        response = test_client.post(
            "/projects/create",
            json=project_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert "success" in data
        assert data["success"] is True


class TestConversationEndpoints:
    """Test conversation management endpoints"""
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_get_conversation_messages(self, mock_db, test_client, auth_headers):
        """Test get conversation messages endpoint"""
        mock_messages = [
            {
                "message_id": "msg-1",
                "role": "user",
                "content": "Hello",
                "timestamp": datetime.utcnow(),
                "metadata": {}
            }
        ]
        mock_db.get_conversation_messages = AsyncMock(return_value=mock_messages)
        
        response = test_client.get(
            "/conversations/test-conv-123/messages",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert "conversation_id" in data
        assert len(data["messages"]) == 1
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_list_conversations(self, mock_db, test_client, auth_headers):
        """Test list conversations endpoint"""
        mock_conversations = [
            {
                "conversation_id": "conv-1",
                "title": "Test Conversation",
                "created_at": datetime.utcnow(),
                "metadata": {"mode": "chat"}
            }
        ]
        mock_db.get_conversations = AsyncMock(return_value=mock_conversations)
        
        response = test_client.get("/conversations", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert len(data["conversations"]) == 1
    
    @pytest.mark.integration
    @patch('mcp_server.main.db_manager')
    def test_create_conversation(self, mock_db, test_client, auth_headers):
        """Test create conversation endpoint"""
        mock_db.create_conversation = AsyncMock(return_value="conv-123")
        
        conv_data = {
            "conversation_id": "conv-123",
            "title": "New Conversation",
            "metadata": {"mode": "chat"}
        }
        
        response = test_client.post(
            "/conversations",
            json=conv_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "success" in data


class TestWebSocketEndpoints:
    """Test WebSocket functionality"""
    
    @pytest.mark.integration
    def test_websocket_connection(self, test_client):
        """Test WebSocket connection"""
        with test_client.websocket_connect("/ws/test-connection-123") as websocket:
            # Send ping message
            ping_data = {
                "type": "ping",
                "content": "test"
            }
            websocket.send_text(json.dumps(ping_data))
            
            # Receive pong response
            response = websocket.receive_text()
            data = json.loads(response)
            
            assert data["type"] == "pong"
            assert "timestamp" in data
    
    @pytest.mark.integration
    @patch('mcp_server.main.ollama_client')
    @patch('mcp_server.main.db_manager')
    def test_websocket_query(self, mock_db, mock_ollama, test_client):
        """Test WebSocket query functionality"""
        # Mock streaming response
        async def mock_stream(request, context):
            from mcp_server.models import StreamChunk
            yield StreamChunk(
                chunk="Hello",
                is_final=False,
                conversation_id="test-conv",
                message_id="msg-123"
            )
            yield StreamChunk(
                chunk=" world!",
                is_final=True,
                conversation_id="test-conv",
                message_id="msg-123"
            )
        
        mock_ollama.generate_stream = AsyncMock(side_effect=mock_stream)
        mock_db.get_conversation_context = AsyncMock(return_value=[])
        
        with test_client.websocket_connect("/ws/test-ws-123") as websocket:
            # Send query message
            query_data = {
                "type": "query",
                "content": "Hello, how are you?",
                "conversation_id": "test-conv"
            }
            websocket.send_text(json.dumps(query_data))
            
            # Receive streaming responses
            responses = []
            while True:
                response_text = websocket.receive_text()
                response_data = json.loads(response_text)
                responses.append(response_data)
                
                if response_data.get("is_final"):
                    break
            
            assert len(responses) >= 2
            assert responses[0]["type"] == "response_chunk"
            assert responses[-1]["is_final"] is True


class TestErrorHandling:
    """Test error handling in MCP server"""
    
    @pytest.mark.integration
    def test_invalid_endpoint(self, test_client):
        """Test invalid endpoint returns 404"""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_invalid_json(self, test_client, auth_headers):
        """Test invalid JSON in request"""
        response = test_client.post(
            "/agent/query",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.integration
    def test_missing_required_fields(self, test_client, auth_headers):
        """Test missing required fields in request"""
        incomplete_data = {"conversation_id": "test"}  # Missing 'message' field
        
        response = test_client.post(
            "/agent/query",
            json=incomplete_data,
            headers=auth_headers
        )
        assert response.status_code == 422
    
    @pytest.mark.integration
    @patch('mcp_server.main.ollama_client')
    def test_service_unavailable(self, mock_ollama, test_client, auth_headers):
        """Test handling of service unavailability"""
        # Mock service failure
        mock_ollama.generate = AsyncMock(side_effect=Exception("Service unavailable"))
        
        query_data = {
            "message": "Hello",
            "conversation_id": "test-conv"
        }
        
        response = test_client.post(
            "/agent/query",
            json=query_data,
            headers=auth_headers
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestPerformance:
    """Test performance characteristics of MCP server"""
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_concurrent_requests(self, test_client, auth_headers):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = test_client.get("/health")
            end_time = time.time()
            
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 10
        assert all(result["status_code"] == 200 for result in results)
        
        # Check that average response time is reasonable
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 1.0  # Should be under 1 second
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_large_request_handling(self, test_client, auth_headers):
        """Test handling of large requests"""
        # Create a large message
        large_message = "Hello! " * 1000  # ~6KB message
        
        query_data = {
            "message": large_message,
            "conversation_id": "large-test-conv"
        }
        
        response = test_client.post(
            "/memory/save",
            json={
                "conversation_id": "large-test",
                "role": "user",
                "content": large_message
            },
            headers=auth_headers
        )
        
        # Should handle large requests gracefully
        assert response.status_code in [200, 413]  # OK or Request Entity Too Large