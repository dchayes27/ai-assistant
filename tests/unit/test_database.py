"""
Unit tests for database operations
Tests the DatabaseManager and VectorStore classes
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any

from memory import DatabaseManager, VectorStore
from memory.models import Conversation, Message, Knowledge, Project


class TestDatabaseManager:
    """Test cases for DatabaseManager"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialization(self, test_db_manager):
        """Test database initialization"""
        assert test_db_manager.db_path is not None
        assert await test_db_manager.health_check() is True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_creation(self, test_db_manager):
        """Test conversation creation"""
        conv_id = "test-conv-123"
        title = "Test Conversation"
        metadata = {"mode": "chat", "test": True}
        
        created_id = await test_db_manager.create_conversation(conv_id, title, metadata)
        assert created_id == conv_id
        
        # Verify conversation exists
        conversations = await test_db_manager.get_conversations(limit=10)
        assert len(conversations) == 1
        assert conversations[0]["conversation_id"] == conv_id
        assert conversations[0]["title"] == title
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_auto_id(self, test_db_manager):
        """Test conversation creation with auto-generated ID"""
        title = "Auto ID Conversation"
        
        conv_id = await test_db_manager.create_conversation(None, title)
        assert conv_id is not None
        assert len(conv_id) > 0
        
        conversations = await test_db_manager.get_conversations()
        assert len(conversations) == 1
        assert conversations[0]["conversation_id"] == conv_id
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_operations(self, test_db_manager):
        """Test message creation and retrieval"""
        # Create conversation first
        conv_id = await test_db_manager.create_conversation(
            "msg-test-conv", "Message Test"
        )
        
        # Add user message
        user_msg_id = await test_db_manager.add_message(
            conv_id, "user", "Hello, world!", metadata={"test": True}
        )
        assert user_msg_id is not None
        
        # Add assistant message
        assistant_msg_id = await test_db_manager.add_message(
            conv_id, "assistant", "Hello! How can I help you?", metadata={"test": True}
        )
        assert assistant_msg_id is not None
        
        # Retrieve messages
        messages = await test_db_manager.get_conversation_messages(conv_id)
        assert len(messages) == 2
        
        # Check message order (should be chronological)
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[0]["content"] == "Hello, world!"
        assert messages[1]["content"] == "Hello! How can I help you?"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_search(self, populated_db_manager):
        """Test full-text search functionality"""
        results = await populated_db_manager.search_messages("hello")
        assert len(results) > 0
        
        # Search should find the greeting message
        found_greeting = any("Hello" in result["content"] for result in results)
        assert found_greeting
        
        # Test case-insensitive search
        results_lower = await populated_db_manager.search_messages("HELLO")
        assert len(results_lower) == len(results)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_context(self, populated_db_manager):
        """Test conversation context retrieval"""
        conversations = await populated_db_manager.get_conversations()
        conv_id = conversations[0]["conversation_id"]
        
        context = await populated_db_manager.get_conversation_context(conv_id, limit=5)
        assert len(context) <= 5
        assert all("role" in msg and "content" in msg for msg in context)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_knowledge_operations(self, test_db_manager):
        """Test knowledge base operations"""
        # Add knowledge item
        knowledge_id = await test_db_manager.add_knowledge(
            title="Test Knowledge",
            content="This is test knowledge about AI",
            category="ai",
            tags=["test", "ai", "knowledge"],
            metadata={"importance": "high"}
        )
        assert knowledge_id is not None
        
        # Search knowledge
        results = await test_db_manager.search_knowledge("AI")
        assert len(results) == 1
        assert results[0]["title"] == "Test Knowledge"
        assert "ai" in results[0]["tags"]
        
        # Get knowledge by category
        ai_knowledge = await test_db_manager.get_knowledge_by_category("ai")
        assert len(ai_knowledge) == 1
        assert ai_knowledge[0]["knowledge_id"] == knowledge_id
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_project_operations(self, test_db_manager):
        """Test project management operations"""
        # Create project
        project_id = await test_db_manager.create_project(
            name="Test Project",
            description="A test project for unit testing",
            metadata={"status": "active", "priority": "high"}
        )
        assert project_id is not None
        
        # Get projects
        projects = await test_db_manager.get_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "Test Project"
        assert projects[0]["project_id"] == project_id
        
        # Update project
        success = await test_db_manager.update_project(
            project_id,
            metadata={"status": "completed", "priority": "high"}
        )
        assert success is True
        
        # Verify update
        updated_projects = await test_db_manager.get_projects()
        assert updated_projects[0]["metadata"]["status"] == "completed"
        
        # Get project count
        count = await test_db_manager.get_projects_count()
        assert count == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_statistics(self, populated_db_manager):
        """Test database statistics"""
        stats = await populated_db_manager.get_stats()
        
        assert "conversations" in stats
        assert "messages" in stats
        assert "knowledge" in stats
        assert "projects" in stats
        
        assert stats["conversations"] >= 2  # From populated fixture
        assert stats["messages"] >= 3
        assert stats["knowledge"] >= 1
        assert stats["projects"] >= 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_archiving(self, populated_db_manager):
        """Test conversation archiving functionality"""
        # Archive conversations older than 1 day
        cutoff_date = datetime.utcnow() - timedelta(days=1)
        archived_count = await populated_db_manager.archive_old_conversations(cutoff_date)
        
        # Should be 0 since our test conversations are new
        assert archived_count == 0
        
        # Test with future date to archive all
        future_date = datetime.utcnow() + timedelta(days=1)
        archived_count = await populated_db_manager.archive_old_conversations(future_date)
        assert archived_count >= 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_database_health_check(self, test_db_manager):
        """Test database health check"""
        is_healthy = await test_db_manager.health_check()
        assert is_healthy is True
        
        # Test with closed database
        await test_db_manager.close()
        is_healthy = await test_db_manager.health_check()
        assert is_healthy is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_operations(self, test_db_manager):
        """Test error handling for invalid operations"""
        # Try to add message to non-existent conversation
        with pytest.raises(Exception):
            await test_db_manager.add_message(
                "non-existent-conv", "user", "test message"
            )
        
        # Try to get messages from non-existent conversation
        messages = await test_db_manager.get_conversation_messages("non-existent-conv")
        assert len(messages) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_filtering(self, populated_db_manager):
        """Test message filtering by date and other criteria"""
        conversations = await populated_db_manager.get_conversations()
        conv_id = conversations[0]["conversation_id"]
        
        # Test date filtering
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)
        
        filtered_messages = await populated_db_manager.search_messages(
            "hello",
            conversation_id=conv_id,
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(filtered_messages, list)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bulk_operations(self, test_db_manager):
        """Test bulk database operations"""
        # Create conversation for bulk messages
        conv_id = await test_db_manager.create_conversation(
            "bulk-test-conv", "Bulk Test"
        )
        
        # Add multiple messages
        message_ids = []
        for i in range(10):
            msg_id = await test_db_manager.add_message(
                conv_id, "user" if i % 2 == 0 else "assistant", 
                f"Bulk message {i}", metadata={"index": i}
            )
            message_ids.append(msg_id)
        
        assert len(message_ids) == 10
        
        # Retrieve all messages
        messages = await test_db_manager.get_conversation_messages(conv_id)
        assert len(messages) == 10
        
        # Test pagination
        limited_messages = await test_db_manager.get_conversation_messages(
            conv_id, limit=5
        )
        assert len(limited_messages) == 5
        
        offset_messages = await test_db_manager.get_conversation_messages(
            conv_id, limit=5, offset=5
        )
        assert len(offset_messages) == 5


class TestVectorStore:
    """Test cases for VectorStore"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, test_vector_store):
        """Test vector store initialization"""
        assert test_vector_store.db_manager is not None
        assert hasattr(test_vector_store, '_embedding_cache')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_storage(self, test_vector_store):
        """Test storing and retrieving embeddings"""
        entity_type = "message"
        entity_id = "test-msg-123"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"test": True, "content": "test message"}
        
        # Store embedding
        success = await test_vector_store.store_embedding(
            entity_type, entity_id, embedding, metadata
        )
        assert success is True
        
        # Retrieve embedding
        retrieved = await test_vector_store.get_embedding(entity_type, entity_id)
        assert retrieved is not None
        assert retrieved["embedding"] == embedding
        assert retrieved["metadata"] == metadata
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_similarity_search(self, test_vector_store):
        """Test similarity search functionality"""
        # Store test embeddings
        embeddings_data = [
            ("msg1", [1.0, 0.0, 0.0], {"content": "hello world"}),
            ("msg2", [0.0, 1.0, 0.0], {"content": "goodbye world"}),
            ("msg3", [0.5, 0.5, 0.0], {"content": "hello goodbye"}),
        ]
        
        for entity_id, embedding, metadata in embeddings_data:
            await test_vector_store.store_embedding(
                "message", entity_id, embedding, metadata
            )
        
        # Search for similar embeddings
        query_embedding = [0.9, 0.1, 0.0]  # Similar to msg1
        results = await test_vector_store.search_similar_embeddings(
            query_embedding, limit=2
        )
        
        assert len(results) <= 2
        assert all("similarity_score" in result for result in results)
        assert all("entity_id" in result for result in results)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_similar_text(self, test_vector_store, mock_ollama_client):
        """Test text-based similarity search"""
        # Mock the embedding generation
        test_vector_store.ollama_client = mock_ollama_client
        
        # Store some test embeddings first
        await test_vector_store.store_embedding(
            "message", "test1", [0.1] * 768, {"content": "machine learning"}
        )
        await test_vector_store.store_embedding(
            "message", "test2", [0.2] * 768, {"content": "artificial intelligence"}
        )
        
        # Search with text query
        results = await test_vector_store.search_similar(
            "AI and ML", limit=2
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_batch_embedding_operations(self, test_vector_store):
        """Test batch operations for embeddings"""
        # Prepare batch data
        batch_data = [
            ("message", f"msg{i}", [i/10.0] * 5, {"index": i})
            for i in range(5)
        ]
        
        # Store batch embeddings
        success = await test_vector_store.store_embeddings_batch(batch_data)
        assert success is True
        
        # Retrieve batch embeddings
        entity_ids = [f"msg{i}" for i in range(5)]
        retrieved = await test_vector_store.get_embeddings_batch("message", entity_ids)
        
        assert len(retrieved) == 5
        assert all(item["entity_id"] in entity_ids for item in retrieved)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_cache(self, test_vector_store):
        """Test embedding cache functionality"""
        entity_type = "message"
        entity_id = "cache-test-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Store embedding (should be cached)
        await test_vector_store.store_embedding(
            entity_type, entity_id, embedding, {}
        )
        
        # Retrieve from cache
        cached = test_vector_store._get_from_cache(entity_type, entity_id)
        assert cached is not None
        assert cached["embedding"] == embedding
        
        # Clear cache and verify
        test_vector_store._clear_cache()
        cached_after_clear = test_vector_store._get_from_cache(entity_type, entity_id)
        assert cached_after_clear is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, test_vector_store):
        """Test duplicate embedding detection"""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]  # Identical
        embedding3 = [0.0, 1.0, 0.0]  # Different
        
        # Store first embedding
        await test_vector_store.store_embedding(
            "message", "dup1", embedding1, {"content": "original"}
        )
        
        # Check for duplicates
        duplicates1 = await test_vector_store.find_duplicate_embeddings(
            embedding2, threshold=0.99
        )
        assert len(duplicates1) > 0
        
        duplicates2 = await test_vector_store.find_duplicate_embeddings(
            embedding3, threshold=0.99
        )
        assert len(duplicates2) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_embedding_deletion(self, test_vector_store):
        """Test embedding deletion"""
        entity_type = "message"
        entity_id = "delete-test-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Store embedding
        await test_vector_store.store_embedding(
            entity_type, entity_id, embedding, {}
        )
        
        # Verify it exists
        retrieved = await test_vector_store.get_embedding(entity_type, entity_id)
        assert retrieved is not None
        
        # Delete embedding
        success = await test_vector_store.delete_embedding(entity_type, entity_id)
        assert success is True
        
        # Verify it's gone
        retrieved_after = await test_vector_store.get_embedding(entity_type, entity_id)
        assert retrieved_after is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_vector_statistics(self, test_vector_store):
        """Test vector store statistics"""
        # Store some test embeddings
        for i in range(3):
            await test_vector_store.store_embedding(
                "message", f"stats-test-{i}", [i/10.0] * 5, {}
            )
        
        # Get statistics
        stats = await test_vector_store.get_statistics()
        
        assert "total_embeddings" in stats
        assert "entity_types" in stats
        assert stats["total_embeddings"] >= 3
        assert "message" in stats["entity_types"]


class TestDatabaseModels:
    """Test cases for database models"""
    
    @pytest.mark.unit
    def test_conversation_model(self):
        """Test Conversation model"""
        conv = Conversation(
            conversation_id="test-123",
            title="Test Conversation",
            metadata={"mode": "chat"}
        )
        
        assert conv.conversation_id == "test-123"
        assert conv.title == "Test Conversation"
        assert conv.metadata["mode"] == "chat"
        assert conv.created_at is not None
    
    @pytest.mark.unit
    def test_message_model(self):
        """Test Message model"""
        msg = Message(
            conversation_id="test-conv",
            role="user",
            content="Hello world",
            metadata={"test": True}
        )
        
        assert msg.conversation_id == "test-conv"
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.metadata["test"] is True
        assert msg.message_id is not None
    
    @pytest.mark.unit
    def test_knowledge_model(self):
        """Test Knowledge model"""
        knowledge = Knowledge(
            title="Test Knowledge",
            content="Test content",
            category="test",
            tags=["tag1", "tag2"],
            metadata={"importance": "high"}
        )
        
        assert knowledge.title == "Test Knowledge"
        assert knowledge.content == "Test content"
        assert knowledge.category == "test"
        assert "tag1" in knowledge.tags
        assert knowledge.knowledge_id is not None
    
    @pytest.mark.unit
    def test_project_model(self):
        """Test Project model"""
        project = Project(
            name="Test Project",
            description="Test description",
            metadata={"status": "active"}
        )
        
        assert project.name == "Test Project"
        assert project.description == "Test description"
        assert project.metadata["status"] == "active"
        assert project.project_id is not None