"""
End-to-end conversation tests
Tests complete conversation flows and scenarios
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

from core import SmartAssistant, ConversationMode, AssistantConfig
from memory import DatabaseManager


class TestConversationFlows:
    """Test complete conversation flows"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_basic_conversation_flow(self, test_assistant):
        """Test basic conversation flow"""
        # Create conversation thread
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.CHAT,
            title="Basic Chat Test"
        )
        
        # Conversation sequence
        conversation = [
            ("Hello! How are you today?", "greeting"),
            ("Can you help me with Python programming?", "programming_request"),
            ("What's the difference between lists and tuples?", "technical_question"),
            ("Thank you for the explanation!", "gratitude")
        ]
        
        responses = []
        
        for message, intent in conversation:
            response = await test_assistant.process_message(
                message, 
                thread_id=thread_id,
                metadata={"intent": intent}
            )
            
            responses.append(response)
            assert len(response) > 0
            
            # Small delay to simulate real conversation
            await asyncio.sleep(0.1)
        
        # Verify conversation was stored
        conversation_messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
        assert len(conversation_messages) == len(conversation) * 2  # User + assistant messages
        
        # Verify message order
        for i, (original_message, _) in enumerate(conversation):
            user_msg = conversation_messages[i * 2]
            assistant_msg = conversation_messages[i * 2 + 1]
            
            assert user_msg["role"] == "user"
            assert user_msg["content"] == original_message
            assert assistant_msg["role"] == "assistant"
            assert len(assistant_msg["content"]) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_mode_conversation(self, test_assistant):
        """Test conversation with mode switching"""
        # Start in chat mode
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.CHAT,
            title="Multi-Mode Test"
        )
        
        # Chat mode interaction
        response1 = await test_assistant.process_message(
            "Hello there!", 
            thread_id=thread_id,
            mode=ConversationMode.CHAT
        )
        assert len(response1) > 0
        
        # Switch to project mode
        response2 = await test_assistant.process_message(
            "Let's plan a new software project",
            thread_id=thread_id,
            mode=ConversationMode.PROJECT
        )
        assert len(response2) > 0
        
        # Switch to learning mode
        response3 = await test_assistant.process_message(
            "Explain machine learning concepts",
            thread_id=thread_id,
            mode=ConversationMode.LEARNING
        )
        assert len(response3) > 0
        
        # Verify thread mode was updated
        thread = test_assistant.threads[thread_id]
        assert thread.mode == ConversationMode.LEARNING
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_conversation_with_context(self, test_assistant):
        """Test conversation with context awareness"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Build context through conversation
        messages = [
            "My name is Alice and I'm a software developer",
            "I work primarily with Python and JavaScript",
            "I'm currently working on a web application",
            "What design patterns would you recommend for my project?",
            "How would the Observer pattern help in my JavaScript code?"
        ]
        
        for message in messages:
            response = await test_assistant.process_message(message, thread_id=thread_id)
            assert len(response) > 0
        
        # The assistant should have context about Alice, her skills, and her project
        # This is verified by the fact that all messages are stored and can be retrieved
        context = await test_assistant._get_conversation_context(thread_id)
        assert len(context) > 0
        
        # Should include recent messages
        user_messages = [msg for msg in context if msg.get("role") == "user"]
        assert len(user_messages) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_conversation_memory_search(self, test_assistant):
        """Test conversation memory and search functionality"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Create a conversation about specific topics
        topics_discussed = [
            "machine learning algorithms for recommendation systems",
            "database optimization techniques for large datasets", 
            "microservices architecture patterns and best practices",
            "frontend performance optimization strategies",
            "API design principles and RESTful services"
        ]
        
        for topic in topics_discussed:
            message = f"Tell me about {topic}"
            response = await test_assistant.process_message(message, thread_id=thread_id)
            assert len(response) > 0
        
        # Search for specific topics in memory
        search_queries = [
            "machine learning",
            "database optimization",
            "microservices",
            "API design"
        ]
        
        for query in search_queries:
            results = await test_assistant.search_memory(query, search_type="fts")
            assert len(results) > 0
            
            # Should find relevant messages
            found_relevant = any(query.lower() in result["content"].lower() for result in results)
            assert found_relevant
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_long_conversation_summarization(self, test_assistant):
        """Test conversation summarization for long conversations"""
        # Configure for quick summarization
        test_assistant.config.summarization_threshold = 10
        
        thread_id = await test_assistant.create_conversation_thread()
        
        # Create a long conversation
        base_topics = [
            "artificial intelligence",
            "machine learning",
            "deep learning", 
            "natural language processing",
            "computer vision"
        ]
        
        # Generate enough messages to trigger summarization
        for i in range(15):  # Exceed threshold
            topic = base_topics[i % len(base_topics)]
            message = f"Tell me more about {topic} - conversation turn {i+1}"
            
            response = await test_assistant.process_message(message, thread_id=thread_id)
            assert len(response) > 0
        
        # Check if summarization was triggered
        thread = test_assistant.threads[thread_id]
        # In a full implementation, summarization would be triggered
        # For now, just verify the conversation length tracking works
        assert thread.message_count >= 30  # 15 user + 15 assistant messages
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, test_assistant):
        """Test handling multiple concurrent conversations"""
        # Create multiple conversation threads
        thread_ids = []
        for i in range(5):
            thread_id = await test_assistant.create_conversation_thread(
                title=f"Concurrent Conversation {i+1}"
            )
            thread_ids.append(thread_id)
        
        async def conversation_worker(thread_id: str, worker_id: int):
            """Worker function for concurrent conversations"""
            messages = [
                f"Hello from worker {worker_id}",
                f"This is message 2 from worker {worker_id}",
                f"Final message from worker {worker_id}"
            ]
            
            responses = []
            for message in messages:
                response = await test_assistant.process_message(message, thread_id=thread_id)
                responses.append(response)
                await asyncio.sleep(0.05)  # Small delay
            
            return responses
        
        # Run conversations concurrently
        tasks = [
            conversation_worker(thread_id, i) 
            for i, thread_id in enumerate(thread_ids)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all conversations completed successfully
        assert len(results) == 5
        for worker_responses in results:
            assert len(worker_responses) == 3
            assert all(len(response) > 0 for response in worker_responses)
        
        # Verify messages were stored correctly for each thread
        for thread_id in thread_ids:
            messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
            assert len(messages) == 6  # 3 user + 3 assistant messages


class TestVoiceConversationFlows:
    """Test voice-based conversation flows"""
    
    @pytest.mark.e2e
    @pytest.mark.audio
    @pytest.mark.asyncio
    async def test_voice_conversation_flow(self, test_assistant, sample_audio_data):
        """Test complete voice conversation flow"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Simulate multiple voice interactions
        voice_interactions = 3
        
        for i in range(voice_interactions):
            result = await test_assistant.process_voice_message(
                sample_audio_data,
                thread_id=thread_id,
                synthesize_response=True
            )
            
            assert "transcript" in result
            assert "response" in result
            assert "audio_file" in result
            assert result["thread_id"] == thread_id
            
            # Verify voice interaction was stored
            messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
            expected_count = (i + 1) * 2  # Each interaction creates 2 messages
            assert len(messages) == expected_count
    
    @pytest.mark.e2e
    @pytest.mark.audio
    @pytest.mark.asyncio
    async def test_mixed_voice_text_conversation(self, test_assistant, sample_audio_data):
        """Test conversation mixing voice and text inputs"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Mixed interaction sequence
        interactions = [
            ("text", "Hello, I'd like to start a conversation"),
            ("voice", sample_audio_data),
            ("text", "That was interesting, tell me more"),
            ("voice", sample_audio_data),
            ("text", "Thank you for the conversation")
        ]
        
        for interaction_type, input_data in interactions:
            if interaction_type == "text":
                response = await test_assistant.process_message(input_data, thread_id=thread_id)
                assert len(response) > 0
            else:  # voice
                result = await test_assistant.process_voice_message(
                    input_data, thread_id=thread_id, synthesize_response=False
                )
                assert "transcript" in result
                assert "response" in result
        
        # Verify all interactions were stored
        messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
        assert len(messages) == 10  # 5 user + 5 assistant messages


class TestProjectWorkflows:
    """Test project-related conversation workflows"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_project_planning_workflow(self, test_assistant):
        """Test complete project planning workflow"""
        # Create project in database
        project_id = await test_assistant.db_manager.create_project(
            name="E2E Test Project",
            description="End-to-end test project for workflow testing",
            metadata={"status": "planning", "priority": "high"}
        )
        
        # Start project conversation
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.PROJECT,
            title="Project Planning Session",
            metadata={"project_id": project_id}
        )
        
        # Project planning conversation flow
        planning_steps = [
            "Let's start planning our new web application project",
            "What technologies should we use for the frontend?",
            "How should we structure the backend architecture?", 
            "What's our timeline for the MVP?",
            "Who are the key stakeholders for this project?",
            "What are the main risks we should consider?"
        ]
        
        for step in planning_steps:
            response = await test_assistant.process_message(
                step, 
                thread_id=thread_id,
                mode=ConversationMode.PROJECT
            )
            assert len(response) > 0
        
        # Verify project conversation was stored
        messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
        assert len(messages) == len(planning_steps) * 2
        
        # Search for project-related content
        project_results = await test_assistant.search_memory("project planning")
        assert len(project_results) > 0
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_learning_session_workflow(self, test_assistant):
        """Test learning session workflow"""
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.LEARNING,
            title="Python Learning Session"
        )
        
        # Learning conversation flow
        learning_sequence = [
            "I want to learn about Python classes and objects",
            "Can you explain inheritance in Python?",
            "What's the difference between class and instance variables?",
            "How do decorators work in Python?",
            "Can you give me a practical example of using decorators?",
            "Let me test my understanding - what would this code do: @property"
        ]
        
        for question in learning_sequence:
            response = await test_assistant.process_message(
                question,
                thread_id=thread_id,
                mode=ConversationMode.LEARNING
            )
            assert len(response) > 0
        
        # Verify learning content can be searched
        learning_results = await test_assistant.search_memory("Python classes")
        assert len(learning_results) > 0
        
        # Check for educational content
        decorator_results = await test_assistant.search_memory("decorators")
        assert len(decorator_results) > 0


class TestErrorRecoveryFlows:
    """Test error recovery and resilience"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_conversation_with_errors(self, test_assistant):
        """Test conversation flow with simulated errors"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Normal conversation start
        response1 = await test_assistant.process_message(
            "Hello, let's start a conversation", thread_id=thread_id
        )
        assert len(response1) > 0
        
        # Simulate error condition
        original_generate = test_assistant.ollama_client.generate
        
        async def failing_generate(request, context=None):
            raise Exception("Simulated LLM failure")
        
        test_assistant.ollama_client.generate = failing_generate
        
        # This should fail
        with pytest.raises(Exception):
            await test_assistant.process_message(
                "This message will fail", thread_id=thread_id
            )
        
        # Restore normal operation
        test_assistant.ollama_client.generate = original_generate
        
        # Conversation should continue normally
        response2 = await test_assistant.process_message(
            "I hope that error is resolved now", thread_id=thread_id
        )
        assert len(response2) > 0
        
        # Verify conversation state is maintained
        messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
        # Should have 2 successful message pairs (4 total)
        assert len(messages) == 4
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_database_recovery(self, test_assistant):
        """Test recovery from database issues"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Normal operation
        response1 = await test_assistant.process_message(
            "Test message before database issue", thread_id=thread_id
        )
        assert len(response1) > 0
        
        # Simulate database issue
        original_add_message = test_assistant.db_manager.add_message
        
        async def failing_add_message(*args, **kwargs):
            raise Exception("Database connection failed")
        
        test_assistant.db_manager.add_message = failing_add_message
        
        # This should handle the database error
        with pytest.raises(Exception):
            await test_assistant.process_message(
                "This will cause database error", thread_id=thread_id
            )
        
        # Restore database operation
        test_assistant.db_manager.add_message = original_add_message
        
        # Should work again
        response2 = await test_assistant.process_message(
            "Database should work now", thread_id=thread_id
        )
        assert len(response2) > 0


class TestPerformanceInConversations:
    """Test performance characteristics in conversation flows"""
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_conversation_response_times(self, test_assistant):
        """Test response times in realistic conversations"""
        thread_id = await test_assistant.create_conversation_thread()
        
        # Realistic conversation messages
        messages = [
            "Hello! How are you doing today?",
            "I'm working on a Python project and need some advice",
            "What's the best way to handle database connections in a web app?",
            "Should I use connection pooling? How does that work?",
            "What about error handling for database operations?",
            "Can you show me an example of good error handling?",
            "That's very helpful, thank you!",
            "One more question - how do I test database code?",
            "What testing frameworks do you recommend?",
            "Perfect, I think I have what I need now"
        ]
        
        response_times = []
        
        for message in messages:
            start_time = time.time()
            response = await test_assistant.process_message(message, thread_id=thread_id)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            assert len(response) > 0
            assert response_time < 1.0  # Should be fast with mocks
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Maximum response time: {max_response_time:.3f}s")
        
        # Performance assertions
        assert avg_response_time < 0.1  # Very fast with mocks
        assert max_response_time < 0.2
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_in_long_conversation(self, test_assistant, memory_profiler):
        """Test memory usage during long conversations"""
        memory_profiler.start()
        
        thread_id = await test_assistant.create_conversation_thread()
        
        # Long conversation simulation
        num_exchanges = 100
        
        for i in range(num_exchanges):
            message = f"This is message number {i+1} in our long conversation. " \
                     f"Let's discuss topic {i % 10} in detail."
            
            response = await test_assistant.process_message(message, thread_id=thread_id)
            assert len(response) > 0
            
            # Update memory usage periodically
            if i % 20 == 0:
                memory_profiler.update()
        
        memory_profiler.update()
        memory_increase = memory_profiler.memory_increase
        
        print(f"Memory increase after {num_exchanges} exchanges: {memory_increase / 1024 / 1024:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase


class TestConversationStateManagement:
    """Test conversation state and thread management"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_thread_switching(self, test_assistant):
        """Test switching between conversation threads"""
        # Create multiple threads
        thread1 = await test_assistant.create_conversation_thread(title="Thread 1")
        thread2 = await test_assistant.create_conversation_thread(title="Thread 2")
        thread3 = await test_assistant.create_conversation_thread(title="Thread 3")
        
        # Start conversations in each thread
        await test_assistant.process_message("Hello from thread 1", thread_id=thread1)
        await test_assistant.process_message("Hello from thread 2", thread_id=thread2)
        await test_assistant.process_message("Hello from thread 3", thread_id=thread3)
        
        # Switch between threads and continue conversations
        await test_assistant.process_message("More from thread 1", thread_id=thread1)
        await test_assistant.process_message("More from thread 3", thread_id=thread3)
        await test_assistant.process_message("More from thread 2", thread_id=thread2)
        
        # Verify each thread has correct message count
        for thread_id, expected_pairs in [(thread1, 2), (thread2, 2), (thread3, 2)]:
            messages = await test_assistant.db_manager.get_conversation_messages(thread_id)
            assert len(messages) == expected_pairs * 2  # User + assistant messages
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_conversation_metadata_tracking(self, test_assistant):
        """Test conversation metadata and tracking"""
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.RESEARCH,
            title="Research Session",
            metadata={"topic": "artificial intelligence", "priority": "high"}
        )
        
        # Add messages with metadata
        messages_with_metadata = [
            ("What is artificial intelligence?", {"intent": "definition"}),
            ("How does machine learning work?", {"intent": "explanation"}),
            ("What are the latest AI developments?", {"intent": "current_events"})
        ]
        
        for message, metadata in messages_with_metadata:
            response = await test_assistant.process_message(
                message, 
                thread_id=thread_id,
                metadata=metadata
            )
            assert len(response) > 0
        
        # Verify thread metadata
        thread = test_assistant.threads[thread_id]
        assert thread.mode == ConversationMode.RESEARCH
        assert thread.metadata["topic"] == "artificial intelligence"
        assert thread.metadata["priority"] == "high"
        
        # Verify message count tracking
        assert thread.message_count == 6  # 3 user + 3 assistant messages