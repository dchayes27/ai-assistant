"""
Performance benchmarks for AI Assistant components
Tests response times, memory usage, and throughput
"""

import pytest
import asyncio
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import statistics

from memory import DatabaseManager, VectorStore
from core import SmartAssistant, ConversationMode, AssistantConfig


class TestDatabasePerformance:
    """Performance benchmarks for database operations"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_message_insertion_performance(self, test_db_manager, performance_timer):
        """Benchmark message insertion performance"""
        # Create test conversation
        conv_id = await test_db_manager.create_conversation(
            "perf-test-conv", "Performance Test"
        )
        
        num_messages = 1000
        performance_timer.start()
        
        # Insert messages
        for i in range(num_messages):
            await test_db_manager.add_message(
                conv_id, 
                "user" if i % 2 == 0 else "assistant",
                f"Performance test message {i}",
                metadata={"index": i}
            )
        
        performance_timer.stop()
        
        # Verify performance
        elapsed_time = performance_timer.elapsed
        messages_per_second = num_messages / elapsed_time
        
        print(f"Inserted {num_messages} messages in {elapsed_time:.2f}s")
        print(f"Rate: {messages_per_second:.1f} messages/second")
        
        # Performance assertions
        assert elapsed_time < 30.0  # Should complete within 30 seconds
        assert messages_per_second > 10  # At least 10 messages per second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_performance(self, populated_db_manager, performance_timer):
        """Benchmark search performance"""
        # Add more test data for meaningful search
        conv_id = await populated_db_manager.create_conversation(
            "search-perf-conv", "Search Performance Test"
        )
        
        # Add varied content for search
        test_phrases = [
            "machine learning algorithms",
            "artificial intelligence research",
            "natural language processing",
            "computer vision techniques",
            "deep learning networks",
            "data science methodology",
            "software engineering practices",
            "database optimization strategies"
        ]
        
        for i, phrase in enumerate(test_phrases * 25):  # 200 messages
            await populated_db_manager.add_message(
                conv_id, "user", f"{phrase} - message {i}", metadata={"test": True}
            )
        
        # Benchmark search operations
        search_queries = [
            "machine learning",
            "artificial intelligence", 
            "natural language",
            "computer vision",
            "deep learning"
        ]
        
        search_times = []
        
        for query in search_queries:
            performance_timer.start()
            results = await populated_db_manager.search_messages(query, limit=50)
            performance_timer.stop()
            
            search_times.append(performance_timer.elapsed)
            assert len(results) > 0  # Should find results
        
        avg_search_time = statistics.mean(search_times)
        max_search_time = max(search_times)
        
        print(f"Average search time: {avg_search_time:.3f}s")
        print(f"Maximum search time: {max_search_time:.3f}s")
        
        # Performance assertions
        assert avg_search_time < 1.0  # Average search under 1 second
        assert max_search_time < 2.0  # No search over 2 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, test_db_manager):
        """Benchmark concurrent database operations"""
        conv_id = await test_db_manager.create_conversation(
            "concurrent-test", "Concurrent Test"
        )
        
        async def insert_messages(start_idx: int, count: int):
            """Insert messages concurrently"""
            for i in range(start_idx, start_idx + count):
                await test_db_manager.add_message(
                    conv_id, "user", f"Concurrent message {i}"
                )
        
        # Create concurrent tasks
        num_tasks = 10
        messages_per_task = 50
        start_time = time.time()
        
        tasks = [
            insert_messages(i * messages_per_task, messages_per_task)
            for i in range(num_tasks)
        ]
        
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_messages = num_tasks * messages_per_task
        throughput = total_messages / elapsed_time
        
        print(f"Concurrent insertion: {total_messages} messages in {elapsed_time:.2f}s")
        print(f"Throughput: {throughput:.1f} messages/second")
        
        # Verify all messages were inserted
        messages = await test_db_manager.get_conversation_messages(conv_id)
        assert len(messages) == total_messages
        
        # Performance assertions
        assert throughput > 50  # At least 50 messages/second with concurrency
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_growth(self, test_db_manager, memory_profiler):
        """Test memory usage growth with large datasets"""
        memory_profiler.start()
        
        # Create multiple conversations with messages
        num_conversations = 50
        messages_per_conv = 100
        
        for conv_idx in range(num_conversations):
            conv_id = await test_db_manager.create_conversation(
                f"memory-test-{conv_idx}", f"Memory Test {conv_idx}"
            )
            
            for msg_idx in range(messages_per_conv):
                await test_db_manager.add_message(
                    conv_id, "user", f"Memory test message {msg_idx} in conv {conv_idx}"
                )
            
            # Update memory profiler periodically
            if conv_idx % 10 == 0:
                memory_profiler.update()
        
        memory_profiler.update()
        memory_increase = memory_profiler.memory_increase
        
        print(f"Memory increase: {memory_increase / 1024 / 1024:.1f} MB")
        print(f"Total messages: {num_conversations * messages_per_conv}")
        
        # Memory usage should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase


class TestVectorStorePerformance:
    """Performance benchmarks for vector operations"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_embedding_storage_performance(self, test_vector_store, performance_timer):
        """Benchmark embedding storage performance"""
        num_embeddings = 1000
        embedding_dim = 768
        
        performance_timer.start()
        
        # Store embeddings
        for i in range(num_embeddings):
            embedding = [0.1 * (i % 10)] * embedding_dim
            await test_vector_store.store_embedding(
                "message", f"perf-msg-{i}", embedding, {"index": i}
            )
        
        performance_timer.stop()
        
        elapsed_time = performance_timer.elapsed
        embeddings_per_second = num_embeddings / elapsed_time
        
        print(f"Stored {num_embeddings} embeddings in {elapsed_time:.2f}s")
        print(f"Rate: {embeddings_per_second:.1f} embeddings/second")
        
        # Performance assertions
        assert embeddings_per_second > 100  # At least 100 embeddings/second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, test_vector_store, performance_timer):
        """Benchmark similarity search performance"""
        # First, populate with test embeddings
        num_embeddings = 5000
        embedding_dim = 768
        
        # Create diverse embeddings for meaningful search
        for i in range(num_embeddings):
            # Create embeddings with different patterns
            embedding = []
            for j in range(embedding_dim):
                val = 0.1 * ((i + j) % 20) / 20.0
                embedding.append(val)
            
            await test_vector_store.store_embedding(
                "message", f"search-msg-{i}", embedding, 
                {"content": f"test message {i}", "category": i % 10}
            )
        
        # Benchmark search operations
        search_times = []
        result_counts = []
        
        for _ in range(20):  # Multiple search queries
            query_embedding = [0.05 * (i % 10) for i in range(embedding_dim)]
            
            performance_timer.start()
            results = await test_vector_store.search_similar_embeddings(
                query_embedding, limit=100
            )
            performance_timer.stop()
            
            search_times.append(performance_timer.elapsed)
            result_counts.append(len(results))
        
        avg_search_time = statistics.mean(search_times)
        avg_results = statistics.mean(result_counts)
        
        print(f"Average search time: {avg_search_time:.3f}s")
        print(f"Average results returned: {avg_results:.1f}")
        print(f"Search rate: {1/avg_search_time:.1f} searches/second")
        
        # Performance assertions
        assert avg_search_time < 0.5  # Under 500ms per search
        assert avg_results > 50  # Should return meaningful results
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, test_vector_store, performance_timer):
        """Benchmark batch embedding operations"""
        batch_size = 500
        embedding_dim = 768
        
        # Prepare batch data
        batch_data = []
        for i in range(batch_size):
            embedding = [0.01 * (i % 100)] * embedding_dim
            batch_data.append((
                "message", f"batch-msg-{i}", embedding, {"batch_index": i}
            ))
        
        # Benchmark batch storage
        performance_timer.start()
        success = await test_vector_store.store_embeddings_batch(batch_data)
        performance_timer.stop()
        
        batch_time = performance_timer.elapsed
        batch_rate = batch_size / batch_time
        
        print(f"Batch storage: {batch_size} embeddings in {batch_time:.2f}s")
        print(f"Batch rate: {batch_rate:.1f} embeddings/second")
        
        assert success is True
        assert batch_rate > 200  # Batch should be faster than individual ops


class TestAssistantPerformance:
    """Performance benchmarks for Smart Assistant"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_message_processing_performance(self, test_assistant, performance_timer):
        """Benchmark message processing performance"""
        # Create conversation thread
        thread_id = await test_assistant.create_conversation_thread(
            mode=ConversationMode.CHAT
        )
        
        num_messages = 50
        response_times = []
        
        for i in range(num_messages):
            message = f"Performance test message {i}: How are you doing today?"
            
            performance_timer.start()
            response = await test_assistant.process_message(message, thread_id=thread_id)
            performance_timer.stop()
            
            response_times.append(performance_timer.elapsed)
            assert len(response) > 0  # Should get a response
        
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Min response time: {min_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")
        print(f"Messages per minute: {60 / avg_response_time:.1f}")
        
        # Performance assertions (with mocked LLM, should be fast)
        assert avg_response_time < 0.1  # Under 100ms with mocks
        assert max_response_time < 0.5  # No response over 500ms
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_conversations(self, test_assistant):
        """Benchmark concurrent conversation handling"""
        num_threads = 10
        messages_per_thread = 20
        
        async def conversation_thread(thread_idx: int):
            """Simulate a conversation thread"""
            thread_id = await test_assistant.create_conversation_thread(
                mode=ConversationMode.CHAT,
                title=f"Concurrent Thread {thread_idx}"
            )
            
            response_times = []
            for msg_idx in range(messages_per_thread):
                message = f"Thread {thread_idx}, message {msg_idx}"
                
                start_time = time.time()
                await test_assistant.process_message(message, thread_id=thread_id)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
            
            return response_times
        
        # Run concurrent conversations
        start_time = time.time()
        tasks = [conversation_thread(i) for i in range(num_threads)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_messages = num_threads * messages_per_thread
        throughput = total_messages / total_time
        
        # Flatten all response times
        all_response_times = [time for thread_times in results for time in thread_times]
        avg_response_time = statistics.mean(all_response_times)
        
        print(f"Concurrent processing: {total_messages} messages in {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} messages/second")
        print(f"Average response time: {avg_response_time:.3f}s")
        
        # Performance assertions
        assert throughput > 50  # At least 50 messages/second concurrently
        assert avg_response_time < 0.2  # Under 200ms average
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_search_performance(self, test_assistant, performance_timer):
        """Benchmark memory search performance"""
        # Create conversation with varied content
        thread_id = await test_assistant.create_conversation_thread()
        
        test_topics = [
            "machine learning algorithms and their applications",
            "artificial intelligence in modern software development", 
            "natural language processing techniques for chatbots",
            "computer vision applications in autonomous vehicles",
            "deep learning frameworks and neural network architectures",
            "data science methodologies for business intelligence",
            "software engineering best practices and design patterns",
            "database optimization strategies for high-traffic applications"
        ]
        
        # Populate with messages
        for i, topic in enumerate(test_topics * 10):  # 80 messages
            await test_assistant.process_message(
                f"Tell me about {topic} - iteration {i}",
                thread_id=thread_id
            )
        
        # Benchmark search operations
        search_queries = [
            "machine learning",
            "artificial intelligence",
            "natural language processing",
            "computer vision",
            "deep learning"
        ]
        
        search_times = []
        
        for query in search_queries:
            performance_timer.start()
            results = await test_assistant.search_memory(
                query, search_type="hybrid", limit=20
            )
            performance_timer.stop()
            
            search_times.append(performance_timer.elapsed)
            assert len(results) > 0  # Should find results
        
        avg_search_time = statistics.mean(search_times)
        
        print(f"Memory search average time: {avg_search_time:.3f}s")
        print(f"Search rate: {1/avg_search_time:.1f} searches/second")
        
        # Performance assertions
        assert avg_search_time < 2.0  # Under 2 seconds per search


class TestResourceUtilization:
    """Test resource utilization and efficiency"""
    
    @pytest.mark.performance
    def test_cpu_utilization(self):
        """Monitor CPU utilization during operations"""
        process = psutil.Process()
        
        # Baseline CPU usage
        baseline_cpu = process.cpu_percent(interval=0.1)
        
        # Simulate CPU-intensive operation
        start_time = time.time()
        while time.time() - start_time < 1.0:
            # Simulate work
            sum(i * i for i in range(1000))
        
        # Measure CPU usage
        cpu_usage = process.cpu_percent(interval=0.1)
        
        print(f"Baseline CPU: {baseline_cpu:.1f}%")
        print(f"Active CPU: {cpu_usage:.1f}%")
        
        # CPU usage should be reasonable
        assert cpu_usage < 90.0  # Should not max out CPU
    
    @pytest.mark.performance
    def test_memory_efficiency(self, memory_profiler):
        """Test memory efficiency of operations"""
        memory_profiler.start()
        
        # Simulate memory-intensive operations
        large_data = []
        for i in range(10000):
            large_data.append({
                "id": i,
                "data": f"test data {i}" * 10,
                "timestamp": time.time()
            })
        
        memory_profiler.update()
        
        # Clean up
        del large_data
        
        memory_increase = memory_profiler.memory_increase
        print(f"Memory increase: {memory_increase / 1024 / 1024:.1f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_efficiency(self, performance_timer):
        """Test efficiency of async operations"""
        async def async_task(task_id: int, duration: float):
            """Simulate async work"""
            await asyncio.sleep(duration)
            return f"Task {task_id} completed"
        
        # Create multiple async tasks
        num_tasks = 20
        task_duration = 0.1
        
        performance_timer.start()
        
        # Run tasks concurrently
        tasks = [async_task(i, task_duration) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)
        
        performance_timer.stop()
        
        elapsed_time = performance_timer.elapsed
        expected_sequential_time = num_tasks * task_duration
        efficiency = expected_sequential_time / elapsed_time
        
        print(f"Async execution time: {elapsed_time:.2f}s")
        print(f"Sequential time would be: {expected_sequential_time:.2f}s")
        print(f"Efficiency gain: {efficiency:.1f}x")
        
        assert len(results) == num_tasks
        assert efficiency > 10  # Should be much faster than sequential


class TestScalabilityBenchmarks:
    """Test scalability with increasing loads"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_database_scalability(self, test_db_manager):
        """Test database performance with increasing data sizes"""
        conv_id = await test_db_manager.create_conversation(
            "scale-test", "Scalability Test"
        )
        
        # Test with increasing numbers of messages
        test_sizes = [100, 500, 1000, 2000]
        insertion_rates = []
        search_times = []
        
        cumulative_messages = 0
        
        for size in test_sizes:
            # Insert messages
            start_time = time.time()
            for i in range(size):
                await test_db_manager.add_message(
                    conv_id, "user", f"Scalability test message {cumulative_messages + i}"
                )
            end_time = time.time()
            
            cumulative_messages += size
            insertion_time = end_time - start_time
            insertion_rate = size / insertion_time
            insertion_rates.append(insertion_rate)
            
            # Test search performance
            search_start = time.time()
            results = await test_db_manager.search_messages("scalability")
            search_end = time.time()
            
            search_time = search_end - search_start
            search_times.append(search_time)
            
            print(f"Size {cumulative_messages}: "
                  f"Insert rate {insertion_rate:.1f}/s, "
                  f"Search time {search_time:.3f}s")
        
        # Performance should not degrade significantly
        for i in range(1, len(insertion_rates)):
            rate_ratio = insertion_rates[i] / insertion_rates[0]
            assert rate_ratio > 0.5  # Should maintain at least 50% of initial rate
        
        # Search times should remain reasonable
        assert all(t < 2.0 for t in search_times)
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_running_stability(self, test_assistant):
        """Test stability over extended operation"""
        # This test runs for longer to check for memory leaks, etc.
        thread_id = await test_assistant.create_conversation_thread()
        
        start_memory = psutil.Process().memory_info().rss
        
        # Run for an extended period
        num_iterations = 200
        for i in range(num_iterations):
            message = f"Long running test iteration {i}"
            response = await test_assistant.process_message(message, thread_id=thread_id)
            
            assert len(response) > 0
            
            # Check memory usage periodically
            if i % 50 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_increase = current_memory - start_memory
                
                print(f"Iteration {i}: Memory increase {memory_increase / 1024 / 1024:.1f} MB")
                
                # Memory should not grow excessively
                assert memory_increase < 100 * 1024 * 1024  # Less than 100MB growth
        
        final_memory = psutil.Process().memory_info().rss
        total_memory_increase = final_memory - start_memory
        
        print(f"Total memory increase: {total_memory_increase / 1024 / 1024:.1f} MB")
        
        # Final memory check
        assert total_memory_increase < 200 * 1024 * 1024  # Less than 200MB total