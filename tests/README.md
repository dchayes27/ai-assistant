# AI Assistant Test Suite

Comprehensive testing framework for the AI Assistant project, including unit tests, integration tests, performance benchmarks, and end-to-end testing.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── pytest.ini              # Pytest settings and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_database.py     # Database operations testing
│   ├── test_llm_mocks.py    # LLM interaction mocking and testing
│   └── test_audio.py        # Audio processing testing
├── integration/             # Integration tests for services
│   └── test_mcp_server.py   # MCP server API testing
├── performance/             # Performance benchmarks
│   └── test_benchmarks.py   # Performance and scalability tests
└── e2e/                     # End-to-end testing
    └── test_conversations.py # Complete conversation flow testing
```

## Test Categories

### 🧪 Unit Tests
- **Database Operations**: Test DatabaseManager and VectorStore functionality
- **LLM Mocking**: Test language model interactions with mocked responses
- **Audio Processing**: Test speech-to-text and text-to-speech components

### 🔗 Integration Tests
- **MCP Server**: Test FastAPI endpoints, WebSocket functionality, and API integration
- **Authentication**: Test security middleware and API key validation
- **Error Handling**: Test error responses and recovery mechanisms

### ⚡ Performance Tests
- **Database Performance**: Benchmark database operations and scalability
- **Memory Usage**: Monitor memory consumption and leak detection
- **Concurrent Operations**: Test multi-threaded and async performance

### 🎯 End-to-End Tests
- **Conversation Flows**: Test complete conversation scenarios
- **Voice Interactions**: Test audio input/output pipelines
- **Multi-Modal**: Test mixed text and voice conversations

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the AI Assistant components are properly set up:
```bash
# Run the installation script
./install_dependencies.sh
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m performance             # Performance benchmarks
pytest -m e2e                     # End-to-end tests

# Run specific test files
pytest tests/unit/test_database.py
pytest tests/integration/test_mcp_server.py
```

### Test Options

```bash
# Run with coverage report
pytest --cov=memory --cov=core --cov=mcp_server --cov=gui

# Run only fast tests (exclude slow/performance tests)
pytest -m "not slow and not performance"

# Run with detailed output
pytest -v --tb=long

# Run specific test method
pytest tests/unit/test_database.py::TestDatabaseManager::test_conversation_creation

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Conditional Test Execution

Some tests have conditional execution based on available services:

```bash
# Skip tests requiring Ollama service
pytest -m "not requires_ollama"

# Skip tests requiring audio hardware
pytest -m "not requires_audio"

# Run only tests that don't require external services
pytest -m "not requires_ollama and not requires_audio"
```

**Automatic Skipping**:
- `@pytest.mark.requires_ollama` tests are skipped if Ollama is not running on localhost:11434
- `@pytest.mark.requires_audio` tests are skipped if PyAudio/audio hardware is unavailable
- Use markers to explicitly include/exclude these tests in CI/CD environments

## Test Configuration

### Markers

The test suite uses pytest markers to categorize tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests across services
- `performance`: Performance benchmarks and scalability tests
- `slow`: Tests that take longer than 5 seconds
- `audio`: Tests requiring audio processing capabilities
- `e2e`: End-to-end workflow tests
- `requires_ollama`: Tests requiring Ollama service
- `requires_audio`: Tests requiring audio hardware/libraries

### Fixtures

Key fixtures available for testing:

#### Database Fixtures
- `test_db_manager`: Clean database manager for testing
- `populated_db_manager`: Database with sample test data
- `test_vector_store`: Vector store with test embeddings

#### Mock Fixtures
- `mock_ollama_client`: Mocked LLM client with predictable responses
- `mock_whisper_model`: Mocked speech-to-text model
- `mock_tts_engine`: Mocked text-to-speech engine

#### Audio Fixtures
- `sample_audio_data`: Generated audio data for testing
- `sample_wav_file`: WAV file with test audio

#### Assistant Fixtures
- `test_assistant`: Fully configured assistant with mocked components
- `test_assistant_config`: Test configuration for assistant

#### Performance Fixtures
- `performance_timer`: Timer for measuring execution time
- `memory_profiler`: Memory usage profiler

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
import pytest
from memory import DatabaseManager

class TestDatabaseOperations:
    """Test database functionality"""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_storage(self, test_db_manager):
        """Test storing and retrieving messages"""
        # Arrange
        conv_id = await test_db_manager.create_conversation("test", "Test Conv")
        
        # Act
        msg_id = await test_db_manager.add_message(
            conv_id, "user", "Hello world"
        )
        
        # Assert
        assert msg_id is not None
        messages = await test_db_manager.get_conversation_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello world"
```

### Async Test Guidelines

Most tests are async due to the async nature of the AI Assistant:

```python
@pytest.mark.asyncio
async def test_async_operation(self, test_assistant):
    """Test async operations"""
    result = await test_assistant.process_message("Hello")
    assert result is not None
```

### Mock Usage

Use the provided mock fixtures for predictable testing:

```python
async def test_llm_interaction(self, mock_ollama_client):
    """Test LLM interaction with mocked responses"""
    request = QueryRequest(message="Test", model="test-model")
    response = await mock_ollama_client.generate(request)
    
    assert response.response == "This is a mock response"
    mock_ollama_client.generate.assert_called_once()
```

## Performance Testing

### Benchmarking Guidelines

Performance tests measure:
- **Response Times**: How fast operations complete
- **Throughput**: Operations per second
- **Memory Usage**: Memory consumption patterns
- **Scalability**: Performance with increasing load

Example performance test:

```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_message_insertion_performance(self, test_db_manager, performance_timer):
    """Benchmark message insertion rate"""
    conv_id = await test_db_manager.create_conversation("perf", "Performance Test")
    
    num_messages = 1000
    performance_timer.start()
    
    for i in range(num_messages):
        await test_db_manager.add_message(conv_id, "user", f"Message {i}")
    
    performance_timer.stop()
    
    # Verify performance
    messages_per_second = num_messages / performance_timer.elapsed
    assert messages_per_second > 100  # At least 100 messages/second
```

### Performance Assertions

Performance tests include assertions for:
- Maximum execution time
- Minimum throughput rates
- Memory usage limits
- Resource utilization bounds

## Continuous Integration

### GitHub Actions

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest -m "not requires_ollama and not requires_audio" --cov
```

### Test Reports

Generate various test reports:

```bash
# HTML coverage report
pytest --cov --cov-report=html

# XML test results for CI
pytest --junitxml=test-results.xml

# Performance benchmark results
pytest -m performance --benchmark-json=benchmark.json
```

## Troubleshooting

### Common Issues

1. **Database Lock Errors**
   ```bash
   # Use separate test databases
   pytest --db-isolation
   ```

2. **Async Test Failures**
   ```bash
   # Ensure proper event loop handling
   pytest --asyncio-mode=auto
   ```

3. **Memory Leaks in Long Tests**
   ```bash
   # Run with memory monitoring
   pytest -m performance --memray
   ```

4. **Flaky Tests**
   ```bash
   # Run tests multiple times
   pytest --count=5 tests/integration/
   ```

### Debug Mode

Run tests in debug mode for detailed information:

```bash
# Maximum verbosity
pytest -vvv --tb=long --capture=no

# Debug specific test
pytest -vvv --pdb tests/unit/test_database.py::test_specific_function
```

## Contributing

### Adding New Tests

1. **Choose the appropriate test category** (unit/integration/performance/e2e)
2. **Use existing fixtures** when possible
3. **Follow naming conventions** and add appropriate markers
4. **Include docstrings** explaining what the test validates
5. **Add performance assertions** for performance tests
6. **Mock external dependencies** in unit tests

### Test Quality Guidelines

- **Isolation**: Tests should not depend on each other
- **Repeatability**: Tests should produce consistent results
- **Speed**: Unit tests should be fast (<1s), integration tests moderate (<10s)
- **Coverage**: Aim for >90% code coverage in critical components
- **Documentation**: Clear test names and docstrings

### Code Review Checklist

- [ ] Tests cover both happy path and error conditions
- [ ] Appropriate markers are applied
- [ ] Mocks are used properly for external dependencies
- [ ] Performance tests include meaningful assertions
- [ ] Tests are properly isolated and don't leak state
- [ ] Documentation is clear and complete

## Test Data Management

### Test Database

Tests use isolated SQLite databases:
- Created fresh for each test
- Populated with consistent test data
- Cleaned up automatically after tests

### Sample Data

Standard test data includes:
- Sample conversations with varied content
- Audio files for speech testing
- Project and knowledge base entries
- User interaction patterns

### Test Environment

Set up test environment variables:

```bash
export TEST_MODE=true
export LOG_LEVEL=DEBUG
export DATABASE_URL=":memory:"
```

This comprehensive test suite ensures the AI Assistant is robust, performant, and reliable across all its components and use cases.