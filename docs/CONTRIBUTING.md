# Contributing to AI Assistant

Thank you for your interest in contributing! This guide will help you get started.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Getting Help](#getting-help)

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-assistant.git
cd ai-assistant

# Add upstream remote
git remote add upstream https://github.com/dchayes27/ai-assistant.git
```

### 2. Create Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
./install_dependencies.sh

# Install development dependencies
pip install -r requirements.txt
pip install pre-commit pytest-cov black flake8 mypy
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run linting and formatting checks before each commit.

### 4. Branch Naming Conventions

Use descriptive branch names with prefixes:

- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

**Examples:**
- `feature/websocket-streaming`
- `bugfix/database-connection-leak`
- `docs/api-documentation`

---

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black (automatic formatting)
- **Linter**: flake8
- **Type checker**: mypy

### Code Formatting

```bash
# Format code with Black
python -m black .

# Check formatting without changes
python -m black --check .
```

### Linting

```bash
# Run flake8
python -m flake8 core/ memory/ mcp_server/ gui/

# Common issues to avoid:
# - Unused imports
# - Undefined variables
# - Line too long (>100 chars)
# - Missing whitespace
```

### Type Hints

Type hints are **required** for all public APIs:

```python
# Good
def process_message(message: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Process a user message with optional context.

    Args:
        message: User message string
        context: Optional conversation context

    Returns:
        Processing result dictionary
    """
    pass

# Bad - missing type hints
def process_message(message, context=None):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def add_message(self, conversation_id: str, role: str, content: str) -> int:
    """Add a message to a conversation.

    Args:
        conversation_id: Unique conversation identifier
        role: Message role ('user' or 'assistant')
        content: Message content text

    Returns:
        The ID of the newly created message

    Raises:
        ValueError: If conversation_id doesn't exist
        DatabaseError: If database operation fails

    Example:
        >>> db = DatabaseManager()
        >>> msg_id = db.add_message("conv_123", "user", "Hello")
        >>> print(msg_id)
        42
    """
    pass
```

### Async/Await Patterns

- Use `async def` for I/O-bound operations
- Use `await` for async calls
- Don't block the event loop with sync operations
- Use `asyncio.to_thread()` for CPU-bound work

```python
# Good
async def fetch_data(self, url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Bad - blocking call in async function
async def fetch_data(self, url: str) -> Dict[str, Any]:
    response = requests.get(url)  # Blocks event loop!
    return response.json()
```

---

## Testing Requirements

### Test Coverage

- **New features**: Must include unit tests
- **Bug fixes**: Must include regression test
- **API changes**: Must include integration tests
- **Performance work**: Must include performance tests
- **Coverage**: Must not decrease overall coverage

### Running Tests

```bash
# All tests
./run_tests.sh

# Specific test types
./run_tests.sh --unit              # Fast unit tests
./run_tests.sh --integration       # Integration tests (needs Ollama)
./run_tests.sh --performance       # Performance benchmarks

# With coverage
./run_tests.sh --coverage
```

### Writing Tests

```python
import pytest
from memory import DatabaseManager

class TestDatabaseOperations:
    """Test database functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_add_message(self, test_db_manager):
        """Test adding a message to conversation."""
        # Arrange
        conv_id = await test_db_manager.create_conversation("test", "Test")

        # Act
        msg_id = await test_db_manager.add_message(conv_id, "user", "Hello")

        # Assert
        assert msg_id is not None
        messages = await test_db_manager.get_conversation_messages(conv_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello"
```

### Test Markers

Use appropriate pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.requires_ollama` - Needs Ollama running
- `@pytest.mark.requires_audio` - Needs audio hardware

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature main
```

### 2. Make Changes

- Write code following style guide
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass locally

### 3. Commit Changes

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
git commit -m "feat: add websocket streaming support"
git commit -m "fix: resolve database connection leak"
git commit -m "docs: update API documentation"
git commit -m "test: add integration tests for streaming"
```

**Commit Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, no logic change)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub with:

**Title**: Clear, descriptive title (use conventional commit format)

**Description Template:**
```markdown
## Summary
Brief description of changes

## Changes Made
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] CHANGELOG.md updated (if user-facing change)
```

### 5. Review Process

- CI/CD checks must pass (tests, linting, type checking)
- At least one approval required
- Address reviewer feedback
- Keep PR focused and reasonably sized (<500 lines preferred)

### 6. After Merge

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Delete feature branch
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

---

## Adding New Features

### Adding New MCP Tools

1. Create tool implementation in `mcp_server/tools.py`:

```python
class MyNewTool:
    """My new tool functionality."""

    async def execute(self, **params) -> Dict[str, Any]:
        """Execute the tool."""
        # Implementation
        pass
```

2. Register tool in tool registry
3. Add tests in `tests/unit/test_tools.py`
4. Document tool in API docs

### Adding New Conversation Modes

1. Add mode to `ConversationMode` enum in `core/smart_assistant.py`
2. Add system prompt in `config/prompt_templates.yaml`
3. Update GUI to include new mode
4. Add tests for mode-specific behavior

### Adding New TTS Providers

1. Create provider class in `core/` or `mcp_server/`
2. Implement TTS interface
3. Add provider to configuration options
4. Add fallback logic if provider fails
5. Test provider with various inputs

### Database Schema Changes

1. Create migration in `memory/migrations.py`:

```python
async def migration_003_add_new_table():
    """Add new feature table."""
    # Migration logic
    pass
```

2. Update models in `memory/models.py`
3. Test migration on clean database
4. Test migration on existing database with data
5. Document breaking changes if any

---

## Getting Help

### Resources

- **Documentation**: See [docs/](../docs/) directory
- **Issues**: Check [GitHub Issues](https://github.com/dchayes27/ai-assistant/issues)
- **Discussions**: Use GitHub Discussions for questions

### Asking Questions

When asking for help:

1. Search existing issues first
2. Provide clear description of problem
3. Include relevant code snippets
4. Share error messages and logs
5. Specify your environment (OS, Python version, etc.)

### Reporting Bugs

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g. Ubuntu 22.04]
 - Python Version: [e.g. 3.11]
 - AI Assistant Version: [e.g. 1.0.0]

**Additional context**
Any other context about the problem.
```

---

## Code Review Guidelines

When reviewing PRs:

- **Be constructive**: Suggest improvements, don't just criticize
- **Be specific**: Point to exact lines and explain why
- **Be timely**: Review within 2-3 business days if possible
- **Test locally**: Check out the branch and test manually
- **Check tests**: Ensure new tests actually test the feature
- **Consider maintainability**: Will this be easy to maintain?

---

## Recognition

Contributors will be:
- Listed in CHANGELOG.md for their contributions
- Mentioned in release notes
- Added to a contributors list (coming soon)

---

Thank you for contributing to AI Assistant! 🎉
