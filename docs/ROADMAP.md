# AI Assistant Improvement Roadmap

**Last Updated**: 2025-10-29
**Purpose**: Track code improvements, security fixes, and feature enhancements

---

## 🚨 Critical (Fix Immediately)

### 1. Hardcoded Development API Key Exposure ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: Development API key `"dev-key-12345"` is hardcoded and logged in plaintext, creating a security vulnerability if accidentally deployed to production.

**Files**:
- `mcp_server/auth.py:54-61`

**What to Do**:
1. Remove hardcoded default key
2. Require explicit API key configuration in development
3. Add warning if running without proper auth in non-dev environment
4. Never log API keys, even in development

**Code Change**:
```python
# Before (BAD):
if not api_keys and os.getenv("ENVIRONMENT", "development") == "development":
    default_key = "dev-key-12345"
    api_keys[default_key] = {...}
    logger.warning(f"Using default development API key: {default_key}")  # SECURITY RISK!

# After (GOOD):
if not api_keys:
    if os.getenv("ENVIRONMENT") == "development":
        logger.warning("No API keys configured. Set API_KEY_1 environment variable.")
    else:
        raise ValueError("API keys must be configured for production")
```

**Time**: 2 hours (includes testing)

---

### 2. JWT Secret Key Regeneration on Restart ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: JWT secret key auto-generates if `JWT_SECRET_KEY` env var not set. This invalidates all issued tokens on server restart, breaking sessions.

**Files**:
- `mcp_server/auth.py:23, 32-34`

**What to Do**:
1. Require `JWT_SECRET_KEY` to be explicitly set
2. Fail fast at startup if not configured in production
3. Add migration guide for existing deployments
4. Add validation that secret is sufficiently complex

**Code Change**:
```python
def __init__(self):
    self.secret_key = os.getenv("JWT_SECRET_KEY")
    if not self.secret_key:
        if os.getenv("ENVIRONMENT") == "production":
            raise ValueError("JWT_SECRET_KEY must be set in production")
        logger.error("JWT_SECRET_KEY not set. Tokens will be invalidated on restart!")
        self.secret_key = self._generate_secret_key()
    elif len(self.secret_key) < 32:
        raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
```

**Time**: 2 hours (includes documentation updates)

---

### 3. Missing scikit-learn Dependency ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: `vector_store.py:388-396` uses sklearn for clustering but it's not in `requirements.txt`. This will cause runtime errors.

**Files**:
- `memory/vector_store.py:388-396`
- `requirements.txt`

**What to Do**:
1. Add `scikit-learn>=1.3.0` to requirements.txt
2. Add try/except import with clear error message
3. Make clustering an optional feature that gracefully degrades
4. Document which features require sklearn

**Code Change**:
```python
# At top of vector_store.py
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Clustering features disabled.")

# In cluster_entities method
def cluster_entities(self, ...):
    if not SKLEARN_AVAILABLE:
        raise ValueError("Clustering requires scikit-learn: pip install scikit-learn")
    # ... rest of implementation
```

**Time**: 1 hour

---

### 4. No Input Validation on Embedding Imports ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: `vector_store.py:565-610` imports embeddings from JSON without validating dimensions or format. Malicious/corrupted files could crash the system or corrupt the database.

**Files**:
- `memory/vector_store.py:565-610`

**What to Do**:
1. Validate JSON schema before import
2. Check embedding dimensions match expected
3. Verify entity IDs exist in database
4. Add dry-run mode to preview imports
5. Wrap imports in transaction for rollback

**Code Change**:
```python
def import_embeddings(self, input_file: str, overwrite: bool = False, dry_run: bool = False) -> int:
    with open(input_file, 'r') as f:
        embeddings_data = json.load(f)

    # Validate schema
    required_fields = {'entity_type', 'entity_id', 'embedding', 'model'}
    for i, data in enumerate(embeddings_data):
        missing = required_fields - set(data.keys())
        if missing:
            raise ValueError(f"Entry {i} missing fields: {missing}")

        # Validate embedding dimension
        if len(data['embedding']) != self.embedding_dim:
            raise ValueError(f"Entry {i} has wrong dimension: {len(data['embedding'])} != {self.embedding_dim}")

    if dry_run:
        logger.info(f"Dry run: would import {len(embeddings_data)} embeddings")
        return len(embeddings_data)

    # ... rest of import with transaction
```

**Time**: 3 hours (includes tests)

---

### 5. No Rate Limiting on Authentication ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: Auth endpoints (`/auth/login`, `/auth/refresh`) have no rate limiting, making them vulnerable to brute force attacks.

**Files**:
- `mcp_server/main.py:228-243`
- `mcp_server/auth.py`

**What to Do**:
1. Add rate limiting middleware using `slowapi`
2. Limit to 5 login attempts per IP per 15 minutes
3. Add exponential backoff after failures
4. Log suspicious activity (many failures)
5. Add CAPTCHA after N failures (optional)

**Code Change**:
```python
# Add to requirements.txt:
# slowapi==0.1.9

# In mcp_server/main.py:
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/auth/login", response_model=AuthResponse)
@limiter.limit("5/15minutes")
async def login_endpoint(request: Request, auth_request: AuthRequest):
    # ... existing code
```

**Time**: 4 hours (includes testing and documentation)

---

### 6. Print Statements in Production Code ✅ **COMPLETED**

**Status**: Fixed in PR #4 (2025-10-27)

**Problem**: `core/smart_assistant.py` uses `print()` instead of logging, which won't be captured by logging infrastructure.

**Files**:
- `core/smart_assistant.py` (2 occurrences)

**What to Do**:
1. Replace all `print()` calls with `logger.info()` or `logger.debug()`
2. Add pre-commit hook to prevent future print statements
3. Audit entire codebase for other print statements

**Code Change**:
```python
# Before:
print(f"Assistant: {response}")
print(f"Metrics: {metrics}")

# After:
logger.info(f"Assistant response generated: {len(response)} chars")
logger.debug(f"Performance metrics: {metrics}")
```

**Time**: 1 hour

---

## 🎯 High Impact (Big Improvements, Reasonable Effort)

### 7. Refactor Large Files (Smart Assistant & GUI)

**Problem**: `core/smart_assistant.py` (979 lines) and `gui/interface.py` (1080 lines) are too large, violating single responsibility principle. Hard to maintain and test.

**Files**:
- `core/smart_assistant.py` (979 lines)
- `gui/interface.py` (1080 lines)

**What to Do**:

**For smart_assistant.py:**
1. Extract audio processing to `core/audio_processor.py`
2. Extract state management to `core/conversation_state.py`
3. Extract metrics/monitoring to `core/metrics.py`
4. Keep only core orchestration in `smart_assistant.py`

**For interface.py:**
1. Split into `gui/tabs/` directory with one file per tab
2. Extract common components to `gui/widgets.py`
3. Extract state management to `gui/app_state.py`
4. Keep only main app setup in `interface.py`

**File Structure After**:
```
core/
├── smart_assistant.py (300 lines - orchestration only)
├── audio_processor.py (250 lines - audio I/O)
├── conversation_state.py (200 lines - state management)
├── metrics.py (150 lines - monitoring)
└── tool_manager.py (existing)

gui/
├── interface.py (200 lines - app setup)
├── app_state.py (150 lines - shared state)
├── widgets.py (200 lines - reusable components)
└── tabs/
    ├── chat_tab.py (250 lines)
    ├── settings_tab.py (200 lines)
    ├── history_tab.py (150 lines)
    └── analytics_tab.py (130 lines)
```

**Time**: 2 weeks (includes extensive testing)

---

### 8. Add Database Transaction Management

**Problem**: Database operations lack explicit transaction management and rollback handling. Partial updates could corrupt data.

**Files**:
- `memory/db_manager.py` (all write operations)
- `memory/vector_store.py` (batch operations)

**What to Do**:
1. Wrap all multi-step operations in transactions
2. Add retry logic with exponential backoff
3. Implement proper error handling with rollback
4. Add transaction context manager
5. Log transaction failures for debugging

**Code Change**:
```python
# In db_manager.py, add:
@contextmanager
def transaction(self):
    """Context manager for database transactions with rollback."""
    conn = self._pool.get_connection()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.commit()
        logger.debug("Transaction committed")
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction rolled back: {e}")
        raise
    finally:
        self._pool.put(conn)

# Usage in methods:
async def add_message_with_embedding(self, conv_id, role, content, embedding):
    with self.transaction() as conn:
        msg_id = self._add_message(conn, conv_id, role, content)
        self._store_embedding(conn, msg_id, embedding)
    return msg_id
```

**Time**: 1 week (includes testing edge cases)

---

### 9. Fix HTTP Client Resource Leaks

**Problem**: `core/tool_manager.py:18` creates `httpx.AsyncClient()` but doesn't close it if `initialize()` fails. Memory leak over time.

**Files**:
- `core/tool_manager.py:16-60`

**What to Do**:
1. Use context manager for client lifecycle
2. Ensure client is closed on error
3. Add cleanup in `__del__` as fallback
4. Implement proper async context manager protocol
5. Add tests for error paths

**Code Change**:
```python
class ToolManager:
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.mcp_server_url = mcp_server_url.rstrip("/")
        self.client: Optional[httpx.AsyncClient] = None
        self.available_tools = []
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=30.0)
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize the tool manager."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            await self._load_available_tools()
            self._initialized = True
            logger.info(f"Tool manager initialized with {len(self.available_tools)} tools")
        except Exception as e:
            await self.close()  # Clean up on failure
            logger.error(f"Failed to initialize tool manager: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    def __del__(self):
        """Fallback cleanup."""
        if self.client and not self.client.is_closed:
            logger.warning("ToolManager not properly closed. Resources may leak.")

# Usage:
async with ToolManager() as tool_manager:
    result = await tool_manager.execute_tool("web_search", query="test")
```

**Time**: 4 hours

---

### 10. Implement LRU Cache for Vector Embeddings

**Problem**: `memory/vector_store.py:125-132` uses simple FIFO cache eviction. LRU would be more efficient for frequently accessed embeddings.

**Files**:
- `memory/vector_store.py:125-132`

**What to Do**:
1. Replace dict cache with `functools.lru_cache` or `cachetools.LRUCache`
2. Add cache hit/miss metrics
3. Make cache size configurable
4. Add cache warming for common entities
5. Monitor cache performance

**Code Change**:
```python
from cachetools import LRUCache
from threading import Lock

class VectorStore:
    def __init__(self, db_manager: Optional[DatabaseManager] = None,
                 embedding_dim: int = 1536,
                 cache_size: int = 1000):
        # ... existing init ...
        self._cache = LRUCache(maxsize=cache_size)
        self._cache_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def _update_cache(self, key: str, embedding: np.ndarray) -> None:
        """Update the LRU cache."""
        with self._cache_lock:
            self._cache[key] = embedding

    def get_embedding(self, entity_type: EntityType, entity_id: int) -> Optional[np.ndarray]:
        cache_key = f"{entity_type.value}_{entity_id}"

        with self._cache_lock:
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]

        self._cache_misses += 1
        # ... fetch from database ...

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self._cache.maxsize
        }
```

**Time**: 3 hours

---

### 11. Add Comprehensive API Metrics Endpoint

**Problem**: Only basic `/health` endpoint exists. No detailed metrics for monitoring performance, usage, or errors.

**Files**:
- `mcp_server/main.py:156-195` (expand health check)
- New file: `mcp_server/metrics.py`

**What to Do**:
1. Add Prometheus-compatible metrics endpoint
2. Track request latency, error rates, throughput
3. Monitor database query performance
4. Track tool execution success/failure
5. Monitor LLM token usage
6. Add Grafana dashboard example

**Implementation**:
```python
# New file: mcp_server/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
active_connections = Gauge('active_websocket_connections', 'Active WebSocket connections')
llm_tokens = Counter('llm_tokens_used', 'LLM tokens consumed', ['model'])
tool_executions = Counter('tool_executions_total', 'Tool executions', ['tool', 'status'])
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', ['operation'])

# In main.py:
from mcp_server.metrics import *
from prometheus_client import generate_latest

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")

# Add middleware to track requests:
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    request_duration.labels(endpoint=request.url.path).observe(duration)

    return response
```

**Time**: 1 week (includes dashboard setup)

---

### 12. WebSocket Authentication Enforcement

**Problem**: WebSocket endpoint `/ws/{connection_id}` accepts anonymous connections. No authentication check before processing messages.

**Files**:
- `mcp_server/main.py:539-612`

**What to Do**:
1. Require authentication token in WebSocket connection
2. Validate token before accepting connection
3. Associate WebSocket with authenticated user
4. Add token refresh mechanism for long connections
5. Implement rate limiting per user

**Code Change**:
```python
@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str, token: str = Query(...)):
    """WebSocket endpoint with authentication."""

    # Validate token
    try:
        payload = auth_manager.verify_token(token)
        user_id = payload.get("sub")
        permissions = payload.get("permissions", [])
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
        return

    # Check permissions
    if "read" not in permissions:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Insufficient permissions")
        return

    await connection_manager.connect(websocket, connection_id, user_id)

    try:
        while True:
            data = await websocket.receive_text()

            # Validate token hasn't expired (check every N messages)
            try:
                auth_manager.verify_token(token)
            except HTTPException:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Token expired. Please reconnect."
                }))
                break

            # ... process message ...

    except WebSocketDisconnect:
        connection_manager.disconnect(connection_id, user_id)
```

**Time**: 6 hours

---

### 13. Add OpenAPI Schema Customization

**Problem**: FastAPI auto-generates OpenAPI schema, but it lacks examples, descriptions, and proper error documentation.

**Files**:
- `mcp_server/main.py` (all endpoints)
- `mcp_server/models.py` (all Pydantic models)

**What to Do**:
1. Add detailed docstrings to all endpoints
2. Add request/response examples to models
3. Document all error responses
4. Add authentication requirements to schema
5. Generate client SDKs from schema

**Code Change**:
```python
# In models.py:
class QueryRequest(BaseModel):
    """Request to query the AI assistant.

    Example:
        {
            "message": "What's the weather like today?",
            "conversation_id": "conv_123",
            "model": "llama3.2:3b",
            "stream": true
        }
    """
    message: str = Field(..., description="User message or query", min_length=1, max_length=10000)
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID for context")
    model: str = Field("llama3.2:3b", description="LLM model to use")
    stream: bool = Field(False, description="Stream response tokens")
    context_length: int = Field(10, description="Number of previous messages to include", ge=0, le=50)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Summarize the key points from our last conversation",
                    "conversation_id": "conv_abc123",
                    "context_length": 20
                }
            ]
        }
    }

# In main.py:
@app.post("/agent/query",
          response_model=QueryResponse,
          summary="Query AI Assistant",
          description="Send a message to the AI assistant and receive a response",
          responses={
              200: {"description": "Successful response", "model": QueryResponse},
              400: {"description": "Invalid request"},
              401: {"description": "Authentication required"},
              500: {"description": "Internal server error"}
          })
async def agent_query(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_read_user)
) -> QueryResponse:
    """
    Query the AI assistant with context and memory.

    This endpoint:
    - Retrieves conversation context from database
    - Generates response using configured LLM
    - Stores conversation history for future context
    - Supports tool execution during response generation

    Args:
        request: Query request with message and optional conversation ID
        current_user: Authenticated user information

    Returns:
        QueryResponse with assistant's response and metadata

    Raises:
        HTTPException: If query fails or user lacks permissions
    """
    # ... implementation ...
```

**Time**: 1 week

---

## 🔧 Quality (Refactoring & Maintainability)

### 14. Add Dependency Injection Pattern

**Problem**: Global instances (`_tool_manager`, `auth_manager`, etc.) make testing difficult and create hidden dependencies.

**Files**:
- `core/tool_manager.py:240-249`
- `mcp_server/auth.py:179-180`
- Multiple files with global instances

**What to Do**:
1. Create dependency injection container
2. Use FastAPI's `Depends()` for DI
3. Make all services injectable
4. Add factory functions for test doubles
5. Document DI patterns

**Implementation**:
```python
# New file: core/container.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceContainer:
    """Dependency injection container."""
    db_manager: Optional[DatabaseManager] = None
    vector_store: Optional[VectorStore] = None
    ollama_client: Optional[OllamaClient] = None
    tool_manager: Optional[ToolManager] = None

    def __post_init__(self):
        if self.db_manager is None:
            self.db_manager = DatabaseManager()
        if self.vector_store is None:
            self.vector_store = VectorStore(self.db_manager)
        # ... initialize others ...

# Global container (only one global)
_container: Optional[ServiceContainer] = None

def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container

def set_container(container: ServiceContainer) -> None:
    """Set container (useful for testing)."""
    global _container
    _container = container

# Usage in FastAPI:
def get_db_manager(container: ServiceContainer = Depends(get_container)) -> DatabaseManager:
    return container.db_manager

@app.post("/agent/query")
async def agent_query(
    request: QueryRequest,
    db: DatabaseManager = Depends(get_db_manager),
    ollama: OllamaClient = Depends(lambda c: c.ollama_client, Depends(get_container))
):
    # Use injected dependencies
    messages = await db.get_conversation_context(request.conversation_id)
    response = await ollama.generate(request, messages)
    # ...
```

**Time**: 2 weeks (major refactoring)

---

### 15. Consolidate Configuration Management

**Problem**: Configuration scattered across `config/`, environment variables, and hardcoded values. Hard to understand what's configurable.

**Files**:
- `config/` (all files)
- Various hardcoded values throughout

**What to Do**:
1. Create single `Settings` class using Pydantic
2. Load from environment variables with validation
3. Provide sensible defaults
4. Add config validation at startup
5. Generate documentation from schema

**Implementation**:
```python
# New file: config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class DatabaseSettings(BaseSettings):
    """Database configuration."""
    path: str = "~/ai-assistant/memory/assistant.db"
    pool_size: int = 5
    backup_interval: int = 3600

    model_config = SettingsConfigDict(env_prefix="DB_")

class ServerSettings(BaseSettings):
    """API server configuration."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    reload: bool = False

    model_config = SettingsConfigDict(env_prefix="SERVER_")

class LLMSettings(BaseSettings):
    """LLM configuration."""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2:3b"
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.7

    model_config = SettingsConfigDict(env_prefix="LLM_")

class Settings(BaseSettings):
    """Application settings."""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    database: DatabaseSettings = DatabaseSettings()
    server: ServerSettings = ServerSettings()
    llm: LLMSettings = LLMSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    def validate_settings(self) -> None:
        """Validate settings and fail fast."""
        if self.environment == "production":
            assert self.debug is False, "Debug must be False in production"
            # ... more validations ...

# Usage:
settings = Settings()
settings.validate_settings()

# Access:
db_path = settings.database.path
api_port = settings.server.port
```

**Time**: 1 week

---

### 16. Add Comprehensive Type Hints

**Problem**: Some functions lack type hints, making code harder to understand and breaking IDE autocomplete.

**Files**:
- Various files throughout codebase

**What to Do**:
1. Run `mypy` in strict mode
2. Add type hints to all function signatures
3. Add return type annotations
4. Use `typing.Protocol` for duck typing
5. Add type hints to class attributes

**Example Fixes**:
```python
# Before:
def process_message(message, context=None):
    if context:
        # ...
    return result

# After:
from typing import Optional, Dict, Any, List

def process_message(
    message: str,
    context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Process a message with optional context.

    Args:
        message: User message string
        context: Optional conversation context

    Returns:
        Processing result dictionary
    """
    if context:
        # ...
    return result
```

**Time**: 1 week (run mypy, fix all errors)

---

### 17. Reduce Code Duplication in Tool Manager

**Problem**: `core/tool_manager.py` has duplicated extraction patterns for different tools. Can be refactored to shared extractors.

**Files**:
- `core/tool_manager.py:111-185`

**What to Do**:
1. Create generic pattern matcher
2. Define extraction rules in configuration
3. Use strategy pattern for different extractors
4. Make patterns configurable/extensible
5. Add tests for all patterns

**Implementation**:
```python
# Refactored approach:
from typing import Protocol, Dict, Any, List
from dataclasses import dataclass

@dataclass
class ExtractionPattern:
    """Pattern for extracting information."""
    name: str
    patterns: List[str]
    extractor: Callable[[str], str]

class PatternExtractor:
    """Generic pattern extractor."""

    def __init__(self):
        self.patterns: Dict[str, ExtractionPattern] = {}

    def register_pattern(self, pattern: ExtractionPattern):
        """Register an extraction pattern."""
        self.patterns[pattern.name] = pattern

    def extract(self, pattern_name: str, text: str) -> Optional[str]:
        """Extract using registered pattern."""
        if pattern_name not in self.patterns:
            return None

        pattern = self.patterns[pattern_name]
        return pattern.extractor(text)

# Register patterns:
extractor = PatternExtractor()

def extract_query(text: str) -> str:
    patterns = [
        r"search (?:the )?web for (.+)",
        r"look up (.+)",
        # ... more patterns
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return text.strip()

extractor.register_pattern(ExtractionPattern(
    name="search_query",
    patterns=[r"search", r"look up", r"find"],
    extractor=extract_query
))

# Usage:
query = extractor.extract("search_query", user_message)
```

**Time**: 4 hours

---

### 18. Improve Error Messages

**Problem**: Generic error messages make debugging difficult. Need more context in errors.

**Files**:
- All exception raising code

**What to Do**:
1. Create custom exception classes
2. Include context (IDs, values) in error messages
3. Add error codes for programmatic handling
4. Include troubleshooting hints
5. Add correlation IDs to trace errors

**Implementation**:
```python
# New file: common/exceptions.py
class AIAssistantError(Exception):
    """Base exception for AI Assistant errors."""

    def __init__(self, message: str, error_code: str, context: Dict[str, Any] = None, hint: str = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.hint = hint
        super().__init__(self.format_message())

    def format_message(self) -> str:
        msg = f"[{self.error_code}] {self.message}"
        if self.context:
            msg += f"\nContext: {json.dumps(self.context, indent=2)}"
        if self.hint:
            msg += f"\nHint: {self.hint}"
        return msg

class DatabaseError(AIAssistantError):
    """Database operation failed."""
    pass

class LLMError(AIAssistantError):
    """LLM generation failed."""
    pass

class AuthenticationError(AIAssistantError):
    """Authentication failed."""
    pass

# Usage:
try:
    message = db.get_message(message_id)
    if not message:
        raise DatabaseError(
            message=f"Message not found",
            error_code="DB_NOT_FOUND",
            context={"message_id": message_id, "conversation_id": conv_id},
            hint="Check if the message was deleted or the ID is correct"
        )
except Exception as e:
    logger.error(f"Database error: {e}", exc_info=True)
    raise
```

**Time**: 1 week

---

### 19. Add Integration Test Coverage

**Problem**: Limited integration tests. Only `test_mcp_server.py` exists. Missing tests for end-to-end flows.

**Files**:
- `tests/integration/` (add new files)

**What to Do**:
1. Add tests for voice input → LLM → TTS flow
2. Test tool execution integration
3. Test database + vector store integration
4. Test WebSocket message flow
5. Test authentication flow
6. Add performance baseline tests

**New Test Files**:
```
tests/integration/
├── test_mcp_server.py (existing)
├── test_voice_flow.py (new)
├── test_tool_integration.py (new)
├── test_memory_system.py (new)
├── test_websocket_flow.py (new)
└── test_auth_flow.py (new)
```

**Example Test**:
```python
# tests/integration/test_voice_flow.py
import pytest
from core import SmartAssistant, AssistantConfig

@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_voice_interaction():
    """Test complete voice flow: audio → STT → LLM → TTS → audio."""
    config = AssistantConfig(
        whisper_model="tiny",  # Fast for testing
        ollama_model="llama3.2:3b"
    )

    async with SmartAssistant(config) as assistant:
        # Simulate audio input
        audio_data = generate_test_audio("Hello assistant")

        # Process through pipeline
        result = await assistant.process_voice_input(audio_data)

        # Verify each stage
        assert result.transcription is not None
        assert len(result.transcription) > 0
        assert result.llm_response is not None
        assert result.tts_audio is not None
        assert len(result.tts_audio) > 0

        # Verify stored in database
        messages = await assistant.db.get_conversation_messages(result.conversation_id)
        assert len(messages) >= 2  # User + assistant
```

**Time**: 2 weeks

---

## 💡 Future Enhancements (Nice-to-Have)

### 20. Real-Time Streaming Voice Pipeline

**Problem**: Current voice pipeline is batch-based (record full sentence → process). Users want real-time streaming like ChatGPT voice mode.

**Files**:
- New directory: `realtime/`
- See `STREAMING_REQUIREMENTS.md` for full spec

**What to Do**:
1. Implement continuous audio capture with VAD
2. Add streaming Whisper STT (chunked processing)
3. Use Ollama streaming API for token-by-token output
4. Implement sentence-level TTS generation
5. Add audio queue for gapless playback
6. Implement interruption handling
7. Add latency monitoring (<500ms target)

**Architecture**:
```
┌─────────────────────────────────────────────┐
│         Real-Time Voice Pipeline            │
├─────────────────────────────────────────────┤
│ 🎤 Continuous Input → VAD → Chunked STT    │
│ ↓                                           │
│ Streaming Ollama (token-by-token)          │
│ ↓                                           │
│ Sentence Detection → Progressive TTS        │
│ ↓                                           │
│ Audio Queue → 🔊 Gapless Playback          │
└─────────────────────────────────────────────┘
```

**Key Components**:
```
realtime/
├── audio_streaming.py      # Continuous audio I/O
├── vad.py                  # Voice activity detection
├── streaming_stt.py        # Chunked Whisper
├── streaming_llm.py        # Token-by-token Ollama
├── streaming_tts.py        # Progressive TTS
├── audio_queue.py          # Gapless playback buffer
├── interruption_handler.py # Handle user interruptions
└── latency_monitor.py      # Track end-to-end latency
```

**Dependencies**:
- `webrtcvad` for speech detection
- `pyaudio` for real-time audio I/O
- `queue` for audio buffering
- Existing Ollama streaming support

**Time**: 1 month (complex, needs careful testing)

---

### 21. Multi-User Support with Isolation

**Problem**: Current design assumes single user. Need multi-user support with data isolation for production use.

**Files**:
- `memory/db_manager.py` (add user_id to all tables)
- `mcp_server/auth.py` (enhance user management)
- All API endpoints (add user filtering)

**What to Do**:
1. Add `user_id` column to all database tables
2. Create user management system
3. Add user registration/login flow
4. Implement data isolation at query level
5. Add user quotas and rate limiting
6. Add admin dashboard for user management
7. Implement organization/team support

**Database Changes**:
```sql
-- Add user_id to all tables
ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE knowledge_base ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE projects ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default';

-- Add users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    quota_tokens_per_day INTEGER DEFAULT 100000,
    metadata TEXT DEFAULT '{}'
);

-- Add indexes
CREATE INDEX idx_conversations_user ON conversations(user_id);
CREATE INDEX idx_messages_user ON messages(user_id);
```

**API Changes**:
```python
# All queries filtered by user
async def get_conversations(self, user_id: str, limit: int = 20):
    """Get conversations for specific user only."""
    with self._pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
```

**Time**: 1 month (includes migration)

---

### 22. Knowledge Graph for Advanced Memory

**Problem**: Current memory is flat (messages, knowledge items). Knowledge graph would enable more sophisticated reasoning.

**Files**:
- New: `memory/knowledge_graph.py`
- New: `memory/graph_schema.py`
- Integrate with existing `vector_store.py`

**What to Do**:
1. Choose graph database (Neo4j or embedded solution)
2. Define entity types (Person, Topic, Project, etc.)
3. Define relationship types (RELATED_TO, MENTIONED_IN, etc.)
4. Extract entities from conversations
5. Build relationships between entities
6. Add graph queries for context retrieval
7. Visualize knowledge graph in GUI

**Schema**:
```python
class EntityType(Enum):
    PERSON = "person"
    TOPIC = "topic"
    PROJECT = "project"
    DOCUMENT = "document"
    CONCEPT = "concept"

class RelationType(Enum):
    RELATED_TO = "related_to"
    MENTIONED_IN = "mentioned_in"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"

# Example graph queries:
# - "Find all topics related to 'machine learning'"
# - "Show me everything we discussed about Project X"
# - "What concepts are connected to this person?"
```

**Visualization**:
- Add graph viewer tab in GUI
- Interactive node exploration
- Path finding between concepts
- Temporal view (how knowledge evolved)

**Time**: 2 months (research + implementation)

---

### 23. Advanced Tool Ecosystem

**Problem**: Limited tools (web search, weather). Need more comprehensive tool system.

**Files**:
- `mcp_server/tools.py` (expand)
- New: `mcp_server/tools/` directory for modular tools

**What to Do**:
1. Create plugin system for tools
2. Add tool marketplace/registry
3. Implement common tools:
   - Code execution (sandboxed)
   - File operations (read/write with permissions)
   - Database queries (SQL with safety)
   - API integrations (REST/GraphQL)
   - Email sending
   - Calendar management
   - Web scraping
   - PDF/document processing
4. Add tool composition (chain tools)
5. Implement tool versioning

**Architecture**:
```python
class Tool(ABC):
    """Base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters against schema."""
        # ... jsonschema validation ...

# Tool registry:
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def discover_tools(self, directory: Path):
        """Auto-discover tools in directory."""
        # Load .py files, instantiate Tool classes

# Example tool:
class CodeExecutionTool(Tool):
    name = "execute_code"
    description = "Execute Python code in sandboxed environment"

    async def execute(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        # Use RestrictedPython or Docker container
        # ... safe execution ...
```

**Time**: 2 months

---

### 24. Performance Monitoring Dashboard

**Problem**: No visibility into system performance, bottlenecks, or usage patterns.

**Files**:
- New: `gui/tabs/monitoring_tab.py`
- Enhance: `mcp_server/metrics.py`

**What to Do**:
1. Create real-time performance dashboard
2. Track key metrics:
   - Request latency (p50, p95, p99)
   - LLM token usage and costs
   - Database query performance
   - Vector search latency
   - Tool execution times
   - Memory usage trends
   - Error rates
   - Active users/sessions
3. Add alerting for anomalies
4. Historical trend analysis
5. Export metrics for external monitoring

**Dashboard Sections**:
```
┌─────────────────────────────────────────────┐
│         Performance Dashboard               │
├─────────────────────────────────────────────┤
│ [Real-time Metrics]                         │
│ • Requests/sec: 45                          │
│ • Avg Latency: 234ms                        │
│ • Active Users: 12                          │
│ • LLM Tokens/min: 1,250                     │
│                                             │
│ [Latency Distribution]                      │
│ ▂▃▅▇█▇▅▃▂▁▁▁▁▁▂▃▅▇█▅▃▂▁ (last hour)        │
│ p50: 180ms │ p95: 450ms │ p99: 820ms       │
│                                             │
│ [Database Performance]                      │
│ • Query time: 12ms avg                      │
│ • Connection pool: 3/5 used                 │
│ • Cache hit rate: 87%                       │
│                                             │
│ [Tool Usage]                                │
│ • web_search: 145 calls (98% success)       │
│ • weather: 23 calls (100% success)          │
│                                             │
│ [Errors (last hour)]                        │
│ • Database timeout: 2                       │
│ • LLM timeout: 1                            │
│ • Auth failure: 5                           │
└─────────────────────────────────────────────┘
```

**Time**: 2 weeks

---

### 25. Automatic Conversation Summarization

**Problem**: Long conversations become unwieldy. Need automatic summarization to maintain context while reducing token usage.

**Files**:
- New: `memory/summarization.py`
- Integrate with `core/smart_assistant.py`

**What to Do**:
1. Detect when conversation exceeds threshold
2. Generate summary of older messages
3. Store summary in database
4. Use summary instead of full history for context
5. Implement hierarchical summarization (summaries of summaries)
6. Add user control over summarization

**Implementation**:
```python
class ConversationSummarizer:
    """Automatic conversation summarization."""

    async def should_summarize(self, conversation_id: str) -> bool:
        """Check if conversation needs summarization."""
        message_count = await self.db.get_message_count(conversation_id)
        return message_count > SUMMARIZATION_THRESHOLD  # e.g., 50 messages

    async def summarize_conversation(self, conversation_id: str,
                                    messages: List[Dict]) -> str:
        """Generate summary of messages."""
        # Use LLM to summarize
        summary_prompt = f"""
        Summarize the following conversation, highlighting:
        1. Main topics discussed
        2. Key decisions made
        3. Important information shared
        4. Action items or follow-ups

        Conversation:
        {self._format_messages(messages)}

        Summary:
        """

        summary = await self.llm.generate(summary_prompt)

        # Store summary
        await self.db.store_summary(conversation_id, summary,
                                   start_msg_id=messages[0]['id'],
                                   end_msg_id=messages[-1]['id'])

        return summary

    async def get_conversation_context(self, conversation_id: str,
                                     max_tokens: int = 4000) -> List[Dict]:
        """Get conversation context with automatic summarization."""
        # Get recent messages
        recent_messages = await self.db.get_recent_messages(conversation_id, limit=10)

        # Check if we need historical context
        if await self.should_use_summary(conversation_id, recent_messages, max_tokens):
            # Get summary of older messages
            summary = await self.db.get_latest_summary(conversation_id)

            # Combine summary + recent messages
            context = [
                {"role": "system", "content": f"Previous conversation summary: {summary}"},
                *recent_messages
            ]
        else:
            # Use all messages if within token budget
            context = await self.db.get_conversation_messages(conversation_id)

        return context
```

**Time**: 2 weeks

---

## Implementation Priority Order

### Phase 1: Critical Fixes (Week 1-2) ✅ **COMPLETED**

**Status**: All items completed in PR #4 (2025-10-27)

1. ✅ Hardcoded API key removal
2. ✅ JWT secret key fix
3. ✅ Add sklearn to requirements
4. ✅ Input validation on imports
5. ✅ Rate limiting on auth
6. ✅ Remove print statements

**Total**: 2 weeks
**Actual**: Completed in 1 day

### Phase 2: High Impact Improvements (Weeks 3-8)
1. Refactor large files
2. Add transaction management
3. Fix HTTP client leaks
4. Implement LRU cache
5. Add metrics endpoint
6. WebSocket authentication
7. OpenAPI customization

**Total**: 6 weeks

### Phase 3: Quality Improvements (Weeks 9-14)
1. Dependency injection
2. Configuration consolidation
3. Type hints
4. Reduce code duplication
5. Improve error messages
6. Integration tests

**Total**: 6 weeks

### Phase 4: Future Features (Ongoing)
1. Real-time streaming (Q2 2025)
2. Multi-user support (Q2 2025)
3. Knowledge graph (Q3 2025)
4. Advanced tools (Q3 2025)
5. Monitoring dashboard (Q2 2025)
6. Auto-summarization (Q3 2025)

---

## Success Metrics

### Security
- [x] Zero hardcoded secrets in codebase ✅ **COMPLETED (PR #4)**
- [x] All auth endpoints rate-limited ✅ **COMPLETED (PR #4)**
- [x] JWT tokens persist across restarts ✅ **COMPLETED (PR #4)**
- [ ] WebSocket authentication enforced

### Performance
- [ ] API latency p95 < 500ms
- [ ] Database queries < 50ms average
- [ ] Vector search < 100ms
- [ ] Memory usage < 2GB under normal load
- [ ] Cache hit rate > 80%

### Code Quality
- [ ] Test coverage > 80%
- [ ] Mypy strict mode passes
- [ ] No files > 500 lines
- [ ] All public APIs documented
- [ ] Zero security warnings from scanners

### Reliability
- [ ] 99.9% uptime
- [ ] Graceful degradation on failures
- [ ] All errors properly logged
- [ ] Transaction rollback on failures

---

## Contributing

When implementing items from this roadmap:
1. Create issue referencing roadmap item number
2. Create branch: `roadmap/item-{number}-{short-description}`
3. Include tests for all changes
4. Update documentation
5. Add entry to CHANGELOG.md

---

**Note**: Estimates are rough and may vary based on complexity discovered during implementation. Always validate changes with comprehensive testing.
