# API Reference

REST API and WebSocket documentation for the AI Assistant.

**Base URL**: `http://localhost:8000`
**API Docs**: `http://localhost:8000/docs` (when server running)

---

## Authentication

### POST /auth/login

Obtain JWT access token.

**Request**:
```json
{
  "api_key": "your-api-key",
  "username": "optional",
  "password": "optional"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "expires_in": 1800,
  "user_id": "user_123"
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"api_key": "dev-key-12345"}'
```

### POST /auth/refresh

Refresh access token using refresh token.

**Request**:
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

---

## Agent Endpoints

### POST /agent/query

Send message to AI assistant and get response.

**Request**:
```json
{
  "message": "What's the weather today?",
  "conversation_id": "conv_123",
  "model": "llama3.2:3b",
  "stream": false,
  "context_length": 10
}
```

**Response**:
```json
{
  "response": "I don't have access to real-time weather...",
  "conversation_id": "conv_123",
  "message_id": 456,
  "model": "llama3.2:3b",
  "tokens_used": 150,
  "processing_time": 1.23
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, assistant!",
    "conversation_id": null
  }'
```

**Python Example**:
```python
import httpx

async def query_assistant(message: str, token: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/agent/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": message}
        )
        return response.json()
```

### POST /agent/stream

Stream assistant responses token by token.

**Request**: Same as `/agent/query` with `stream: true`

**Response**: Server-Sent Events (SSE)
```
data: {"token": "Hello", "done": false}
data: {"token": " there", "done": false}
data: {"token": "!", "done": true, "conversation_id": "conv_123"}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/agent/stream \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "stream": true}'
```

---

## Conversation Endpoints

### GET /conversations

List all conversations.

**Query Parameters**:
- `limit` (int, default: 20): Number of conversations
- `offset` (int, default: 0): Pagination offset
- `mode` (string, optional): Filter by mode

**Response**:
```json
{
  "conversations": [
    {
      "id": "conv_123",
      "mode": "chat",
      "title": "General Discussion",
      "message_count": 15,
      "created_at": "2025-10-27T10:00:00Z",
      "updated_at": "2025-10-27T12:30:00Z"
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### GET /conversations/{conversation_id}

Get conversation details.

**Response**:
```json
{
  "id": "conv_123",
  "mode": "chat",
  "title": "General Discussion",
  "created_at": "2025-10-27T10:00:00Z",
  "updated_at": "2025-10-27T12:30:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "Hello!",
      "created_at": "2025-10-27T10:00:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "Hi there! How can I help?",
      "created_at": "2025-10-27T10:00:05Z"
    }
  ]
}
```

### DELETE /conversations/{conversation_id}

Delete a conversation.

**Response**:
```json
{
  "success": true,
  "message": "Conversation deleted"
}
```

---

## Memory Endpoints

### POST /memory/search

Semantic search across knowledge base.

**Request**:
```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "entity_types": ["knowledge", "message"]
}
```

**Response**:
```json
{
  "results": [
    {
      "entity_type": "knowledge",
      "entity_id": 42,
      "score": 0.89,
      "content": "Machine learning algorithms can be categorized...",
      "metadata": {}
    }
  ],
  "query": "machine learning algorithms",
  "total_results": 3
}
```

### POST /memory/knowledge

Add knowledge base entry.

**Request**:
```json
{
  "title": "Python Best Practices",
  "content": "Use type hints, write tests...",
  "tags": ["python", "programming"],
  "metadata": {}
}
```

---

## Tool Endpoints

### GET /tools/list

List available MCP tools.

**Response**:
```json
{
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web using Brave API",
      "parameters": {
        "query": "string",
        "count": "integer"
      }
    },
    {
      "name": "weather",
      "description": "Get weather information",
      "parameters": {
        "location": "string"
      }
    }
  ]
}
```

### POST /tools/execute

Execute a tool.

**Request**:
```json
{
  "tool_name": "web_search",
  "parameters": {
    "query": "latest AI news",
    "count": 5
  }
}
```

**Response**:
```json
{
  "tool_name": "web_search",
  "result": {
    "results": [
      {
        "title": "Latest AI Breakthrough",
        "url": "https://example.com/ai-news",
        "snippet": "Researchers announce..."
      }
    ]
  },
  "execution_time": 0.45
}
```

---

## Health Check

### GET /health

Check API server health.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "database": "connected",
  "ollama": "available",
  "models": ["llama3.2:3b"]
}
```

---

## WebSocket Interface

### Connection

**URL**: `ws://localhost:8000/ws/{connection_id}?token=YOUR_TOKEN`

**Connection Example** (JavaScript):
```javascript
const ws = new WebSocket(
  'ws://localhost:8000/ws/client-123?token=YOUR_TOKEN'
);

ws.onopen = () => {
  console.log('Connected');
  ws.send(JSON.stringify({
    type: 'query',
    message: 'Hello!',
    conversation_id: null
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Message Types

**Query Message** (Client → Server):
```json
{
  "type": "query",
  "message": "What is the capital of France?",
  "conversation_id": "conv_123",
  "stream": true
}
```

**Response Chunk** (Server → Client):
```json
{
  "type": "response_chunk",
  "token": "Paris",
  "done": false
}
```

**Complete Response** (Server → Client):
```json
{
  "type": "response_complete",
  "response": "The capital of France is Paris.",
  "conversation_id": "conv_123",
  "message_id": 456
}
```

**Ping/Pong** (Keepalive):
```json
{
  "type": "ping"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error: message is required",
  "errors": [
    {
      "loc": ["body", "message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 401 Unauthorized
```json
{
  "detail": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "detail": "Insufficient permissions"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "error_id": "err_abc123"
}
```

---

## Rate Limiting

Currently not implemented. Planned for future release.

**Planned Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1635724800
```

---

## API Versioning

Currently API is unversioned. Future versions will use:
- URL versioning: `/api/v2/agent/query`
- Header versioning: `Accept: application/vnd.ai-assistant.v2+json`

---

## Client Libraries

### Python

```python
import httpx

class AIAssistantClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.token = None

    async def login(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/auth/login",
                json={"api_key": self.api_key}
            )
            data = response.json()
            self.token = data["access_token"]

    async def query(self, message: str, conversation_id: str = None):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agent/query",
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "message": message,
                    "conversation_id": conversation_id
                }
            )
            return response.json()

# Usage
client = AIAssistantClient("http://localhost:8000", "dev-key-12345")
await client.login()
result = await client.query("Hello, assistant!")
print(result["response"])
```

### JavaScript

```javascript
class AIAssistantClient {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this.token = null;
  }

  async login() {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({api_key: this.apiKey})
    });
    const data = await response.json();
    this.token = data.access_token;
  }

  async query(message, conversationId = null) {
    const response = await fetch(`${this.baseUrl}/agent/query`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: message,
        conversation_id: conversationId
      })
    });
    return await response.json();
  }
}

// Usage
const client = new AIAssistantClient('http://localhost:8000', 'dev-key-12345');
await client.login();
const result = await client.query('Hello, assistant!');
console.log(result.response);
```

---

For interactive API documentation, start the server and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
