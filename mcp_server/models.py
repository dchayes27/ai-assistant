"""
Pydantic models for MCP server requests and responses
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Message type enumeration"""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    FILE = "file"


class QueryRequest(BaseModel):
    """Request model for agent queries"""
    message: str = Field(..., description="The user's message/query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    context_length: Optional[int] = Field(10, description="Number of previous messages to include")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    model: Optional[str] = Field("llama3.2:3b", description="Ollama model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="Maximum tokens to generate")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class QueryResponse(BaseModel):
    """Response model for agent queries"""
    response: str = Field(..., description="The agent's response")
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Unique message ID")
    model_used: str = Field(..., description="Model that generated the response")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    response_time: float = Field(..., description="Response time in seconds")
    context_used: List[Dict[str, Any]] = Field(default_factory=list, description="Context messages used")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class MemorySearchRequest(BaseModel):
    """Request model for memory search"""
    query: str = Field(..., description="Search query")
    search_type: str = Field("fts", description="Search type: 'fts', 'semantic', or 'hybrid'")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results to return")
    conversation_id: Optional[str] = Field(None, description="Filter by conversation ID")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    include_system: Optional[bool] = Field(False, description="Include system messages")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class MemorySearchResult(BaseModel):
    """Single search result"""
    message_id: str = Field(..., description="Message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    similarity_score: Optional[float] = Field(None, description="Similarity score (for semantic search)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")


class MemorySearchResponse(BaseModel):
    """Response model for memory search"""
    results: List[MemorySearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of matching results")
    search_time: float = Field(..., description="Search time in seconds")
    query: str = Field(..., description="Original search query")
    search_type: str = Field(..., description="Search type used")


class MemorySaveRequest(BaseModel):
    """Request model for saving memory"""
    conversation_id: str = Field(..., description="Conversation ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    message_type: MessageType = Field(MessageType.TEXT, description="Message type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")
    embedding: Optional[List[float]] = Field(None, description="Message embedding vector")


class MemorySaveResponse(BaseModel):
    """Response model for saving memory"""
    message_id: str = Field(..., description="Saved message ID")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime = Field(..., description="Save timestamp")
    success: bool = Field(..., description="Save success status")


class ProjectInfo(BaseModel):
    """Project information model"""
    project_id: str = Field(..., description="Project ID")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: str = Field(..., description="Project status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Project tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Project metadata")


class ProjectListResponse(BaseModel):
    """Response model for project list"""
    projects: List[ProjectInfo] = Field(..., description="List of projects")
    total_count: int = Field(..., description="Total number of projects")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Page size")


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field("1.0.0", description="API version")
    database_status: str = Field(..., description="Database connection status")
    ollama_status: str = Field(..., description="Ollama service status")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage stats")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class StreamChunk(BaseModel):
    """Streaming response chunk"""
    chunk: str = Field(..., description="Response chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Message ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class AuthRequest(BaseModel):
    """Authentication request model"""
    username: Optional[str] = Field(None, description="Username (if using username/password)")
    password: Optional[str] = Field(None, description="Password (if using username/password)")
    api_key: Optional[str] = Field(None, description="API key (if using API key auth)")
    token: Optional[str] = Field(None, description="JWT token (if using token auth)")


class AuthResponse(BaseModel):
    """Authentication response model"""
    access_token: str = Field(..., description="Access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_id: Optional[str] = Field(None, description="User ID")


class ConversationSummary(BaseModel):
    """Conversation summary model"""
    conversation_id: str = Field(..., description="Conversation ID")
    title: Optional[str] = Field(None, description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    participants: List[str] = Field(default_factory=list, description="Conversation participants")
    tags: List[str] = Field(default_factory=list, description="Conversation tags")
    summary: Optional[str] = Field(None, description="Conversation summary")


class BatchOperation(BaseModel):
    """Batch operation model"""
    operation: str = Field(..., description="Operation type")
    items: List[Dict[str, Any]] = Field(..., description="Items to process")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation options")


class BatchResponse(BaseModel):
    """Batch operation response"""
    operation: str = Field(..., description="Operation type")
    total_items: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    processing_time: float = Field(..., description="Processing time in seconds")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Operation results")