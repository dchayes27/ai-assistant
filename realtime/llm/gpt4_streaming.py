"""
OpenAI GPT-4o Streaming Client

Implements streaming LLM interaction with token-by-token reception,
sentence boundary detection, and MCP tool integration.

Usage:
    client = StreamingGPT4Client(api_key="your-key")
    async for token in client.stream_completion(messages):
        print(token, end="", flush=True)
"""

import asyncio
import logging
import re
import time
from typing import AsyncIterator, Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

try:
    import openai
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI library not available")


class StreamingState(Enum):
    """Streaming states for the LLM client."""
    IDLE = "idle"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class StreamingConfig:
    """Configuration for OpenAI streaming client."""
    model: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Streaming settings
    sentence_buffer_size: int = 50  # tokens
    chunk_timeout: float = 0.1  # seconds
    max_context_length: int = 8192
    
    # Tool settings
    enable_tools: bool = True
    tool_timeout: float = 30.0
    
    # Performance settings
    stream_timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class StreamedToken:
    """Represents a streamed token with metadata."""
    content: str
    timestamp: float
    is_complete: bool = False
    is_sentence_end: bool = False
    token_index: int = 0
    latency_ms: float = 0.0


@dataclass
class StreamedResponse:
    """Complete streamed response with metadata."""
    content: str
    tokens: List[StreamedToken]
    total_tokens: int
    completion_time: float
    first_token_latency: float
    tool_calls: List[Dict[str, Any]]
    interrupted: bool = False


class StreamingGPT4Client:
    """
    Streaming client for OpenAI GPT-4o with real-time token processing.
    
    Features:
    - Token-by-token streaming with sentence detection
    - Smart chunking for TTS integration
    - MCP tool calling support
    - Interruption handling
    - Performance monitoring
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[StreamingConfig] = None):
        if not HAS_OPENAI:
            raise ImportError("OpenAI library required for streaming client")
        
        self.config = config or StreamingConfig()
        self.client = AsyncOpenAI(api_key=api_key)
        self.state = StreamingState.IDLE
        
        # Streaming state
        self.current_stream = None
        self.token_buffer = []
        self.sentence_buffer = []
        self.total_tokens = 0
        self.start_time = 0
        self.first_token_time = 0
        
        # Callbacks
        self.on_token: Optional[Callable[[StreamedToken], None]] = None
        self.on_sentence: Optional[Callable[[str], None]] = None
        self.on_tool_call: Optional[Callable[[Dict], Any]] = None
        self.on_complete: Optional[Callable[[StreamedResponse], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Context management
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Tool registry (to be connected with MCP)
        self.available_tools: Dict[str, Dict] = {}
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "average_latency_ms": 0,
            "errors": 0,
            "interruptions": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], 
                     handler: Callable[[Dict], Any]):
        """Register a tool for function calling."""
        self.available_tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
                "handler": handler
            }
        }
        self.logger.info(f"Registered tool: {name}")
    
    def _detect_sentence_boundary(self, text: str) -> bool:
        """Detect if text ends with a sentence boundary."""
        # Common sentence endings
        sentence_endings = r'[.!?]\s*$'
        
        # Also consider code blocks and lists as boundaries
        code_endings = r'```\s*$|^\s*[-*+]\s|^\s*\d+\.\s'
        
        return bool(re.search(sentence_endings, text.strip()) or 
                   re.search(code_endings, text.strip()))
    
    def _should_chunk_for_tts(self, buffer: List[str]) -> bool:
        """Determine if current buffer should be sent to TTS."""
        text = ''.join(buffer).strip()
        
        # Chunk on sentence boundaries
        if self._detect_sentence_boundary(text):
            return True
        
        # Chunk if buffer is getting large
        if len(buffer) >= self.config.sentence_buffer_size:
            return True
        
        # Chunk on natural pause points
        pause_points = [', ', '; ', ' - ', ' and ', ' but ', ' however ']
        for point in pause_points:
            if text.endswith(point):
                return True
        
        return False
    
    async def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool/function call."""
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        
        if function_name not in self.available_tools:
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": f"Error: Tool '{function_name}' not available"
            }
        
        try:
            self.state = StreamingState.TOOL_CALLING
            
            # Call the tool handler
            tool_handler = self.available_tools[function_name]["function"]["handler"]
            
            if self.on_tool_call:
                result = await self.on_tool_call(tool_call)
            else:
                # Call handler directly
                if asyncio.iscoroutinefunction(tool_handler):
                    result = await tool_handler(function_args)
                else:
                    result = tool_handler(function_args)
            
            self.logger.debug(f"Tool {function_name} executed successfully")
            
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": str(result)
            }
            
        except Exception as e:
            self.logger.error(f"Tool execution error: {e}")
            return {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "content": f"Error executing tool: {str(e)}"
            }
        finally:
            self.state = StreamingState.STREAMING
    
    async def stream_completion(self, messages: List[Dict[str, Any]], 
                              **kwargs) -> AsyncIterator[StreamedToken]:
        """
        Stream completion tokens from OpenAI GPT-4o.
        
        Args:
            messages: Conversation messages
            **kwargs: Additional OpenAI API parameters
        
        Yields:
            StreamedToken objects with content and metadata
        """
        if self.state != StreamingState.IDLE:
            raise RuntimeError(f"Client busy in state: {self.state}")
        
        self.state = StreamingState.STREAMING
        self.start_time = time.time()
        self.first_token_time = 0
        self.token_buffer.clear()
        self.sentence_buffer.clear()
        self.total_tokens = 0
        
        # Prepare tools for the request
        tools = None
        if self.config.enable_tools and self.available_tools:
            tools = [tool for tool in self.available_tools.values()]
        
        try:
            # Create streaming request
            stream_params = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": True,
                **kwargs
            }
            
            if tools:
                stream_params["tools"] = tools
                stream_params["tool_choice"] = "auto"
            
            self.current_stream = await self.client.chat.completions.create(**stream_params)
            
            token_index = 0
            
            async for chunk in self.current_stream:
                if self.state == StreamingState.INTERRUPTED:
                    break
                
                current_time = time.time()
                
                # Record first token time
                if self.first_token_time == 0:
                    self.first_token_time = current_time
                
                # Process the chunk
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function:
                                # Tool call detected - handle it
                                tool_response = await self._handle_tool_call(tool_call)
                                # Continue streaming with tool response
                                updated_messages = messages + [
                                    {"role": "assistant", "tool_calls": [tool_call]},
                                    tool_response
                                ]
                                async for token in self.stream_completion(updated_messages):
                                    yield token
                                return
                    
                    # Handle regular content
                    if delta.content:
                        content = delta.content
                        latency = (current_time - self.start_time) * 1000
                        
                        # Create token object
                        token = StreamedToken(
                            content=content,
                            timestamp=current_time,
                            token_index=token_index,
                            latency_ms=latency
                        )
                        
                        self.token_buffer.append(content)
                        self.sentence_buffer.append(content)
                        self.total_tokens += 1
                        token_index += 1
                        
                        # Check for sentence boundaries
                        current_text = ''.join(self.sentence_buffer)
                        if self._should_chunk_for_tts(self.sentence_buffer):
                            token.is_sentence_end = True
                            
                            # Call sentence callback
                            if self.on_sentence:
                                self.on_sentence(current_text.strip())
                            
                            self.sentence_buffer.clear()
                        
                        # Call token callback
                        if self.on_token:
                            self.on_token(token)
                        
                        yield token
                    
                    # Check if completion is finished
                    if choice.finish_reason:
                        # Mark final token
                        if self.token_buffer:
                            final_token = StreamedToken(
                                content="",
                                timestamp=current_time,
                                is_complete=True,
                                token_index=token_index,
                                latency_ms=(current_time - self.start_time) * 1000
                            )
                            yield final_token
                        break
            
            # Create final response
            completion_time = time.time() - self.start_time
            first_token_latency = (self.first_token_time - self.start_time) * 1000 if self.first_token_time else 0
            
            response = StreamedResponse(
                content=''.join(self.token_buffer),
                tokens=[],  # Could store all tokens if needed
                total_tokens=self.total_tokens,
                completion_time=completion_time,
                first_token_latency=first_token_latency,
                tool_calls=[],
                interrupted=(self.state == StreamingState.INTERRUPTED)
            )
            
            # Update stats
            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += self.total_tokens
            self.stats["average_latency_ms"] = (
                (self.stats["average_latency_ms"] * (self.stats["total_requests"] - 1) + 
                 first_token_latency) / self.stats["total_requests"]
            )
            
            if self.on_complete:
                self.on_complete(response)
            
            self.logger.info(f"Streaming completed: {self.total_tokens} tokens in {completion_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            self.stats["errors"] += 1
            
            if self.on_error:
                self.on_error(e)
            
            raise
        
        finally:
            self.state = StreamingState.IDLE
            self.current_stream = None
    
    async def interrupt_stream(self):
        """Interrupt the current streaming operation."""
        if self.state == StreamingState.STREAMING:
            self.state = StreamingState.INTERRUPTED
            self.stats["interruptions"] += 1
            self.logger.debug("Stream interrupted")
    
    def add_to_conversation(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim context if too long
        while len(self.conversation_history) > self.config.max_context_length:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self, max_messages: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversation context for the API."""
        recent_messages = self.conversation_history[-max_messages:]
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent_messages]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming performance statistics."""
        return self.stats.copy()
    
    async def test_streaming(self, test_message: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test the streaming functionality."""
        messages = [{"role": "user", "content": test_message}]
        
        tokens = []
        start_time = time.time()
        
        try:
            async for token in self.stream_completion(messages):
                tokens.append(token)
                if token.is_complete:
                    break
            
            end_time = time.time()
            
            return {
                "success": True,
                "total_tokens": len(tokens),
                "duration_ms": (end_time - start_time) * 1000,
                "first_token_latency_ms": tokens[0].latency_ms if tokens else 0,
                "content": ''.join(t.content for t in tokens if t.content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration_ms": (time.time() - start_time) * 1000
            }


# Convenience function for testing
async def test_gpt4_streaming():
    """Test the GPT-4o streaming implementation."""
    try:
        # This would need a real API key in production
        client = StreamingGPT4Client()
        
        # Test without actual API call
        config = StreamingConfig()
        print(f"GPT-4o Streaming Client initialized with model: {config.model}")
        print(f"Config: max_tokens={config.max_tokens}, temperature={config.temperature}")
        
        # Test sentence boundary detection
        test_texts = [
            "Hello there.",
            "This is a test, and it continues",
            "Here's some code:\n```python\nprint('hello')\n```",
            "A list:\n- Item 1\n- Item 2"
        ]
        
        temp_client = StreamingGPT4Client()
        for text in test_texts:
            is_boundary = temp_client._detect_sentence_boundary(text)
            print(f"'{text}' -> sentence boundary: {is_boundary}")
        
        return {"status": "test_completed", "structure_valid": True}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run basic tests
    result = asyncio.run(test_gpt4_streaming())
    print(f"Test result: {result}")