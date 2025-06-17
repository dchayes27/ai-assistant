"""
Smart Assistant - Main AI assistant class with integrated speech, LLM, and TTS capabilities
"""

import os
import sys
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json

import whisper
import httpx
from loguru import logger
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import DatabaseManager, VectorStore
from mcp_server.ollama_client import OllamaClient
from .tool_manager import get_tool_manager


class ConversationMode(str, Enum):
    """Conversation mode enumeration"""
    CHAT = "chat"
    PROJECT = "project"
    LEARNING = "learning"
    RESEARCH = "research"
    DEBUG = "debug"


class AssistantState(str, Enum):
    """Assistant state enumeration"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    THINKING = "thinking"
    ERROR = "error"


@dataclass
class AssistantConfig:
    """Configuration for the Smart Assistant"""
    
    # Model configurations
    whisper_model: str = "medium"
    ollama_model: str = "llama3.2:3b"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    embedding_model: str = "nomic-embed-text"
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    audio_timeout: float = 5.0
    silence_threshold: float = 0.01
    
    # Conversation settings
    max_context_length: int = 20
    context_window_tokens: int = 4000
    summarization_threshold: int = 50  # messages
    memory_retention_days: int = 30
    
    # Performance settings
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    
    # TTS settings
    speech_rate: int = 150
    speech_volume: float = 0.9
    openai_api_key: str = ""  # OpenAI API key for professional TTS
    
    # Monitoring
    performance_logging: bool = True
    metrics_interval: int = 60  # seconds
    
    # Conversation modes
    default_mode: ConversationMode = ConversationMode.CHAT
    mode_prompts: Dict[ConversationMode, str] = field(default_factory=lambda: {
        ConversationMode.CHAT: """You are a helpful AI assistant with access to persistent memory and information storage. 

You can:
- Remember information from previous conversations
- Store important information for future reference
- Search through conversation history
- Access stored knowledge to provide better context

Be conversational and friendly while leveraging your memory capabilities.""",

        ConversationMode.PROJECT: """You are a project management AI with database access for tracking projects and tasks.

You can:
- Store and retrieve project information
- Track project progress over time
- Access project history and context
- Organize tasks and deliverables

Focus on helping manage projects efficiently using your persistent storage.""",

        ConversationMode.LEARNING: """You are a learning companion with knowledge storage capabilities.

You can:
- Store learning concepts and explanations
- Remember previous learning sessions
- Track learning progress
- Build upon past educational content

Help users learn effectively by maintaining educational continuity.""",

        ConversationMode.RESEARCH: """You are a research assistant with comprehensive information management.

You can:
- Store research findings and sources
- Maintain research project continuity
- Access previous research context
- Organize information systematically

Provide thorough research support with persistent information storage.""",

        ConversationMode.DEBUG: """You are a technical assistant with problem-solving capabilities.

You can:
- Store solutions and troubleshooting steps
- Remember previous technical discussions
- Access relevant technical context
- Build knowledge of common issues

Help solve technical problems systematically."""
    })


@dataclass
class ConversationThread:
    """Represents a conversation thread"""
    thread_id: str
    mode: ConversationMode
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    context_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0
    speech_to_text_time: float = 0.0
    llm_processing_time: float = 0.0
    text_to_speech_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class SmartAssistant:
    """
    Smart AI Assistant with integrated speech, LLM, TTS, and memory capabilities
    """
    
    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        self.state = AssistantState.IDLE
        self.current_thread_id: Optional[str] = None
        self.threads: Dict[str, ConversationThread] = {}
        self.metrics = PerformanceMetrics()
        
        # Core components (initialized in setup)
        self.db_manager: Optional[DatabaseManager] = None
        self.vector_store: Optional[VectorStore] = None
        self.ollama_client: Optional[OllamaClient] = None
        self.whisper_model: Optional[Any] = None
        self.tts_engine: Optional[Any] = None
        self.tool_manager: Optional[Any] = None
        
        # State management
        self._shutdown_event = asyncio.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._initialized = False
        
        # Callbacks
        self.state_change_callbacks: List[Callable[[AssistantState], None]] = []
        self.message_callbacks: List[Callable[[str, str, Dict[str, Any]], None]] = []
        
        logger.info("Smart Assistant initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()
    
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        logger.info("Initializing Smart Assistant components...")
        
        try:
            # Initialize database
            self.db_manager = DatabaseManager()
            
            # Initialize vector store
            self.vector_store = VectorStore(self.db_manager)
            
            # Initialize Ollama client
            self.ollama_client = OllamaClient()
            await self.ollama_client.initialize()
            
            # Initialize Whisper
            await self._initialize_whisper()
            
            # Initialize TTS
            await self._initialize_tts()
            
            # Initialize tool manager
            self.tool_manager = await get_tool_manager()
            
            # Start monitoring
            if self.config.performance_logging:
                self._monitoring_task = asyncio.create_task(self._performance_monitor())
            
            self._initialized = True
            await self._set_state(AssistantState.IDLE)
            
            logger.info("Smart Assistant fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Smart Assistant: {e}")
            await self._set_state(AssistantState.ERROR)
            raise
    
    async def shutdown(self):
        """Shutdown the assistant and cleanup resources"""
        logger.info("Shutting down Smart Assistant...")
        
        self._shutdown_event.set()
        
        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close components
        if self.ollama_client:
            await self.ollama_client.close()
        
        if self.tool_manager:
            await self.tool_manager.close()
        
        if self.db_manager:
            self.db_manager.close()
        
        logger.info("Smart Assistant shutdown complete")
    
    async def _initialize_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.whisper_model = whisper.load_model(
                self.config.whisper_model,
                download_root="models/whisper"
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    async def _initialize_tts(self):
        """Initialize TTS engine"""
        try:
            # Try Coqui TTS first, fallback to pyttsx3
            try:
                from TTS.api import TTS
                logger.info(f"Loading Coqui TTS model: {self.config.tts_model}")
                self.tts_engine = TTS(
                    model_name=self.config.tts_model,
                    progress_bar=False
                )
                logger.info("Coqui TTS loaded successfully")
            except ImportError:
                # Fallback to pyttsx3
                import pyttsx3
                logger.info("Using pyttsx3 TTS engine")
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.config.speech_rate)
                self.tts_engine.setProperty('volume', self.config.speech_volume)
                
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    async def _set_state(self, new_state: AssistantState):
        """Set assistant state and notify callbacks"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"State changed: {old_state} -> {new_state}")
            
            # Notify callbacks
            for callback in self.state_change_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff"""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
        
        raise last_exception
    
    async def _performance_monitor(self):
        """Monitor performance metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Update system metrics
                process = psutil.Process()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.cpu_usage_percent = process.cpu_percent()
                self.metrics.last_updated = datetime.utcnow()
                
                # Log metrics periodically
                logger.debug(f"Performance metrics: "
                           f"Memory: {self.metrics.memory_usage_mb:.1f}MB, "
                           f"CPU: {self.metrics.cpu_usage_percent:.1f}%, "
                           f"Avg response: {self.metrics.average_response_time:.2f}s")
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.config.metrics_interval)
    
    def add_state_callback(self, callback: Callable[[AssistantState], None]):
        """Add state change callback"""
        self.state_change_callbacks.append(callback)
    
    def add_message_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """Add message callback"""
        self.message_callbacks.append(callback)
    
    async def create_conversation_thread(
        self,
        mode: ConversationMode = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation thread"""
        thread_id = str(uuid.uuid4())
        mode = mode or self.config.default_mode
        
        thread = ConversationThread(
            thread_id=thread_id,
            mode=mode,
            title=title or f"{mode.value.title()} Conversation",
            metadata=metadata or {}
        )
        
        self.threads[thread_id] = thread
        
        # Create conversation in database
        self.db_manager.create_conversation(
            thread_id,
            thread.title,
            {"mode": mode.value, **thread.metadata}
        )
        
        logger.info(f"Created conversation thread: {thread_id} ({mode.value})")
        return thread_id
    
    async def switch_thread(self, thread_id: str) -> bool:
        """Switch to a different conversation thread"""
        if thread_id not in self.threads:
            logger.error(f"Thread {thread_id} not found")
            return False
        
        self.current_thread_id = thread_id
        self.threads[thread_id].last_activity = datetime.utcnow()
        
        logger.info(f"Switched to thread: {thread_id}")
        return True
    
    async def get_or_create_default_thread(self) -> str:
        """Get current thread or create default one"""
        if self.current_thread_id and self.current_thread_id in self.threads:
            return self.current_thread_id
        
        # Create default thread
        thread_id = await self.create_conversation_thread(
            mode=self.config.default_mode,
            title="Default Conversation"
        )
        self.current_thread_id = thread_id
        return thread_id
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text using Whisper"""
        await self._set_state(AssistantState.PROCESSING)
        
        try:
            start_time = time.time()
            
            # Save audio to temporary file
            temp_file = f"temp/audio_{int(time.time())}.wav"
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # Transcribe using Whisper
            result = await self._retry_with_backoff(
                self.whisper_model.transcribe,
                temp_file,
                language="en"
            )
            
            transcript = result["text"].strip()
            
            # Cleanup
            os.remove(temp_file)
            
            # Update metrics
            self.metrics.speech_to_text_time = time.time() - start_time
            
            logger.info(f"Transcribed audio: '{transcript[:50]}...'")
            return transcript
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise
        finally:
            await self._set_state(AssistantState.IDLE)
    
    async def synthesize_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Synthesize speech from text"""
        await self._set_state(AssistantState.SPEAKING)
        
        try:
            start_time = time.time()
            
            if not output_file:
                # Use system temp directory for Gradio compatibility
                import tempfile
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, f"speech_{int(time.time())}.wav")
            
            # Check if using OpenAI TTS
            voice_model = str(self.config.tts_model)
            if voice_model.startswith("openai:"):
                # OpenAI TTS
                openai_voice = voice_model.split(":")[1]
                openai_api_key = getattr(self.config, 'openai_api_key', '')
                
                if not openai_api_key:
                    logger.warning("OpenAI API key not set, falling back to local TTS")
                    voice_model = "tts_models/en/ljspeech/tacotron2-DDC"
                else:
                    # Use OpenAI TTS API
                    import httpx
                    
                    headers = {
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "tts-1",
                        "input": text,
                        "voice": openai_voice,
                        "response_format": "wav"
                    }
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.openai.com/v1/audio/speech",
                            headers=headers,
                            json=data,
                            timeout=30.0
                        )
                        
                        if response.status_code == 200:
                            # Save the audio content
                            with open(output_file, "wb") as f:
                                f.write(response.content)
                            
                            logger.info(f"OpenAI TTS synthesis successful: {openai_voice}")
                        else:
                            logger.error(f"OpenAI TTS failed: {response.status_code} - {response.text}")
                            raise RuntimeError(f"OpenAI TTS failed: {response.text}")
            
            # If not OpenAI or fallback needed, use local TTS
            if not voice_model.startswith("openai:"):
                # Use appropriate TTS engine
                if hasattr(self.tts_engine, 'tts_to_file'):
                    # Coqui TTS
                    await self._retry_with_backoff(
                        self.tts_engine.tts_to_file,
                        text=text,
                        file_path=output_file
                    )
                else:
                    # Use macOS's built-in 'say' command as fallback
                    import subprocess
                    import platform
                    
                    if platform.system() == "Darwin":  # macOS
                        try:
                            # Use say command to generate audio file with explicit format
                            cmd = ["say", "-o", output_file, "--data-format=LEF32@22050", text]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                            
                            if result.returncode != 0:
                                logger.error(f"say command failed: {result.stderr}")
                                raise RuntimeError(f"macOS say command failed: {result.stderr}")
                                
                        except subprocess.TimeoutExpired:
                            logger.error("say command timed out")
                            raise TimeoutError("Speech synthesis timed out")
                    else:
                        # Fallback for non-macOS systems
                        self.tts_engine.save_to_file(text, output_file)
                        try:
                            self.tts_engine.runAndWait()
                        except RuntimeError as e:
                            if "run loop already started" in str(e):
                                logger.warning("Skipping runAndWait due to event loop conflict")
                            else:
                                raise
            
            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"Audio file created: {output_file} ({file_size} bytes)")
            else:
                logger.error(f"Audio file was not created: {output_file}")
                raise FileNotFoundError(f"TTS failed to create audio file: {output_file}")
            
            # Update metrics
            self.metrics.text_to_speech_time = time.time() - start_time
            
            logger.info(f"Synthesized speech: '{text[:50]}...'")
            return output_file
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
        finally:
            await self._set_state(AssistantState.IDLE)
    
    async def _get_conversation_context(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation context for LLM"""
        try:
            thread = self.threads.get(thread_id)
            if not thread:
                return []
            
            # Get recent messages
            context_data = self.db_manager.get_conversation_context(
                thread_id,
                max_messages=self.config.max_context_length
            )
            messages = context_data.get("messages", [])
            
            # Add mode-specific system prompt
            system_prompt = self.config.mode_prompts.get(
                thread.mode,
                self.config.mode_prompts[ConversationMode.CHAT]
            )
            
            context = [{"role": "system", "content": system_prompt}]
            
            # Add conversation summary if available
            if thread.context_summary:
                context.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {thread.context_summary}"
                })
            
            # Add recent messages
            context.extend(messages)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []
    
    async def _should_summarize_conversation(self, thread_id: str) -> bool:
        """Check if conversation should be summarized"""
        thread = self.threads.get(thread_id)
        if not thread:
            return False
        
        return thread.message_count >= self.config.summarization_threshold
    
    async def _summarize_conversation(self, thread_id: str) -> Optional[str]:
        """Summarize conversation for long-term memory"""
        try:
            logger.info(f"Summarizing conversation: {thread_id}")
            
            # Get all messages in the conversation
            messages = self.db_manager.get_conversation_messages(
                thread_id,
                limit=self.config.summarization_threshold
            )
            
            if not messages:
                return None
            
            # Create summarization prompt
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])
            
            summarization_prompt = f"""
            Please provide a concise summary of the following conversation, focusing on:
            1. Main topics discussed
            2. Key decisions or conclusions
            3. Important information or insights
            4. Any action items or follow-ups
            
            Conversation:
            {conversation_text}
            
            Summary:
            """
            
            # Generate summary using LLM
            context = [{"role": "user", "content": summarization_prompt}]
            response = await self.ollama_client.generate(
                QueryRequest(
                    message=summarization_prompt,
                    conversation_id=thread_id,
                    model="llama3.2:3b",
                    temperature=0.3,
                    max_tokens=500
                ),
                context
            )
            
            summary = response.response.strip()
            
            # Update thread summary
            thread = self.threads.get(thread_id)
            if thread:
                thread.context_summary = summary
            
            # Store summary in knowledge base
            self.db_manager.add_knowledge(
                title=f"Conversation Summary: {thread_id}",
                content=summary,
                category="conversation_summary",
                tags=["summary", thread.mode.value],
                metadata={"thread_id": thread_id, "message_count": len(messages)}
            )
            
            logger.info(f"Conversation summarized: {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Conversation summarization failed: {e}")
            return None
    
    async def process_message(
        self,
        message: str,
        thread_id: Optional[str] = None,
        mode: Optional[ConversationMode] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a text message and generate response"""
        await self._set_state(AssistantState.THINKING)
        
        try:
            start_time = time.time()
            self.metrics.total_queries += 1
            
            # Get or create thread
            if not thread_id:
                thread_id = await self.get_or_create_default_thread()
            elif thread_id not in self.threads:
                thread_id = await self.create_conversation_thread(
                    mode=mode or self.config.default_mode
                )
            
            thread = self.threads[thread_id]
            
            # Switch mode if requested
            if mode and mode != thread.mode:
                thread.mode = mode
                logger.info(f"Switched mode to {mode.value} for thread {thread_id}")
            
            # Process message with tools if needed
            enhanced_message = message
            tool_results = []
            
            if self.tool_manager:
                try:
                    enhanced_message, tool_results = await self.tool_manager.process_message_with_tools(message)
                    logger.info(f"Processed message with {len(tool_results)} tool results")
                except Exception as e:
                    logger.error(f"Tool processing failed: {e}")
            
            # Get conversation context
            context = await self._get_conversation_context(thread_id)
            
            # Add current message to context (use enhanced message with tool results)
            context.append({"role": "user", "content": enhanced_message})
            
            # Generate response using Ollama
            from mcp_server.models import QueryRequest
            
            request = QueryRequest(
                message=message,
                conversation_id=thread_id,
                model=self.config.ollama_model,
                temperature=0.7,
                max_tokens=2000,
                metadata=metadata or {}
            )
            
            # Process with retry logic
            response = await self._retry_with_backoff(
                self.ollama_client.generate,
                request,
                context[:-1]  # Don't include the current message in context
            )
            
            assistant_response = response.response
            
            # Save messages to database
            self.db_manager.add_message(
                thread_id,
                "user",
                message,
                metadata=metadata
            )
            
            self.db_manager.add_message(
                thread_id,
                "assistant",
                assistant_response,
                metadata={
                    "model": self.config.ollama_model,
                    "tokens": response.tokens_used,
                    "response_time": response.response_time
                }
            )
            
            # Update thread stats
            thread.message_count += 2
            thread.last_activity = datetime.utcnow()
            
            # Check if conversation should be summarized
            if await self._should_summarize_conversation(thread_id):
                asyncio.create_task(self._summarize_conversation(thread_id))
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.successful_queries += 1
            self.metrics.total_response_time += response_time
            self.metrics.average_response_time = (
                self.metrics.total_response_time / self.metrics.successful_queries
            )
            self.metrics.llm_processing_time = response.response_time
            
            # Notify callbacks
            for callback in self.message_callbacks:
                try:
                    callback("assistant", assistant_response, {"thread_id": thread_id})
                except Exception as e:
                    logger.error(f"Message callback error: {e}")
            
            logger.info(f"Processed message in {response_time:.2f}s")
            return assistant_response
            
        except Exception as e:
            self.metrics.failed_queries += 1
            logger.error(f"Message processing failed: {e}")
            raise
        finally:
            await self._set_state(AssistantState.IDLE)
    
    async def process_voice_message(
        self,
        audio_data: bytes,
        thread_id: Optional[str] = None,
        synthesize_response: bool = True
    ) -> Dict[str, Any]:
        """Process voice input and optionally return voice response"""
        await self._set_state(AssistantState.LISTENING)
        
        try:
            # Transcribe audio
            transcript = await self.transcribe_audio(audio_data)
            
            if not transcript:
                return {"error": "No speech detected"}
            
            # Process message
            response_text = await self.process_message(transcript, thread_id)
            
            result = {
                "transcript": transcript,
                "response": response_text,
                "thread_id": thread_id or self.current_thread_id
            }
            
            # Synthesize speech response if requested
            if synthesize_response:
                audio_file = await self.synthesize_speech(response_text)
                result["audio_file"] = audio_file
            
            return result
            
        except Exception as e:
            logger.error(f"Voice message processing failed: {e}")
            raise
    
    async def search_memory(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search conversation memory"""
        try:
            if search_type == "fts":
                results = self.db_manager.search_messages(query, limit=limit)
            elif search_type == "semantic":
                results = await self.vector_store.search_similar(query, limit=limit)
            else:  # hybrid
                fts_results = self.db_manager.search_messages(query, limit=limit//2)
                semantic_results = await self.vector_store.search_similar(query, limit=limit//2)
                
                # Combine and deduplicate
                seen = set()
                results = []
                for result in fts_results + semantic_results:
                    if result["message_id"] not in seen:
                        seen.add(result["message_id"])
                        results.append(result)
                
                results = results[:limit]
            
            logger.info(f"Memory search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def get_conversation_threads(self) -> List[Dict[str, Any]]:
        """Get all conversation threads"""
        return [
            {
                "thread_id": thread.thread_id,
                "mode": thread.mode.value,
                "title": thread.title,
                "created_at": thread.created_at.isoformat(),
                "last_activity": thread.last_activity.isoformat(),
                "message_count": thread.message_count,
                "active": thread.active
            }
            for thread in self.threads.values()
        ]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "total_queries": self.metrics.total_queries,
            "successful_queries": self.metrics.successful_queries,
            "failed_queries": self.metrics.failed_queries,
            "success_rate": (
                self.metrics.successful_queries / max(self.metrics.total_queries, 1)
            ),
            "average_response_time": self.metrics.average_response_time,
            "speech_to_text_time": self.metrics.speech_to_text_time,
            "llm_processing_time": self.metrics.llm_processing_time,
            "text_to_speech_time": self.metrics.text_to_speech_time,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "cpu_usage_percent": self.metrics.cpu_usage_percent,
            "last_updated": self.metrics.last_updated.isoformat(),
            "state": self.state.value,
            "active_threads": len([t for t in self.threads.values() if t.active])
        }
    
    async def cleanup_old_conversations(self, days: int = None):
        """Cleanup old conversations based on retention policy"""
        days = days or self.config.memory_retention_days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            # Mark old threads as inactive
            for thread in self.threads.values():
                if thread.last_activity < cutoff_date:
                    thread.active = False
            
            # Archive old conversations in database
            archived_count = self.db_manager.archive_old_conversations(cutoff_date)
            
            logger.info(f"Archived {archived_count} old conversations")
            
        except Exception as e:
            logger.error(f"Conversation cleanup failed: {e}")


# Example usage and testing
async def main():
    """Example usage of the Smart Assistant"""
    config = AssistantConfig(
        whisper_model="small",  # Use smaller model for testing
        performance_logging=True
    )
    
    async with SmartAssistant(config) as assistant:
        # Create a conversation thread
        thread_id = await assistant.create_conversation_thread(
            mode=ConversationMode.CHAT,
            title="Test Conversation"
        )
        
        # Process a text message
        response = await assistant.process_message(
            "Hello! How are you today?",
            thread_id=thread_id
        )
        
        print(f"Assistant: {response}")
        
        # Get performance metrics
        metrics = await assistant.get_performance_metrics()
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/smart_assistant.log", rotation="10 MB", retention="7 days")
    
    # Run example
    asyncio.run(main())