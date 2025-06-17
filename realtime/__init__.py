"""
Real-time Streaming Voice Assistant

This package implements the real-time streaming components for the AI assistant,
providing sub-500ms voice-to-voice latency with continuous conversation flow.

Architecture:
- audio/: Audio pipeline and processing
- llm/: LLM streaming and response handling  
- tts/: Text-to-speech streaming
- state/: Real-time conversation state management
- monitoring/: Performance monitoring and metrics

Usage:
    from realtime import StreamingVoiceAgent
    
    agent = StreamingVoiceAgent()
    await agent.start_streaming()
"""

__version__ = "0.1.0"
__author__ = "AI Assistant Project"

# Import main components when they're implemented
try:
    from .streaming_voice_agent import StreamingVoiceAgent
except ImportError:
    # Components not yet implemented
    StreamingVoiceAgent = None

try:
    from .unified_interface import UnifiedStreamingInterface
except ImportError:
    UnifiedStreamingInterface = None

__all__ = [
    "StreamingVoiceAgent",
    "UnifiedStreamingInterface",
]