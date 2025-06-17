"""
Advanced Gradio GUI for AI Assistant
Modern, multi-tab interface with comprehensive features
"""

import os
import sys
import asyncio
import json
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import zipfile

import gradio as gr
import pandas as pd
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SmartAssistant, ConversationMode, AssistantConfig
from memory import DatabaseManager


class AssistantGUI:
    """Advanced Gradio GUI for AI Assistant"""
    
    def __init__(self):
        self.assistant: Optional[SmartAssistant] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.current_thread_id: Optional[str] = None
        self.current_project_id: Optional[str] = None
        self.theme = "default"
        self.settings = self._load_default_settings()
        self.conversation_history: List[Dict[str, Any]] = []
        
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default settings"""
        defaults = {
            "whisper_model": "medium",
            "ollama_model": "llama3.2:3b",
            "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
            "openai_api_key": "",
            "voice_model": "openai:alloy",
            "voice_enabled": True,
            "auto_save": True,
            "max_context_length": 20,
            "theme": "default",
            "language": "en"
        }
        
        # Try to load saved settings
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "gui_settings.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_settings = json.load(f)
                    defaults.update(saved_settings)
                    logger.info("Loaded saved settings from config file")
        except Exception as e:
            logger.warning(f"Failed to load saved settings: {e}")
        
        return defaults
    
    async def initialize_assistant(self) -> bool:
        """Initialize the assistant"""
        try:
            if self.assistant is None:
                config = AssistantConfig(
                    whisper_model=self.settings["whisper_model"],
                    ollama_model=self.settings["ollama_model"],
                    tts_model=self.settings.get("voice_model", self.settings["tts_model"]),
                    max_context_length=self.settings["max_context_length"],
                    openai_api_key=self.settings.get("openai_api_key", "")
                )
                
                self.assistant = SmartAssistant(config)
                await self.assistant.initialize()
                
                self.db_manager = self.assistant.db_manager
                
                logger.info("Assistant initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {e}")
            return False
    
    def create_interface(self) -> gr.Interface:
        """Create the main Gradio interface"""
        
        # Custom CSS for modern styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .chat-container {
            height: 600px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            background: #fafafa;
        }
        
        .message-user {
            background: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 18px 18px 5px 18px;
            margin: 5px 0;
            margin-left: 50px;
            text-align: right;
        }
        
        .message-assistant {
            background: #e9ecef;
            color: #333;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 5px;
            margin: 5px 0;
            margin-right: 50px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #28a745; }
        .status-processing { background-color: #ffc107; }
        .status-offline { background-color: #dc3545; }
        
        .audio-visualizer {
            height: 100px;
            background: linear-gradient(45deg, #007bff, #0056b3);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 5px;
            text-align: center;
        }
        
        .project-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .tab-nav {
            background: #f8f9fa;
            border-bottom: 2px solid #007bff;
        }
        
        .search-result {
            background: #fff;
            border-left: 4px solid #007bff;
            padding: 12px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .settings-section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .theme-dark {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        .theme-dark .chat-container {
            background: #2d2d2d;
            border-color: #555;
        }
        
        .theme-dark .message-assistant {
            background: #3a3a3a;
            color: #ffffff;
        }
        """
        
        with gr.Blocks(
            css=custom_css,
            title="AI Assistant",
            theme=gr.themes.Soft()
        ) as interface:
            
            # Global state components
            assistant_state = gr.State()
            theme_state = gr.State("default")
            
            # Header
            with gr.Row():
                gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1 style="color: #007bff; margin-bottom: 10px;">🤖 Advanced AI Assistant</h1>
                    <p style="color: #666; font-size: 16px;">Your intelligent companion with voice, memory, and project management</p>
                </div>
                """)
            
            # Status bar
            with gr.Row():
                status_indicator = gr.HTML(
                    '<span class="status-indicator status-offline"></span>Initializing...',
                    elem_classes=["status-bar"]
                )
                theme_toggle = gr.Button("🌙 Dark Mode", size="sm", variant="secondary")
            
            # Main interface with tabs
            with gr.Tabs() as main_tabs:
                
                # ==================== CHAT TAB ====================
                with gr.Tab("💬 Chat", id="chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Chat display
                            chatbot = gr.Chatbot(
                                height=500,
                                bubble_full_width=False,
                                show_copy_button=True,
                                elem_classes=["chat-container"],
                                type="messages"
                            )
                            
                            # Input area
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="Type your message here...",
                                    container=False,
                                    scale=4,
                                    max_lines=3
                                )
                                send_btn = gr.Button("Send", variant="primary", scale=1)
                            
                            # Audio input section
                            with gr.Row():
                                with gr.Column():
                                    audio_input = gr.Audio(
                                        type="filepath",
                                        label="Voice Message"
                                    )
                                    
                                with gr.Column():
                                    audio_visualizer = gr.HTML(
                                        '<div class="audio-visualizer">🎤 Ready to listen</div>'
                                    )
                            
                            # Audio output for TTS responses
                            audio_output = gr.Audio(
                                label="🔊 Assistant Voice Response - Click to Play",
                                visible=True,
                                autoplay=False,
                                interactive=True
                            )
                            
                            # Audio controls
                            with gr.Row():
                                voice_toggle = gr.Checkbox(
                                    label="Enable voice responses",
                                    value=True
                                )
                                auto_speech = gr.Checkbox(
                                    label="Auto-play responses",
                                    value=False
                                )
                        
                        with gr.Column(scale=1):
                            # Conversation controls
                            gr.Markdown("### Conversation Controls")
                            
                            mode_selector = gr.Dropdown(
                                choices=[mode.value for mode in ConversationMode],
                                value=ConversationMode.CHAT.value,
                                label="Conversation Mode"
                            )
                            
                            thread_selector = gr.Dropdown(
                                choices=[],
                                label="Conversation Thread",
                                interactive=True
                            )
                            
                            new_thread_btn = gr.Button("New Conversation", variant="secondary")
                            clear_chat_btn = gr.Button("Clear Chat", variant="secondary")
                            
                            # Export options
                            gr.Markdown("### Export")
                            export_format = gr.Radio(
                                choices=["JSON", "Markdown", "PDF"],
                                value="JSON",
                                label="Format"
                            )
                            export_btn = gr.Button("Export Conversation", variant="secondary")
                            export_file = gr.File(label="Download", visible=False)
                            
                            # Performance metrics
                            gr.Markdown("### Performance")
                            metrics_display = gr.HTML("")
                
                # ==================== PROJECTS TAB ====================
                with gr.Tab("📋 Projects", id="projects"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("## Project Management")
                            
                            # Project selector
                            project_dropdown = gr.Dropdown(
                                choices=[],
                                label="Select Project",
                                interactive=True
                            )
                            
                            # Project details
                            project_name = gr.Textbox(label="Project Name", placeholder="Enter project name")
                            project_description = gr.Textbox(
                                label="Description",
                                placeholder="Project description...",
                                lines=3
                            )
                            project_status = gr.Dropdown(
                                choices=["Planning", "Active", "On Hold", "Completed"],
                                value="Planning",
                                label="Status"
                            )
                            project_tags = gr.Textbox(
                                label="Tags",
                                placeholder="tag1, tag2, tag3"
                            )
                            
                            # Project actions
                            with gr.Row():
                                create_project_btn = gr.Button("Create Project", variant="primary")
                                update_project_btn = gr.Button("Update Project", variant="secondary")
                                delete_project_btn = gr.Button("Delete Project", variant="stop")
                        
                        with gr.Column(scale=3):
                            # Project list
                            gr.Markdown("## Active Projects")
                            projects_table = gr.DataFrame(
                                headers=["Name", "Status", "Created", "Last Updated"],
                                datatype=["str", "str", "str", "str"],
                                interactive=False
                            )
                            
                            refresh_projects_btn = gr.Button("Refresh Projects", variant="secondary")
                            
                            # Project conversations
                            gr.Markdown("## Project Conversations")
                            project_conversations = gr.HTML("<p>Select a project to view conversations</p>")
                
                # ==================== MEMORY BROWSER TAB ====================
                with gr.Tab("🧠 Memory Browser", id="memory"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Search Memory")
                            
                            search_query = gr.Textbox(
                                label="Search Query",
                                placeholder="What are you looking for?",
                                lines=2
                            )
                            
                            search_type = gr.Radio(
                                choices=["Semantic", "Full-text", "Hybrid"],
                                value="Hybrid",
                                label="Search Type"
                            )
                            
                            search_filters = gr.CheckboxGroup(
                                choices=["Conversations", "Projects", "Knowledge Base"],
                                value=["Conversations"],
                                label="Search In"
                            )
                            
                            with gr.Row():
                                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2024-01-01")
                                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="2024-12-31")
                            
                            search_btn = gr.Button("Search Memory", variant="primary")
                            
                            # Memory stats
                            gr.Markdown("## Memory Statistics")
                            memory_stats = gr.HTML("")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("## Search Results")
                            
                            search_results = gr.HTML(
                                "<p>Enter a search query to explore your memory</p>"
                            )
                            
                            # Result actions
                            with gr.Row():
                                export_results_btn = gr.Button("Export Results", variant="secondary")
                                clear_results_btn = gr.Button("Clear Results", variant="secondary")
                
                # ==================== SETTINGS TAB ====================
                with gr.Tab("⚙️ Settings", id="settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Model Settings", elem_classes=["settings-section"])
                            
                            whisper_model_setting = gr.Dropdown(
                                choices=["tiny", "base", "small", "medium", "large"],
                                value="medium",
                                label="Whisper Model (Speech-to-Text)"
                            )
                            
                            ollama_model_setting = gr.Dropdown(
                                choices=["llama3.2:3b", "llama3.2:7b", "mistral:7b", "codellama:7b"],
                                value="llama3.2:3b",
                                label="Ollama Model (Language Model)"
                            )
                            
                            tts_model_setting = gr.Dropdown(
                                choices=[
                                    "tts_models/en/ljspeech/tacotron2-DDC",
                                    "tts_models/en/ljspeech/glow-tts",
                                    "pyttsx3"
                                ],
                                value="tts_models/en/ljspeech/tacotron2-DDC",
                                label="TTS Model (Text-to-Speech)"
                            )
                            
                            openai_api_key_setting = gr.Textbox(
                                label="OpenAI API Key (for Professional TTS)",
                                placeholder="sk-...",
                                type="password",
                                value=""
                            )
                        
                        with gr.Column():
                            gr.Markdown("## Voice Settings", elem_classes=["settings-section"])
                            
                            voice_enabled_setting = gr.Checkbox(
                                label="Enable Voice Features",
                                value=True
                            )
                            
                            speech_rate_setting = gr.Slider(
                                minimum=50,
                                maximum=300,
                                value=150,
                                label="Speech Rate (words/min)"
                            )
                            
                            speech_volume_setting = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                label="Speech Volume"
                            )
                            
                            # Voice model selection
                            voice_model_setting = gr.Dropdown(
                                choices=[
                                    # OpenAI TTS (Premium)
                                    ("🎭 OpenAI Alloy (Professional)", "openai:alloy"),
                                    ("🎵 OpenAI Echo (Professional)", "openai:echo"),
                                    ("📚 OpenAI Fable (Professional)", "openai:fable"),
                                    ("🎤 OpenAI Nova (Professional)", "openai:nova"),
                                    ("💎 OpenAI Onyx (Professional)", "openai:onyx"),
                                    ("✨ OpenAI Shimmer (Professional)", "openai:shimmer"),
                                    # Local Coqui Models (Free)
                                    ("LJSpeech Tacotron2 (Local)", "tts_models/en/ljspeech/tacotron2-DDC"),
                                    ("LJSpeech VITS (Local)", "tts_models/en/ljspeech/vits"),
                                    ("VCTK Multi-Speaker (Local)", "tts_models/en/vctk/vits"),
                                    ("Jenny (Local)", "tts_models/en/jenny/jenny"),
                                    ("XTTS v2 (Local)", "tts_models/multilingual/multi-dataset/xtts_v2"),
                                    ("Bark (Local)", "tts_models/multilingual/multi-dataset/bark"),
                                    ("Fast Pitch (Local)", "tts_models/en/ljspeech/fast_pitch"),
                                    ("Glow-TTS (Local)", "tts_models/en/ljspeech/glow-tts")
                                ],
                                value="openai:alloy",
                                label="Voice Model"
                            )
                            
                            # Audio device selection (placeholder)
                            audio_device_setting = gr.Dropdown(
                                choices=["Default", "System Default"],
                                value="Default",
                                label="Audio Device"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Performance Settings", elem_classes=["settings-section"])
                            
                            max_context_setting = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                label="Max Context Length"
                            )
                            
                            auto_save_setting = gr.Checkbox(
                                label="Auto-save Conversations",
                                value=True
                            )
                            
                            performance_logging_setting = gr.Checkbox(
                                label="Enable Performance Logging",
                                value=True
                            )
                        
                        with gr.Column():
                            gr.Markdown("## Interface Settings", elem_classes=["settings-section"])
                            
                            language_setting = gr.Dropdown(
                                choices=["English", "Spanish", "French", "German"],
                                value="English",
                                label="Language"
                            )
                            
                            theme_setting = gr.Radio(
                                choices=["Light", "Dark", "Auto"],
                                value="Light",
                                label="Theme"
                            )
                            
                            animations_setting = gr.Checkbox(
                                label="Enable Animations",
                                value=True
                            )
                    
                    # Settings actions
                    with gr.Row():
                        save_settings_btn = gr.Button("Save Settings", variant="primary")
                        reset_settings_btn = gr.Button("Reset to Defaults", variant="secondary")
                        export_settings_btn = gr.Button("Export Settings", variant="secondary")
                        import_settings_btn = gr.Button("Import Settings", variant="secondary")
                    
                    settings_status = gr.HTML("")
            
            # ==================== EVENT HANDLERS ====================
            
            # Initialize assistant on load
            interface.load(
                fn=self._initialize_interface,
                outputs=[status_indicator, thread_selector, projects_table, memory_stats]
            )
            
            # Chat functionality
            msg_input.submit(
                fn=self._send_message,
                inputs=[msg_input, thread_selector, mode_selector, voice_toggle],
                outputs=[chatbot, msg_input, metrics_display, audio_output]
            )
            
            send_btn.click(
                fn=self._send_message,
                inputs=[msg_input, thread_selector, mode_selector, voice_toggle],
                outputs=[chatbot, msg_input, metrics_display, audio_output]
            )
            
            # Audio input
            audio_input.change(
                fn=self._process_audio,
                inputs=[audio_input, thread_selector, voice_toggle],
                outputs=[chatbot, audio_visualizer, metrics_display, audio_output]
            )
            
            # Thread management
            new_thread_btn.click(
                fn=self._create_new_thread,
                inputs=[mode_selector],
                outputs=[thread_selector, chatbot]
            )
            
            clear_chat_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, msg_input]
            )
            
            # Export functionality
            export_btn.click(
                fn=self._export_conversation,
                inputs=[export_format, thread_selector],
                outputs=[export_file]
            )
            
            # Project management
            create_project_btn.click(
                fn=self._create_project,
                inputs=[project_name, project_description, project_status, project_tags],
                outputs=[projects_table, project_dropdown]
            )
            
            refresh_projects_btn.click(
                fn=self._refresh_projects,
                outputs=[projects_table, project_dropdown]
            )
            
            # Memory search
            search_btn.click(
                fn=self._search_memory,
                inputs=[search_query, search_type, search_filters, start_date, end_date],
                outputs=[search_results]
            )
            
            # Settings
            save_settings_btn.click(
                fn=self._save_settings,
                inputs=[
                    whisper_model_setting, ollama_model_setting, tts_model_setting, openai_api_key_setting,
                    voice_enabled_setting, speech_rate_setting, speech_volume_setting, voice_model_setting,
                    max_context_setting, auto_save_setting, performance_logging_setting,
                    language_setting, theme_setting, animations_setting
                ],
                outputs=[settings_status]
            )
            
            # Theme toggle
            theme_toggle.click(
                fn=self._toggle_theme,
                inputs=[theme_state],
                outputs=[theme_state, theme_toggle],
                js="(theme) => { document.body.classList.toggle('theme-dark'); return [theme === 'dark' ? 'light' : 'dark', theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode']; }"
            )
            
            # Auto-refresh metrics
            interface.load(
                fn=self._update_metrics,
                outputs=[metrics_display]
            )
        
        return interface
    
    # ==================== EVENT HANDLER METHODS ====================
    
    async def _initialize_interface(self):
        """Initialize the interface"""
        try:
            success = await self.initialize_assistant()
            
            if success:
                # Get initial data
                threads = await self.assistant.get_conversation_threads()
                thread_choices = [(f"{t['title']} ({t['mode']})", t['thread_id']) for t in threads]
                
                projects_df = await self._get_projects_dataframe()
                
                memory_stats_html = await self._get_memory_stats_html()
                
                status_html = '<span class="status-indicator status-online"></span>Assistant Ready'
                
                return status_html, gr.Dropdown(choices=thread_choices), projects_df, memory_stats_html
            else:
                status_html = '<span class="status-indicator status-offline"></span>Initialization Failed'
                return status_html, gr.Dropdown(choices=[]), gr.DataFrame(), "<p>Failed to load stats</p>"
                
        except Exception as e:
            logger.error(f"Interface initialization failed: {e}")
            status_html = '<span class="status-indicator status-offline"></span>Error'
            return status_html, gr.Dropdown(choices=[]), gr.DataFrame(), "<p>Error loading stats</p>"
    
    async def _send_message(self, message: str, thread_id: str, mode: str, voice_enabled: bool):
        """Send a text message"""
        if not message.strip():
            return self.conversation_history, "", "", None
        
        try:
            if not self.assistant:
                await self.initialize_assistant()
            
            # Convert mode string to enum
            conv_mode = ConversationMode(mode.lower())
            
            response = await self.assistant.process_message(
                message,
                thread_id=thread_id,
                mode=conv_mode
            )
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Generate voice response if enabled
            audio_file = None
            if voice_enabled:
                try:
                    audio_file = await self.assistant.synthesize_speech(response)
                    logger.info(f"Generated audio file: {audio_file}")
                    if audio_file and os.path.exists(audio_file):
                        logger.info(f"Audio file exists, size: {os.path.getsize(audio_file)} bytes")
                    else:
                        logger.error(f"Audio file not found: {audio_file}")
                except Exception as e:
                    logger.error(f"Voice synthesis failed: {e}")
            
            # Get updated metrics
            metrics_html = await self._get_metrics_html()
            
            return self.conversation_history, "", metrics_html, audio_file
            
        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            error_response = f"Sorry, I encountered an error: {str(e)}"
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": error_response})
            return self.conversation_history, "", "", None
    
    async def _process_audio(self, audio_file: str, thread_id: str, voice_enabled: bool):
        """Process audio input"""
        if not audio_file:
            return self.conversation_history, "🎤 Ready to listen", "", None
        
        try:
            if not self.assistant:
                await self.initialize_assistant()
            
            # Update visualizer
            visualizer_html = '<div class="audio-visualizer">🔄 Processing audio...</div>'
            
            # Read audio file
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            
            # Process voice message
            result = await self.assistant.process_voice_message(
                audio_data,
                thread_id=thread_id,
                synthesize_response=voice_enabled
            )
            
            if "error" in result:
                error_msg = f"Audio processing error: {result['error']}"
                self.conversation_history.append({"role": "user", "content": "[Audio Input]"})
                self.conversation_history.append({"role": "assistant", "content": error_msg})
                visualizer_html = '<div class="audio-visualizer">❌ Error processing audio</div>'
                metrics_html = await self._get_metrics_html()
                return self.conversation_history, visualizer_html, metrics_html, None
            else:
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": f"🎤 {result['transcript']}"})
                self.conversation_history.append({"role": "assistant", "content": result['response']})
                visualizer_html = '<div class="audio-visualizer">✅ Audio processed successfully</div>'
            
            # Get audio file for playback
            audio_response = None
            if voice_enabled and "audio_file" in result:
                audio_response = result["audio_file"]
            
            metrics_html = await self._get_metrics_html()
            return self.conversation_history, visualizer_html, metrics_html, audio_response
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            error_response = f"Audio processing failed: {str(e)}"
            self.conversation_history.append({"role": "user", "content": "[Audio Input]"})
            self.conversation_history.append({"role": "assistant", "content": error_response})
            visualizer_html = '<div class="audio-visualizer">❌ Processing failed</div>'
            return self.conversation_history, visualizer_html, "", None
    
    async def _create_new_thread(self, mode: str):
        """Create a new conversation thread"""
        try:
            if not self.assistant:
                await self.initialize_assistant()
            
            conv_mode = ConversationMode(mode.lower())
            thread_id = await self.assistant.create_conversation_thread(
                mode=conv_mode,
                title=f"New {mode} Conversation"
            )
            
            self.current_thread_id = thread_id
            self.conversation_history = []
            
            # Update thread selector
            threads = await self.assistant.get_conversation_threads()
            thread_choices = [(f"{t['title']} ({t['mode']})", t['thread_id']) for t in threads]
            
            return gr.Dropdown(choices=thread_choices, value=thread_id), []
            
        except Exception as e:
            logger.error(f"Thread creation failed: {e}")
            return gr.Dropdown(choices=[]), []
    
    async def _export_conversation(self, format_type: str, thread_id: str):
        """Export conversation in specified format"""
        try:
            if not self.assistant or not thread_id:
                return None
            
            messages = await self.assistant.db_manager.get_conversation_messages(thread_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "JSON":
                filename = f"conversation_{timestamp}.json"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(messages, f, indent=2, default=str)
                    return f.name
                    
            elif format_type == "Markdown":
                filename = f"conversation_{timestamp}.md"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                    f.write(f"# Conversation Export\n\n")
                    f.write(f"**Exported:** {datetime.now().isoformat()}\n\n")
                    
                    for msg in messages:
                        role = msg['role'].title()
                        content = msg['content']
                        timestamp = msg.get('timestamp', '')
                        f.write(f"## {role}\n")
                        f.write(f"*{timestamp}*\n\n")
                        f.write(f"{content}\n\n---\n\n")
                    
                    return f.name
            
            return None
            
        except Exception as e:
            logger.error(f"Conversation export failed: {e}")
            return None
    
    async def _create_project(self, name: str, description: str, status: str, tags: str):
        """Create a new project"""
        try:
            if not self.assistant or not name.strip():
                return gr.DataFrame(), gr.Dropdown(choices=[])
            
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            
            project_id = await self.assistant.db_manager.create_project(
                name=name,
                description=description,
                metadata={"status": status, "tags": tag_list}
            )
            
            # Refresh project list
            projects_df = await self._get_projects_dataframe()
            
            # Update project dropdown
            projects = await self.assistant.db_manager.get_projects()
            project_choices = [(p['name'], p['project_id']) for p in projects]
            
            return projects_df, gr.Dropdown(choices=project_choices)
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            return gr.DataFrame(), gr.Dropdown(choices=[])
    
    async def _refresh_projects(self):
        """Refresh project list"""
        try:
            projects_df = await self._get_projects_dataframe()
            
            if self.assistant:
                projects = await self.assistant.db_manager.get_projects()
                project_choices = [(p['name'], p['project_id']) for p in projects]
            else:
                project_choices = []
            
            return projects_df, gr.Dropdown(choices=project_choices)
            
        except Exception as e:
            logger.error(f"Project refresh failed: {e}")
            return gr.DataFrame(), gr.Dropdown(choices=[])
    
    async def _search_memory(self, query: str, search_type: str, filters: List[str], start_date: str, end_date: str):
        """Search memory"""
        try:
            if not self.assistant or not query.strip():
                return "<p>Please enter a search query</p>"
            
            search_type_map = {
                "Semantic": "semantic",
                "Full-text": "fts",
                "Hybrid": "hybrid"
            }
            
            results = await self.assistant.search_memory(
                query=query,
                search_type=search_type_map[search_type],
                limit=20
            )
            
            if not results:
                return "<p>No results found</p>"
            
            # Format results as HTML
            html_content = f"<div style='margin-bottom: 20px;'><strong>Found {len(results)} results for:</strong> <em>{query}</em></div>"
            
            for i, result in enumerate(results):
                relevance = result.get('similarity_score', 0) * 100 if result.get('similarity_score') else 'N/A'
                timestamp = result.get('timestamp', 'Unknown')
                content = result.get('content', '')[:200] + ('...' if len(result.get('content', '')) > 200 else '')
                
                html_content += f"""
                <div class="search-result">
                    <div style="display: flex; justify-content: between; margin-bottom: 8px;">
                        <strong>Result #{i+1}</strong>
                        <span style="color: #666; font-size: 12px;">{timestamp}</span>
                    </div>
                    <div style="margin-bottom: 8px;">{content}</div>
                    <div style="font-size: 12px; color: #888;">
                        Relevance: {relevance}% | Thread: {result.get('conversation_id', 'Unknown')}
                    </div>
                </div>
                """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return f"<p>Search failed: {str(e)}</p>"
    
    async def _save_settings(self, *settings_values):
        """Save settings"""
        try:
            # Update settings dictionary
            setting_keys = [
                "whisper_model", "ollama_model", "tts_model", "openai_api_key",
                "voice_enabled", "speech_rate", "speech_volume", "voice_model",
                "max_context_length", "auto_save", "performance_logging",
                "language", "theme", "animations"
            ]
            
            for key, value in zip(setting_keys, settings_values):
                self.settings[key] = value
            
            # Save to file
            settings_file = "config/gui_settings.json"
            os.makedirs("config", exist_ok=True)
            
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            # Reinitialize assistant if model settings changed
            if any(key in ["whisper_model", "ollama_model", "tts_model", "voice_model"] for key in setting_keys):
                if self.assistant:
                    await self.assistant.shutdown()
                    self.assistant = None
                await self.initialize_assistant()
            
            return '<span style="color: green;">✅ Settings saved successfully!</span>'
            
        except Exception as e:
            logger.error(f"Settings save failed: {e}")
            return f'<span style="color: red;">❌ Failed to save settings: {str(e)}</span>'
    
    def _toggle_theme(self, current_theme: str):
        """Toggle between light and dark theme"""
        new_theme = "light" if current_theme == "dark" else "dark"
        button_text = "☀️ Light Mode" if new_theme == "dark" else "🌙 Dark Mode"
        return new_theme, button_text
    
    # ==================== UTILITY METHODS ====================
    
    async def _get_projects_dataframe(self):
        """Get projects as DataFrame"""
        try:
            if self.assistant:
                projects = await self.assistant.db_manager.get_projects()
                if projects:
                    data = []
                    for p in projects:
                        data.append([
                            p['name'],
                            p.get('metadata', {}).get('status', 'Unknown'),
                            p['created_at'].strftime('%Y-%m-%d') if p.get('created_at') else 'Unknown',
                            p['updated_at'].strftime('%Y-%m-%d') if p.get('updated_at') else 'Unknown'
                        ])
                    return pd.DataFrame(data, columns=["Name", "Status", "Created", "Last Updated"])
            
            return pd.DataFrame(columns=["Name", "Status", "Created", "Last Updated"])
            
        except Exception as e:
            logger.error(f"Failed to get projects DataFrame: {e}")
            return pd.DataFrame(columns=["Name", "Status", "Created", "Last Updated"])
    
    async def _get_memory_stats_html(self):
        """Get memory statistics as HTML"""
        try:
            if self.assistant:
                stats = self.assistant.db_manager.get_stats()
                return f"""
                <div class="metric-card">
                    <h4>Memory Statistics</h4>
                    <p><strong>Conversations:</strong> {stats.get('conversations', 0)}</p>
                    <p><strong>Messages:</strong> {stats.get('messages', 0)}</p>
                    <p><strong>Knowledge Items:</strong> {stats.get('knowledge', 0)}</p>
                    <p><strong>Projects:</strong> {stats.get('projects', 0)}</p>
                </div>
                """
            return "<p>Memory statistics unavailable</p>"
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return "<p>Error loading memory statistics</p>"
    
    async def _get_metrics_html(self):
        """Get performance metrics as HTML"""
        try:
            if self.assistant:
                metrics = await self.assistant.get_performance_metrics()
                return f"""
                <div class="metric-card">
                    <h4>Performance</h4>
                    <p><strong>Queries:</strong> {metrics.get('total_queries', 0)}</p>
                    <p><strong>Success Rate:</strong> {metrics.get('success_rate', 0):.1%}</p>
                    <p><strong>Avg Response:</strong> {metrics.get('average_response_time', 0):.2f}s</p>
                    <p><strong>Memory:</strong> {metrics.get('memory_usage_mb', 0):.1f}MB</p>
                </div>
                """
            return ""
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return ""
    
    async def _update_metrics(self):
        """Update performance metrics display"""
        return await self._get_metrics_html()


# ==================== INTERFACE CREATION FUNCTIONS ====================

def create_interface() -> gr.Interface:
    """Create and return the Gradio interface"""
    gui = AssistantGUI()
    return gui.create_interface()


def launch_interface(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    debug: bool = False
):
    """Launch the Gradio interface"""
    logger.info("Launching AI Assistant GUI...")
    
    interface = create_interface()
    
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        debug=debug,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/gui.log", rotation="10 MB", retention="7 days")
    
    # Launch interface
    launch_interface(debug=True)