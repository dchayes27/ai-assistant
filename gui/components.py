"""
Reusable GUI components for the AI Assistant interface
"""

import gradio as gr
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime


class ChatMessage:
    """Represents a chat message with rich formatting"""
    
    def __init__(self, role: str, content: str, timestamp: datetime = None, metadata: Dict[str, Any] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_html(self) -> str:
        """Convert message to HTML representation"""
        role_class = f"message-{self.role.lower()}"
        time_str = self.timestamp.strftime("%H:%M")
        
        return f"""
        <div class="{role_class}">
            <div class="message-content">{self.content}</div>
            <div class="message-time">{time_str}</div>
        </div>
        """


class StatusIndicator:
    """Status indicator component"""
    
    @staticmethod
    def create(status: str = "offline", message: str = "Disconnected") -> gr.HTML:
        """Create a status indicator"""
        status_class = f"status-{status}"
        return gr.HTML(
            f'<span class="status-indicator {status_class}"></span>{message}',
            elem_classes=["status-bar"]
        )


class MetricsCard:
    """Performance metrics display card"""
    
    @staticmethod
    def create(title: str = "Metrics") -> gr.HTML:
        """Create a metrics card"""
        return gr.HTML(
            f"""
            <div class="metric-card">
                <h4>{title}</h4>
                <div id="metrics-content">Loading...</div>
            </div>
            """,
            elem_classes=["metrics-display"]
        )


class ProjectCard:
    """Project information card"""
    
    @staticmethod
    def create(project_data: Dict[str, Any]) -> str:
        """Create HTML for a project card"""
        name = project_data.get("name", "Unnamed Project")
        status = project_data.get("status", "Unknown")
        description = project_data.get("description", "No description")
        created = project_data.get("created_at", "Unknown")
        
        status_color = {
            "Planning": "#6c757d",
            "Active": "#28a745", 
            "On Hold": "#ffc107",
            "Completed": "#007bff"
        }.get(status, "#6c757d")
        
        return f"""
        <div class="project-card">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 10px;">
                <h5 style="margin: 0; color: #333;">{name}</h5>
                <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">{status}</span>
            </div>
            <p style="color: #666; margin: 8px 0; font-size: 14px;">{description}</p>
            <small style="color: #999;">Created: {created}</small>
        </div>
        """


class SearchResult:
    """Search result component"""
    
    @staticmethod
    def create(result_data: Dict[str, Any]) -> str:
        """Create HTML for a search result"""
        content = result_data.get("content", "")
        timestamp = result_data.get("timestamp", "Unknown")
        relevance = result_data.get("similarity_score", 0)
        thread_id = result_data.get("conversation_id", "Unknown")
        
        # Truncate content if too long
        display_content = content[:200] + ("..." if len(content) > 200 else "")
        
        relevance_percent = f"{relevance * 100:.1f}%" if relevance else "N/A"
        
        return f"""
        <div class="search-result">
            <div style="display: flex; justify-content: between; margin-bottom: 8px;">
                <strong>Search Result</strong>
                <span style="color: #666; font-size: 12px;">{timestamp}</span>
            </div>
            <div style="margin-bottom: 8px; line-height: 1.4;">{display_content}</div>
            <div style="font-size: 12px; color: #888;">
                Relevance: {relevance_percent} | Thread: {thread_id[:8]}...
            </div>
        </div>
        """


class AudioVisualizer:
    """Audio input visualizer component"""
    
    @staticmethod
    def create_ready() -> str:
        """Create ready state visualizer"""
        return '<div class="audio-visualizer">🎤 Ready to listen</div>'
    
    @staticmethod
    def create_processing() -> str:
        """Create processing state visualizer"""
        return '<div class="audio-visualizer">🔄 Processing audio...</div>'
    
    @staticmethod
    def create_success() -> str:
        """Create success state visualizer"""
        return '<div class="audio-visualizer">✅ Audio processed successfully</div>'
    
    @staticmethod
    def create_error(message: str = "Error") -> str:
        """Create error state visualizer"""
        return f'<div class="audio-visualizer">❌ {message}</div>'


class ThemeManager:
    """Theme management utilities"""
    
    @staticmethod
    def get_custom_css() -> str:
        """Get custom CSS for the interface"""
        return """
        /* Main container styling */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Chat styling */
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
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .message-assistant {
            background: #e9ecef;
            color: #333;
            padding: 10px 15px;
            border-radius: 18px 18px 18px 5px;
            margin: 5px 0;
            margin-right: 50px;
            max-width: 80%;
            word-wrap: break-word;
        }
        
        .message-time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 4px;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-online { background-color: #28a745; }
        .status-processing { background-color: #ffc107; }
        .status-offline { background-color: #dc3545; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        /* Audio visualizer */
        .audio-visualizer {
            height: 100px;
            background: linear-gradient(45deg, #007bff, #0056b3);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(0,123,255,0.3);
            transition: all 0.3s ease;
        }
        
        .audio-visualizer:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,123,255,0.4);
        }
        
        /* Cards */
        .metric-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 5px;
            text-align: center;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .project-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .project-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Search results */
        .search-result {
            background: #fff;
            border-left: 4px solid #007bff;
            padding: 12px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .search-result:hover {
            transform: translateX(4px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        
        /* Settings sections */
        .settings-section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid #e9ecef;
        }
        
        /* Buttons */
        .gr-button {
            transition: all 0.2s ease;
        }
        
        .gr-button:hover {
            transform: translateY(-1px);
        }
        
        /* Tabs */
        .tab-nav {
            background: #f8f9fa;
            border-bottom: 2px solid #007bff;
        }
        
        /* Dark theme */
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
        
        .theme-dark .metric-card {
            background: #2d2d2d;
            border-color: #555;
            color: #ffffff;
        }
        
        .theme-dark .project-card {
            background: #2d2d2d;
            border-color: #555;
        }
        
        .theme-dark .search-result {
            background: #2d2d2d;
            color: #ffffff;
        }
        
        .theme-dark .settings-section {
            background: #2d2d2d;
            border-color: #555;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 10px;
            }
            
            .message-user, .message-assistant {
                margin-left: 10px;
                margin-right: 10px;
                max-width: 90%;
            }
            
            .audio-visualizer {
                height: 80px;
                font-size: 14px;
            }
        }
        
        /* Animations */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .slide-in {
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        """


class UtilityComponents:
    """Utility component builders"""
    
    @staticmethod
    def create_header(title: str, subtitle: str = "") -> gr.HTML:
        """Create a styled header"""
        subtitle_html = f'<p style="color: #666; font-size: 16px; margin-top: 10px;">{subtitle}</p>' if subtitle else ""
        
        return gr.HTML(f"""
        <div style="text-align: center; padding: 20px; border-bottom: 1px solid #e0e0e0; margin-bottom: 20px;">
            <h1 style="color: #007bff; margin-bottom: 10px; font-size: 2.5em;">{title}</h1>
            {subtitle_html}
        </div>
        """)
    
    @staticmethod
    def create_section_header(title: str, icon: str = "") -> gr.Markdown:
        """Create a section header"""
        return gr.Markdown(f"## {icon} {title}" if icon else f"## {title}")
    
    @staticmethod
    def create_info_box(content: str, type: str = "info") -> gr.HTML:
        """Create an information box"""
        colors = {
            "info": "#d1ecf1",
            "success": "#d4edda", 
            "warning": "#fff3cd",
            "error": "#f8d7da"
        }
        
        border_colors = {
            "info": "#bee5eb",
            "success": "#c3e6cb",
            "warning": "#ffeaa7",
            "error": "#f5c6cb"
        }
        
        return gr.HTML(f"""
        <div style="
            background-color: {colors.get(type, colors['info'])};
            border: 1px solid {border_colors.get(type, border_colors['info'])};
            border-radius: 6px;
            padding: 12px;
            margin: 10px 0;
        ">
            {content}
        </div>
        """)
    
    @staticmethod
    def create_loading_spinner() -> gr.HTML:
        """Create a loading spinner"""
        return gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <div style="
                border: 4px solid #f3f3f3;
                border-top: 4px solid #007bff;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            "></div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <p style="margin-top: 10px; color: #666;">Loading...</p>
        </div>
        """)


# Component factory functions
def create_chat_interface() -> Dict[str, Any]:
    """Create chat interface components"""
    return {
        "chatbot": gr.Chatbot(
            height=500,
            bubble_full_width=False,
            show_copy_button=True,
            elem_classes=["chat-container"],
            type="messages"
        ),
        "message_input": gr.Textbox(
            placeholder="Type your message here...",
            container=False,
            scale=4,
            max_lines=3
        ),
        "send_button": gr.Button("Send", variant="primary", scale=1),
        "audio_input": gr.Audio(
            type="filepath",
            label="Voice Message"
        )
    }


def create_project_interface() -> Dict[str, Any]:
    """Create project management interface components"""
    return {
        "project_dropdown": gr.Dropdown(
            choices=[],
            label="Select Project",
            interactive=True
        ),
        "project_name": gr.Textbox(label="Project Name"),
        "project_description": gr.Textbox(label="Description", lines=3),
        "project_status": gr.Dropdown(
            choices=["Planning", "Active", "On Hold", "Completed"],
            value="Planning",
            label="Status"
        ),
        "projects_table": gr.DataFrame(
            headers=["Name", "Status", "Created", "Last Updated"],
            interactive=False
        )
    }


def create_memory_interface() -> Dict[str, Any]:
    """Create memory browser interface components"""
    return {
        "search_query": gr.Textbox(
            label="Search Query",
            placeholder="What are you looking for?",
            lines=2
        ),
        "search_type": gr.Radio(
            choices=["Semantic", "Full-text", "Hybrid"],
            value="Hybrid",
            label="Search Type"
        ),
        "search_results": gr.HTML(
            "<p>Enter a search query to explore your memory</p>"
        )
    }