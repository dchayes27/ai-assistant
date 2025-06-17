"""Memory management module for AI Assistant.

This module provides database management, models, migrations, backups, and vector storage
for persisting conversations, knowledge, and context.
"""

from .db_manager import DatabaseManager, get_db_manager
from .models import Conversation, Message, Knowledge, Project, Embedding
from .migrations import MigrationManager
from .backup import BackupManager
from .vector_store import VectorStore

__all__ = [
    # Database Management
    'DatabaseManager',
    'get_db_manager',
    
    # Models
    'Conversation',
    'Message',
    'Knowledge',
    'Project',
    'Embedding',
    
    # Migration System
    'MigrationManager',
    
    # Backup System
    'BackupManager',
    
    # Vector Storage
    'VectorStore',
]

__version__ = '1.0.0'