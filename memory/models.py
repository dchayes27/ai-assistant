"""SQLAlchemy ORM models for AI Assistant memory storage.

Defines the data models for conversations, messages, knowledge, projects, and embeddings.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import json

Base = declarative_base()


class Conversation(Base):
    """Model for storing conversation information."""
    
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    meta_data = Column(JSON, default=dict)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, conversation_id='{self.conversation_id}', title='{self.title}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.meta_data or {},
            'message_count': len(self.messages) if self.messages else 0
        }
    
    @validates('conversation_id')
    def validate_conversation_id(self, key: str, value: str) -> str:
        """Validate conversation ID."""
        if not value or not value.strip():
            raise ValueError("Conversation ID cannot be empty")
        return value.strip()


class Message(Base):
    """Model for storing messages within conversations."""
    
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(255), ForeignKey('conversations.conversation_id'), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    meta_data = Column(JSON, default=dict)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, conversation_id='{self.conversation_id}', role='{self.role}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.meta_data or {}
        }
    
    @validates('role')
    def validate_role(self, key: str, value: str) -> str:
        """Validate message role."""
        valid_roles = ['user', 'assistant', 'system']
        if value not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return value
    
    @validates('content')
    def validate_content(self, key: str, value: str) -> str:
        """Validate message content."""
        if not value or not value.strip():
            raise ValueError("Message content cannot be empty")
        return value


class Knowledge(Base):
    """Model for storing knowledge base entries."""
    
    __tablename__ = 'knowledge_base'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=True, index=True)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    meta_data = Column(JSON, default=dict)
    
    # Relationships
    embeddings = relationship("Embedding", 
                            primaryjoin="and_(Embedding.entity_type=='knowledge', "
                                      "Embedding.entity_id==Knowledge.id)",
                            foreign_keys="[Embedding.entity_id]",
                            cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Knowledge(id={self.id}, title='{self.title}', category='{self.category}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'tags': self.tags or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.meta_data or {}
        }
    
    @validates('title')
    def validate_title(self, key: str, value: str) -> str:
        """Validate knowledge title."""
        if not value or not value.strip():
            raise ValueError("Knowledge title cannot be empty")
        return value.strip()
    
    @validates('content')
    def validate_content(self, key: str, value: str) -> str:
        """Validate knowledge content."""
        if not value or not value.strip():
            raise ValueError("Knowledge content cannot be empty")
        return value
    
    @validates('tags')
    def validate_tags(self, key: str, value: Any) -> List[str]:
        """Validate and normalize tags."""
        if isinstance(value, str):
            # If JSON string, parse it
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                # If not JSON, treat as single tag
                value = [value]
        
        if not isinstance(value, list):
            value = [value] if value else []
        
        # Ensure all tags are strings and remove duplicates
        return list(set(str(tag).strip() for tag in value if tag))


class Project(Base):
    """Model for storing project information."""
    
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    path = Column(String(500), nullable=True)
    status = Column(String(50), default='active')  # 'active', 'archived', 'completed'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    meta_data = Column(JSON, default=dict)
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, project_id='{self.project_id}', name='{self.name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'path': self.path,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.meta_data or {}
        }
    
    @validates('project_id')
    def validate_project_id(self, key: str, value: str) -> str:
        """Validate project ID."""
        if not value or not value.strip():
            raise ValueError("Project ID cannot be empty")
        return value.strip()
    
    @validates('name')
    def validate_name(self, key: str, value: str) -> str:
        """Validate project name."""
        if not value or not value.strip():
            raise ValueError("Project name cannot be empty")
        return value.strip()
    
    @validates('status')
    def validate_status(self, key: str, value: str) -> str:
        """Validate project status."""
        valid_statuses = ['active', 'archived', 'completed']
        if value not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return value


class Embedding(Base):
    """Model for storing vector embeddings."""
    
    __tablename__ = 'embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(50), nullable=False, index=True)  # 'message', 'knowledge', 'project'
    entity_id = Column(Integer, nullable=False, index=True)
    embedding = Column(JSON, nullable=False)  # Store as JSON array
    model = Column(String(100), default='text-embedding-ada-002')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, entity_type='{self.entity_type}', entity_id={self.entity_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'embedding': self.embedding,
            'model': self.model,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @validates('entity_type')
    def validate_entity_type(self, key: str, value: str) -> str:
        """Validate entity type."""
        valid_types = ['message', 'knowledge', 'project', 'conversation']
        if value not in valid_types:
            raise ValueError(f"Entity type must be one of: {', '.join(valid_types)}")
        return value
    
    @validates('embedding')
    def validate_embedding(self, key: str, value: Any) -> List[float]:
        """Validate embedding vector."""
        if isinstance(value, str):
            # If JSON string, parse it
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid embedding format")
        
        if not isinstance(value, list):
            raise ValueError("Embedding must be a list of floats")
        
        if not value:
            raise ValueError("Embedding cannot be empty")
        
        # Ensure all values are floats
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            raise ValueError("All embedding values must be numeric")
    
    def get_vector(self) -> List[float]:
        """Get the embedding vector as a list of floats."""
        if isinstance(self.embedding, str):
            return json.loads(self.embedding)
        return self.embedding