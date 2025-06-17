"""Database migration system for AI Assistant memory.

Provides version tracking, schema updates, and rollback support for database migrations.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from contextlib import contextmanager


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, description: str, 
                 up: Callable[[sqlite3.Connection], None],
                 down: Optional[Callable[[sqlite3.Connection], None]] = None):
        self.version = version
        self.description = description
        self.up = up
        self.down = down
        self.applied_at: Optional[datetime] = None
    
    def __repr__(self) -> str:
        return f"<Migration(version='{self.version}', description='{self.description}')>"


class MigrationManager:
    """Manages database migrations with version tracking and rollback support."""
    
    def __init__(self, db_path: str = "~/ai-assistant/memory/assistant.db"):
        self.db_path = Path(db_path).expanduser()
        self.migrations: List[Migration] = []
        self._initialize_migration_table()
        self._register_migrations()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_migration_table(self) -> None:
        """Create the migrations tracking table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_sql TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.commit()
            logger.info("Migration table initialized")
    
    def _register_migrations(self) -> None:
        """Register all available migrations."""
        # Migration 001: Add user preferences table
        self.add_migration(
            Migration(
                version="001",
                description="Add user preferences table",
                up=self._migration_001_up,
                down=self._migration_001_down
            )
        )
        
        # Migration 002: Add conversation summary field
        self.add_migration(
            Migration(
                version="002",
                description="Add summary field to conversations",
                up=self._migration_002_up,
                down=self._migration_002_down
            )
        )
        
        # Migration 003: Add knowledge source tracking
        self.add_migration(
            Migration(
                version="003",
                description="Add source tracking to knowledge base",
                up=self._migration_003_up,
                down=self._migration_003_down
            )
        )
        
        # Migration 004: Add conversation tags
        self.add_migration(
            Migration(
                version="004",
                description="Add tags to conversations",
                up=self._migration_004_up,
                down=self._migration_004_down
            )
        )
        
        # Migration 005: Add embedding dimensions and indexes
        self.add_migration(
            Migration(
                version="005",
                description="Add embedding dimensions and performance indexes",
                up=self._migration_005_up,
                down=self._migration_005_down
            )
        )
    
    def add_migration(self, migration: Migration) -> None:
        """Add a migration to the manager."""
        self.migrations.append(migration)
        logger.debug(f"Registered migration: {migration.version} - {migration.description}")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row[0] for row in cursor.fetchall()]
    
    def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = set(self.get_applied_migrations())
        return [m for m in self.migrations if m.version not in applied]
    
    def migrate(self, target_version: Optional[str] = None) -> int:
        """Apply pending migrations up to target version."""
        pending = self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations")
            return 0
        
        applied_count = 0
        for migration in pending:
            self._apply_migration(migration)
            applied_count += 1
        
        logger.info(f"Applied {applied_count} migrations")
        return applied_count
    
    def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        logger.info(f"Applying migration {migration.version}: {migration.description}")
        
        with self._get_connection() as conn:
            try:
                # Begin transaction
                conn.execute("BEGIN TRANSACTION")
                
                # Apply the migration
                migration.up(conn)
                
                # Record the migration
                rollback_sql = self._generate_rollback_sql(migration)
                conn.execute("""
                    INSERT INTO schema_migrations (version, description, rollback_sql)
                    VALUES (?, ?, ?)
                """, (migration.version, migration.description, rollback_sql))
                
                # Commit transaction
                conn.commit()
                logger.info(f"Migration {migration.version} applied successfully")
                
            except Exception as e:
                # Rollback on error
                conn.rollback()
                logger.error(f"Failed to apply migration {migration.version}: {e}")
                raise
    
    def rollback(self, steps: int = 1) -> int:
        """Rollback the specified number of migrations."""
        applied = self.get_applied_migrations()
        
        if not applied:
            logger.info("No migrations to rollback")
            return 0
        
        # Get migrations to rollback (most recent first)
        to_rollback = applied[-steps:] if steps <= len(applied) else applied
        to_rollback.reverse()
        
        rolled_back = 0
        for version in to_rollback:
            migration = next((m for m in self.migrations if m.version == version), None)
            if migration:
                self._rollback_migration(migration)
                rolled_back += 1
            else:
                logger.warning(f"Migration {version} not found in registered migrations")
        
        logger.info(f"Rolled back {rolled_back} migrations")
        return rolled_back
    
    def _rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration."""
        logger.info(f"Rolling back migration {migration.version}: {migration.description}")
        
        with self._get_connection() as conn:
            try:
                # Begin transaction
                conn.execute("BEGIN TRANSACTION")
                
                if migration.down:
                    # Use custom down migration if provided
                    migration.down(conn)
                else:
                    # Use stored rollback SQL
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT rollback_sql FROM schema_migrations 
                        WHERE version = ?
                    """, (migration.version,))
                    row = cursor.fetchone()
                    if row and row[0]:
                        for sql in json.loads(row[0]):
                            conn.execute(sql)
                
                # Remove migration record
                conn.execute("""
                    DELETE FROM schema_migrations WHERE version = ?
                """, (migration.version,))
                
                # Commit transaction
                conn.commit()
                logger.info(f"Migration {migration.version} rolled back successfully")
                
            except Exception as e:
                # Rollback on error
                conn.rollback()
                logger.error(f"Failed to rollback migration {migration.version}: {e}")
                raise
    
    def _generate_rollback_sql(self, migration: Migration) -> str:
        """Generate rollback SQL for a migration (simplified)."""
        # This is a placeholder - in practice, you'd analyze the migration
        # to generate appropriate rollback SQL
        return json.dumps([])
    
    def status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version, description, applied_at 
                FROM schema_migrations 
                ORDER BY version DESC 
                LIMIT 1
            """)
            latest = cursor.fetchone()
        
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(applied),
            "pending_migrations": len(pending),
            "latest_migration": dict(latest) if latest else None,
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending]
        }
    
    def reset(self) -> None:
        """Reset all migrations (DANGEROUS - will lose all data)."""
        logger.warning("Resetting all migrations - this will delete all data!")
        
        # Rollback all migrations
        applied_count = len(self.get_applied_migrations())
        if applied_count > 0:
            self.rollback(steps=applied_count)
        
        # Drop all tables
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Drop each table
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            conn.commit()
        
        logger.info("Database reset complete")
    
    # Migration definitions
    def _migration_001_up(self, conn: sqlite3.Connection) -> None:
        """Add user preferences table."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                type TEXT DEFAULT 'string',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX idx_preferences_key ON user_preferences(key)
        """)
    
    def _migration_001_down(self, conn: sqlite3.Connection) -> None:
        """Remove user preferences table."""
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS user_preferences")
    
    def _migration_002_up(self, conn: sqlite3.Connection) -> None:
        """Add summary field to conversations."""
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE conversations 
            ADD COLUMN summary TEXT
        """)
    
    def _migration_002_down(self, conn: sqlite3.Connection) -> None:
        """Remove summary field from conversations."""
        # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE conversations_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            INSERT INTO conversations_new (id, conversation_id, title, created_at, updated_at, metadata)
            SELECT id, conversation_id, title, created_at, updated_at, metadata
            FROM conversations
        """)
        cursor.execute("DROP TABLE conversations")
        cursor.execute("ALTER TABLE conversations_new RENAME TO conversations")
    
    def _migration_003_up(self, conn: sqlite3.Connection) -> None:
        """Add source tracking to knowledge base."""
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE knowledge_base 
            ADD COLUMN source TEXT
        """)
        cursor.execute("""
            ALTER TABLE knowledge_base 
            ADD COLUMN source_url TEXT
        """)
    
    def _migration_003_down(self, conn: sqlite3.Connection) -> None:
        """Remove source tracking from knowledge base."""
        # Recreate table without source columns
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE knowledge_base_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            INSERT INTO knowledge_base_new (id, title, content, category, tags, created_at, updated_at, metadata)
            SELECT id, title, content, category, tags, created_at, updated_at, metadata
            FROM knowledge_base
        """)
        cursor.execute("DROP TABLE knowledge_base")
        cursor.execute("ALTER TABLE knowledge_base_new RENAME TO knowledge_base")
    
    def _migration_004_up(self, conn: sqlite3.Connection) -> None:
        """Add tags to conversations."""
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE conversations 
            ADD COLUMN tags TEXT DEFAULT '[]'
        """)
    
    def _migration_004_down(self, conn: sqlite3.Connection) -> None:
        """Remove tags from conversations."""
        # Recreate table without tags column
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE conversations_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                title TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            INSERT INTO conversations_new (id, conversation_id, title, summary, created_at, updated_at, metadata)
            SELECT id, conversation_id, title, summary, created_at, updated_at, metadata
            FROM conversations
        """)
        cursor.execute("DROP TABLE conversations")
        cursor.execute("ALTER TABLE conversations_new RENAME TO conversations")
    
    def _migration_005_up(self, conn: sqlite3.Connection) -> None:
        """Add embedding dimensions and performance indexes."""
        cursor = conn.cursor()
        
        # Add dimension column to embeddings
        cursor.execute("""
            ALTER TABLE embeddings 
            ADD COLUMN dimension INTEGER
        """)
        
        # Add composite index for faster similarity searches
        cursor.execute("""
            CREATE INDEX idx_embeddings_type_dimension 
            ON embeddings(entity_type, dimension)
        """)
        
        # Add index for conversation tags
        cursor.execute("""
            CREATE INDEX idx_conversations_tags 
            ON conversations(tags)
        """)
    
    def _migration_005_down(self, conn: sqlite3.Connection) -> None:
        """Remove embedding dimensions and performance indexes."""
        cursor = conn.cursor()
        
        # Drop indexes
        cursor.execute("DROP INDEX IF EXISTS idx_embeddings_type_dimension")
        cursor.execute("DROP INDEX IF EXISTS idx_conversations_tags")
        
        # Recreate embeddings table without dimension column
        cursor.execute("""
            CREATE TABLE embeddings_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                model TEXT DEFAULT 'text-embedding-ada-002',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO embeddings_new (id, entity_type, entity_id, embedding, model, created_at)
            SELECT id, entity_type, entity_id, embedding, model, created_at
            FROM embeddings
        """)
        cursor.execute("DROP TABLE embeddings")
        cursor.execute("ALTER TABLE embeddings_new RENAME TO embeddings")
        cursor.execute("CREATE INDEX idx_embeddings_entity ON embeddings(entity_type, entity_id)")


def get_migration_manager(db_path: Optional[str] = None) -> MigrationManager:
    """Get or create a migration manager instance."""
    return MigrationManager(db_path or "~/ai-assistant/memory/assistant.db")