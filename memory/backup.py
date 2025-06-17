"""Backup and restore functionality for AI Assistant memory.

Provides automated backup, restore, scheduling, and compression support for the database.
"""

import sqlite3
import shutil
import gzip
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import threading
import schedule
import time
from loguru import logger


class BackupManager:
    """Manages database backups with compression and scheduling support."""
    
    def __init__(self, db_path: str = "~/ai-assistant/memory/assistant.db",
                 backup_dir: str = "~/ai-assistant/memory/backups"):
        self.db_path = Path(db_path).expanduser()
        self.backup_dir = Path(backup_dir).expanduser()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_running = False
        self._backup_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"Backup manager initialized. Backup directory: {self.backup_dir}")
    
    def create_backup(self, name: Optional[str] = None, compress: bool = True,
                     include_metadata: bool = True) -> str:
        """Create a backup of the database.
        
        Args:
            name: Optional backup name. If not provided, timestamp will be used.
            compress: Whether to compress the backup.
            include_metadata: Whether to include metadata about the backup.
            
        Returns:
            Path to the created backup file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"
        
        # Ensure .db extension is not in the name (we'll add it)
        backup_name = backup_name.replace('.db', '').replace('.gz', '').replace('.tar', '')
        
        try:
            if compress:
                backup_path = self._create_compressed_backup(backup_name, include_metadata)
            else:
                backup_path = self._create_simple_backup(backup_name, include_metadata)
            
            logger.info(f"Backup created successfully: {backup_path}")
            
            # Call backup callbacks
            for callback in self._backup_callbacks:
                try:
                    callback(str(backup_path))
                except Exception as e:
                    logger.error(f"Backup callback error: {e}")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _create_simple_backup(self, backup_name: str, include_metadata: bool) -> Path:
        """Create a simple backup by copying the database file."""
        backup_path = self.backup_dir / f"{backup_name}.db"
        
        # Use SQLite backup API for consistency
        source_conn = sqlite3.connect(str(self.db_path))
        backup_conn = sqlite3.connect(str(backup_path))
        
        try:
            with source_conn:
                source_conn.backup(backup_conn)
            logger.debug(f"Database backed up to {backup_path}")
        finally:
            source_conn.close()
            backup_conn.close()
        
        if include_metadata:
            self._save_backup_metadata(backup_name, backup_path, compressed=False)
        
        return backup_path
    
    def _create_compressed_backup(self, backup_name: str, include_metadata: bool) -> Path:
        """Create a compressed backup archive."""
        temp_backup = self.backup_dir / f"{backup_name}_temp.db"
        archive_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # First create a database backup
        source_conn = sqlite3.connect(str(self.db_path))
        backup_conn = sqlite3.connect(str(temp_backup))
        
        try:
            with source_conn:
                source_conn.backup(backup_conn)
        finally:
            source_conn.close()
            backup_conn.close()
        
        # Create a tar.gz archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(temp_backup, arcname=f"{backup_name}.db")
            
            if include_metadata:
                # Add metadata to the archive
                metadata = self._generate_backup_metadata(backup_name, compressed=True)
                metadata_path = self.backup_dir / f"{backup_name}_metadata.json"
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                tar.add(metadata_path, arcname="metadata.json")
                metadata_path.unlink()  # Clean up temp metadata file
        
        # Clean up temporary backup
        temp_backup.unlink()
        
        return archive_path
    
    def _generate_backup_metadata(self, backup_name: str, compressed: bool) -> Dict[str, Any]:
        """Generate metadata for a backup."""
        # Get database statistics
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        stats = {}
        tables = ['conversations', 'messages', 'knowledge_base', 'projects', 'embeddings']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]
        
        # Get database size
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        db_size = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "database_path": str(self.db_path),
            "database_size": db_size,
            "compressed": compressed,
            "statistics": stats,
            "version": "1.0.0"
        }
    
    def _save_backup_metadata(self, backup_name: str, backup_path: Path, compressed: bool) -> None:
        """Save metadata file alongside the backup."""
        metadata = self._generate_backup_metadata(backup_name, compressed)
        metadata_path = backup_path.with_suffix('.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def restore_backup(self, backup_path: str, create_pre_restore_backup: bool = True) -> None:
        """Restore database from a backup.
        
        Args:
            backup_path: Path to the backup file.
            create_pre_restore_backup: Whether to create a backup before restoring.
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            # Create a pre-restore backup if requested
            if create_pre_restore_backup:
                pre_restore_name = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_backup(name=pre_restore_name, compress=False)
                logger.info(f"Created pre-restore backup: {pre_restore_name}")
            
            # Determine backup type and restore
            if backup_file.suffix == '.gz' or backup_file.suffixes == ['.tar', '.gz']:
                self._restore_compressed_backup(backup_file)
            elif backup_file.suffix == '.db':
                self._restore_simple_backup(backup_file)
            else:
                raise ValueError(f"Unknown backup format: {backup_file.suffix}")
            
            logger.info(f"Database restored successfully from {backup_path}")
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
    
    def _restore_simple_backup(self, backup_file: Path) -> None:
        """Restore from a simple database backup."""
        # Close any existing connections
        temp_restore = self.db_path.with_suffix('.restore')
        
        # Copy backup to temp location
        shutil.copy2(backup_file, temp_restore)
        
        # Verify the backup is a valid SQLite database
        try:
            conn = sqlite3.connect(str(temp_restore))
            conn.execute("SELECT 1")
            conn.close()
        except sqlite3.Error as e:
            temp_restore.unlink()
            raise ValueError(f"Invalid SQLite database: {e}")
        
        # Replace the current database
        if self.db_path.exists():
            self.db_path.unlink()
        temp_restore.rename(self.db_path)
    
    def _restore_compressed_backup(self, backup_file: Path) -> None:
        """Restore from a compressed backup archive."""
        temp_dir = self.backup_dir / "temp_restore"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract the archive
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            # Find the database file
            db_files = list(temp_dir.glob("*.db"))
            if not db_files:
                raise ValueError("No database file found in backup archive")
            
            # Restore the database
            self._restore_simple_backup(db_files[0])
            
            # Check for metadata
            metadata_file = temp_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                logger.info(f"Restored backup created at: {metadata.get('created_at')}")
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    def list_backups(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """List all available backups.
        
        Args:
            include_metadata: Whether to include metadata for each backup.
            
        Returns:
            List of backup information dictionaries.
        """
        backups = []
        
        # Find all backup files
        for backup_file in self.backup_dir.iterdir():
            if backup_file.suffix in ['.db', '.gz'] or backup_file.suffixes == ['.tar', '.gz']:
                backup_info = {
                    "name": backup_file.stem.replace('.tar', ''),
                    "path": str(backup_file),
                    "size": backup_file.stat().st_size,
                    "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "compressed": backup_file.suffix == '.gz' or backup_file.suffixes == ['.tar', '.gz']
                }
                
                if include_metadata:
                    # Check for metadata file
                    metadata_file = backup_file.with_suffix('.json')
                    if not metadata_file.exists() and backup_info["compressed"]:
                        # Try to extract metadata from archive
                        try:
                            with tarfile.open(backup_file, "r:gz") as tar:
                                for member in tar.getmembers():
                                    if member.name == "metadata.json":
                                        f = tar.extractfile(member)
                                        if f:
                                            backup_info["metadata"] = json.load(f)
                                            break
                        except Exception as e:
                            logger.debug(f"Could not extract metadata from {backup_file}: {e}")
                    elif metadata_file.exists():
                        with open(metadata_file) as f:
                            backup_info["metadata"] = json.load(f)
                
                backups.append(backup_info)
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        
        return backups
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup.
        
        Args:
            backup_name: Name of the backup to delete.
            
        Returns:
            True if backup was deleted, False if not found.
        """
        deleted = False
        
        # Try different file extensions
        for ext in ['.db', '.tar.gz', '.json']:
            backup_file = self.backup_dir / f"{backup_name}{ext}"
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"Deleted backup file: {backup_file}")
                deleted = True
        
        return deleted
    
    def cleanup_old_backups(self, keep_days: int = 7, keep_min: int = 3) -> int:
        """Clean up old backups.
        
        Args:
            keep_days: Number of days to keep backups.
            keep_min: Minimum number of backups to keep regardless of age.
            
        Returns:
            Number of backups deleted.
        """
        backups = self.list_backups(include_metadata=False)
        
        if len(backups) <= keep_min:
            logger.info(f"Keeping all {len(backups)} backups (minimum: {keep_min})")
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0
        
        # Keep the most recent keep_min backups
        backups_to_check = backups[keep_min:]
        
        for backup in backups_to_check:
            backup_date = datetime.fromisoformat(backup["created_at"])
            if backup_date < cutoff_date:
                if self.delete_backup(backup["name"]):
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup['name']} (created: {backup['created_at']})")
        
        logger.info(f"Cleanup complete. Deleted {deleted_count} old backups")
        return deleted_count
    
    def schedule_backup(self, schedule_func: Callable, *args, **kwargs) -> None:
        """Schedule automatic backups.
        
        Example:
            # Daily backup at 2 AM
            manager.schedule_backup(schedule.every().day.at("02:00").do)
            
            # Hourly backup
            manager.schedule_backup(schedule.every().hour.do)
        """
        # Create a job that runs create_backup
        job = schedule_func(self.create_backup, *args, **kwargs)
        logger.info(f"Scheduled backup job: {job}")
    
    def start_scheduler(self) -> None:
        """Start the backup scheduler in a background thread."""
        if self._scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Backup scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the backup scheduler."""
        if not self._scheduler_running:
            return
        
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("Backup scheduler stopped")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler loop."""
        while self._scheduler_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def add_backup_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback to be called after each backup.
        
        Args:
            callback: Function that receives the backup path as argument.
        """
        self._backup_callbacks.append(callback)
    
    def verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """Verify the integrity of a backup.
        
        Args:
            backup_path: Path to the backup file.
            
        Returns:
            Dictionary with verification results.
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            return {"valid": False, "error": "Backup file not found"}
        
        results = {
            "valid": True,
            "path": str(backup_file),
            "size": backup_file.stat().st_size,
            "errors": []
        }
        
        try:
            if backup_file.suffix == '.db':
                # Verify SQLite database
                conn = sqlite3.connect(str(backup_file))
                cursor = conn.cursor()
                
                # Check integrity
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                if integrity != "ok":
                    results["valid"] = False
                    results["errors"].append(f"Integrity check failed: {integrity}")
                
                # Check tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                expected_tables = ['conversations', 'messages', 'knowledge_base', 'projects', 'embeddings']
                
                for table in expected_tables:
                    if table not in tables:
                        results["errors"].append(f"Missing table: {table}")
                
                conn.close()
                
            elif backup_file.suffix == '.gz' or backup_file.suffixes == ['.tar', '.gz']:
                # Verify compressed archive
                try:
                    with tarfile.open(backup_file, "r:gz") as tar:
                        members = tar.getnames()
                        if not any(m.endswith('.db') for m in members):
                            results["valid"] = False
                            results["errors"].append("No database file found in archive")
                except Exception as e:
                    results["valid"] = False
                    results["errors"].append(f"Archive error: {e}")
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Verification error: {e}")
        
        return results


def get_backup_manager(db_path: Optional[str] = None, 
                      backup_dir: Optional[str] = None) -> BackupManager:
    """Get or create a backup manager instance."""
    return BackupManager(
        db_path or "~/ai-assistant/memory/assistant.db",
        backup_dir or "~/ai-assistant/memory/backups"
    )