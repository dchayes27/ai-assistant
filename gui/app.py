"""
Main application launcher for the AI Assistant GUI
"""

import os
import sys
import argparse
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.interface import launch_interface


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="AI Assistant GUI")
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/gui.log",
        rotation="10 MB",
        retention="7 days",
        level=args.log_level
    )
    
    logger.info("Starting AI Assistant GUI...")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Share: {args.share}, Debug: {args.debug}")
    
    try:
        launch_interface(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("GUI shutdown requested by user")
    except Exception as e:
        logger.error(f"GUI failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()