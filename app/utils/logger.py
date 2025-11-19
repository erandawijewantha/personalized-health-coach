"""
Logging utility for the health coach application.
Provides simple, consistent logging across all modules.
Senior note: Keep logging minimal - only critical operations (API calls, agent execution, errors).
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str = "health_coach") -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        Configured logger instance
    
    Senior note: Using basicConfig for simplicity; file + console output for debugging.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get log configuration from environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE", "logs/health_coach.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(name)
    return logger


# Default logger instance
logger = setup_logger()