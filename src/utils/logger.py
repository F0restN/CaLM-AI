import sys
import os
from pathlib import Path
from loguru import logger

"""
Configure global logger settings with chronological logging

Args:
    log_path: Directory path for log files
    rotation: When to rotate the log file (size or time)
    retention: How long to keep log files
    level: Minimum log level to record
"""

log_path: str = "logs"
rotation: str = "500 MB"
retention: str = "10 days"
level: str = "INFO"

# Create logs directory if it doesn't exist
project_dir = Path(os.path.abspath(__file__)).parent.parent.parent
log_path = os.path.join(project_dir, "logs")
os.makedirs(log_path, exist_ok=True)

# Remove default logger
logger.remove(0)

# Add console logger with color
# Configure multiple log handlers
handlers = [
    # Console log handler with timestamp
    {
        "sink": sys.stdout,
        "colorize": True,
        "format": "<blue>{time:YYYY-MM-DD @ HH:mm:ss}</blue> | <level>{level: <8}</level> | <cyan>{function}</cyan> | <level>{message}</level>",
        "level": level,
        "backtrace": True,
        "diagnose": True
    },
    # Chronological log file handler (daily rotation)
    {
        "sink": f"{log_path}/{{time:YYYY-MM-DD}}.log",
        "rotation": "00:00",  # Rotate at midnight
        "retention": retention,
        "format": "{time:YYYY-MM-DD @ HH:mm:ss} | <level>{level: <8}</level> | <cyan>{function}</cyan> | <level>{message}</level>",
        "level": level,
        "encoding": "utf-8",
        "enqueue": True  # Thread-safe logging
    },
    # Error log file handler with chronological entries
    {
        "sink": f"{log_path}/errors.log",
        "rotation": rotation,
        "retention": retention,
        "format": "{time:YYYY-MM-DD @ HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        "level": "ERROR",
        "encoding": "utf-8",
        "enqueue": True  # Thread-safe logging
    }
]

# Configure all log handlers with a single call
for handler in handlers:
    logger.add(**handler)

logger.info("Logger initialized")
