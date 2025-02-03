import sys
import os
from pathlib import Path
from loguru import logger

"""
Configure global logger settings

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

# Define log path
project_dir = Path(os.path.abspath(__file__)).parent.parent.parent
log_path = os.path.join(project_dir, "logs")

# Remove default logger
logger.remove()

# Add console logger with color
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD @ HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=level
)

# Add file logger
logger.add(
    f"{log_path}/{{time:YYYY-MM-DD}}.log",
    rotation=rotation,
    retention=retention,
    format="{time:YYYY-MM-DD @ HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=level,
    encoding="utf-8"
)

# Add error logger for errors only
logger.add(
    f"{log_path}/errors.log",
    rotation=rotation,
    retention=retention,
    format="{time:YYYY-MM-DD @ HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    encoding="utf-8"
)
