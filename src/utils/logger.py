"""
Logging utility for the house price prediction app.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name="house_price_app", log_level=logging.INFO):
    """
    Set up and configure logger.
    
    Parameters:
    -----------
    name : str, default="house_price_app"
        Name of the logger
    log_level : int, default=logging.INFO
        Logging level
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    return logger 