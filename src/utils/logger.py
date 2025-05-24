"""
Logging utility for the house price prediction app.
"""
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path

class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs."""
    
    def __init__(self):
        super().__init__()
        # Patterns for sensitive data
        self.patterns = [
            (re.compile(r'password\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), 'password=*****'),
            (re.compile(r'api_key\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), 'api_key=*****'),
            (re.compile(r'token\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), 'token=*****'),
            (re.compile(r'secret\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), 'secret=*****'),
            (re.compile(r'credit_card\s*=\s*[\'"][^\'"]+[\'"]', re.IGNORECASE), 'credit_card=*****'),
            # Add more patterns as needed
        ]
    
    def filter(self, record):
        if isinstance(record.msg, str):
            for pattern, replacement in self.patterns:
                record.msg = pattern.sub(replacement, record.msg)
                
        if hasattr(record, 'args') and record.args:
            record.args = self._filter_args(record.args)
                
        return True
    
    def _filter_args(self, args):
        """Recursively filter sensitive data from log arguments."""
        if isinstance(args, dict):
            return {k: self._filter_args(v) for k, v in args.items()}
        elif isinstance(args, (list, tuple)):
            return type(args)(self._filter_args(v) for v in args)
        elif isinstance(args, str):
            result = args
            for pattern, replacement in self.patterns:
                result = pattern.sub(replacement, result)
            return result
        else:
            return args

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
        # File handler with proper permissions
        file_handler = logging.FileHandler(log_file, mode='a')
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        
        # Add sensitive data filter
        sensitive_filter = SensitiveDataFilter()
        file_handler.addFilter(sensitive_filter)
        
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        console_handler.setFormatter(console_format)
        console_handler.addFilter(sensitive_filter)
        logger.addHandler(console_handler)
    
    # Set proper permissions for log file
    try:
        os.chmod(log_file, 0o640)  # Read/write for owner, read for group, nothing for others
    except Exception:
        # If we can't set permissions, log a warning but continue
        logger.warning(f"Could not set permissions on log file: {log_file}")
    
    return logger 