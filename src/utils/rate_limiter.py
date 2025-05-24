"""
Rate limiter utility for the house price prediction app.
"""
import time
import hashlib
import socket
from collections import defaultdict
from threading import Lock
import logging

import streamlit as st

from src.config.security_config import RATE_LIMIT_CONFIG

logger = logging.getLogger("house_price_app")

def get_client_id():
    """
    Generate a client ID for rate limiting.
    
    In a production environment, this would use the client's IP address.
    For local development, we use a combination of hostname and session state.
    
    Returns:
    --------
    str
        Client identifier
    """
    # In production, you would use something like:
    # client_ip = st.experimental_get_query_params().get('client_ip', ['unknown'])[0]
    
    # For local development/demo, use a combination of hostname and session state
    hostname = socket.gethostname()
    session_id = st.session_state.get('session_id', 'unknown')
    
    # Create a hash of the combined values
    client_id = hashlib.md5(f"{hostname}:{session_id}".encode()).hexdigest()
    
    return client_id

class RateLimiter:
    """Rate limiter to prevent abuse."""
    
    def __init__(self, max_requests=100, time_window=60):
        """
        Initialize rate limiter.
        
        Parameters:
        -----------
        max_requests : int, default=100
            Maximum number of requests allowed in the time window
        time_window : int, default=60
            Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_counts = defaultdict(list)
        self.lock = Lock()
        
    def is_allowed(self, client_id):
        """
        Check if request is allowed based on rate limits.
        
        Parameters:
        -----------
        client_id : str
            Identifier for the client (e.g., IP address)
            
        Returns:
        --------
        bool
            True if request is allowed, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # Remove expired timestamps
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if current_time - timestamp < self.time_window
            ]
            
            # Check if rate limit is exceeded
            if len(self.request_counts[client_id]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return False
            
            # Add current timestamp
            self.request_counts[client_id].append(current_time)
            return True
    
    def get_remaining_requests(self, client_id):
        """
        Get remaining requests for a client.
        
        Parameters:
        -----------
        client_id : str
            Identifier for the client
            
        Returns:
        --------
        int
            Number of remaining requests
        """
        with self.lock:
            current_time = time.time()
            
            # Remove expired timestamps
            self.request_counts[client_id] = [
                timestamp for timestamp in self.request_counts[client_id]
                if current_time - timestamp < self.time_window
            ]
            
            return max(0, self.max_requests - len(self.request_counts[client_id]))

# Create a global rate limiter instance
rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_CONFIG["max_requests"],
    time_window=RATE_LIMIT_CONFIG["time_window"]
)

def check_rate_limit(client_id):
    """
    Check if a request is allowed based on rate limits.
    
    Parameters:
    -----------
    client_id : str
        Identifier for the client (e.g., IP address)
        
    Returns:
    --------
    bool
        True if request is allowed, False otherwise
    """
    if not RATE_LIMIT_CONFIG["enabled"]:
        return True
        
    return rate_limiter.is_allowed(client_id) 