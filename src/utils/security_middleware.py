"""
Security middleware for the house price prediction app.
"""
import re
import logging
from functools import wraps

import streamlit as st

from src.config.security_config import INPUT_VALIDATION
from src.utils.rate_limiter import check_rate_limit, get_client_id

logger = logging.getLogger("house_price_app")

def sanitize_input(input_value):
    """
    Sanitize input to prevent injection attacks.
    
    Parameters:
    -----------
    input_value : any
        Input value to sanitize
        
    Returns:
    --------
    any
        Sanitized input value
    """
    if isinstance(input_value, str):
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>(){}[\]\'";]', '', input_value)
        return sanitized
    return input_value

def validate_input(name, value):
    """
    Validate input against defined rules.
    
    Parameters:
    -----------
    name : str
        Name of the input field
    value : any
        Value to validate
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if name not in INPUT_VALIDATION:
        return True
        
    validation_rules = INPUT_VALIDATION[name]
    
    # Check type
    if validation_rules.get("type") == "integer":
        try:
            value = int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid type for {name}: expected integer, got {type(value).__name__}")
            return False
    
    # Check min/max
    if "min" in validation_rules and value < validation_rules["min"]:
        logger.warning(f"Value for {name} below minimum: {value} < {validation_rules['min']}")
        return False
        
    if "max" in validation_rules and value > validation_rules["max"]:
        logger.warning(f"Value for {name} above maximum: {value} > {validation_rules['max']}")
        return False
        
    return True

def security_middleware(func):
    """
    Decorator to apply security measures to a function.
    
    Parameters:
    -----------
    func : callable
        Function to wrap
        
    Returns:
    --------
    callable
        Wrapped function with security measures
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check rate limit
        client_id = get_client_id()
        if not check_rate_limit(client_id):
            st.error("Rate limit exceeded. Please try again later.")
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return None
            
        # Validate and sanitize inputs
        sanitized_kwargs = {}
        for name, value in kwargs.items():
            # Sanitize input
            sanitized_value = sanitize_input(value)
            
            # Validate input
            if not validate_input(name, sanitized_value):
                st.error(f"Invalid input for {name}")
                logger.warning(f"Invalid input for {name}: {value}")
                return None
                
            sanitized_kwargs[name] = sanitized_value
            
        # Call the original function with sanitized inputs
        return func(*args, **sanitized_kwargs)
        
    return wrapper

def secure_endpoint(func):
    """
    Decorator to secure a Streamlit endpoint.
    
    Parameters:
    -----------
    func : callable
        Function to wrap
        
    Returns:
    --------
    callable
        Wrapped function with security measures
    """
    @wraps(func)
    def wrapper():
        try:
            # Initialize session state for security
            if 'security_checks_passed' not in st.session_state:
                st.session_state.security_checks_passed = False
                
            # Apply rate limiting
            client_id = get_client_id()
            if not check_rate_limit(client_id):
                st.error("Rate limit exceeded. Please try again later.")
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                return
                
            # Mark security checks as passed
            st.session_state.security_checks_passed = True
            
            # Call the original function
            return func()
            
        except Exception as e:
            logger.error(f"Security error: {str(e)}", exc_info=True)
            st.error("A security error occurred. Please try again later.")
            return None
            
    return wrapper 