"""
Security configuration for the house price prediction app.
"""

# Content Security Policy
CSP_CONFIG = {
    "default-src": ["'self'"],
    "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://cdn.jsdelivr.net"],
    "style-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
    "img-src": ["'self'", "data:", "https:"],
    "connect-src": ["'self'", "https:"],
    "font-src": ["'self'", "https:"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "frame-ancestors": ["'self'"]
}

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "enabled": True,
    "max_requests": 100,
    "time_window": 60  # seconds
}

# Input validation rules
INPUT_VALIDATION = {
    "Size": {
        "min": 100,
        "max": 10000,
        "type": "integer"
    },
    "Bedrooms": {
        "min": 1,
        "max": 10,
        "type": "integer"
    },
    "Bathrooms": {
        "min": 1,
        "max": 10,
        "type": "integer"
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "sensitive_fields": [
        "password",
        "api_key",
        "token",
        "secret",
        "credit_card"
    ],
    "log_file_permissions": 0o640  # Read/write for owner, read for group, nothing for others
}

# Cookie security settings
COOKIE_CONFIG = {
    "secure": True,
    "httponly": True,
    "samesite": "Lax",
    "max_age": 3600  # 1 hour
}

def get_csp_header():
    """
    Generate Content Security Policy header value from configuration.
    
    Returns:
    --------
    str
        Formatted CSP header value
    """
    csp_parts = []
    for directive, sources in CSP_CONFIG.items():
        csp_parts.append(f"{directive} {' '.join(sources)}")
    return "; ".join(csp_parts)

def get_all_security_headers():
    """
    Get all security headers including CSP.
    
    Returns:
    --------
    dict
        Dictionary of all security headers
    """
    headers = SECURITY_HEADERS.copy()
    headers["Content-Security-Policy"] = get_csp_header()
    return headers 