"""Utilities for generating client fingerprints"""

import hashlib
from fastapi import Request


def generate_fingerprint(request: Request) -> str:
    """Generate a fingerprint hash from IP address and user-agent
    
    Args:
        request: FastAPI request object containing client information
        
    Returns:
        SHA256 hash of the client fingerprint
    """
    # Get client IP (handling proxies)
    client_ip = request.client.host if request.client else "unknown"
    
    # Get user-agent
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Combine and hash
    fingerprint_data = f"{client_ip}:{user_agent}"
    fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()
    
    return fingerprint_hash


