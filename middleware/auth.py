"""
middleware/auth.py
Simple API-key authentication via the X-Api-Key request header.
Set the API_KEY environment variable to enable protection.
If API_KEY is empty/unset, auth is skipped (development mode).
"""

import os
from fastapi import Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(x_api_key: str = Security(_api_key_header)) -> None:
    """
    Dependency injected into protected routes.
    Raises HTTP 401 when:
      - API_KEY env var is set AND
      - the request does not supply a matching X-Api-Key header.
    """
    if not API_KEY:
        # Development mode: no key configured, all requests pass through.
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")