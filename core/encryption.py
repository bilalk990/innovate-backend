"""
core/encryption.py — Field-level encryption for sensitive data (#46)
Uses Fernet symmetric encryption for Google OAuth tokens.
"""
import base64
import os
from django.conf import settings


def _get_fernet():
    """Get or create Fernet cipher. Lazy import to avoid startup issues."""
    try:
        from cryptography.fernet import Fernet
        key = getattr(settings, 'ENCRYPTION_KEY', None)
        if not key:
            # Fallback: derive from SECRET_KEY (not ideal but functional)
            import hashlib
            raw = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
            key = base64.urlsafe_b64encode(raw)
        if isinstance(key, str):
            key = key.encode()
        return Fernet(key)
    except ImportError:
        return None


def encrypt_dict(data: dict) -> str:
    """Encrypt a dict to a base64 string. Fails if encryption unavailable."""
    import json
    fernet = _get_fernet()
    if not fernet:
        raise RuntimeError('Encryption not available. Set ENCRYPTION_KEY in .env!')
    raw = json.dumps(data).encode()
    return fernet.encrypt(raw).decode()


def decrypt_dict(encrypted: str) -> dict:
    """Decrypt an encrypted string back to dict."""
    import json
    if not encrypted:
        return {}
    fernet = _get_fernet()
    if not fernet:
        raise RuntimeError('Encryption not available. Set ENCRYPTION_KEY in .env!')
    try:
        raw = encrypted.encode() if isinstance(encrypted, str) else encrypted
        decrypted = fernet.decrypt(raw)
        return json.loads(decrypted.decode())
    except Exception as e:
        # Log error but return empty dict to prevent crashes
        import logging
        logging.getLogger('innovaite').error(f'Decryption failed: {e}')
        return {}
