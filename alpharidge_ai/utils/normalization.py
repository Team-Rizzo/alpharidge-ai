"""
Text normalization utilities shared across miner, validator, and API layers.

This module provides consistent text normalization functions to ensure that
miners, validators, and the API all use the same normalization logic for
comparing text content.
"""

import unicodedata
import re


def norm_text(s: str) -> str:
    """
    Normalize text for comparison to handle encoding differences, line endings, and whitespace.
    
    This ensures that minor formatting differences don't cause false mismatches:
    - Unicode normalization (NFC) handles different encodings of the same characters
    - Converts all line endings to \n
    - Collapses multiple whitespace characters to single spaces
    - Trims leading/trailing whitespace
    
    This normalization is used consistently across:
    - Miner: Normalizes content before analysis and submission
    - Validator: Normalizes live X API text for comparison
    - API: Normalizes content before storage
    
    Args:
        s: Raw text string
        
    Returns:
        Normalized text string ready for comparison
        
    Example:
        >>> norm_text("Hello\\r\\n  World  ")
        'Hello World'
        >>> norm_text("Café\\t\\t\\nCafé")  # Different unicode encodings normalized
        'Café Café'
    """
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_author(s: str) -> str:
    """
    Normalize author username: lowercase and strip whitespace.
    
    This ensures consistent comparison of author usernames across the system.
    
    Args:
        s: Raw author username string
        
    Returns:
        Normalized author username (lowercase, stripped)
        
    Example:
        >>> norm_author("  @JohnDoe  ")
        'johndoe'
    """
    return (s or "").strip().lower()

