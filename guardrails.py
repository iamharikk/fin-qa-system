import re
from typing import Tuple

def validate_content(text: str) -> Tuple[bool, str]:
    banned_words = [
        'spam', 'hack', 'malicious', 'attack', 'exploit', 
        'virus', 'malware', 'phishing', 'scam', 'fraud'
    ]
    
    text_lower = text.lower()
    for word in banned_words:
        if word in text_lower:
            return False, f"Input contains inappropriate content: '{word}'"
    
    suspicious_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'sql.*drop',
        r'union.*select',
        r'exec\s*\(',
        r'eval\s*\(',
        r'setTimeout\s*\(',
        r'setInterval\s*\(',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False, "Input contains suspicious patterns that may be malicious"
    
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\s\.\,\?\!\-\:\;\(\)]', text))
    if len(text) > 0 and special_char_count / len(text) > 0.3:
        return False, "Input contains too many special characters"
    
    return True, ""

def validate_length(text: str, min_length: int = 5, max_length: int = 1000) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "Query cannot be empty or contain only whitespace"
    
    if len(text.strip()) < min_length:
        return False, f"Query too short (minimum {min_length} characters, got {len(text.strip())})"
    
    if len(text) > max_length:
        return False, f"Query too long (maximum {max_length} characters, got {len(text)})"
    
    return True, ""

def validate_method(method: str) -> Tuple[bool, str]:
    valid_methods = [
        "RAG System",
        "Fine Tuned Model"
    ]
    
    if not method:
        return False, "No method selected"
    
    if method not in valid_methods:
        return False, f"Invalid method selected: '{method}'"
    
    return True, ""

def validate_all_inputs(text: str, method: str) -> Tuple[bool, str]:
    is_valid, error = validate_content(text)
    if not is_valid:
        return False, error
    
    is_valid, error = validate_length(text)
    if not is_valid:
        return False, error
    
    is_valid, error = validate_method(method)
    if not is_valid:
        return False, error
    
    return True, "All validations passed"