import re
import uuid

def sanitize_index_name(name: str) -> str:
    """Sanitizes a string to be a valid Pinecone index name."""
    # Convert to lowercase
    sanitized = name.lower()
    # Replace spaces and invalid characters with hyphens
    sanitized = re.sub(r'[^a-z0-9-]+', '-', sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    # Ensure it's not empty after sanitization
    if not sanitized:
        # Generate a fallback name if sanitization results in an empty string
        sanitized = f"index-{uuid.uuid4().hex[:8]}"
    # Pinecone index names must be between 3 and 63 characters
    return sanitized[:63]
