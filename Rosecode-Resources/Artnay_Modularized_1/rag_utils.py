"""Utility functions for the RAG toolkit.

This module provides helper functions for common tasks like text processing,
file handling, and progress tracking.
"""

import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from functools import wraps
import time
from contextlib import contextmanager

# Optional imports for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def setup_logging(level: str = "INFO", 
                 log_file: Optional[Path] = None,
                 format_string: Optional[str] = None):
    """Set up logging configuration for the RAG toolkit."""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )


def truncate_text(text: str, 
                 max_length: int, 
                 suffix: str = "...",
                 strategy: str = "end") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    
    if strategy == "end":
        return text[:max_length - len(suffix)] + suffix
    elif strategy == "middle":
        half = (max_length - len(suffix)) // 2
        return text[:half] + suffix + text[-half:]
    else:
        return text[:max_length]


def clean_text(text: str, 
              remove_extra_whitespace: bool = True,
              remove_urls: bool = False,
              remove_emails: bool = False,
              lowercase: bool = False) -> str:
    """Clean text by removing unwanted elements."""
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    if remove_emails:
        text = re.sub(r'\S+@\S+', '', text)
    
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    if lowercase:
        text = text.lower()
    
    return text


def chunk_text(text: str, 
              chunk_size: int = 1000, 
              chunk_overlap: int = 200,
              separator: str = " ",
              preserve_sentences: bool = True) -> List[str]:
    """Split text into overlapping chunks."""
    if preserve_sentences:
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split(separator))
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    overlap_size += len(s.split(separator))
                    overlap_sentences.insert(0, s)
                    if overlap_size >= chunk_overlap:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    else:
        # Simple word-based chunking
        words = text.split(separator)
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = words[i:i + chunk_size]
            chunks.append(separator.join(chunk))
        
        return chunks


def calculate_hash(text: str, algorithm: str = "md5") -> str:
    """Calculate hash of text."""
    if algorithm == "md5":
        return hashlib.md5(text.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def estimate_tokens(text: str, method: str = "words") -> int:
    """Estimate the number of tokens in text."""
    if method == "words":
        # Rough estimate: 1 token ≈ 0.75 words
        return int(len(text.split()) / 0.75)
    elif method == "chars":
        # Rough estimate: 1 token ≈ 4 characters
        return int(len(text) / 4)
    else:
        raise ValueError(f"Unknown estimation method: {method}")


def batch_items(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield batches of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@contextmanager
def timer(name: str = "Operation", logger: Optional[logging.Logger] = None):
    """Context manager for timing operations."""
    start_time = time.time()
    
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        message = f"{name} completed in {elapsed_time:.2f} seconds"
        
        if logger:
            logger.info(message)
        else:
            print(message)


def retry(max_attempts: int = 3, 
         delay: float = 1.0,
         backoff: float = 2.0,
         exceptions: tuple = (Exception,)):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    
                    logging.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
            
            return None
        return wrapper
    return decorator


class ProgressTracker:
    """Track progress of operations."""
    
    def __init__(self, 
                 total: int, 
                 description: str = "Processing",
                 disable: bool = False):
        self.total = total
        self.description = description
        self.disable = disable
        self.current = 0
        
        if TQDM_AVAILABLE and not disable:
            self.pbar = tqdm(total=total, desc=description)
        else:
            self.pbar = None
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        
        if self.pbar:
            self.pbar.update(n)
        elif not self.disable:
            # Simple text progress
            percent = (self.current / self.total) * 100
            print(f"\r{self.description}: {percent:.1f}% ({self.current}/{self.total})", end="")
    
    def close(self):
        """Close progress tracker."""
        if self.pbar:
            self.pbar.close()
        elif not self.disable:
            print()  # New line after progress
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def merge_dicts(base: Dict[str, Any], 
               update: Dict[str, Any], 
               deep: bool = True) -> Dict[str, Any]:
    """Merge two dictionaries."""
    result = base.copy()
    
    if deep:
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    else:
        result.update(update)
    
    return result


def load_text_file(file_path: Union[str, Path], 
                  encoding: str = "utf-8") -> str:
    """Load text from a file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def save_text_file(content: str, 
                  file_path: Union[str, Path], 
                  encoding: str = "utf-8",
                  create_dirs: bool = True):
    """Save text to a file."""
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def format_metadata(metadata: Dict[str, Any], 
                   format: str = "yaml") -> str:
    """Format metadata for display or storage."""
    if format == "yaml":
        try:
            import yaml
            return yaml.dump(metadata, default_flow_style=False)
        except ImportError:
            # Fallback to simple format
            format = "simple"
    
    if format == "json":
        import json
        return json.dumps(metadata, indent=2)
    
    if format == "simple":
        lines = []
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    raise ValueError(f"Unknown format: {format}")


def extract_code_blocks(text: str, 
                       language: Optional[str] = None) -> List[Dict[str, str]]:
    """Extract code blocks from markdown text."""
    pattern = r'```(\w*)\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    blocks = []
    for lang, code in matches:
        if language is None or lang == language:
            blocks.append({
                "language": lang or "plain",
                "code": code.strip()
            })
    
    return blocks


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n\n+', '\n\n', text)
    
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    
    return '\n'.join(lines).strip()


def split_into_sentences(text: str, 
                        language: str = "en") -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter for English
    # For production, consider using NLTK or spaCy
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def create_summary_statistics(documents: List[Any]) -> Dict[str, Any]:
    """Create summary statistics for a collection of documents."""
    from .retriever import Document
    
    if not documents:
        return {
            "count": 0,
            "total_length": 0,
            "avg_length": 0,
            "sources": []
        }
    
    total_length = 0
    sources = set()
    metadata_keys = set()
    
    for doc in documents:
        if isinstance(doc, Document):
            total_length += len(doc.content)
            sources.add(doc.source)
            metadata_keys.update(doc.metadata.keys())
        elif isinstance(doc, str):
            total_length += len(doc)
        elif isinstance(doc, dict) and "content" in doc:
            total_length += len(doc["content"])
    
    return {
        "count": len(documents),
        "total_length": total_length,
        "avg_length": total_length / len(documents) if documents else 0,
        "sources": list(sources),
        "metadata_keys": list(metadata_keys)
    }


class TokenBudget:
    """Manage token budget for context windows."""
    
    def __init__(self, max_tokens: int, reserve: int = 100):
        self.max_tokens = max_tokens
        self.reserve = reserve
        self.used_tokens = 0
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens."""
        return max(0, self.max_tokens - self.used_tokens - self.reserve)
    
    def use(self, tokens: int):
        """Use tokens from budget."""
        self.used_tokens += tokens
    
    def can_fit(self, text: str, method: str = "words") -> bool:
        """Check if text fits in remaining budget."""
        estimated_tokens = estimate_tokens(text, method)
        return estimated_tokens <= self.available_tokens
    
    def reset(self):
        """Reset token budget."""
        self.used_tokens = 0