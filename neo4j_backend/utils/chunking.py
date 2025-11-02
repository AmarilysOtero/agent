"""Text extraction and chunking utilities"""

import os
from typing import List, Tuple

try:
    from pypdf import PdfReader  # lightweight PDF text extractor
    _pdf_available = True
except Exception:
    _pdf_available = False


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".html", ".pdf"}


def is_supported_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in SUPPORTED_EXTENSIONS


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    if not _pdf_available:
        return ""
    try:
        reader = PdfReader(path)
        texts: List[str] = []
        for page in reader.pages:
            # extract_text returns string or None
            content = page.extract_text() or ""
            texts.append(content)
        return "\n".join(texts)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Defaults are conservative and model-agnostic.
    """
    if chunk_size <= 0:
        return [text]
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def extract_and_chunk(path: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    if not is_supported_file(path):
        return []
    _, ext = os.path.splitext(path.lower())
    if ext == ".pdf":
        text = read_pdf_file(path)
    else:
        text = read_text_file(path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

"""Document chunking utilities"""

import logging
from typing import List, Dict, Any
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Utility for chunking documents into smaller pieces"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        """Initialize chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try when splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            " ",     # Word breaks
            ""       # Character-by-character as last resort
        ]
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with 'text', 'index', and metadata
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Try splitting by separators in order of preference
        text_parts = [text]
        for separator in self.separators:
            if not text_parts or len(text_parts[0]) <= self.chunk_size:
                break
            
            new_parts = []
            for part in text_parts:
                if len(part) <= self.chunk_size:
                    new_parts.append(part)
                else:
                    if separator:
                        splits = part.split(separator)
                        current_chunk = splits[0]
                        for split in splits[1:]:
                            if len(current_chunk) + len(separator) + len(split) <= self.chunk_size:
                                current_chunk += separator + split
                            else:
                                if current_chunk:
                                    new_parts.append(current_chunk)
                                current_chunk = split
                        if current_chunk:
                            new_parts.append(current_chunk)
                    else:
                        # Character-by-character split as last resort
                        for i in range(0, len(part), self.chunk_size):
                            new_parts.append(part[i:i + self.chunk_size])
            
            text_parts = new_parts
        
        # Create chunks with overlap
        for i, part in enumerate(text_parts):
            if i > 0 and self.chunk_overlap > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[-1]['text']
                overlap_start = max(0, len(prev_chunk) - self.chunk_overlap)
                overlap_text = prev_chunk[overlap_start:]
                part = overlap_text + part
            
            # If chunk is still too large, split it further
            if len(part) > self.chunk_size:
                for j in range(0, len(part), self.chunk_size - self.chunk_overlap):
                    chunk_text = part[j:j + self.chunk_size]
                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'index': len(chunks),
                            **metadata
                        })
            else:
                if part.strip():
                    chunks.append({
                        'text': part,
                        'index': len(chunks),
                        **metadata
                    })
        
        return chunks
    
    def chunk_file(
        self,
        file_path: str,
        content: str,
        file_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Chunk a file's content
        
        Args:
            file_path: Path to the file
            content: File content as string
            file_metadata: Metadata about the file
            
        Returns:
            List of chunk dictionaries
        """
        file_metadata = file_metadata or {}
        
        # Add file-specific metadata
        chunk_metadata = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'file_extension': Path(file_path).suffix,
            **file_metadata
        }
        
        return self.chunk_text(content, chunk_metadata)


def read_file_content(file_path: str) -> str:
    """Read file content based on file type
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
    
    try:
        # Text files
        if extension in ['.txt', '.md', '.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.xml', '.html', '.css', '.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # For other file types, try to read as text (may not work for binary files)
        # In production, you might want to use libraries like:
        # - PyPDF2/pdfplumber for PDFs
        # - python-docx for Word documents
        # - openpyxl for Excel files
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Check if content looks like binary
                if '\x00' in content[:1024]:
                    logger.warning(f"File {file_path} appears to be binary, skipping")
                    return ""
                return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

