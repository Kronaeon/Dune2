"""Content retrieval strategies for the RAG toolkit.

This module provides flexible retrieval mechanisms for various data sources,
supporting single files, directories, and custom retrieval strategies.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from .config import RetrieverConfig


@dataclass
class Document:
    """Represents a retrieved document with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    source: str
    
    def __str__(self):
        return f"Document(source={self.source}, length={len(self.content)})"


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize cache if enabled
        if self.config.cache_enabled:
            self.config.cache_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve documents based on the query."""
        pass
    
    def _extract_metadata(self, content: str, source: str) -> Dict[str, Any]:
        """Extract metadata from content (e.g., YAML frontmatter)."""
        metadata = {"source": source}
        
        # Check for YAML frontmatter
        yaml_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if yaml_match:
            try:
                import yaml
                yaml_data = yaml.safe_load(yaml_match.group(1))
                if isinstance(yaml_data, dict):
                    metadata.update(yaml_data)
            except Exception as e:
                self.logger.warning(f"Failed to parse YAML metadata: {e}")
        
        # Check for simple key-value metadata at the beginning
        for line in content.split('\n')[:10]:
            for field in self.config.metadata_fields:
                pattern = f"^{field}:\\s*(.+)$"
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    metadata[field.lower()] = match.group(1).strip()
        
        return metadata
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks based on configuration."""
        if not self.config.chunk_size:
            return [content]
        
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk_words = words[i:i + self.config.chunk_size]
            chunks.append(' '.join(chunk_words))
            
            if self.config.max_chunks and len(chunks) >= self.config.max_chunks:
                break
        
        return chunks


class FileRetriever(BaseRetriever):
    """Retrieve content from single or multiple files."""
    
    def __init__(self, 
                 file_paths: Union[str, Path, List[Union[str, Path]]],
                 config: Optional[RetrieverConfig] = None):
        super().__init__(config)
        
        # Normalize file paths
        if isinstance(file_paths, (str, Path)):
            self.file_paths = [Path(file_paths)]
        else:
            self.file_paths = [Path(p) for p in file_paths]
        
        # Validate paths
        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
    
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve documents from configured files."""
        documents = []
        
        for file_path in self.file_paths:
            try:
                content = self._read_file(file_path)
                
                # Remove frontmatter from content if present
                content_without_meta = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
                
                # Extract metadata
                metadata = self._extract_metadata(content, str(file_path))
                
                # Create document
                doc = Document(
                    content=content_without_meta.strip(),
                    metadata=metadata,
                    source=str(file_path)
                )
                documents.append(doc)
                
                self.logger.info(f"Retrieved document from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
        
        return documents
    
    def _read_file(self, file_path: Path) -> str:
        """Read content from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


class DirectoryRetriever(BaseRetriever):
    """Retrieve content from all matching files in a directory."""
    
    def __init__(self,
                 directory: Union[str, Path],
                 pattern: str = "*.txt",
                 recursive: bool = True,
                 config: Optional[RetrieverConfig] = None):
        super().__init__(config)
        self.directory = Path(directory)
        self.pattern = pattern
        self.recursive = recursive
        
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
    
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve documents from directory."""
        documents = []
        
        # Find matching files
        if self.recursive:
            files = list(self.directory.rglob(self.pattern))
        else:
            files = list(self.directory.glob(self.pattern))
        
        self.logger.info(f"Found {len(files)} files matching pattern '{self.pattern}'")
        
        # Use FileRetriever for actual file reading
        if files:
            file_retriever = FileRetriever(files, self.config)
            documents = file_retriever.retrieve(query)
        
        return documents


class ConsolidatedRetriever(BaseRetriever):
    """Retrieve and consolidate content from multiple sources."""
    
    def __init__(self,
                 retrievers: List[BaseRetriever],
                 consolidation_strategy: str = "concatenate",
                 config: Optional[RetrieverConfig] = None):
        super().__init__(config)
        self.retrievers = retrievers
        self.consolidation_strategy = consolidation_strategy
    
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve and consolidate documents from all retrievers."""
        all_documents = []
        
        # Collect documents from all retrievers
        for retriever in self.retrievers:
            documents = retriever.retrieve(query)
            all_documents.extend(documents)
        
        # Apply consolidation strategy
        if self.consolidation_strategy == "concatenate":
            return self._concatenate_documents(all_documents)
        elif self.consolidation_strategy == "merge_by_source":
            return self._merge_by_source(all_documents)
        else:
            return all_documents
    
    def _concatenate_documents(self, documents: List[Document]) -> List[Document]:
        """Concatenate all documents into a single document."""
        if not documents:
            return []
        
        combined_content = "\n\n".join(doc.content for doc in documents)
        combined_metadata = {
            "sources": [doc.source for doc in documents],
            "document_count": len(documents)
        }
        
        # Merge common metadata fields
        for field in self.config.metadata_fields:
            values = [doc.metadata.get(field) for doc in documents if field in doc.metadata]
            if values:
                combined_metadata[field] = values[0] if len(set(values)) == 1 else values
        
        return [Document(
            content=combined_content,
            metadata=combined_metadata,
            source="consolidated"
        )]
    
    def _merge_by_source(self, documents: List[Document]) -> List[Document]:
        """Merge documents by their source."""
        source_groups = {}
        
        for doc in documents:
            source = doc.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        merged_documents = []
        for source, docs in source_groups.items():
            if len(docs) == 1:
                merged_documents.append(docs[0])
            else:
                # Merge documents from same source
                merged_content = "\n\n".join(doc.content for doc in docs)
                merged_metadata = docs[0].metadata.copy()
                merged_metadata["merged_count"] = len(docs)
                
                merged_documents.append(Document(
                    content=merged_content,
                    metadata=merged_metadata,
                    source=source
                ))
        
        return merged_documents


class ChunkedRetriever(BaseRetriever):
    """Retrieve content in chunks for better context management."""
    
    def __init__(self,
                 base_retriever: BaseRetriever,
                 chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 config: Optional[RetrieverConfig] = None):
        super().__init__(config)
        self.base_retriever = base_retriever
        
        # Override chunk settings if provided
        if chunk_size is not None:
            self.config.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.config.chunk_overlap = chunk_overlap
    
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve documents and split them into chunks."""
        documents = self.base_retriever.retrieve(query)
        chunked_documents = []
        
        for doc in documents:
            chunks = self._chunk_content(doc.content)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_source": doc.source
                })
                
                chunked_doc = Document(
                    content=chunk,
                    metadata=chunk_metadata,
                    source=f"{doc.source}_chunk_{i}"
                )
                chunked_documents.append(chunked_doc)
        
        return chunked_documents


class FilteredRetriever(BaseRetriever):
    """Apply filters to retrieved documents."""
    
    def __init__(self,
                 base_retriever: BaseRetriever,
                 filter_func: Optional[Callable[[Document], bool]] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 required_metadata: Optional[List[str]] = None,
                 config: Optional[RetrieverConfig] = None):
        super().__init__(config)
        self.base_retriever = base_retriever
        self.filter_func = filter_func
        self.min_length = min_length
        self.max_length = max_length
        self.required_metadata = required_metadata or []
    
    def retrieve(self, query: Optional[str] = None) -> List[Document]:
        """Retrieve and filter documents."""
        documents = self.base_retriever.retrieve(query)
        filtered_documents = []
        
        for doc in documents:
            # Apply custom filter function
            if self.filter_func and not self.filter_func(doc):
                continue
            
            # Apply length filters
            doc_length = len(doc.content)
            if self.min_length and doc_length < self.min_length:
                continue
            if self.max_length and doc_length > self.max_length:
                continue
            
            # Check required metadata
            if self.required_metadata:
                if not all(field in doc.metadata for field in self.required_metadata):
                    continue
            
            filtered_documents.append(doc)
        
        self.logger.info(f"Filtered {len(documents)} documents to {len(filtered_documents)}")
        return filtered_documents