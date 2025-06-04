"""RAG Toolkit - A flexible, modular library for Retrieval-Augmented Generation.

This library provides composable components for building RAG applications:
- Flexible retrieval strategies from various data sources
- Customizable prompt augmentation
- Support for multiple LLM backends (local and API-based)
- Structured output parsing
- Complete pipeline orchestration

Example:
    >>> from rag_toolkit import FileRetriever, TemplateAugmenter, LlamaCppGenerator, RAGPipeline
    >>> 
    >>> # Create components
    >>> retriever = FileRetriever("data.txt")
    >>> augmenter = TemplateAugmenter()
    >>> generator = LlamaCppGenerator("model.gguf")
    >>> 
    >>> # Create and run pipeline
    >>> pipeline = RAGPipeline(retriever, augmenter, generator)
    >>> result = pipeline.run("What is the main topic?")
    >>> print(result.output)
"""

__version__ = "1.0.0"
__author__ = "RAG Toolkit Contributors"

# Import configuration
from .config import (
    RAGConfig,
    RetrieverConfig,
    AugmenterConfig,
    GeneratorConfig
)

# Import retrievers
from .retriever import (
    BaseRetriever,
    Document,
    FileRetriever,
    DirectoryRetriever,
    ConsolidatedRetriever,
    ChunkedRetriever,
    FilteredRetriever
)

# Import augmenters
from .augmenter import (
    BaseAugmenter,
    TemplateAugmenter,
    StyleAugmenter,
    ConversationalAugmenter,
    ChainAugmenter,
    DynamicAugmenter
)

# Import generators
from .generator import (
    BaseGenerator,
    GenerationResult,
    LlamaCppGenerator,
    APIGenerator,
    HybridGenerator
)

# Import parsers
from .parser import (
    BaseParser,
    ParsedOutput,
    RegexParser,
    StructuredParser,
    JSONParser,
    TemplateParser,
    ValidationParser,
    ChainParser
)

# Import pipeline
from .pipeline import (
    RAGPipeline,
    PipelineResult,
    ParallelPipeline,
    CachedPipeline,
    AdaptivePipeline
)

# Import utilities
from .utils import (
    setup_logging,
    truncate_text,
    clean_text,
    chunk_text,
    calculate_hash,
    estimate_tokens,
    batch_items,
    timer,
    retry,
    ProgressTracker,
    TokenBudget
)

# Define what's available when using "from rag_toolkit import *"
__all__ = [
    # Config
    "RAGConfig",
    "RetrieverConfig", 
    "AugmenterConfig",
    "GeneratorConfig",
    
    # Retrievers
    "BaseRetriever",
    "Document",
    "FileRetriever",
    "DirectoryRetriever",
    "ConsolidatedRetriever",
    "ChunkedRetriever",
    "FilteredRetriever",
    
    # Augmenters
    "BaseAugmenter",
    "TemplateAugmenter",
    "StyleAugmenter",
    "ConversationalAugmenter",
    "ChainAugmenter",
    "DynamicAugmenter",
    
    # Generators
    "BaseGenerator",
    "GenerationResult",
    "LlamaCppGenerator",
    "APIGenerator",
    "HybridGenerator",
    
    # Parsers
    "BaseParser",
    "ParsedOutput",
    "RegexParser",
    "StructuredParser",
    "JSONParser",
    "TemplateParser",
    "ValidationParser",
    "ChainParser",
    
    # Pipeline
    "RAGPipeline",
    "PipelineResult",
    "ParallelPipeline",
    "CachedPipeline",
    "AdaptivePipeline",
    
    # Utils
    "setup_logging",
    "truncate_text",
    "clean_text",
    "chunk_text",
    "timer",
    "ProgressTracker",
    "TokenBudget"
]


def create_pipeline(config_path: str = None, **kwargs) -> RAGPipeline:
    """Convenience function to create a pipeline from configuration.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Override configuration values
        
    Returns:
        Configured RAGPipeline
    """
    # Load configuration
    if config_path:
        config = RAGConfig.from_file(config_path)
    else:
        config = RAGConfig()
    
    # Apply overrides
    if kwargs:
        override_config = RAGConfig.from_dict(kwargs)
        config = config.merge(override_config)
    
    # Create components based on configuration
    # This is a simplified example - in practice, you'd have more logic here
    raise NotImplementedError(
        "create_pipeline is a placeholder. "
        "Please create components manually for now."
    )