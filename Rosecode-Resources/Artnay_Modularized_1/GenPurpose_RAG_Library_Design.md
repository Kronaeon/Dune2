# General-Purpose RAG Library Design

## Analysis of Source Scripts

### tubeShortsScriptGen.py - RAG Components:
1. **Retrieval**: 
   - Multi-strategy loading (topic_overview.md or individual clean_*.md files)
   - Metadata extraction from YAML frontmatter
   - Content consolidation from multiple sources

2. **Augmentation**:
   - Dynamic prompt templates based on style parameter
   - Content truncation (1500 chars) for context window management
   - Structured prompt formatting with specific requirements

3. **Generation**:
   - LLM integration via llama.cpp
   - Structured output parsing and validation
   - Multiple generation styles support

### Lsc-2.py - RAG Components:
1. **Retrieval**:
   - File system traversal for content files
   - Metadata extraction from content headers
   - Aggregation of all processed files

2. **Augmentation**:
   - Simple prompt construction
   - Larger context window usage (6000 chars)
   - Task-specific instructions (summarization)

3. **Generation**:
   - Dual backend support (local LLM and API)
   - Topic overview generation

## Proposed Library Structure

```
rag_toolkit/
├── __init__.py
├── config.py           # Configuration management
├── retriever.py        # Content retrieval strategies
├── augmenter.py        # Prompt construction and augmentation
├── generator.py        # LLM interaction and generation
├── parser.py           # Output parsing and validation
├── pipeline.py         # RAG pipeline orchestration
├── utils.py            # Utility functions
└── examples/
    ├── simple_qa.py
    ├── document_summary.py
    └── creative_generation.py
```

## Core Components Design

### 1. Retriever Module
- **Base Retriever Class**: Abstract interface for all retrievers
- **FileRetriever**: Load from single/multiple files
- **DirectoryRetriever**: Scan directories with pattern matching
- **AggregateRetriever**: Combine multiple retrieval strategies
- **Features**:
  - Metadata extraction
  - Content filtering
  - Chunking strategies
  - Caching support

### 2. Augmenter Module
- **Base Augmenter Class**: Abstract interface for prompt construction
- **TemplateAugmenter**: Use customizable templates
- **ContextWindowAugmenter**: Manage context size limits
- **StyleAugmenter**: Apply different prompt styles
- **Features**:
  - Dynamic template rendering
  - Context window management
  - Metadata injection
  - Multi-turn conversation support

### 3. Generator Module
- **Base Generator Class**: Abstract LLM interface
- **LlamaCppGenerator**: Local model via llama.cpp
- **APIGenerator**: Remote API-based models
- **Features**:
  - Unified generation interface
  - Configurable parameters
  - Streaming support
  - Error handling and retries

### 4. Parser Module
- **Base Parser Class**: Abstract output parser
- **StructuredParser**: Extract structured data from text
- **ValidationParser**: Validate output against schemas
- **Features**:
  - Flexible parsing strategies
  - Output validation
  - Error correction

### 5. Pipeline Module
- **RAGPipeline**: Orchestrate retrieval, augmentation, and generation
- **Features**:
  - Component composition
  - Batch processing
  - Progress tracking
  - Result caching

## Key Design Principles

1. **Modularity**: Each component can be used independently or combined
2. **Configurability**: All parameters exposed through config or constructors
3. **Extensibility**: Easy to add new retrievers, augmenters, or generators
4. **Type Safety**: Use type hints throughout
5. **Error Handling**: Comprehensive error handling with informative messages
6. **Logging**: Configurable logging at multiple levels
7. **Performance**: Efficient data handling, caching, and parallel processing