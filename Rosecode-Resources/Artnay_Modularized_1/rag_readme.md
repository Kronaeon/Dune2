# RAG Toolkit

A flexible, modular Python library for building Retrieval-Augmented Generation (RAG) applications. This toolkit provides composable components for retrieving information, augmenting prompts, generating responses, and parsing outputs - all with a focus on extensibility and ease of use.

## Features

- **Modular Architecture**: Mix and match components to build custom RAG pipelines
- **Multiple Retrieval Strategies**: File-based, directory scanning, chunking, filtering, and more
- **Flexible Prompt Augmentation**: Templates, styles, conversational formats, and dynamic augmentation
- **Multi-Backend Generation**: Support for local models (llama.cpp) and API-based models
- **Structured Output Parsing**: Extract and validate structured data from generated text
- **Pipeline Orchestration**: Complete RAG workflows with caching, parallel execution, and adaptation
- **Comprehensive Configuration**: Flexible configuration through files, environment variables, or code
- **Production Ready**: Logging, error handling, progress tracking, and performance optimization

## Installation

### Basic Installation

```bash
pip install rag-toolkit
```

### With Local Model Support (llama.cpp)

```bash
pip install rag-toolkit[local]
```

### Development Installation

```bash
git clone https://github.com/yourusername/rag-toolkit.git
cd rag-toolkit
pip install -e ".[dev]"
```

## Quick Start

```python
from rag_toolkit import (
    FileRetriever,
    StyleAugmenter,
    LlamaCppGenerator,
    RAGPipeline
)

# Create components
retriever = FileRetriever("data.txt")
augmenter = StyleAugmenter(style="qa")
generator = LlamaCppGenerator("model.gguf")

# Build pipeline
pipeline = RAGPipeline(retriever, augmenter, generator)

# Run query
result = pipeline.run("What are the main points?")
print(result.output)
```

## Core Components

### 1. Retrievers

Retrievers fetch relevant documents from various sources:

```python
# Single file
retriever = FileRetriever("document.txt")

# Multiple files
retriever = FileRetriever(["doc1.txt", "doc2.txt"])

# Directory of files
retriever = DirectoryRetriever("docs/", pattern="*.md")

# Chunked retrieval
base_retriever = FileRetriever("large_doc.txt")
retriever = ChunkedRetriever(base_retriever, chunk_size=500)

# Filtered retrieval
retriever = FilteredRetriever(
    base_retriever,
    min_length=100,
    required_metadata=["title", "date"]
)
```

### 2. Augmenters

Augmenters create prompts by combining queries with retrieved documents:

```python
# Template-based augmentation
augmenter = TemplateAugmenter(
    template="Context: ${context}\n\nQuestion: ${query}\n\nAnswer:"
)

# Style-based augmentation
augmenter = StyleAugmenter(style="summary")  # qa, summary, creative, analytical

# Conversational augmentation
augmenter = ConversationalAugmenter(
    system_message="You are a helpful assistant."
)

# Dynamic augmentation
augmenter = DynamicAugmenter()  # Adapts based on content
```

### 3. Generators

Generators produce text using various LLM backends:

```python
# Local model (llama.cpp)
generator = LlamaCppGenerator(
    model_path="path/to/model.gguf",
    n_gpu_layers=35,
    max_tokens=512
)

# API-based model
generator = APIGenerator(
    api_key="your-key",
    model_name="gpt-3.5-turbo"
)

# Hybrid with fallback
generator = HybridGenerator(
    generators=[local_gen, api_gen],
    strategy="fallback"
)
```

### 4. Parsers

Parsers extract structured data from generated text:

```python
# Regex parsing
parser = RegexParser(
    patterns={
        "title": r"Title: (.+)",
        "summary": r"Summary: (.+)",
    }
)

# JSON parsing
parser = JSONParser(schema={"required": ["name", "value"]})

# Structured sections
parser = StructuredParser()  # Parses markdown-like structures

# Template-based parsing
parser = TemplateParser(
    template="Name: {{name}}\nAge: {{age}}"
)
```

### 5. Pipelines

Pipelines orchestrate the complete RAG workflow:

```python
# Basic pipeline
pipeline = RAGPipeline(retriever, augmenter, generator, parser)

# Cached pipeline
pipeline = CachedPipeline(
    retriever, augmenter, generator,
    cache_dir="cache/",
    cache_ttl=3600
)

# Adaptive pipeline
pipeline = AdaptivePipeline(
    retriever, augmenter, generator,
    fallback_retriever=backup_retriever,
    min_documents=2
)

# Parallel pipelines
pipelines = ParallelPipeline(
    [pipeline1, pipeline2, pipeline3],
    strategy="all"  # or "race"
)
```

## Configuration

### Using Configuration Files

```yaml
# config.yaml
retriever:
  chunk_size: 1000
  chunk_overlap: 200
  
augmenter:
  max_context_length: 2048
  include_metadata: true
  
generator:
  model_path: /path/to/model.gguf
  max_tokens: 512
  temperature: 0.7
  n_gpu_layers: 35
```

```python
# Load configuration
config = RAGConfig.from_file("config.yaml")

# Use with components
retriever = FileRetriever("data.txt", config=config.retriever)
```

### Environment Variables

```bash
export RAG_MODEL_PATH=/path/to/model.gguf
export RAG_MAX_TOKENS=1024
export RAG_TEMPERATURE=0.8
```

```python
config = RAGConfig.from_env()
```

## Advanced Usage

### Streaming Responses

```python
for chunk in pipeline.run_stream("Your question"):
    if chunk["type"] == "token":
        print(chunk["content"], end="", flush=True)
```

### Custom Components

```python
class CustomRetriever(BaseRetriever):
    def retrieve(self, query=None):
        # Your custom retrieval logic
        documents = []
        # ... fetch documents ...
        return documents

class CustomAugmenter(BaseAugmenter):
    def augment(self, documents, query=None, **kwargs):
        # Your custom augmentation logic
        prompt = f"Custom prompt with {len(documents)} documents"
        return prompt
```

### Pipeline Hooks

```python
def log_documents(documents=None, **kwargs):
    print(f"Retrieved {len(documents)} documents")

pipeline.add_hook("retrieve", log_documents, position="post")
```

### Batch Processing

```python
from rag_toolkit.utils import batch_items, ProgressTracker

questions = ["Q1", "Q2", "Q3", ...]

with ProgressTracker(len(questions), "Processing") as tracker:
    for batch in batch_items(questions, batch_size=10):
        results = [pipeline.run(q) for q in batch]
        tracker.update(len(batch))
```

## Examples

The library includes comprehensive examples for various use cases:

- **Question Answering**: `examples/simple_qa.py`
- **Document Summarization**: `examples/document_summary.py`
- **Creative Generation**: `examples/creative_generation.py`

### Simple Q&A System

```python
from rag_toolkit import FileRetriever, StyleAugmenter, LlamaCppGenerator, RAGPipeline

# Set up components
retriever = FileRetriever("knowledge_base.txt")
augmenter = StyleAugmenter(style="qa")
generator = LlamaCppGenerator("llama-model.gguf")

# Create pipeline
pipeline = RAGPipeline(retriever, augmenter, generator)

# Ask questions
response = pipeline.run("What is machine learning?")
print(response.output)
```

### Multi-Document Summarization

```python
from rag_toolkit import DirectoryRetriever, TemplateAugmenter, APIGenerator, RAGPipeline

# Retrieve all documents
retriever = DirectoryRetriever("research_papers/", pattern="*.pdf")

# Custom summarization prompt
augmenter = TemplateAugmenter(
    template="""Summarize these research papers:
    
${context}

Focus on:
1. Key findings
2. Methodologies
3. Common themes

Summary:"""
)

# Use API for generation
generator = APIGenerator(api_key="your-key")

# Run pipeline
pipeline = RAGPipeline(retriever, augmenter, generator)
result = pipeline.run()
```

## Best Practices

1. **Choose the Right Retriever**: 
   - Use `ChunkedRetriever` for long documents
   - Use `FilteredRetriever` to ensure quality
   - Use `ConsolidatedRetriever` for multiple sources

2. **Optimize Context Windows**:
   ```python
   augmenter = TemplateAugmenter(
       config=AugmenterConfig(
           max_context_length=2048,
           truncation_strategy="smart"
       )
   )
   ```

3. **Handle Errors Gracefully**:
   ```python
   from rag_toolkit.utils import retry
   
   @retry(max_attempts=3, delay=1.0)
   def generate_with_retry(pipeline, query):
       return pipeline.run(query)
   ```

4. **Monitor Performance**:
   ```python
   result = pipeline.run("Question")
   print(f"Execution time: {result.execution_time:.2f}s")
   print(f"Stage timings: {result.stage_timings}")
   ```

5. **Use Appropriate Temperatures**:
   - Low (0.0-0.3): Factual, consistent responses
   - Medium (0.4-0.7): Balanced creativity
   - High (0.8-1.0): Creative, varied responses

## Performance Optimization

### GPU Acceleration

```python
generator = LlamaCppGenerator(
    model_path="model.gguf",
    n_gpu_layers=-1,  # Use all layers on GPU
    n_batch=512       # Larger batch for GPU
)
```

### Caching

```python
pipeline = CachedPipeline(
    retriever, augmenter, generator,
    cache_dir=".cache",
    cache_ttl=3600  # 1 hour
)
```

### Parallel Processing

```python
# Run multiple pipelines in parallel
results = ParallelPipeline(
    [pipeline1, pipeline2],
    strategy="race"  # Return first successful
).run(query)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `n_gpu_layers` or `max_tokens`
   - Use chunking for large documents
   - Enable CPU offloading

2. **Slow Generation**:
   - Increase `n_gpu_layers` for GPU acceleration
   - Reduce `max_tokens` if appropriate
   - Use caching for repeated queries

3. **Poor Quality Results**:
   - Tune temperature and top_p parameters
   - Improve retrieval with better filtering
   - Refine prompts with custom templates

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Setup development environment
git clone https://github.com/yourusername/rag-toolkit.git
cd rag-toolkit
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black rag_toolkit/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the RAG implementations in the provided scripts
- Built on top of excellent libraries like llama-cpp-python
- Thanks to all contributors and the open-source community

## Citation

If you use this library in your research, please cite:

```bibtex
@software{rag_toolkit,
  title = {RAG Toolkit: A Flexible Library for Retrieval-Augmented Generation},
  year = {2024},
  url = {https://github.com/yourusername/rag-toolkit}
}
```