
# MADRAG_Mod_1.py Created from modifying the working RagTest6.py 
- (RagTest6 is actually the 9th variant but don't talk about that. Ref Arakis if you're Arthur, or RoseCode Repo if you're anyone else.)


## Key Features Implemented

**Class Structure**: Created a RAGSystem class with all required camelCase methods:
- `__init__` for configurable initialization  
- `loadDocuments` for flexible document loading from files/directories
- `splitDocuments` with configurable chunking parameters
- `createVectorstore` supporting multiple embedding models and vector stores
- `setupQAChain` with support for Hugging Face, VLLM, and Llama.cpp models
- `query` and `queryWithDirectContext` for different querying approaches

**Configuration Flexibility**: Eliminated all hard-coded values:
- Configurable document paths (files or directories) with custom glob patterns
- Adjustable text splitting parameters (chunk size, overlap, minimum size)
- Flexible embedding model selection with custom device placement
- Support for multiple vector store types (Chroma, FAISS) with extensibility
- Model-agnostic design supporting different backends with custom parameters

**Model Backend Support**: Implemented comprehensive model handling:
- **Hugging Face**: Full integration with AutoModelForCausalLM and pipeline setup
- **VLLM**: Framework ready for VLLM integration 
- **Llama.cpp**: Framework ready for Llama.cpp integration
- Custom device mapping and memory management
- Flexible generation parameters for each model type

**Extensibility Design**: The architecture supports future expansion:
- Abstract model loading methods that can be extended
- Pluggable vector store system
- Configurable text splitter types
- Custom retrieval strategies with deduplication

**Production Features**: Enhanced the original functionality:
- Comprehensive error handling and validation
- Memory management and GPU allocation optimization
- Debug modes with detailed logging
- System information reporting
- Document deduplication and content hashing

## Test Bench

The included test bench demonstrates complete usage:
- System initialization with GPU configuration
- Document loading with fallback to demo data
- Vector store creation with custom embeddings
- QA chain setup with model configuration
- Multiple query examples including debug mode

The library can now be imported and used in any project with:
```python
from rag_system import RAGSystem
rag = RAGSystem(verbose=True)
```

Users simply need to configure their specific paths, models, and parameters without modifying the core library code. The implementation maintains full functionality while providing the flexibility and modularity you requested.




