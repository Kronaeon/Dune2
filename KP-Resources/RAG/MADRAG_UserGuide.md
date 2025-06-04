# RAG System Library - Complete Usage Guide

A flexible, production-ready Retrieval-Augmented Generation (RAG) system that supports multiple model backends, embedding systems, and vector stores.

## 📁 File Structure

```
your_project/
├── rag_system.py          # Main library (import this)
├── rag_demo.py            # Comprehensive examples & advanced use cases  
├── quick_start.py         # Simple examples to get started immediately
└── README.md              # This guide
```

## 🚀 Quick Start (30 seconds)

```python
from rag_system import RAGSystem

# 1. Initialize
rag = RAGSystem(verbose=True)

# 2. Load documents  
rag.loadDocuments("./my_documents/")

# 3. Process documents
rag.splitDocuments()
rag.createVectorstore()

# 4. Setup model
rag.setupQAChain(
    model_type="huggingface",
    model_path="/path/to/your/model"
)

# 5. Ask questions
answer = rag.query("What are the main topics?")
print(answer)
```

## 📋 Installation Requirements

```bash
pip install torch transformers langchain langchain-community langchain-huggingface chromadb

# Optional for additional model backends:
pip install vllm                    # For VLLM support
pip install llama-cpp-python        # For Llama.cpp support  
pip install faiss-cpu              # For FAISS vector store
```

## 🎯 Use Cases & Examples

### 1. Basic Document Q&A
**Perfect for**: Simple document collections, personal notes, small knowledge bases

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.loadDocuments("./documents/", glob_pattern="**/*.txt")
rag.splitDocuments(chunk_size=800)
rag.createVectorstore()
rag.setupQAChain(model_type="huggingface", model_path="microsoft/DialoGPT-medium")

answer = rag.query("What is the main topic discussed?")
```

### 2. Research Paper Analysis  
**Perfect for**: Academic papers, technical documents, literature reviews

```python
rag = RAGSystem(verbose=True)
rag.loadDocuments("./research_papers/", file_extensions=["pdf", "txt"])
rag.splitDocuments(chunk_size=1500, chunk_overlap=300)  # Larger chunks for papers
rag.createVectorstore(
    embedding_model_name="sentence-transformers/allenai-specter"  # Scientific embeddings
)
rag.setupQAChain(model_type="huggingface", model_path="allenai/scibert_scivocab_uncased")

answer = rag.query("What are the main methodologies discussed?")
```

### 3. Code Documentation Assistant
**Perfect for**: API docs, code repositories, technical guides

```python
rag = RAGSystem()
rag.loadDocuments(["./docs/", "./README.md"], file_extensions=["md", "rst"])
rag.splitDocuments(chunk_size=800, chunk_overlap=100)
rag.createVectorstore(embedding_model_name="microsoft/codebert-base")
rag.setupQAChain(model_type="huggingface", model_path="microsoft/CodeGPT-small-py")

answer = rag.query("How do I implement authentication?")
```

### 4. Multi-GPU High Performance
**Perfect for**: Large document collections, production deployments

```python
rag = RAGSystem(cuda_visible_devices="0,1", verbose=True)
rag.loadDocuments("./large_dataset/")
rag.splitDocuments(chunk_size=1200, chunk_overlap=200)
rag.createVectorstore(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    embedding_kwargs={"model_kwargs": {"device": "cuda:1"}}  # Separate GPU
)

# Custom device mapping for large models
device_map = {"model.embed_tokens": 0, "lm_head": 0}
for i in range(32): device_map[f"model.layers.{i}"] = 0 if i < 24 else 1

rag.setupQAChain(
    model_type="huggingface",
    model_path="/path/to/large/model",
    model_kwargs={"device_map": device_map, "torch_dtype": torch.float16}
)
```

## 🔧 Configuration Options

### Model Types Supported
```python
# Hugging Face models (transformers)
rag.setupQAChain(model_type="huggingface", model_path="microsoft/DialoGPT-medium")

# VLLM for high-performance inference  
rag.setupQAChain(model_type="vllm", model_path="meta-llama/Llama-2-7b-hf")

# Llama.cpp for CPU/quantized models
rag.setupQAChain(model_type="llamacpp", model_path="./model.gguf")
```

### Vector Store Options
```python
# Chroma (default, persistent)
rag.createVectorstore(vectorstore_type='chroma')

# FAISS (in-memory, fast)  
rag.createVectorstore(vectorstore_type='faiss')
```

### Text Splitting Strategies
```python
# For general documents
rag.splitDocuments(chunk_size=800, chunk_overlap=100)

# For academic papers (larger context)
rag.splitDocuments(chunk_size=1500, chunk_overlap=300)

# For code documentation (smaller, precise chunks)
rag.splitDocuments(chunk_size=600, chunk_overlap=80)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No documents found" | Check file paths and glob patterns |
| CUDA out of memory | Reduce chunk_size, use smaller model, enable quantization |
| Model loading fails | Verify model path, check disk space, try smaller model |
| Slow performance | Use GPU, reduce max_new_tokens, enable quantization |
| Poor answer quality | Increase chunk_size, use better embedding model |

## 💡 Performance Tips

### Memory Optimization
```python
# For limited memory (<8GB GPU)
rag.setupQAChain(
    model_kwargs={
        "torch_dtype": torch.float16,
        "load_in_8bit": True,  # Quantization
        "device_map": "auto"
    },
    generation_kwargs={"max_new_tokens": 128}
)
```

### Speed Optimization
```python
# Use lightweight embeddings for faster processing
rag.createVectorstore(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Reduce retrieval docs for faster responses
rag.setupQAChain(retriever_kwargs={"max_docs": 3, "min_docs": 1})
```

## 📚 Advanced Features

### Debug Mode
```python
# See which documents are retrieved for each query
answer = rag.query("My question", debug=True)
```

### Direct Context Queries
```python
# Bypass retrieval, use all documents as context
unique_docs = rag.getAllUniqueDocuments()
context = "\n".join([doc.page_content for doc in unique_docs])
answer = rag.queryWithDirectContext(context, "Summarize everything")
```

### System Information
```python
# Get system status and configuration
info = rag.getSystemInfo()
print(f"Documents loaded: {info['documents_loaded']}")
print(f"GPU memory: {info['gpu_info']}")
```

## 🔄 Common Workflow Patterns

### Research Workflow
1. Load academic papers → Use scientific embeddings → Large chunks → Specialized model
2. Query for methodologies, findings, comparisons
3. Use direct context for comprehensive summaries

### Development Workflow  
1. Load code docs → Use code embeddings → Small chunks → Code-aware model
2. Query for implementation details, examples, troubleshooting
3. Use debug mode to verify relevant sections retrieved

### Production Workflow
1. Load enterprise docs → Multi-GPU setup → Optimized chunks → Large model
2. Implement error handling and logging
3. Monitor performance and memory usage

## 📖 File Guide

- **`rag_system.py`**: Import this in your project - contains the main RAGSystem class
- **`quick_start.py`**: Run this first to see basic examples and verify setup  
- **`rag_demo.py`**: Comprehensive examples showing all advanced features
- **This README**: Complete reference guide

## 🚀 Getting Started Checklist

- [ ] Install required packages
- [ ] Download/prepare your model files
- [ ] Organize your documents in directories  
- [ ] Run `python quick_start.py` to test basic functionality
- [ ] Explore `python rag_demo.py` for advanced examples
- [ ] Import `from rag_system import RAGSystem` in your project
- [ ] Configure paths and parameters for your specific use case

## 🤝 Integration Examples

### Web API Integration
```python
from flask import Flask, request, jsonify
from rag_system import RAGSystem

app = Flask(__name__)
rag = RAGSystem()
# ... setup rag system ...

@app.route('/query', methods=['POST'])
def query_documents():
    question = request.json['question']
    answer = rag.query(question)
    return jsonify({'answer': answer})
```

### Batch Processing
```python
questions = ["What is X?", "How does Y work?", "Compare A and B"]
answers = []

for question in questions:
    answer = rag.query(question)
    answers.append({'question': question, 'answer': answer})
```

The RAGSystem library is designed to be flexible, performant, and easy to integrate into existing workflows. Start with the quick start guide, then explore the comprehensive demos to find patterns that match your specific use case.