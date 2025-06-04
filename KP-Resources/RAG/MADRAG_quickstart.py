"""
RAG System Quick Start Guide

This file shows the fastest way to get started with the RAGSystem library.
Perfect for users who want to see immediate results with minimal configuration.

Requirements:
- rag_system.py
- pip install torch transformers langchain langchain-community langchain-huggingface chromadb
"""

from rag_system import RAGSystem

def quick_start_example():
    """
    Minimal example to get RAG system working in under 10 lines of code
    """
    print("🚀 RAG System Quick Start")
    print("-" * 40)
    
    # 1. Initialize RAG system
    rag = RAGSystem(verbose=True)
    
    # 2. Load your documents (replace with your actual document path)
    try:
        # Option A: Load from directory
        rag.loadDocuments("./my_documents/")
        
        # Option B: Load specific files
        # rag.loadDocuments(["./doc1.txt", "./doc2.pdf", "./notes.md"])
        
    except FileNotFoundError:
        # Create sample documents if none found
        print("📝 Creating sample documents for demo...")
        from langchain.schema import Document
        rag.documents = [
            Document(page_content="Python is great for data science and AI development.", 
                    metadata={"filename": "python.txt"}),
            Document(page_content="JavaScript powers modern web applications and Node.js servers.", 
                    metadata={"filename": "javascript.txt"}),
            Document(page_content="Docker containers make application deployment consistent across environments.", 
                    metadata={"filename": "docker.txt"})
        ]
    
    # 3. Process documents
    rag.splitDocuments()
    rag.createVectorstore()
    
    # 4. Setup model (replace model_path with your actual model)
    try:
        rag.setupQAChain(
            model_type="huggingface",
            model_path="microsoft/DialoGPT-medium",  # Replace with your model path
            generation_kwargs={"max_new_tokens": 200}
        )
        
        # 5. Ask questions!
        questions = [
            "What programming languages are mentioned?",
            "Tell me about Docker",
            "What are the main topics covered?"
        ]
        
        for question in questions:
            print(f"\n❓ {question}")
            answer = rag.query(question)
            print(f"💡 {answer}")
            
    except Exception as e:
        print(f"⚠️  Model setup failed: {e}")
        print("💡 To fix: Replace 'model_path' with path to your local model")

def practical_configurations():
    """
    Common real-world configurations users typically need
    """
    print("\n" + "=" * 50)
    print("🔧 Practical Configuration Examples")
    print("=" * 50)
    
    # Configuration 1: PDF Documents + Research Papers
    print("\n📚 Configuration 1: Research Papers & PDFs")
    rag_research = RAGSystem(verbose=True)
    
    try:
        # Load PDF and text files from research directory
        rag_research.loadDocuments(
            paths="./research_papers/",
            file_extensions=["pdf", "txt", "md"]
        )
        
        # Use larger chunks for academic papers
        rag_research.splitDocuments(chunk_size=1500, chunk_overlap=300)
        
        # Use scientific embeddings for better research content understanding
        rag_research.createVectorstore(
            embedding_model_name="sentence-transformers/allenai-specter",
            vectorstore_kwargs={"persist_directory": "./research_db"}
        )
        
        print("✅ Research configuration ready")
        
    except Exception as e:
        print(f"⚠️  Research config failed: {e}")
    
    # Configuration 2: Code Documentation
    print("\n💻 Configuration 2: Code Documentation")
    rag_code = RAGSystem(verbose=True)
    
    try:
        # Load code-related files
        rag_code.loadDocuments(
            paths=["./docs/", "./README.md", "./api_reference/"],
            file_extensions=["md", "rst", "txt"]
        )
        
        # Smaller chunks for code snippets
        rag_code.splitDocuments(chunk_size=800, chunk_overlap=100)
        
        # Use code-aware embeddings
        rag_code.createVectorstore(
            embedding_model_name="microsoft/codebert-base",
            vectorstore_kwargs={"persist_directory": "./code_db"}
        )
        
        print("✅ Code documentation configuration ready")
        
    except Exception as e:
        print(f"⚠️  Code config failed: {e}")
    
    # Configuration 3: Multi-GPU High Performance
    print("\n🚀 Configuration 3: High Performance Multi-GPU")
    rag_performance = RAGSystem(
        cuda_visible_devices="0,1",  # Use multiple GPUs
        verbose=True
    )
    
    try:
        # Large document processing
        rag_performance.loadDocuments("./large_documents/")
        rag_performance.splitDocuments(chunk_size=1200, chunk_overlap=200)
        
        # Distribute workload across GPUs
        rag_performance.createVectorstore(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2",
            embedding_kwargs={
                "model_kwargs": {"device": "cuda:1"}  # Use second GPU for embeddings
            }
        )
        
        # Setup large model with custom device mapping
        custom_device_map = {
            "model.embed_tokens": 0,
            "lm_head": 0,
            "model.norm": 0,
        }
        
        # Distribute model layers (example for 32-layer model)
        for i in range(32):
            custom_device_map[f"model.layers.{i}"] = 0 if i < 24 else 1
        
        rag_performance.setupQAChain(
            model_type="huggingface",
            model_path="/path/to/large/model",  # Replace with actual large model
            model_kwargs={
                "device_map": custom_device_map,
                "torch_dtype": torch.float16
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.7
            }
        )
        
        print("✅ High performance configuration ready")
        
    except Exception as e:
        print(f"⚠️  High performance config failed: {e}")

def common_usage_patterns():
    """
    Show common ways users interact with the RAG system
    """
    print("\n" + "=" * 50)
    print("📋 Common Usage Patterns")
    print("=" * 50)
    
    # Setup basic system for examples
    rag = RAGSystem(verbose=False)  # Quiet mode for cleaner output
    
    # Create sample documents
    from langchain.schema import Document
    rag.documents = [
        Document(page_content="FastAPI is a modern web framework for Python APIs. It's fast, easy to use, and automatically generates documentation.", metadata={"filename": "fastapi.txt"}),
        Document(page_content="React is a JavaScript library for building user interfaces. It uses a component-based architecture and virtual DOM.", metadata={"filename": "react.txt"}),
        Document(page_content="PostgreSQL is a powerful relational database. It supports JSON, full-text search, and advanced indexing.", metadata={"filename": "postgresql.txt"})
    ]
    
    rag.splitDocuments()
    rag.createVectorstore()
    
    try:
        rag.setupQAChain(
            model_type="huggingface",
            model_path="microsoft/DialoGPT-small",
            generation_kwargs={"max_new_tokens": 150}
        )
        
        # Pattern 1: Simple Q&A
        print("\n🎯 Pattern 1: Simple Question & Answer")
        answer = rag.query("What is FastAPI?")
        print(f"Q: What is FastAPI?\nA: {answer}")
        
        # Pattern 2: Comparative Questions
        print("\n🔄 Pattern 2: Comparative Analysis")
        answer = rag.query("Compare React and FastAPI")
        print(f"Q: Compare React and FastAPI\nA: {answer}")
        
        # Pattern 3: Debug Mode for Development
        print("\n🐛 Pattern 3: Debug Mode (shows retrieved documents)")
        answer = rag.query("Tell me about databases", debug=True)
        
        # Pattern 4: Direct Context (for comprehensive answers)
        print("\n📖 Pattern 4: Direct Context Query")
        unique_docs = rag.getAllUniqueDocuments()
        context = "\n\n".join([f"{doc.metadata['filename']}: {doc.page_content}" 
                              for doc in unique_docs])
        answer = rag.queryWithDirectContext(context, "Summarize all technologies mentioned")
        print(f"Q: Summarize all technologies\nA: {answer}")
        
        # Pattern 5: System Information
        print("\n📊 Pattern 5: System Information")
        info = rag.getSystemInfo()
        print("System Status:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"⚠️  Usage patterns demo failed: {e}")

def troubleshooting_guide():
    """
    Common issues and solutions
    """
    print("\n" + "=" * 50)
    print("🔧 Troubleshooting Guide")
    print("=" * 50)
    
    print("\n❌ Common Issues & Solutions:")
    
    print("\n1. 'No documents found' error:")
    print("   ✅ Check file paths exist")
    print("   ✅ Verify file extensions match glob pattern")
    print("   ✅ Use absolute paths if relative paths fail")
    
    print("\n2. CUDA out of memory:")
    print("   ✅ Reduce chunk_size in splitDocuments()")
    print("   ✅ Use smaller embedding model")
    print("   ✅ Enable model quantization (load_in_8bit=True)")
    print("   ✅ Use CPU for embeddings")
    
    print("\n3. Model loading fails:")
    print("   ✅ Verify model path is correct")
    print("   ✅ Check model format matches model_type")
    print("   ✅ Ensure sufficient disk space and memory")
    print("   ✅ Try smaller model first")
    
    print("\n4. Slow performance:")
    print("   ✅ Use GPU if available")
    print("   ✅ Reduce max_new_tokens")
    print("   ✅ Use faster embedding model")
    print("   ✅ Enable model quantization")
    
    print("\n5. Poor answer quality:")
    print("   ✅ Increase chunk_size and overlap")
    print("   ✅ Use better embedding model")
    print("   ✅ Adjust retriever max_docs parameter")
    print("   ✅ Try different prompt template")

def main():
    """
    Run the quick start demonstration
    """
    # Run quick start example
    quick_start_example()
    
    # Show practical configurations
    practical_configurations()
    
    # Show common usage patterns
    common_usage_patterns()
    
    # Show troubleshooting guide
    troubleshooting_guide()
    
    print("\n" + "=" * 50)
    print("🎉 Quick Start Complete!")
    print("=" * 50)
    print("\n📝 Next Steps:")
    print("1. Replace sample paths with your actual document directories")
    print("2. Replace model_path with your local model")
    print("3. Adjust chunk_size and other parameters for your use case")
    print("4. Check the comprehensive demo file for advanced examples")
    
    print("\n📚 Files you should have:")
    print("• rag_system.py (main library)")
    print("• rag_demo.py (comprehensive examples)")
    print("• quick_start.py (this file)")
    
    print("\n🔗 Import in your project:")
    print("from rag_system import RAGSystem")

if __name__ == "__main__":
    main()