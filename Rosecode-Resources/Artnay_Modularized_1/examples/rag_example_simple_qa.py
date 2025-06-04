"""Simple Question-Answering Example using RAG Toolkit.

This example demonstrates how to build a basic Q&A system that retrieves
information from documents and generates answers.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports (if running from examples folder)
sys.path.append(str(Path(__file__).parent.parent))

from rag_toolkit import (
    FileRetriever,
    DirectoryRetriever,
    StyleAugmenter,
    LlamaCppGenerator,
    APIGenerator,
    RAGPipeline,
    setup_logging
)


def simple_qa_example():
    """Basic Q&A using a single document."""
    print("=== Simple Q&A Example ===\n")
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Create retriever for a single document
    retriever = FileRetriever("data/example_document.txt")
    
    # Create augmenter with Q&A style
    augmenter = StyleAugmenter(style="qa")
    
    # Create generator (choose one based on your setup)
    # Option 1: Local model
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35
    )
    
    # Option 2: API-based model (uncomment to use)
    # generator = APIGenerator(
    #     api_key="your-api-key",
    #     model_name="gpt-3.5-turbo"
    # )
    
    # Create pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Ask questions
    questions = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the implications discussed?"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        result = pipeline.run(query=question)
        print(f"Answer: {result.output}\n")
        print(f"(Generated in {result.execution_time:.2f} seconds)\n")
        print("-" * 50 + "\n")


def multi_document_qa_example():
    """Q&A across multiple documents in a directory."""
    print("=== Multi-Document Q&A Example ===\n")
    
    # Create retriever for directory of documents
    retriever = DirectoryRetriever(
        directory="data/documents",
        pattern="*.txt",
        recursive=True
    )
    
    # Create augmenter that includes document sources
    augmenter = StyleAugmenter(
        style="qa",
        config=AugmenterConfig(include_metadata=True)
    )
    
    # Create generator
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35
    )
    
    # Create pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Ask questions across documents
    question = "What are the common themes across all documents?"
    
    print(f"Question: {question}")
    result = pipeline.run(query=question)
    
    print(f"Answer: {result.output}\n")
    print(f"Documents used: {len(result.retrieved_documents)}")
    for doc in result.retrieved_documents:
        print(f"  - {doc.source}")
    print()


def streaming_qa_example():
    """Q&A with streaming response."""
    print("=== Streaming Q&A Example ===\n")
    
    # Set up components
    retriever = FileRetriever("data/example_document.txt")
    augmenter = StyleAugmenter(style="qa")
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Stream response
    question = "Explain the main concepts in detail"
    print(f"Question: {question}")
    print("Answer: ", end="", flush=True)
    
    for chunk in pipeline.run_stream(query=question):
        if chunk["type"] == "token":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "complete":
            print("\n\nStreaming complete!")


def custom_prompt_qa_example():
    """Q&A with custom prompt template."""
    print("=== Custom Prompt Q&A Example ===\n")
    
    from rag_toolkit import TemplateAugmenter
    
    # Create custom template
    custom_template = """You are an expert analyst. Your task is to provide detailed, analytical answers.

Context Information:
${context}

Question: ${query}

Instructions:
1. Analyze the context carefully
2. Provide a comprehensive answer
3. Include specific examples from the context
4. Conclude with actionable insights

Answer:"""
    
    # Set up components
    retriever = FileRetriever("data/example_document.txt")
    augmenter = TemplateAugmenter(template=custom_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Ask question
    question = "What are the strategic implications?"
    result = pipeline.run(query=question)
    
    print(f"Question: {question}")
    print(f"Answer: {result.output}")


def conversational_qa_example():
    """Multi-turn conversational Q&A."""
    print("=== Conversational Q&A Example ===\n")
    
    from rag_toolkit import ConversationalAugmenter
    
    # Set up components
    retriever = FileRetriever("data/example_document.txt")
    augmenter = ConversationalAugmenter(
        system_message="You are a helpful assistant with access to document information."
    )
    generator = APIGenerator(
        api_key="your-api-key",
        model_name="gpt-3.5-turbo"
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Simulate conversation
    conversation = [
        "What is this document about?",
        "Can you elaborate on the first point?",
        "How does this relate to current trends?"
    ]
    
    for turn, question in enumerate(conversation):
        print(f"User: {question}")
        
        result = pipeline.run(query=question)
        print(f"Assistant: {result.output}\n")
        
        # Add to conversation history
        augmenter.add_turn("user", question)
        augmenter.add_turn("assistant", result.output)


if __name__ == "__main__":
    # Create example data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create example document
    example_doc = data_dir / "example_document.txt"
    if not example_doc.exists():
        example_doc.write_text("""
Title: Introduction to Artificial Intelligence

Artificial Intelligence (AI) represents one of the most transformative technologies 
of our time. This document explores the fundamental concepts, applications, and 
implications of AI in modern society.

Key Concepts:
- Machine Learning: Systems that learn from data
- Neural Networks: Architectures inspired by the human brain
- Natural Language Processing: Understanding and generating human language
- Computer Vision: Interpreting visual information

Applications:
AI is being applied across numerous domains including healthcare, finance, 
transportation, and education. In healthcare, AI assists in diagnosis and 
treatment planning. In finance, it powers fraud detection and algorithmic trading.

Implications:
The widespread adoption of AI raises important questions about employment, 
privacy, bias, and the need for ethical guidelines. As AI systems become more 
sophisticated, ensuring they align with human values becomes increasingly critical.

Future Directions:
Research continues in areas such as explainable AI, federated learning, and 
artificial general intelligence. The goal is to create AI systems that are not 
only powerful but also transparent, fair, and beneficial to humanity.
""")
    
    # Note: You'll need to provide your own model file or API key
    print("Note: Please update the model_path or api_key in the examples before running.\n")
    
    # Run examples (comment out those you don't want to run)
    try:
        # simple_qa_example()
        # multi_document_qa_example()
        # streaming_qa_example()
        # custom_prompt_qa_example()
        # conversational_qa_example()
        
        print("To run examples, uncomment the desired example function calls above.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your model path or API credentials.")