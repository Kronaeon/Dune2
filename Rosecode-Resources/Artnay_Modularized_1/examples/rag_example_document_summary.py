"""Document Summarization Example using RAG Toolkit.

This example demonstrates various summarization strategies including
single document, multi-document, and hierarchical summarization.
"""

import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from rag_toolkit import (
    FileRetriever,
    DirectoryRetriever,
    ChunkedRetriever,
    ConsolidatedRetriever,
    StyleAugmenter,
    TemplateAugmenter,
    LlamaCppGenerator,
    APIGenerator,
    RAGPipeline,
    StructuredParser,
    RegexParser,
    setup_logging,
    ProgressTracker
)


def single_document_summary():
    """Summarize a single document."""
    print("=== Single Document Summary ===\n")
    
    # Set up components
    retriever = FileRetriever("data/long_document.txt")
    augmenter = StyleAugmenter(style="summary")
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=500
    )
    
    # Create pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    # Generate summary
    result = pipeline.run()
    
    print("Summary:")
    print(result.output)
    print(f"\nOriginal length: {len(result.retrieved_documents[0].content)} chars")
    print(f"Summary length: {len(result.output)} chars")
    print(f"Compression ratio: {len(result.output)/len(result.retrieved_documents[0].content):.2%}")


def chunked_summary():
    """Summarize a long document by processing it in chunks."""
    print("=== Chunked Document Summary ===\n")
    
    # Create chunked retriever
    base_retriever = FileRetriever("data/very_long_document.txt")
    chunked_retriever = ChunkedRetriever(
        base_retriever=base_retriever,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Template for chunk summaries
    chunk_template = """Summarize this section of a larger document:

${context}

Provide a concise summary focusing on the main points:"""
    
    augmenter = TemplateAugmenter(template=chunk_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=200
    )
    
    # Process chunks
    chunks = chunked_retriever.retrieve()
    chunk_summaries = []
    
    print(f"Processing {len(chunks)} chunks...")
    
    with ProgressTracker(total=len(chunks), description="Summarizing chunks") as tracker:
        for chunk in chunks:
            # Create mini-pipeline for each chunk
            result = generator.generate(
                augmenter.augment([chunk])
            )
            chunk_summaries.append(result.text)
            tracker.update()
    
    # Consolidate chunk summaries
    print("\nConsolidating chunk summaries...")
    
    consolidation_template = """Based on these section summaries, create a comprehensive summary of the entire document:

${context}

Final Summary:"""
    
    # Create document from summaries
    from rag_toolkit import Document
    summary_doc = Document(
        content="\n\n".join(chunk_summaries),
        metadata={"type": "chunk_summaries"},
        source="consolidated_chunks"
    )
    
    final_augmenter = TemplateAugmenter(template=consolidation_template)
    final_result = generator.generate(
        final_augmenter.augment([summary_doc])
    )
    
    print("\nFinal Summary:")
    print(final_result.text)


def multi_document_summary():
    """Summarize multiple related documents."""
    print("=== Multi-Document Summary ===\n")
    
    # Retrieve multiple documents
    retriever = DirectoryRetriever(
        directory="data/research_papers",
        pattern="*.txt"
    )
    
    # Custom template for multi-doc summary
    multi_doc_template = """You are tasked with creating a comprehensive summary across multiple documents.

Documents provided:
${context}

Create a unified summary that:
1. Identifies common themes across documents
2. Highlights unique contributions from each document
3. Synthesizes the information into a coherent narrative
4. Notes any conflicting information

Comprehensive Summary:"""
    
    augmenter = TemplateAugmenter(
        template=multi_doc_template,
        config=AugmenterConfig(
            include_metadata=True,
            max_context_length=4000
        )
    )
    
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=800
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    result = pipeline.run()
    
    print("Multi-Document Summary:")
    print(result.output)
    print(f"\nDocuments summarized: {len(result.retrieved_documents)}")
    for doc in result.retrieved_documents:
        print(f"  - {Path(doc.source).name}")


def structured_summary():
    """Generate a structured summary with specific sections."""
    print("=== Structured Summary ===\n")
    
    # Template for structured output
    structured_template = """Create a structured summary of the following content:

${context}

Format your response EXACTLY as follows:

# MAIN TOPIC: [One sentence describing the main topic]

## KEY POINTS
- [First key point]
- [Second key point]
- [Third key point]

## DETAILED SUMMARY
[2-3 paragraphs providing a comprehensive summary]

## IMPLICATIONS
[1-2 paragraphs discussing implications or significance]

## CONCLUSION
[Brief concluding statement]"""
    
    # Set up pipeline
    retriever = FileRetriever("data/technical_document.txt")
    augmenter = TemplateAugmenter(template=structured_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=1000
    )
    
    # Create parser for structured output
    parser = StructuredParser(
        section_pattern=r"^#+\s*(.+?):\s*(.*)$",
        list_pattern=r"^-\s+(.+)$"
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator,
        parser=parser
    )
    
    result = pipeline.run()
    
    print("Structured Summary:")
    if result.parsed_output and result.parsed_output.is_valid:
        # Display parsed sections
        for section, content in result.parsed_output.data.items():
            print(f"\n{section.upper()}:")
            if isinstance(content, list):
                for item in content:
                    print(f"  - {item}")
            else:
                print(f"  {content}")
    else:
        # Fallback to raw output
        print(result.output)


def comparative_summary():
    """Generate a comparative summary of multiple documents."""
    print("=== Comparative Summary ===\n")
    
    # Retrieve documents to compare
    doc1_retriever = FileRetriever("data/approach_1.txt")
    doc2_retriever = FileRetriever("data/approach_2.txt")
    
    # Consolidate retrievers
    retriever = ConsolidatedRetriever(
        retrievers=[doc1_retriever, doc2_retriever],
        consolidation_strategy="keep_separate"  # Keep documents separate
    )
    
    # Comparative template
    comparison_template = """Compare and contrast the following documents:

${context}

Create a comparative analysis that includes:

1. SIMILARITIES
   - What common themes or approaches do both documents share?
   - What conclusions do they both reach?

2. DIFFERENCES
   - How do their approaches differ?
   - What unique points does each document make?

3. STRENGTHS & WEAKNESSES
   - What are the strengths of each approach?
   - What are the limitations or weaknesses?

4. SYNTHESIS
   - Which approach is more compelling and why?
   - How might they complement each other?

Comparative Analysis:"""
    
    augmenter = TemplateAugmenter(
        template=comparison_template,
        config=AugmenterConfig(include_metadata=True)
    )
    
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=1200
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator
    )
    
    result = pipeline.run()
    
    print("Comparative Summary:")
    print(result.output)


def executive_summary():
    """Generate an executive summary with key metrics."""
    print("=== Executive Summary ===\n")
    
    # Executive summary template
    exec_template = """You are creating an executive summary for senior leadership.

Document:
${context}

Create an EXECUTIVE SUMMARY that includes:

OVERVIEW: [1-2 sentences capturing the essence]

KEY FINDINGS:
• [Most important finding]
• [Second important finding]
• [Third important finding]

METRICS & DATA:
• [Key metric 1]
• [Key metric 2]
• [Key metric 3]

RECOMMENDATIONS:
1. [Primary recommendation]
2. [Secondary recommendation]

NEXT STEPS: [Brief action items]

Executive Summary:"""
    
    # Create pipeline with parser for metrics
    retriever = FileRetriever("data/quarterly_report.txt")
    augmenter = TemplateAugmenter(template=exec_template)
    generator = LlamaCppGenerator(
        model_path="/path/to/your/model.gguf",
        n_gpu_layers=35,
        max_tokens=600
    )
    
    # Parser to extract metrics
    metric_parser = RegexParser(
        patterns={
            "metrics": r"(?:•\s*)([\d.]+%?[^\n]+)",
            "recommendations": r"(?:\d+\.\s*)([^\n]+)",
            "overview": r"OVERVIEW:\s*([^\n]+)"
        }
    )
    
    pipeline = RAGPipeline(
        retriever=retriever,
        augmenter=augmenter,
        generator=generator,
        parser=metric_parser
    )
    
    result = pipeline.run()
    
    print("Executive Summary Generated:")
    print(result.output)
    
    if result.parsed_output and result.parsed_output.is_valid:
        print("\nExtracted Key Information:")
        for key, value in result.parsed_output.data.items():
            print(f"{key.title()}: {value}")


if __name__ == "__main__":
    # Set up logging
    setup_logging(level="INFO")
    
    # Create example data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create example long document
    long_doc = data_dir / "long_document.txt"
    if not long_doc.exists():
        long_doc.write_text("""
The Evolution of Machine Learning: A Comprehensive Overview

Machine learning has undergone remarkable transformation since its inception in the 1950s. 
This document traces the journey from early perceptrons to modern deep learning systems,
examining key breakthroughs, challenges, and future directions.

Historical Context:
The field began with Arthur Samuel's checkers program in 1959, which coined the term 
"machine learning." The 1960s saw the development of the perceptron by Frank Rosenblatt,
followed by a period known as the "AI winter" when limitations became apparent.

The resurgence came in the 1980s with backpropagation and neural networks. The 1990s
brought support vector machines and ensemble methods. The 2000s saw the rise of 
statistical learning theory and kernel methods.

Modern Era:
The 2010s marked the deep learning revolution, powered by:
- Increased computational power (GPUs)
- Large datasets (ImageNet, Common Crawl)
- Algorithmic innovations (dropout, batch normalization)
- New architectures (CNNs, RNNs, Transformers)

Current State:
Today, machine learning powers numerous applications:
- Natural language processing (GPT, BERT)
- Computer vision (object detection, segmentation)
- Reinforcement learning (game playing, robotics)
- Generative models (GANs, diffusion models)

Challenges remain in areas like interpretability, fairness, robustness, and 
computational efficiency. Research continues on federated learning, few-shot
learning, and neural architecture search.

Future Directions:
The field is moving toward more efficient models, better interpretability,
and stronger theoretical foundations. Quantum machine learning and neuromorphic
computing represent frontier areas. The ultimate goal remains artificial general
intelligence, though the path remains uncertain.

This evolution demonstrates both the power of persistence in research and the
importance of interdisciplinary collaboration in advancing the field.
""")
    
    print("Note: Please update model_path in the examples before running.\n")
    
    # Run examples (uncomment to run)
    try:
        # single_document_summary()
        # chunked_summary()
        # multi_document_summary()
        # structured_summary()
        # comparative_summary()
        # executive_summary()
        
        print("To run examples, uncomment the desired example function calls above.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set up your model path or API credentials.")