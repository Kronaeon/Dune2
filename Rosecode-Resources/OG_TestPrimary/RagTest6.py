import os
import gc
import multiprocessing
import torch
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import hashlib

# Set CUDA device order and memory management
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Set CUDA_VISIBLE_DEVICES once at the beginning and don't change it
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Make both GPUs visible throughout the script

def load_documents():
    from langchain_community.document_loaders import DirectoryLoader
    
    # Load documents
    loader = DirectoryLoader('./documents/', glob='**/*.txt')
    documents = loader.load()
    
    # Add source filename as metadata
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        # Extract just the filename from the source path
        if 'source' in doc.metadata:
            doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        
    print(f"Loaded {len(documents)} documents")
    for doc in documents:
        print(f"Document: {doc.metadata.get('filename', 'unknown')} - {len(doc.page_content)} chars")
    
    return documents

def split_documents(documents):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # For very small documents, don't split them at all
    MIN_CHUNK_SIZE = 50
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    
    small_docs = []
    large_docs = []
    
    # Separate small documents from large ones
    for doc in documents:
        if len(doc.page_content) <= CHUNK_SIZE:
            small_docs.append(doc)
        else:
            large_docs.append(doc)
    
    # Only split the large documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Split large documents
    split_docs = text_splitter.split_documents(large_docs) if large_docs else []
    
    # Combine with small documents
    all_docs = small_docs + split_docs
    
    print(f"Split into {len(all_docs)} chunks")
    print(f" - {len(small_docs)} documents kept whole (under {CHUNK_SIZE} chars)")
    print(f" - {len(large_docs)} documents split into {len(split_docs)} chunks")
    
    return all_docs

def create_vectorstore(texts):
    from langchain_community.vectorstores import Chroma
    
    # Use the newer import if available
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Instead of changing CUDA_VISIBLE_DEVICES, use device mapping
    # Use device 1 (second GPU) for embeddings
    embedding_device = 1  # This corresponds to the second GPU
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": f"cuda:{embedding_device}"}
    )
    
    # Create vector store
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./chroma_db"
    )
    
    return db, texts  # Return both the vector store and the original texts

def custom_retriever(db, max_docs=5, min_docs=2):
    """
    Create a custom retriever that deduplicates results
    """
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain.retrievers import ContextualCompressionRetriever
    
    class DedupDocumentCompressor(BaseDocumentCompressor):
        """Document compressor that deduplicates documents based on their content."""

        def compress_documents(self, documents, query, callbacks=None):
            unique_docs = {}
            unique_doc_list = []

            for doc in documents:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in unique_docs:
                    unique_docs[content_hash] = doc
                    unique_doc_list.append(doc)

            print(f"Deduplicated from {len(documents)} to {len(unique_doc_list)} documents")
            return unique_doc_list

        async def acompress_documents(self, documents, query, callbacks=None):
            return self.compress_documents(documents, query, callbacks)
    
    # Create the deduplication compressor
    dedup_compressor = DedupDocumentCompressor()
    
    # Determine appropriate k value based on collection size
    try:
        collection_size = db._collection.count()
        k = min(max_docs, max(min_docs, collection_size // 2))
    except:
        # Fallback if collection size cannot be determined
        k = max_docs
        print(f"Could not determine collection size, using k={k}")
    
    # Use the dedup compressor directly in a ContextualCompressionRetriever
    retriever = ContextualCompressionRetriever(
        base_retriever=db.as_retriever(search_kwargs={"k": k}),
        base_compressor=dedup_compressor
    )
    
    return retriever

def setup_hf_qa_chain(db):
    from langchain.chains import RetrievalQA
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
    import time
    import re
    
    # Define custom stopping criteria to prevent repetition
    class RepetitionStoppingCriteria(StoppingCriteria):
        def __init__(self, tokenizer, min_length=50, window_size=20, threshold=0.8):
            self.tokenizer = tokenizer
            self.min_length = min_length
            self.window_size = window_size
            self.threshold = threshold
            self.start_length = 0
            
        def __call__(self, input_ids, scores, **kwargs):
            # Don't stop if below minimum length
            if len(input_ids[0]) < self.min_length:
                return False
                
            # Get the generated text
            generated_text = self.tokenizer.decode(input_ids[0])
            
            # Check for repetition in the last part of the text
            if len(generated_text) > 2 * self.window_size:
                # Look at the last portion of generated text
                text_to_check = generated_text[-2 * self.window_size:]
                
                # Check for repeating patterns
                for pattern_length in range(5, self.window_size):
                    pattern = text_to_check[-pattern_length:]
                    # Look for this pattern in the preceding text
                    previous_text = text_to_check[:-pattern_length]
                    if pattern in previous_text:
                        # Found repeating pattern
                        return True
                        
                # Check for structural repetition (lists, sections, etc.)
                if re.search(r'(### End.*\n.*### End|Note:.*\n.*Note:|Please.*\n.*Please)', text_to_check):
                    return True
            
            return False
    
    # Disable FlashAttention to ensure compatibility with GPU 1
    torch.backends.cuda.enable_flash_sdp(False)
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model path
    model_path = ""
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create a custom device map that better utilizes the 16GB GPU
    # Put more layers on GPU 0 (the RTX 4060 Ti with 16GB)
    custom_device_map = {
        "model.embed_tokens": 0,
        "lm_head": 0,  # This is critical for performance - keep on GPU
        "model.norm": 0,
        "model.rotary_emb": 0
    }
    
    # Assign more layers to GPU 0 (the larger GPU)
    for i in range(26):
        if i < 22:  # Put first 22 layers on the primary GPU
            custom_device_map[f"model.layers.{i}"] = 0
        else:  # Put remaining layers on the secondary GPU
            custom_device_map[f"model.layers.{i}"] = 1
    
    # Any remaining layers go on secondary GPU if possible
    for i in range(26, 32):
        custom_device_map[f"model.layers.{i}"] = 1
    
    print("\nLoading model with custom device map optimized for your GPUs...")
    print("This will place more of the model on your 16GB RTX 4060 Ti")
    
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=custom_device_map,  # Use our custom mapping
        torch_dtype=torch.float16,     # Use half precision
    )
    end_time = time.time()
    
    # Verify where model parts are placed
    print(f"\nModel loaded in {end_time - start_time:.2f} seconds.")
    print("\n=== MODEL DEVICE ALLOCATION ===")
    
    # Get model's device map
    if hasattr(model, 'hf_device_map'):
        print("Model device map:")
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")
        
        # Check for CPU allocation
        if any(device == 'cpu' for device in model.hf_device_map.values()):
            print("\nWARNING: Some parts of the model are still on CPU!")
            proceed = input("Do you want to continue? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting...")
                exit()
    else:
        print("Could not retrieve detailed device map.")
        print(f"Model is on: {next(model.parameters()).device}")
    
    # Memory usage after model loading
    print("\n=== GPU MEMORY USAGE AFTER MODEL LOADING ===")
    for i in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        except RuntimeError as e:
            print(f"Error getting memory info for GPU {i}: {e}")
    
    # Create a custom stopping criteria instance
    stopping_criteria = RepetitionStoppingCriteria(tokenizer)
    
    # Create HuggingFace pipeline with better generation parameters
    print("\nCreating text generation pipeline...")
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Reduced from 1024 to 512
        temperature=0.7,     # Increased from 0.1 to 0.7 for more variety
        top_p=0.9,           # Add top_p sampling
        top_k=50,            # Add top_k sampling
        do_sample=True,      # Enable sampling to reduce repetition
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  # Explicitly set EOS token
        stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        return_full_text=False
    )
    
    # Create LangChain HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Create a custom retriever with deduplication
    retriever = custom_retriever(db)
    
    # Define an improved template for better responses (with format that encourages conciseness)
    template = """<|system|>
You are a helpful AI assistant that provides clear, concise answers based only on the given context.
</|system|>

<|user|>
I need an answer to the following question using only the provided context information:

Context information:
--------------------
{context}
--------------------

Question: {question}
</|user|>

<|assistant|>
Based on the provided context, I'll answer your question concisely:

"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    print("\nCreating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("QA chain created successfully.")
    return qa_chain, llm  # Return both the chain and the LLM for direct use

def debug_retrieval(db, question):
    """Debug function to see which documents are being retrieved for a question"""
    print(f"\n=== DEBUG: Documents retrieved for question: '{question}' ===")
    
    # Create a custom retriever with deduplication
    retriever = custom_retriever(db)
    
    # Use invoke method for newer LangChain versions, fallback to get_relevant_documents
    try:
        # Try newer method first
        docs = retriever.invoke(question)
    except (AttributeError, TypeError):
        # Fallback to older method
        docs = retriever.get_relevant_documents(question)
    
    for i, doc in enumerate(docs):
        print(f"Document {i+1}:")
        print(f"  Source: {doc.metadata.get('filename', 'unknown')}")
        print(f"  Content: {doc.page_content}")
    
    return docs

def query_documents(qa_chain, question):
    print(f"\nQuestion: {question}")
    
    try:
        # Try newer method first
        response = qa_chain.invoke({"query": question})
        if isinstance(response, dict) and "result" in response:
            return response["result"].strip()
        return response.strip()
    except (AttributeError, TypeError) as e:
        # Fallback to older method
        print(f"Falling back to run() method: {str(e)}")
        return qa_chain.run(question).strip()

def query_with_direct_context(llm, context, question):
    """
    Query the LLM directly with a specific context, bypassing retrieval
    """
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    
    # Use the same template as before but directly populate the context
    template = """<|system|>
You are a helpful AI assistant that provides clear, concise answers based only on the given context.
</|system|>

<|user|>
I need an answer to the following question using only the provided context information:

Context information:
--------------------
{context}
--------------------

Question: {question}
</|user|>

<|assistant|>
Based on the provided context, I'll answer your question concisely:

"""
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create a simple LLMChain
    chain = LLMChain(llm=llm, prompt=PROMPT)
    
    # Run the chain
    result = chain.run(context=context, question=question)
    
    return result.strip()

def get_all_unique_documents(original_texts):
    """
    Helper function to extract all unique documents from the original texts
    Instead of relying on similarity search, we'll directly use the original texts
    """
    # First deduplicate by content
    unique_docs = {}
    for doc in original_texts:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        if content_hash not in unique_docs:
            unique_docs[content_hash] = doc
    
    # Convert back to list
    unique_doc_list = list(unique_docs.values())
    
    return unique_doc_list

def main():
    # Process documents and create vector store
    documents = load_documents()
    texts = split_documents(documents)
    db, original_texts = create_vectorstore(texts)  # Keep original texts
    
    # Important: manually clear the cache before creating the HF model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Debug retrieval first
    query = "What is the main topic of these documents?"
    debug_retrieval(db, query)
    
    # Create QA chain with HuggingFace
    qa_chain, llm = setup_hf_qa_chain(db)
    
    # Example query
    print("\n\n=== QUERYING DOCUMENTS ===\n\n")
    answer = query_documents(qa_chain, query)
    print("\nAnswer:")
    print(answer)
    
    # Try another query
    query2 = "Compare Python and JavaScript based on the documents."
    debug_retrieval(db, query2)
    answer2 = query_documents(qa_chain, query2)
    print("\nAnswer:")
    print(answer2)
    
    # Try to get information about all documents - THIS IS THE FIXED PART
    query3 = "What topics are covered across all the documents?"
    
    # Get truly unique documents directly from our original texts
    unique_docs = get_all_unique_documents(original_texts)
    print(f"\nRetrieved {len(unique_docs)} unique documents directly from corpus")
    
    # Build context string with document sources for better traceability
    context_parts = []
    for i, doc in enumerate(unique_docs):
        source = doc.metadata.get('filename', f'document_{i+1}')
        context_parts.append(f"Document '{source}':\n{doc.page_content}")
    
    # Join with clear document separators
    context = "\n\n---\n\n".join(context_parts)
    print(f"\n\n=== ALL DOCUMENTS CONTEXT ===\n{context}\n")
    
    # Query using direct context rather than retrieval
    answer3 = query_with_direct_context(llm, context, query3)
    print("\nAnswer:")
    print(answer3)

if __name__ == "__main__":
    # Required for multiprocessing when using spawn method
    multiprocessing.freeze_support()
    main()