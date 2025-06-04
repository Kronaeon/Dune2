"""
RAG System Library - A flexible and configurable Retrieval-Augmented Generation system

This module provides a RAGSystem class that encapsulates document loading, processing,
vector store creation, and querying functionality with support for multiple model types
and embedding systems.

Author: Generated from RagTest6.py
License: MIT
"""

import os
import gc
import multiprocessing
import torch
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

class RAGSystem:
    """
    A flexible RAG (Retrieval-Augmented Generation) system that supports multiple
    model backends, embedding systems, and vector stores.
    
    This class encapsulates the entire RAG pipeline from document loading to querying,
    with configurable components for maximum flexibility and reusability.
    """
    
    def __init__(self, 
                 cuda_device_order: str = "PCI_BUS_ID",
                 pytorch_cuda_alloc_conf: str = "expandable_segments:True",
                 cuda_visible_devices: str = "0,1",
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize the RAG System with configuration parameters.
        
        Args:
            cuda_device_order: CUDA device ordering strategy
            pytorch_cuda_alloc_conf: PyTorch CUDA allocation configuration
            cuda_visible_devices: Visible CUDA devices (comma-separated)
            verbose: Enable verbose logging and debugging output
            **kwargs: Additional configuration parameters
        """
        # Set CUDA environment variables
        os.environ["CUDA_DEVICE_ORDER"] = cuda_device_order
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        self.verbose = verbose
        self.config = kwargs
        
        # Initialize storage for components
        self.documents = None
        self.split_texts = None
        self.vectorstore = None
        self.original_texts = None
        self.qa_chain = None
        self.llm = None
        self.retriever = None
        
        if self.verbose:
            print(f"RAGSystem initialized with CUDA devices: {cuda_visible_devices}")
    
    def loadDocuments(self, 
                     paths: Union[str, List[str]], 
                     glob_pattern: str = '**/*.txt',
                     file_extensions: Optional[List[str]] = None) -> List:
        """
        Load documents from specified files or directories.
        
        Args:
            paths: Single path string or list of paths to files/directories
            glob_pattern: Glob pattern for directory loading (e.g., '**/*.txt')
            file_extensions: List of file extensions to load (overrides glob_pattern)
            
        Returns:
            List of loaded documents with metadata
            
        Raises:
            FileNotFoundError: If specified paths do not exist
            ValueError: If no documents are found
        """
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
        
        # Ensure paths is a list
        if isinstance(paths, str):
            paths = [paths]
        
        all_documents = []
        
        for path in paths:
            path_obj = Path(path)
            
            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")
            
            if path_obj.is_file():
                # Load single file
                loader = TextLoader(str(path_obj))
                documents = loader.load()
            else:
                # Load from directory
                if file_extensions:
                    # Use specific extensions
                    for ext in file_extensions:
                        pattern = f'**/*.{ext.lstrip(".")}'
                        loader = DirectoryLoader(str(path_obj), glob=pattern)
                        documents = loader.load()
                        all_documents.extend(documents)
                    continue
                else:
                    # Use glob pattern
                    loader = DirectoryLoader(str(path_obj), glob=glob_pattern)
                    documents = loader.load()
            
            all_documents.extend(documents)
        
        if not all_documents:
            raise ValueError(f"No documents found in specified paths: {paths}")
        
        # Add source filename as metadata
        for doc in all_documents:
            if not doc.metadata:
                doc.metadata = {}
            # Extract just the filename from the source path
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        
        self.documents = all_documents
        
        if self.verbose:
            print(f"Loaded {len(all_documents)} documents")
            for doc in all_documents:
                filename = doc.metadata.get('filename', 'unknown')
                print(f"Document: {filename} - {len(doc.page_content)} chars")
        
        return all_documents
    
    def splitDocuments(self, 
                      documents: Optional[List] = None,
                      chunk_size: int = 800,
                      chunk_overlap: int = 100,
                      min_chunk_size: int = 50,
                      text_splitter_type: str = 'recursive',
                      **splitter_kwargs) -> List:
        """
        Split documents into chunks based on configurable parameters.
        
        Args:
            documents: List of documents to split (uses self.documents if None)
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum size to consider splitting a document
            text_splitter_type: Type of text splitter ('recursive', 'character', etc.)
            **splitter_kwargs: Additional arguments for the text splitter
            
        Returns:
            List of split document chunks
            
        Raises:
            ValueError: If no documents are provided or available
        """
        if documents is None:
            if self.documents is None:
                raise ValueError("No documents provided. Call loadDocuments() first.")
            documents = self.documents
        
        # Import appropriate text splitter
        if text_splitter_type.lower() == 'recursive':
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter_class = RecursiveCharacterTextSplitter
        elif text_splitter_type.lower() == 'character':
            from langchain.text_splitter import CharacterTextSplitter
            text_splitter_class = CharacterTextSplitter
        else:
            # Default to recursive
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter_class = RecursiveCharacterTextSplitter
        
        # Separate small documents from large ones
        small_docs = []
        large_docs = []
        
        for doc in documents:
            if len(doc.page_content) <= chunk_size:
                small_docs.append(doc)
            else:
                large_docs.append(doc)
        
        # Create text splitter with provided parameters
        splitter_params = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            **splitter_kwargs
        }
        
        text_splitter = text_splitter_class(**splitter_params)
        
        # Split large documents
        split_docs = text_splitter.split_documents(large_docs) if large_docs else []
        
        # Combine with small documents
        all_docs = small_docs + split_docs
        
        self.split_texts = all_docs
        
        if self.verbose:
            print(f"Split into {len(all_docs)} chunks")
            print(f" - {len(small_docs)} documents kept whole (under {chunk_size} chars)")
            print(f" - {len(large_docs)} documents split into {len(split_docs)} chunks")
        
        return all_docs
    
    def createVectorstore(self, 
                         texts: Optional[List] = None,
                         embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         embedding_kwargs: Optional[Dict] = None,
                         vectorstore_type: str = 'chroma',
                         vectorstore_kwargs: Optional[Dict] = None) -> tuple:
        """
        Create a vector store using specified embedding model and vector store type.
        
        Args:
            texts: List of texts to vectorize (uses self.split_texts if None)
            embedding_model_name: Name or path of the embedding model
            embedding_kwargs: Additional kwargs for embedding model (e.g., device)
            vectorstore_type: Type of vector store ('chroma', 'faiss', etc.)
            vectorstore_kwargs: Additional kwargs for vector store initialization
            
        Returns:
            Tuple of (vectorstore, original_texts)
            
        Raises:
            ValueError: If no texts are provided or available
            ImportError: If required vector store library is not installed
        """
        if texts is None:
            if self.split_texts is None:
                raise ValueError("No texts provided. Call splitDocuments() first.")
            texts = self.split_texts
        
        # Set default embedding kwargs
        if embedding_kwargs is None:
            embedding_kwargs = {}
        
        # Set default device if not specified
        if 'model_kwargs' not in embedding_kwargs:
            embedding_kwargs['model_kwargs'] = {}
        if 'device' not in embedding_kwargs['model_kwargs']:
            # Use first available CUDA device or CPU
            if torch.cuda.is_available():
                embedding_kwargs['model_kwargs']['device'] = 'cuda:0'
            else:
                embedding_kwargs['model_kwargs']['device'] = 'cpu'
        
        # Create embeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            **embedding_kwargs
        )
        
        # Set default vectorstore kwargs
        if vectorstore_kwargs is None:
            vectorstore_kwargs = {}
        
        # Create vector store based on type
        if vectorstore_type.lower() == 'chroma':
            from langchain_community.vectorstores import Chroma
            
            # Set default persist directory if not specified
            if 'persist_directory' not in vectorstore_kwargs:
                vectorstore_kwargs['persist_directory'] = "./chroma_db"
            
            vectorstore = Chroma.from_documents(
                texts,
                embeddings,
                **vectorstore_kwargs
            )
            
        elif vectorstore_type.lower() == 'faiss':
            try:
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.from_documents(texts, embeddings, **vectorstore_kwargs)
            except ImportError:
                raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
                
        else:
            raise ValueError(f"Unsupported vectorstore type: {vectorstore_type}")
        
        self.vectorstore = vectorstore
        self.original_texts = texts
        
        if self.verbose:
            print(f"Created {vectorstore_type} vector store with {len(texts)} documents")
        
        return vectorstore, texts
    
    def _createCustomRetriever(self, 
                              vectorstore, 
                              max_docs: int = 5, 
                              min_docs: int = 2):
        """
        Create a custom retriever with deduplication capabilities.
        
        Args:
            vectorstore: The vector store to create retriever from
            max_docs: Maximum number of documents to retrieve
            min_docs: Minimum number of documents to retrieve
            
        Returns:
            ContextualCompressionRetriever with deduplication
        """
        from langchain_core.documents.compressor import BaseDocumentCompressor
        from langchain.retrievers import ContextualCompressionRetriever
        
        class DedupDocumentCompressor(BaseDocumentCompressor):
            """Document compressor that deduplicates documents based on content hash."""

            def compress_documents(self, documents, query, callbacks=None):
                unique_docs = {}
                unique_doc_list = []

                for doc in documents:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in unique_docs:
                        unique_docs[content_hash] = doc
                        unique_doc_list.append(doc)

                if self.verbose:
                    print(f"Deduplicated from {len(documents)} to {len(unique_doc_list)} documents")
                return unique_doc_list

            async def acompress_documents(self, documents, query, callbacks=None):
                return self.compress_documents(documents, query, callbacks)
        
        # Create the deduplication compressor
        dedup_compressor = DedupDocumentCompressor()
        
        # Determine appropriate k value based on collection size
        try:
            collection_size = vectorstore._collection.count()
            k = min(max_docs, max(min_docs, collection_size // 2))
        except:
            # Fallback if collection size cannot be determined
            k = max_docs
            if self.verbose:
                print(f"Could not determine collection size, using k={k}")
        
        # Create retriever with deduplication
        retriever = ContextualCompressionRetriever(
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            base_compressor=dedup_compressor
        )
        
        return retriever
    
    def setupQAChain(self, 
                    vectorstore=None,
                    model_type: str = "huggingface",
                    model_path: str = None,
                    model_kwargs: Optional[Dict] = None,
                    generation_kwargs: Optional[Dict] = None,
                    prompt_template: Optional[str] = None,
                    retriever_kwargs: Optional[Dict] = None):
        """
        Set up the QA chain with specified model type and parameters.
        
        Args:
            vectorstore: Vector store to use (uses self.vectorstore if None)
            model_type: Type of model backend ('huggingface', 'vllm', 'llamacpp')
            model_path: Path to the model files
            model_kwargs: Model-specific loading parameters
            generation_kwargs: Text generation parameters
            prompt_template: Custom prompt template string
            retriever_kwargs: Parameters for the document retriever
            
        Returns:
            Tuple of (qa_chain, llm) for the created QA chain and language model
            
        Raises:
            ValueError: If vectorstore is not available or model_type is unsupported
            ImportError: If required model libraries are not installed
        """
        if vectorstore is None:
            if self.vectorstore is None:
                raise ValueError("No vectorstore provided. Call createVectorstore() first.")
            vectorstore = self.vectorstore
        
        # Set default parameters
        if model_kwargs is None:
            model_kwargs = {}
        if generation_kwargs is None:
            generation_kwargs = {}
        if retriever_kwargs is None:
            retriever_kwargs = {}
        
        # Clear GPU memory before model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Setup model based on type
        if model_type.lower() == "huggingface":
            llm = self._setupHuggingFaceModel(model_path, model_kwargs, generation_kwargs)
        elif model_type.lower() == "vllm":
            llm = self._setupVLLMModel(model_path, model_kwargs, generation_kwargs)
        elif model_type.lower() == "llamacpp":
            llm = self._setupLlamaCppModel(model_path, model_kwargs, generation_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: 'huggingface', 'vllm', 'llamacpp'")
        
        # Create custom retriever
        retriever_params = {'max_docs': 5, 'min_docs': 2, **retriever_kwargs}
        retriever = self._createCustomRetriever(vectorstore, **retriever_params)
        
        # Set up prompt template
        if prompt_template is None:
            prompt_template = """<|system|>
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
        
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        self.qa_chain = qa_chain
        self.llm = llm
        self.retriever = retriever
        
        if self.verbose:
            print(f"QA chain created successfully with {model_type} model")
        
        return qa_chain, llm
    
    def _setupHuggingFaceModel(self, model_path: str, model_kwargs: Dict, generation_kwargs: Dict):
        """Setup Hugging Face model with pipeline."""
        if model_path is None:
            raise ValueError("model_path is required for Hugging Face models")
        
        try:
            from langchain_huggingface import HuggingFacePipeline
        except ImportError:
            from langchain_community.llms import HuggingFacePipeline
            
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import time
        
        # Load tokenizer
        if self.verbose:
            print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set default model loading parameters
        default_model_kwargs = {
            'torch_dtype': torch.float16,
            'device_map': 'auto'
        }
        default_model_kwargs.update(model_kwargs)
        
        # Load model
        if self.verbose:
            print("Loading model...")
            start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(model_path, **default_model_kwargs)
        
        if self.verbose:
            end_time = time.time()
            print(f"Model loaded in {end_time - start_time:.2f} seconds")
            
            # Display memory usage
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                    except RuntimeError as e:
                        print(f"Error getting memory info for GPU {i}: {e}")
        
        # Set default generation parameters
        default_generation_kwargs = {
            'max_new_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
            'return_full_text': False
        }
        default_generation_kwargs.update(generation_kwargs)
        
        # Create pipeline
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **default_generation_kwargs
        )
        
        return HuggingFacePipeline(pipeline=hf_pipeline)
    
    def _setupVLLMModel(self, model_path: str, model_kwargs: Dict, generation_kwargs: Dict):
        """Setup VLLM model (placeholder for future implementation)."""
        try:
            from langchain_community.llms import VLLM
            
            # Set default VLLM parameters
            default_vllm_kwargs = {
                'model': model_path,
                'trust_remote_code': True,
                'max_new_tokens': 512,
                'top_k': 10,
                'top_p': 0.95,
                'temperature': 0.8,
            }
            default_vllm_kwargs.update(model_kwargs)
            default_vllm_kwargs.update(generation_kwargs)
            
            return VLLM(**default_vllm_kwargs)
            
        except ImportError:
            raise ImportError("VLLM not installed. Install with: pip install vllm")
    
    def _setupLlamaCppModel(self, model_path: str, model_kwargs: Dict, generation_kwargs: Dict):
        """Setup Llama.cpp model (placeholder for future implementation)."""
        try:
            from langchain_community.llms import LlamaCpp
            
            # Set default Llama.cpp parameters
            default_llamacpp_kwargs = {
                'model_path': model_path,
                'max_tokens': 512,
                'temperature': 0.7,
                'top_p': 0.9,
                'n_ctx': 2048,
                'verbose': self.verbose,
            }
            default_llamacpp_kwargs.update(model_kwargs)
            default_llamacpp_kwargs.update(generation_kwargs)
            
            return LlamaCpp(**default_llamacpp_kwargs)
            
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    def query(self, question: str, debug: bool = False) -> str:
        """
        Query the documents using the QA chain.
        
        Args:
            question: The question to ask
            debug: Enable debug output showing retrieved documents
            
        Returns:
            String response from the QA system
            
        Raises:
            ValueError: If QA chain is not set up
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not set up. Call setupQAChain() first.")
        
        if debug or self.verbose:
            self.debugRetrieval(question)
        
        if self.verbose:
            print(f"\nQuestion: {question}")
        
        try:
            # Try newer method first
            response = self.qa_chain.invoke({"query": question})
            if isinstance(response, dict) and "result" in response:
                result = response["result"].strip()
            else:
                result = str(response).strip()
        except (AttributeError, TypeError) as e:
            # Fallback to older method
            if self.verbose:
                print(f"Falling back to run() method: {str(e)}")
            result = self.qa_chain.run(question).strip()
        
        return result
    
    def queryWithDirectContext(self, context: str, question: str) -> str:
        """
        Query the LLM directly with specific context, bypassing retrieval.
        
        Args:
            context: The context text to use for answering
            question: The question to ask
            
        Returns:
            String response from the language model
            
        Raises:
            ValueError: If LLM is not set up
        """
        if self.llm is None:
            raise ValueError("LLM not set up. Call setupQAChain() first.")
        
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
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
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create a simple LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run the chain
        result = chain.run(context=context, question=question)
        
        return result.strip()
    
    def debugRetrieval(self, question: str) -> List:
        """
        Debug function to see which documents are retrieved for a question.
        
        Args:
            question: The question to debug retrieval for
            
        Returns:
            List of retrieved documents
            
        Raises:
            ValueError: If retriever is not set up
        """
        if self.retriever is None:
            raise ValueError("Retriever not set up. Call setupQAChain() first.")
        
        print(f"\n=== DEBUG: Documents retrieved for question: '{question}' ===")
        
        try:
            # Try newer method first
            docs = self.retriever.invoke(question)
        except (AttributeError, TypeError):
            # Fallback to older method
            docs = self.retriever.get_relevant_documents(question)
        
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(f"  Source: {doc.metadata.get('filename', 'unknown')}")
            print(f"  Content: {doc.page_content}")
        
        return docs
    
    def getAllUniqueDocuments(self) -> List:
        """
        Get all unique documents from the original texts based on content hash.
        
        Returns:
            List of unique documents
            
        Raises:
            ValueError: If original texts are not available
        """
        if self.original_texts is None:
            raise ValueError("Original texts not available. Call createVectorstore() first.")
        
        # Deduplicate by content hash
        unique_docs = {}
        for doc in self.original_texts:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in unique_docs:
                unique_docs[content_hash] = doc
        
        return list(unique_docs.values())
    
    def getSystemInfo(self) -> Dict[str, Any]:
        """
        Get information about the current system state and configuration.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'documents_loaded': len(self.documents) if self.documents else 0,
            'split_texts_count': len(self.split_texts) if self.split_texts else 0,
            'vectorstore_created': self.vectorstore is not None,
            'qa_chain_ready': self.qa_chain is not None,
            'verbose_mode': self.verbose
        }
        
        if torch.cuda.is_available():
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_info[f'gpu_{i}'] = {
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated_gb': torch.cuda.memory_allocated(i) / 1024**3,
                        'memory_reserved_gb': torch.cuda.memory_reserved(i) / 1024**3
                    }
                except:
                    gpu_info[f'gpu_{i}'] = {'error': 'Could not retrieve info'}
            info['gpu_info'] = gpu_info
        
        return info


# Test bench and example usage
if __name__ == "__main__":
    # Required for multiprocessing when using spawn method
    multiprocessing.freeze_support()
    
    print("=== RAG System Library Test Bench ===\n")
    
    # Example configuration - users should replace these with their own paths
    DOCUMENT_PATHS = ["./sample_documents/", "./additional_docs/file.txt"]  # Replace with actual paths
    MODEL_PATH = "/path/to/your/model"  # Replace with actual model path
    
    try:
        # Initialize RAG System with verbose output
        rag_system = RAGSystem(
            cuda_visible_devices="0,1",  # Adjust based on available GPUs
            verbose=True
        )
        
        print("1. System Info:")
        system_info = rag_system.getSystemInfo()
        for key, value in system_info.items():
            print(f"   {key}: {value}")
        
        print("\n2. Loading documents...")
        # Load documents from multiple sources
        # NOTE: Replace DOCUMENT_PATHS with actual document paths
        try:
            documents = rag_system.loadDocuments(
                paths=DOCUMENT_PATHS,
                glob_pattern='**/*.txt'
            )
            print(f"   Loaded {len(documents)} documents successfully")
        except FileNotFoundError:
            print("   Sample document paths not found. Creating dummy documents for demo...")
            # Create some dummy documents for demonstration
            from langchain.schema import Document
            dummy_docs = [
                Document(page_content="Python is a programming language known for its simplicity.", 
                        metadata={"filename": "python_intro.txt"}),
                Document(page_content="JavaScript is used for web development and runs in browsers.", 
                        metadata={"filename": "javascript_basics.txt"}),
                Document(page_content="Machine learning involves training algorithms on data.", 
                        metadata={"filename": "ml_overview.txt"})
            ]
            rag_system.documents = dummy_docs
            print(f"   Created {len(dummy_docs)} dummy documents for demonstration")
        
        print("\n3. Splitting documents...")
        split_texts = rag_system.splitDocuments(
            chunk_size=500,
            chunk_overlap=50
        )
        print(f"   Split into {len(split_texts)} chunks")
        
        print("\n4. Creating vector store...")
        vectorstore, original_texts = rag_system.createVectorstore(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_kwargs={
                "model_kwargs": {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
            },
            vectorstore_type='chroma',
            vectorstore_kwargs={"persist_directory": "./demo_chroma_db"}
        )
        print("   Vector store created successfully")
        
        print("\n5. Setting up QA Chain...")
        # NOTE: Replace MODEL_PATH with actual model path for real usage
        try:
            qa_chain, llm = rag_system.setupQAChain(
                model_type="huggingface",
                model_path=MODEL_PATH,  # Replace with actual path
                model_kwargs={
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                },
                generation_kwargs={
                    "max_new_tokens": 256,
                    "temperature": 0.7
                }
            )
            print("   QA Chain setup completed")
            
            print("\n6. Testing queries...")
            
            # Test query 1
            question1 = "What programming languages are mentioned in the documents?"
            print(f"\nQuestion 1: {question1}")
            answer1 = rag_system.query(question1, debug=True)
            print(f"Answer: {answer1}")
            
            # Test query 2
            question2 = "What topics are covered in these documents?"
            print(f"\nQuestion 2: {question2}")
            
            # Get all unique documents for comprehensive context
            unique_docs = rag_system.getAllUniqueDocuments()
            context_parts = []
            for doc in unique_docs:
                source = doc.metadata.get('filename', 'unknown')
                context_parts.append(f"Document '{source}':\n{doc.page_content}")
            
            full_context = "\n\n---\n\n".join(context_parts)
            answer2 = rag_system.queryWithDirectContext(full_context, question2)
            print(f"Answer: {answer2}")
            
        except ValueError as e:
            print(f"   Skipping QA setup - Model path not configured: {e}")
            print("   To test QA functionality, provide a valid model_path")
        
        print("\n=== Test completed successfully ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        print("Please ensure you have the required dependencies installed:")
        print("pip install torch transformers langchain langchain-community langchain-huggingface chromadb")
        
    print("\nTo use this library in your own project:")
    print("from rag_system import RAGSystem")
    print("rag = RAGSystem(verbose=True)")
    print("# Configure with your own paths and models...")