"""RAG pipeline orchestration for the RAG toolkit.

This module provides a unified interface for composing retrieval, augmentation,
generation, and parsing components into complete RAG workflows.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import RAGConfig
from .retriever import BaseRetriever, Document
from .augmenter import BaseAugmenter
from .generator import BaseGenerator, GenerationResult
from .parser import BaseParser, ParsedOutput


@dataclass
class PipelineResult:
    """Container for complete pipeline execution results."""
    
    # Core results
    output: Union[str, Dict[str, Any]]
    parsed_output: Optional[ParsedOutput] = None
    
    # Intermediate results
    retrieved_documents: List[Document] = field(default_factory=list)
    augmented_prompt: str = ""
    generation_result: Optional[GenerationResult] = None
    
    # Metadata
    execution_time: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if pipeline executed successfully."""
        return bool(self.output)
    
    def __repr__(self):
        return f"PipelineResult(success={self.success}, execution_time={self.execution_time:.2f}s)"


class RAGPipeline:
    """Main pipeline for orchestrating RAG workflows."""
    
    def __init__(self,
                 retriever: BaseRetriever,
                 augmenter: BaseAugmenter,
                 generator: BaseGenerator,
                 parser: Optional[BaseParser] = None,
                 config: Optional[RAGConfig] = None):
        self.retriever = retriever
        self.augmenter = augmenter
        self.generator = generator
        self.parser = parser
        self.config = config or RAGConfig()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hooks
        self.pre_retrieve_hooks: List[Callable] = []
        self.post_retrieve_hooks: List[Callable] = []
        self.pre_augment_hooks: List[Callable] = []
        self.post_augment_hooks: List[Callable] = []
        self.pre_generate_hooks: List[Callable] = []
        self.post_generate_hooks: List[Callable] = []
        self.pre_parse_hooks: List[Callable] = []
        self.post_parse_hooks: List[Callable] = []
    
    def run(self,
            query: Optional[str] = None,
            return_parsed: bool = True,
            **kwargs) -> PipelineResult:
        """Execute the complete RAG pipeline."""
        start_time = time.time()
        stage_timings = {}
        
        try:
            # Stage 1: Retrieval
            self.logger.info("Stage 1: Retrieving documents")
            stage_start = time.time()
            
            self._run_hooks(self.pre_retrieve_hooks, query=query)
            documents = self.retriever.retrieve(query)
            self._run_hooks(self.post_retrieve_hooks, documents=documents)
            
            stage_timings["retrieval"] = time.time() - stage_start
            self.logger.info(f"Retrieved {len(documents)} documents in {stage_timings['retrieval']:.2f}s")
            
            # Stage 2: Augmentation
            self.logger.info("Stage 2: Augmenting prompt")
            stage_start = time.time()
            
            self._run_hooks(self.pre_augment_hooks, documents=documents, query=query)
            augmented_prompt = self.augmenter.augment(documents, query, **kwargs)
            self._run_hooks(self.post_augment_hooks, prompt=augmented_prompt)
            
            stage_timings["augmentation"] = time.time() - stage_start
            self.logger.info(f"Created prompt ({len(augmented_prompt)} chars) in {stage_timings['augmentation']:.2f}s")
            
            # Stage 3: Generation
            self.logger.info("Stage 3: Generating response")
            stage_start = time.time()
            
            self._run_hooks(self.pre_generate_hooks, prompt=augmented_prompt)
            generation_result = self.generator.generate(augmented_prompt, **kwargs)
            self._run_hooks(self.post_generate_hooks, result=generation_result)
            
            stage_timings["generation"] = time.time() - stage_start
            self.logger.info(f"Generated {len(generation_result.text)} chars in {stage_timings['generation']:.2f}s")
            
            # Stage 4: Parsing (optional)
            parsed_output = None
            if self.parser and return_parsed:
                self.logger.info("Stage 4: Parsing output")
                stage_start = time.time()
                
                self._run_hooks(self.pre_parse_hooks, text=generation_result.text)
                parsed_output = self.parser.parse(generation_result)
                self._run_hooks(self.post_parse_hooks, parsed=parsed_output)
                
                stage_timings["parsing"] = time.time() - stage_start
                self.logger.info(f"Parsed output (valid={parsed_output.is_valid}) in {stage_timings['parsing']:.2f}s")
            
            # Determine final output
            if parsed_output and parsed_output.is_valid:
                output = parsed_output.data
            else:
                output = generation_result.text
            
            # Create result
            execution_time = time.time() - start_time
            
            return PipelineResult(
                output=output,
                parsed_output=parsed_output,
                retrieved_documents=documents,
                augmented_prompt=augmented_prompt,
                generation_result=generation_result,
                execution_time=execution_time,
                stage_timings=stage_timings,
                metadata={
                    "query": query,
                    "document_count": len(documents),
                    "prompt_length": len(augmented_prompt),
                    "output_length": len(str(output)),
                    "kwargs": kwargs
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise
    
    def run_stream(self,
                   query: Optional[str] = None,
                   **kwargs):
        """Execute pipeline with streaming generation."""
        # Stages 1-2: Retrieval and Augmentation (same as run())
        documents = self.retriever.retrieve(query)
        augmented_prompt = self.augmenter.augment(documents, query, **kwargs)
        
        # Stage 3: Streaming Generation
        self.logger.info("Streaming generation")
        
        # Yield metadata first
        yield {
            "type": "metadata",
            "documents": len(documents),
            "prompt_length": len(augmented_prompt)
        }
        
        # Stream tokens
        for token in self.generator.generate_stream(augmented_prompt, **kwargs):
            yield {
                "type": "token",
                "content": token
            }
        
        yield {
            "type": "complete"
        }
    
    def add_hook(self, stage: str, hook: Callable, position: str = "pre"):
        """Add a hook to a specific pipeline stage."""
        hook_attr = f"{position}_{stage}_hooks"
        if hasattr(self, hook_attr):
            getattr(self, hook_attr).append(hook)
        else:
            raise ValueError(f"Invalid hook: {position}_{stage}")
    
    def _run_hooks(self, hooks: List[Callable], **kwargs):
        """Execute a list of hooks."""
        for hook in hooks:
            try:
                hook(**kwargs)
            except Exception as e:
                self.logger.warning(f"Hook error: {e}")


class ParallelPipeline:
    """Execute multiple pipelines in parallel."""
    
    def __init__(self,
                 pipelines: List[RAGPipeline],
                 strategy: str = "all",
                 config: Optional[RAGConfig] = None):
        self.pipelines = pipelines
        self.strategy = strategy
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self,
            query: Optional[str] = None,
            **kwargs) -> List[PipelineResult]:
        """Execute pipelines based on strategy."""
        if self.strategy == "all":
            return self._run_all(query, **kwargs)
        elif self.strategy == "race":
            return self._run_race(query, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _run_all(self,
                 query: Optional[str] = None,
                 **kwargs) -> List[PipelineResult]:
        """Run all pipelines and return all results."""
        results = []
        
        for i, pipeline in enumerate(self.pipelines):
            self.logger.info(f"Running pipeline {i+1}/{len(self.pipelines)}")
            try:
                result = pipeline.run(query, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Pipeline {i+1} failed: {e}")
                # Add failed result
                results.append(PipelineResult(
                    output="",
                    metadata={"error": str(e), "pipeline_index": i}
                ))
        
        return results
    
    def _run_race(self,
                  query: Optional[str] = None,
                  **kwargs) -> List[PipelineResult]:
        """Run pipelines until one succeeds."""
        for i, pipeline in enumerate(self.pipelines):
            self.logger.info(f"Trying pipeline {i+1}/{len(self.pipelines)}")
            try:
                result = pipeline.run(query, **kwargs)
                if result.success:
                    return [result]
            except Exception as e:
                self.logger.warning(f"Pipeline {i+1} failed: {e}")
                continue
        
        # All failed
        return [PipelineResult(
            output="",
            metadata={"error": "All pipelines failed"}
        )]


class CachedPipeline(RAGPipeline):
    """Pipeline with caching support."""
    
    def __init__(self,
                 retriever: BaseRetriever,
                 augmenter: BaseAugmenter,
                 generator: BaseGenerator,
                 parser: Optional[BaseParser] = None,
                 cache_dir: Optional[Path] = None,
                 cache_ttl: int = 3600,
                 config: Optional[RAGConfig] = None):
        super().__init__(retriever, augmenter, generator, parser, config)
        
        self.cache_dir = cache_dir or Path(".rag_cache")
        self.cache_ttl = cache_ttl
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Simple in-memory cache
        self._cache: Dict[str, Any] = {}
    
    def run(self,
            query: Optional[str] = None,
            use_cache: bool = True,
            **kwargs) -> PipelineResult:
        """Execute pipeline with caching."""
        # Generate cache key
        cache_key = self._generate_cache_key(query, **kwargs)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            if time.time() - cached_result["timestamp"] < self.cache_ttl:
                self.logger.info("Returning cached result")
                return cached_result["result"]
        
        # Run pipeline
        result = super().run(query, **kwargs)
        
        # Cache result
        if use_cache and result.success:
            self._cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
        
        return result
    
    def _generate_cache_key(self, query: Optional[str], **kwargs) -> str:
        """Generate a cache key from query and parameters."""
        import hashlib
        import json
        
        key_data = {
            "query": query,
            "kwargs": kwargs,
            "retriever": self.retriever.__class__.__name__,
            "augmenter": self.augmenter.__class__.__name__,
            "generator": self.generator.__class__.__name__
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class AdaptivePipeline(RAGPipeline):
    """Pipeline that adapts based on intermediate results."""
    
    def __init__(self,
                 retriever: BaseRetriever,
                 augmenter: BaseAugmenter,
                 generator: BaseGenerator,
                 parser: Optional[BaseParser] = None,
                 fallback_retriever: Optional[BaseRetriever] = None,
                 min_documents: int = 1,
                 config: Optional[RAGConfig] = None):
        super().__init__(retriever, augmenter, generator, parser, config)
        
        self.fallback_retriever = fallback_retriever
        self.min_documents = min_documents
    
    def run(self,
            query: Optional[str] = None,
            **kwargs) -> PipelineResult:
        """Execute pipeline with adaptive behavior."""
        # Initial retrieval
        documents = self.retriever.retrieve(query)
        
        # Check if we need fallback retrieval
        if len(documents) < self.min_documents and self.fallback_retriever:
            self.logger.info(f"Only {len(documents)} documents found, using fallback retriever")
            fallback_docs = self.fallback_retriever.retrieve(query)
            documents.extend(fallback_docs)
        
        # Check document quality
        total_content_length = sum(len(doc.content) for doc in documents)
        if total_content_length < 100:  # Arbitrary threshold
            self.logger.warning("Retrieved documents have very little content")
            # Could implement additional strategies here
        
        # Continue with augmentation
        augmented_prompt = self.augmenter.augment(documents, query, **kwargs)
        
        # Adaptive generation parameters based on prompt length
        if len(augmented_prompt) > self.config.generator.n_ctx * 0.8:
            self.logger.info("Prompt is long, reducing max_tokens")
            kwargs["max_tokens"] = min(
                kwargs.get("max_tokens", self.config.generator.max_tokens),
                self.config.generator.max_tokens // 2
            )
        
        # Generate
        generation_result = self.generator.generate(augmented_prompt, **kwargs)
        
        # Adaptive parsing
        parsed_output = None
        if self.parser:
            parsed_output = self.parser.parse(generation_result)
            
            # If parsing fails, try with a more lenient parser
            if not parsed_output.is_valid and hasattr(self, "fallback_parser"):
                self.logger.info("Primary parser failed, trying fallback")
                parsed_output = self.fallback_parser.parse(generation_result)
        
        # Create result
        output = parsed_output.data if parsed_output and parsed_output.is_valid else generation_result.text
        
        return PipelineResult(
            output=output,
            parsed_output=parsed_output,
            retrieved_documents=documents,
            augmented_prompt=augmented_prompt,
            generation_result=generation_result,
            execution_time=0,  # Would be calculated in real implementation
            metadata={
                "adaptive_actions": [
                    "fallback_retrieval" if len(documents) > len(self.retriever.retrieve(query)) else None,
                    "reduced_max_tokens" if "max_tokens" in kwargs else None
                ]
            }
        )