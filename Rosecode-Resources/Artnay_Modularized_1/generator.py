"""LLM interaction and generation for the RAG toolkit.

This module provides unified interfaces for various LLM backends,
including local models (via llama.cpp) and API-based models.
"""

import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Generator
from pathlib import Path

from .config import GeneratorConfig


class GenerationResult:
    """Container for generation results with metadata."""
    
    def __init__(self,
                 text: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 usage: Optional[Dict[str, int]] = None,
                 timing: Optional[Dict[str, float]] = None):
        self.text = text
        self.metadata = metadata or {}
        self.usage = usage or {}
        self.timing = timing or {}
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"GenerationResult(text_length={len(self.text)}, metadata={self.metadata})"


class BaseGenerator(ABC):
    """Abstract base class for all generation strategies."""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self,
                 prompt: Union[str, List[Dict[str, str]]],
                 **kwargs) -> GenerationResult:
        """Generate text from the prompt."""
        pass
    
    @abstractmethod
    def generate_stream(self,
                        prompt: Union[str, List[Dict[str, str]]],
                        **kwargs) -> Generator[str, None, None]:
        """Generate text in streaming mode."""
        pass
    
    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Merge kwargs with config values."""
        params = {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        params.update(kwargs)
        return params


class LlamaCppGenerator(BaseGenerator):
    """Generate using local models via llama.cpp."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config: Optional[GeneratorConfig] = None):
        super().__init__(config)
        
        # Override model path if provided
        if model_path:
            self.config.model_path = model_path
        
        if not self.config.model_path:
            raise ValueError("Model path must be specified for LlamaCppGenerator")
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the llama.cpp model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        
        self.logger.info(f"Loading model from {self.config.model_path}")
        start_time = time.time()
        
        # Initialize with configuration
        self.model = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            verbose=self.config.verbose
        )
        
        load_time = time.time() - start_time
        self.logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    def generate(self,
                 prompt: Union[str, List[Dict[str, str]]],
                 **kwargs) -> GenerationResult:
        """Generate text using llama.cpp."""
        params = self._merge_kwargs(**kwargs)
        start_time = time.time()
        
        # Convert chat format to string if necessary
        if isinstance(prompt, list):
            prompt = self._format_chat_prompt(prompt)
        
        try:
            # Generate with llama.cpp
            output = self.model(
                prompt,
                max_tokens=params.get("max_tokens", self.config.max_tokens),
                temperature=params.get("temperature", self.config.temperature),
                top_p=params.get("top_p", self.config.top_p),
                stop=params.get("stop", []),
                stream=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract text and metadata
            text = output["choices"][0]["text"]
            
            # Create result
            result = GenerationResult(
                text=text.strip(),
                metadata={
                    "model": self.config.model_path,
                    "finish_reason": output["choices"][0].get("finish_reason", "unknown")
                },
                usage={
                    "prompt_tokens": output.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": output.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": output.get("usage", {}).get("total_tokens", 0)
                },
                timing={
                    "generation_time": generation_time
                }
            )
            
            self.logger.info(f"Generated {len(result.text)} characters in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            raise
    
    def generate_stream(self,
                        prompt: Union[str, List[Dict[str, str]]],
                        **kwargs) -> Generator[str, None, None]:
        """Generate text in streaming mode."""
        params = self._merge_kwargs(**kwargs)
        
        # Convert chat format to string if necessary
        if isinstance(prompt, list):
            prompt = self._format_chat_prompt(prompt)
        
        try:
            # Generate with streaming
            stream = self.model(
                prompt,
                max_tokens=params.get("max_tokens", self.config.max_tokens),
                temperature=params.get("temperature", self.config.temperature),
                top_p=params.get("top_p", self.config.top_p),
                stop=params.get("stop", []),
                stream=True
            )
            
            for chunk in stream:
                token = chunk["choices"][0]["text"]
                yield token
                
        except Exception as e:
            self.logger.error(f"Streaming generation error: {e}")
            raise
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        # Simple format - can be customized based on model requirements
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)


class APIGenerator(BaseGenerator):
    """Generate using API-based models."""
    
    def __init__(self,
                 api_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 config: Optional[GeneratorConfig] = None):
        super().__init__(config)
        
        # Override API settings if provided
        if api_endpoint:
            self.config.api_endpoint = api_endpoint
        if api_key:
            self.config.api_key = api_key
        
        self.model_name = model_name
        
        if not self.config.api_endpoint:
            # Default to OpenAI endpoint
            self.config.api_endpoint = "https://api.openai.com/v1/chat/completions"
        
        if not self.config.api_key:
            raise ValueError("API key must be specified for APIGenerator")
    
    def generate(self,
                 prompt: Union[str, List[Dict[str, str]]],
                 **kwargs) -> GenerationResult:
        """Generate text using API."""
        params = self._merge_kwargs(**kwargs)
        start_time = time.time()
        
        # Prepare messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        data = {
            "model": params.get("model", self.model_name),
            "messages": messages,
            "max_tokens": params.get("max_tokens", self.config.max_tokens),
            "temperature": params.get("temperature", self.config.temperature),
            "top_p": params.get("top_p", self.config.top_p),
            "stream": False
        }
        
        # Add optional parameters
        if "stop" in params:
            data["stop"] = params["stop"]
        if "presence_penalty" in params:
            data["presence_penalty"] = params["presence_penalty"]
        if "frequency_penalty" in params:
            data["frequency_penalty"] = params["frequency_penalty"]
        
        try:
            # Make API request
            response = requests.post(
                self.config.api_endpoint,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            generation_time = time.time() - start_time
            result_data = response.json()
            
            # Extract text
            text = result_data["choices"][0]["message"]["content"]
            
            # Create result
            result = GenerationResult(
                text=text.strip(),
                metadata={
                    "model": result_data.get("model", self.model_name),
                    "finish_reason": result_data["choices"][0].get("finish_reason", "unknown")
                },
                usage={
                    "prompt_tokens": result_data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": result_data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": result_data.get("usage", {}).get("total_tokens", 0)
                },
                timing={
                    "generation_time": generation_time
                }
            )
            
            self.logger.info(f"Generated {len(result.text)} characters in {generation_time:.2f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            raise
    
    def generate_stream(self,
                        prompt: Union[str, List[Dict[str, str]]],
                        **kwargs) -> Generator[str, None, None]:
        """Generate text in streaming mode."""
        params = self._merge_kwargs(**kwargs)
        
        # Prepare messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        data = {
            "model": params.get("model", self.model_name),
            "messages": messages,
            "max_tokens": params.get("max_tokens", self.config.max_tokens),
            "temperature": params.get("temperature", self.config.temperature),
            "top_p": params.get("top_p", self.config.top_p),
            "stream": True
        }
        
        try:
            # Make streaming request
            response = requests.post(
                self.config.api_endpoint,
                headers=headers,
                json=data,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # Process stream
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Streaming API request error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Streaming generation error: {e}")
            raise


class HybridGenerator(BaseGenerator):
    """Use multiple generators with fallback support."""
    
    def __init__(self,
                 generators: List[BaseGenerator],
                 strategy: str = "fallback",
                 config: Optional[GeneratorConfig] = None):
        super().__init__(config)
        self.generators = generators
        self.strategy = strategy
        
        if not self.generators:
            raise ValueError("At least one generator must be provided")
    
    def generate(self,
                 prompt: Union[str, List[Dict[str, str]]],
                 **kwargs) -> GenerationResult:
        """Generate using configured strategy."""
        if self.strategy == "fallback":
            return self._generate_with_fallback(prompt, **kwargs)
        elif self.strategy == "ensemble":
            return self._generate_ensemble(prompt, **kwargs)
        else:
            # Default to first generator
            return self.generators[0].generate(prompt, **kwargs)
    
    def generate_stream(self,
                        prompt: Union[str, List[Dict[str, str]]],
                        **kwargs) -> Generator[str, None, None]:
        """Stream from the first available generator."""
        for generator in self.generators:
            try:
                yield from generator.generate_stream(prompt, **kwargs)
                return
            except Exception as e:
                self.logger.warning(f"Generator {generator.__class__.__name__} failed: {e}")
                continue
        
        raise RuntimeError("All generators failed")
    
    def _generate_with_fallback(self,
                                prompt: Union[str, List[Dict[str, str]]],
                                **kwargs) -> GenerationResult:
        """Try generators in order until one succeeds."""
        errors = []
        
        for i, generator in enumerate(self.generators):
            try:
                self.logger.info(f"Trying generator {i+1}/{len(self.generators)}: {generator.__class__.__name__}")
                return generator.generate(prompt, **kwargs)
            except Exception as e:
                self.logger.warning(f"Generator {generator.__class__.__name__} failed: {e}")
                errors.append((generator.__class__.__name__, str(e)))
                continue
        
        # All failed
        error_msg = "\n".join([f"{name}: {error}" for name, error in errors])
        raise RuntimeError(f"All generators failed:\n{error_msg}")
    
    def _generate_ensemble(self,
                           prompt: Union[str, List[Dict[str, str]]],
                           **kwargs) -> GenerationResult:
        """Generate from all generators and combine results."""
        results = []
        timings = {}
        
        for generator in self.generators:
            try:
                result = generator.generate(prompt, **kwargs)
                results.append(result)
                timings[generator.__class__.__name__] = result.timing.get("generation_time", 0)
            except Exception as e:
                self.logger.warning(f"Generator {generator.__class__.__name__} failed in ensemble: {e}")
        
        if not results:
            raise RuntimeError("No generators succeeded in ensemble")
        
        # Simple combination - concatenate with separators
        # This could be made more sophisticated
        combined_text = "\n\n--- Alternative Response ---\n\n".join([r.text for r in results])
        
        # Combine metadata
        combined_metadata = {
            "ensemble_size": len(results),
            "generators": [g.__class__.__name__ for g in self.generators[:len(results)]]
        }
        
        # Combine usage stats
        combined_usage = {
            "total_prompt_tokens": sum(r.usage.get("prompt_tokens", 0) for r in results),
            "total_completion_tokens": sum(r.usage.get("completion_tokens", 0) for r in results),
            "total_tokens": sum(r.usage.get("total_tokens", 0) for r in results)
        }
        
        return GenerationResult(
            text=combined_text,
            metadata=combined_metadata,
            usage=combined_usage,
            timing=timings
        )