"""Configuration management for the RAG toolkit.

This module provides flexible configuration options through multiple methods:
- Constructor parameters
- Configuration dictionaries
- Environment variables
- Configuration files (JSON/YAML)
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrieverConfig:
    """Configuration for retrieval components."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks: Optional[int] = None
    metadata_fields: list = field(default_factory=lambda: ["title", "source", "date"])
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".rag_cache"))


@dataclass
class AugmenterConfig:
    """Configuration for augmentation components."""
    max_context_length: int = 2048
    template_dir: Optional[Path] = None
    default_template: str = "default"
    include_metadata: bool = True
    truncation_strategy: str = "end"  # "end", "middle", "smart"


@dataclass
class GeneratorConfig:
    """Configuration for generation components."""
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    n_gpu_layers: int = -1  # -1 for all layers
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 8
    verbose: bool = False


@dataclass
class RAGConfig:
    """Main configuration class for the RAG toolkit."""
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    augmenter: AugmenterConfig = field(default_factory=AugmenterConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    
    # General settings
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("rag_output"))
    enable_progress_bar: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create configuration from a dictionary."""
        retriever_cfg = RetrieverConfig(**config_dict.get("retriever", {}))
        augmenter_cfg = AugmenterConfig(**config_dict.get("augmenter", {}))
        generator_cfg = GeneratorConfig(**config_dict.get("generator", {}))
        
        # Extract general settings
        general_settings = {
            k: v for k, v in config_dict.items() 
            if k not in ["retriever", "augmenter", "generator"]
        }
        
        return cls(
            retriever=retriever_cfg,
            augmenter=augmenter_cfg,
            generator=generator_cfg,
            **general_settings
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "RAGConfig":
        """Load configuration from a JSON or YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config structure
        env_mapping = {
            "RAG_CHUNK_SIZE": ("retriever", "chunk_size", int),
            "RAG_CHUNK_OVERLAP": ("retriever", "chunk_overlap", int),
            "RAG_MAX_CONTEXT": ("augmenter", "max_context_length", int),
            "RAG_MODEL_PATH": ("generator", "model_path", str),
            "RAG_API_ENDPOINT": ("generator", "api_endpoint", str),
            "RAG_API_KEY": ("generator", "api_key", str),
            "RAG_MAX_TOKENS": ("generator", "max_tokens", int),
            "RAG_TEMPERATURE": ("generator", "temperature", float),
            "RAG_LOG_LEVEL": (None, "log_level", str),
        }
        
        for env_var, (section, key, type_func) in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                if section:
                    if section not in config_dict:
                        config_dict[section] = {}
                    config_dict[section][key] = type_func(value)
                else:
                    config_dict[key] = type_func(value)
        
        return cls.from_dict(config_dict) if config_dict else cls()
    
    def merge(self, other: "RAGConfig") -> "RAGConfig":
        """Merge another configuration into this one."""
        # Deep merge implementation
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        base_dict = self.__dict__.copy()
        update_dict = other.__dict__.copy()
        merged_dict = deep_merge(base_dict, update_dict)
        
        return RAGConfig.from_dict(merged_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "retriever": self.retriever.__dict__,
            "augmenter": self.augmenter.__dict__,
            "generator": self.generator.__dict__,
            "log_level": self.log_level,
            "output_dir": str(self.output_dir),
            "enable_progress_bar": self.enable_progress_bar
        }
    
    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)

