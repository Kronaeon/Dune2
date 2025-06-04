"""Prompt construction and augmentation for the RAG toolkit.

This module provides flexible mechanisms for creating prompts with retrieved content,
supporting various templates, styles, and context management strategies.
"""

import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from string import Template

from .config import AugmenterConfig
from .retriever import Document


class BaseAugmenter(ABC):
    """Abstract base class for all augmentation strategies."""
    
    def __init__(self, config: Optional[AugmenterConfig] = None):
        self.config = config or AugmenterConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize template directory if specified
        if self.config.template_dir:
            self.config.template_dir = Path(self.config.template_dir)
            if not self.config.template_dir.exists():
                self.logger.warning(f"Template directory not found: {self.config.template_dir}")
    
    @abstractmethod
    def augment(self, 
                documents: List[Document],
                query: Optional[str] = None,
                **kwargs) -> str:
        """Augment the query with retrieved documents to create a prompt."""
        pass
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content based on the configured strategy."""
        if len(content) <= max_length:
            return content
        
        if self.config.truncation_strategy == "end":
            return content[:max_length] + "..."
        
        elif self.config.truncation_strategy == "middle":
            # Keep beginning and end
            half = max_length // 2
            return content[:half] + "\n...[truncated]...\n" + content[-half:]
        
        elif self.config.truncation_strategy == "smart":
            # Try to truncate at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', content)
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) > max_length:
                    break
                truncated += sentence + " "
            return truncated.strip() + "..."
        
        else:
            return content[:max_length]
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for inclusion in prompts."""
        if not metadata or not self.config.include_metadata:
            return ""
        
        # Filter to only include configured fields
        filtered_metadata = {
            k: v for k, v in metadata.items() 
            if k in self.config.metadata_fields
        }
        
        if not filtered_metadata:
            return ""
        
        # Format as key-value pairs
        lines = []
        for key, value in filtered_metadata.items():
            lines.append(f"{key.title()}: {value}")
        
        return "\n".join(lines)


class TemplateAugmenter(BaseAugmenter):
    """Augment using customizable templates."""
    
    def __init__(self, 
                 template: Optional[str] = None,
                 template_path: Optional[Path] = None,
                 config: Optional[AugmenterConfig] = None):
        super().__init__(config)
        
        # Load template
        if template:
            self.template = template
        elif template_path:
            self.template = self._load_template(template_path)
        else:
            # Default template
            self.template = """${task_instruction}

Based on the following context:
${context}

${query_section}

Please provide a comprehensive response."""
    
    def augment(self,
                documents: List[Document],
                query: Optional[str] = None,
                task_instruction: str = "You are a helpful assistant.",
                **kwargs) -> str:
        """Create a prompt using the template."""
        # Prepare context from documents
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format document content
            doc_text = doc.content
            
            # Add metadata if configured
            if self.config.include_metadata:
                metadata_text = self._format_metadata(doc.metadata)
                if metadata_text:
                    doc_text = f"{metadata_text}\n\n{doc_text}"
            
            # Add document separator
            if len(documents) > 1:
                doc_text = f"--- Document {i+1} ---\n{doc_text}\n--- End Document {i+1} ---"
            
            context_parts.append(doc_text)
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        # Truncate if necessary
        if self.config.max_context_length:
            full_context = self._truncate_content(full_context, self.config.max_context_length)
        
        # Prepare query section
        query_section = f"Query: {query}" if query else ""
        
        # Create template substitution dictionary
        template_vars = {
            "task_instruction": task_instruction,
            "context": full_context,
            "query_section": query_section,
            "query": query or "",
            **kwargs  # Allow additional custom variables
        }
        
        # Substitute template
        template_obj = Template(self.template)
        prompt = template_obj.safe_substitute(**template_vars)
        
        return prompt.strip()
    
    def _load_template(self, template_path: Path) -> str:
        """Load template from file."""
        template_path = Path(template_path)
        
        # Check in template directory first
        if self.config.template_dir and not template_path.is_absolute():
            template_path = self.config.template_dir / template_path
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()


class StyleAugmenter(BaseAugmenter):
    """Augment prompts with specific styles or formats."""
    
    # Predefined style templates
    STYLES = {
        "qa": {
            "template": """Answer the following question based on the provided context.

Context:
${context}

Question: ${query}

Answer:""",
            "instruction": "You are a helpful question-answering assistant."
        },
        "summary": {
            "template": """Summarize the following content:

${context}

Summary:""",
            "instruction": "You are an expert at creating concise, informative summaries."
        },
        "creative": {
            "template": """Using the following information as inspiration:

${context}

${query}

Create:""",
            "instruction": "You are a creative writer who produces engaging content."
        },
        "analytical": {
            "template": """Analyze the following information:

${context}

Analysis Request: ${query}

Provide a detailed analysis:""",
            "instruction": "You are an analytical expert who provides thorough insights."
        },
        "instructional": {
            "template": """Based on the following information:

${context}

Create a step-by-step guide for: ${query}

Guide:""",
            "instruction": "You are an instructional designer creating clear, actionable guides."
        }
    }
    
    def __init__(self,
                 style: str = "qa",
                 custom_styles: Optional[Dict[str, Dict[str, str]]] = None,
                 config: Optional[AugmenterConfig] = None):
        super().__init__(config)
        
        # Add custom styles if provided
        if custom_styles:
            self.STYLES.update(custom_styles)
        
        # Validate style
        if style not in self.STYLES:
            raise ValueError(f"Unknown style: {style}. Available: {list(self.STYLES.keys())}")
        
        self.style = style
        self.style_config = self.STYLES[style]
    
    def augment(self,
                documents: List[Document],
                query: Optional[str] = None,
                **kwargs) -> str:
        """Create a styled prompt."""
        # Prepare context
        context_parts = []
        for doc in documents:
            content = doc.content
            if self.config.include_metadata:
                metadata = self._format_metadata(doc.metadata)
                if metadata:
                    content = f"{metadata}\n{content}"
            context_parts.append(content)
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate if necessary
        if self.config.max_context_length:
            full_context = self._truncate_content(full_context, self.config.max_context_length)
        
        # Get template and instruction for the style
        template = Template(self.style_config["template"])
        
        # Prepare variables
        template_vars = {
            "context": full_context,
            "query": query or "",
            **kwargs
        }
        
        # Generate prompt
        prompt = template.safe_substitute(**template_vars)
        
        # Optionally prepend instruction
        if kwargs.get("include_instruction", True):
            instruction = kwargs.get("instruction", self.style_config["instruction"])
            prompt = f"{instruction}\n\n{prompt}"
        
        return prompt.strip()


class ConversationalAugmenter(BaseAugmenter):
    """Augment prompts for conversational/chat-based interactions."""
    
    def __init__(self,
                 system_message: Optional[str] = None,
                 config: Optional[AugmenterConfig] = None):
        super().__init__(config)
        self.system_message = system_message or "You are a helpful AI assistant."
        self.conversation_history = []
    
    def augment(self,
                documents: List[Document],
                query: Optional[str] = None,
                include_history: bool = True,
                **kwargs) -> Union[str, List[Dict[str, str]]]:
        """Create a conversational prompt."""
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.system_message
        })
        
        # Add conversation history if requested
        if include_history:
            messages.extend(self.conversation_history)
        
        # Prepare context from documents
        if documents:
            context_parts = []
            for doc in documents:
                content = doc.content
                if self.config.include_metadata:
                    metadata = self._format_metadata(doc.metadata)
                    if metadata:
                        content = f"{metadata}\n{content}"
                context_parts.append(content)
            
            context = "\n\n".join(context_parts)
            
            # Truncate if necessary
            if self.config.max_context_length:
                context = self._truncate_content(context, self.config.max_context_length)
            
            # Add context as a system or user message
            context_message = f"Here is some relevant context:\n\n{context}"
            messages.append({
                "role": "system",
                "content": context_message
            })
        
        # Add user query
        if query:
            messages.append({
                "role": "user",
                "content": query
            })
        
        # Return based on format preference
        if kwargs.get("return_list", True):
            return messages
        else:
            # Convert to string format
            prompt_parts = []
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                prompt_parts.append(f"{role}: {content}")
            return "\n\n".join(prompt_parts)
    
    def add_turn(self, role: str, content: str):
        """Add a conversation turn to history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


class ChainAugmenter(BaseAugmenter):
    """Chain multiple augmenters together."""
    
    def __init__(self,
                 augmenters: List[BaseAugmenter],
                 config: Optional[AugmenterConfig] = None):
        super().__init__(config)
        self.augmenters = augmenters
    
    def augment(self,
                documents: List[Document],
                query: Optional[str] = None,
                **kwargs) -> str:
        """Apply augmenters in sequence."""
        current_output = None
        
        for i, augmenter in enumerate(self.augmenters):
            if i == 0:
                # First augmenter gets the original documents
                current_output = augmenter.augment(documents, query, **kwargs)
            else:
                # Subsequent augmenters get the output as a document
                output_doc = Document(
                    content=current_output,
                    metadata={"augmenter_index": i-1},
                    source="previous_augmenter"
                )
                current_output = augmenter.augment([output_doc], query, **kwargs)
        
        return current_output


class DynamicAugmenter(BaseAugmenter):
    """Dynamically adjust augmentation based on document characteristics."""
    
    def __init__(self,
                 base_augmenter: Optional[BaseAugmenter] = None,
                 config: Optional[AugmenterConfig] = None):
        super().__init__(config)
        self.base_augmenter = base_augmenter or TemplateAugmenter(config=config)
    
    def augment(self,
                documents: List[Document],
                query: Optional[str] = None,
                **kwargs) -> str:
        """Dynamically adjust augmentation strategy."""
        # Analyze documents
        total_length = sum(len(doc.content) for doc in documents)
        doc_count = len(documents)
        avg_length = total_length / doc_count if doc_count > 0 else 0
        
        # Adjust context length based on document characteristics
        if total_length > self.config.max_context_length * 2:
            # For very long content, use aggressive truncation
            self.logger.info("Using aggressive truncation due to long content")
            old_strategy = self.config.truncation_strategy
            self.config.truncation_strategy = "smart"
            result = self.base_augmenter.augment(documents, query, **kwargs)
            self.config.truncation_strategy = old_strategy
            return result
        
        elif doc_count > 5:
            # For many documents, consider summarizing each first
            self.logger.info(f"Processing {doc_count} documents with summarization")
            # This could be extended to actually summarize documents
            # For now, just use the base augmenter with metadata focus
            old_include_metadata = self.config.include_metadata
            self.config.include_metadata = True
            result = self.base_augmenter.augment(documents, query, **kwargs)
            self.config.include_metadata = old_include_metadata
            return result
        
        else:
            # Default augmentation
            return self.base_augmenter.augment(documents, query, **kwargs)