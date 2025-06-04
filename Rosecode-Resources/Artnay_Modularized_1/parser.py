"""Output parsing and validation for the RAG toolkit.

This module provides flexible mechanisms for parsing structured output from LLMs,
validating results, and extracting specific information patterns.
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Pattern, Callable
from dataclasses import dataclass, field

from .generator import GenerationResult


@dataclass
class ParsedOutput:
    """Container for parsed output with validation results."""
    
    data: Dict[str, Any]
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"ParsedOutput(valid={self.is_valid}, errors={len(self.validation_errors)})"


class BaseParser(ABC):
    """Abstract base class for all parsing strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def parse(self, 
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Parse the generated text into structured output."""
        pass
    
    def _extract_text(self, input_data: Union[str, GenerationResult]) -> str:
        """Extract text from input."""
        if isinstance(input_data, GenerationResult):
            return input_data.text
        return input_data


class RegexParser(BaseParser):
    """Parse output using regular expressions."""
    
    def __init__(self,
                 patterns: Dict[str, Union[str, Pattern]],
                 required_fields: Optional[List[str]] = None,
                 multiline: bool = True):
        super().__init__()
        
        # Compile patterns
        self.patterns = {}
        flags = re.MULTILINE | re.DOTALL if multiline else 0
        
        for name, pattern in patterns.items():
            if isinstance(pattern, str):
                self.patterns[name] = re.compile(pattern, flags)
            else:
                self.patterns[name] = pattern
        
        self.required_fields = required_fields or []
    
    def parse(self,
              text: Union[str, GenerationResult],
              validate: bool = True,
              **kwargs) -> ParsedOutput:
        """Parse text using regex patterns."""
        text_content = self._extract_text(text)
        
        # Extract data using patterns
        data = {}
        for field_name, pattern in self.patterns.items():
            match = pattern.search(text_content)
            if match:
                # Support both single captures and groups
                if match.groups():
                    if len(match.groups()) == 1:
                        data[field_name] = match.group(1)
                    else:
                        data[field_name] = match.groups()
                else:
                    data[field_name] = match.group(0)
        
        # Validate if requested
        validation_errors = []
        if validate:
            validation_errors = self._validate(data)
        
        return ParsedOutput(
            data=data,
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            raw_text=text_content,
            metadata={"parser": "regex", "patterns": len(self.patterns)}
        )
    
    def _validate(self, data: Dict[str, Any]) -> List[str]:
        """Validate extracted data."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in data or not data[field]:
                errors.append(f"Required field missing: {field}")
        
        return errors


class StructuredParser(BaseParser):
    """Parse structured output with sections and hierarchy."""
    
    def __init__(self,
                 section_pattern: str = r"^#+\s*(.+)$",
                 field_pattern: str = r"^(\w+):\s*(.+)$",
                 list_pattern: str = r"^[-*]\s+(.+)$"):
        super().__init__()
        
        self.section_pattern = re.compile(section_pattern, re.MULTILINE)
        self.field_pattern = re.compile(field_pattern, re.MULTILINE)
        self.list_pattern = re.compile(list_pattern, re.MULTILINE)
    
    def parse(self,
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Parse structured text with sections."""
        text_content = self._extract_text(text)
        
        # Split into sections
        sections = self._extract_sections(text_content)
        
        # Parse each section
        data = {}
        for section_name, section_content in sections.items():
            section_data = self._parse_section(section_content)
            if section_data:
                data[section_name] = section_data
        
        return ParsedOutput(
            data=data,
            is_valid=len(data) > 0,
            validation_errors=[] if data else ["No structured content found"],
            raw_text=text_content,
            metadata={"parser": "structured", "sections": len(data)}
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from text."""
        sections = {}
        current_section = "main"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            section_match = self.section_pattern.match(line)
            if section_match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = section_match.group(1).lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_section(self, content: str) -> Dict[str, Any]:
        """Parse content within a section."""
        data = {}
        current_list = None
        current_list_name = None
        
        lines = content.split('\n')
        
        for line in lines:
            # Check for field
            field_match = self.field_pattern.match(line)
            if field_match:
                # Save any pending list
                if current_list is not None:
                    data[current_list_name] = current_list
                    current_list = None
                
                field_name = field_match.group(1).lower()
                field_value = field_match.group(2).strip()
                data[field_name] = field_value
                continue
            
            # Check for list item
            list_match = self.list_pattern.match(line)
            if list_match:
                if current_list is None:
                    current_list = []
                    current_list_name = "items"
                current_list.append(list_match.group(1).strip())
        
        # Save any pending list
        if current_list is not None:
            data[current_list_name] = current_list
        
        return data


class JSONParser(BaseParser):
    """Parse JSON-formatted output."""
    
    def __init__(self,
                 strict: bool = False,
                 schema: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.strict = strict
        self.schema = schema
    
    def parse(self,
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Parse JSON from text."""
        text_content = self._extract_text(text)
        
        # Try to extract JSON from text
        json_text = self._extract_json(text_content)
        
        if not json_text:
            return ParsedOutput(
                data={},
                is_valid=False,
                validation_errors=["No JSON content found"],
                raw_text=text_content
            )
        
        # Parse JSON
        try:
            data = json.loads(json_text)
            
            # Validate against schema if provided
            validation_errors = []
            if self.schema:
                validation_errors = self._validate_schema(data, self.schema)
            
            return ParsedOutput(
                data=data,
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                raw_text=text_content,
                metadata={"parser": "json", "extracted_length": len(json_text)}
            )
            
        except json.JSONDecodeError as e:
            return ParsedOutput(
                data={},
                is_valid=False,
                validation_errors=[f"JSON parsing error: {str(e)}"],
                raw_text=text_content
            )
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON content from text."""
        # Look for JSON code blocks
        json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for JSON objects or arrays
        json_patterns = [
            r'(\{[^{}]*\})',  # Simple object
            r'(\[[^\[\]]*\])',  # Simple array
            r'(\{.*\})',  # Complex object
            r'(\[.*\])'  # Complex array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    json.loads(match)
                    return match
                except:
                    continue
        
        # If strict mode, don't try the entire text
        if not self.strict:
            # Try the entire text
            try:
                json.loads(text)
                return text
            except:
                pass
        
        return None
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Simple schema validation."""
        errors = []
        
        if "required" in schema:
            for field in schema["required"]:
                if isinstance(data, dict) and field not in data:
                    errors.append(f"Required field missing: {field}")
        
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
            elif expected_type == "array" and not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
        
        return errors


class TemplateParser(BaseParser):
    """Parse output based on a template structure."""
    
    def __init__(self,
                 template: str,
                 markers: Optional[Dict[str, str]] = None):
        super().__init__()
        self.template = template
        self.markers = markers or {
            "start": "{{",
            "end": "}}"
        }
        
        # Extract fields from template
        self.fields = self._extract_template_fields()
    
    def parse(self,
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Parse text based on template."""
        text_content = self._extract_text(text)
        
        # Build regex pattern from template
        pattern = self._build_pattern()
        
        # Extract data
        match = pattern.search(text_content)
        if match:
            data = match.groupdict()
            
            return ParsedOutput(
                data=data,
                is_valid=True,
                validation_errors=[],
                raw_text=text_content,
                metadata={"parser": "template", "fields": list(data.keys())}
            )
        else:
            return ParsedOutput(
                data={},
                is_valid=False,
                validation_errors=["Text does not match template structure"],
                raw_text=text_content
            )
    
    def _extract_template_fields(self) -> List[str]:
        """Extract field names from template."""
        start = re.escape(self.markers["start"])
        end = re.escape(self.markers["end"])
        pattern = f"{start}(\\w+){end}"
        
        fields = re.findall(pattern, self.template)
        return fields
    
    def _build_pattern(self) -> Pattern:
        """Build regex pattern from template."""
        # Escape special regex characters in template
        pattern_text = re.escape(self.template)
        
        # Replace field markers with capture groups
        start = re.escape(self.markers["start"])
        end = re.escape(self.markers["end"])
        
        for field in self.fields:
            field_pattern = f"{start}{field}{end}"
            # Use non-greedy capture
            pattern_text = pattern_text.replace(field_pattern, f"(?P<{field}>.*?)")
        
        return re.compile(pattern_text, re.DOTALL)


class ValidationParser(BaseParser):
    """Parse with custom validation rules."""
    
    def __init__(self,
                 base_parser: BaseParser,
                 validators: Dict[str, Callable[[Any], bool]],
                 error_messages: Optional[Dict[str, str]] = None):
        super().__init__()
        self.base_parser = base_parser
        self.validators = validators
        self.error_messages = error_messages or {}
    
    def parse(self,
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Parse and validate output."""
        # First parse with base parser
        result = self.base_parser.parse(text, **kwargs)
        
        # Apply additional validation
        validation_errors = result.validation_errors.copy()
        
        for field_name, validator in self.validators.items():
            if field_name in result.data:
                value = result.data[field_name]
                if not validator(value):
                    error_msg = self.error_messages.get(
                        field_name,
                        f"Validation failed for field: {field_name}"
                    )
                    validation_errors.append(error_msg)
        
        # Update validation status
        result.validation_errors = validation_errors
        result.is_valid = len(validation_errors) == 0
        
        return result


class ChainParser(BaseParser):
    """Chain multiple parsers together."""
    
    def __init__(self,
                 parsers: List[BaseParser],
                 strategy: str = "first_valid"):
        super().__init__()
        self.parsers = parsers
        self.strategy = strategy
    
    def parse(self,
              text: Union[str, GenerationResult],
              **kwargs) -> ParsedOutput:
        """Apply parsers based on strategy."""
        if self.strategy == "first_valid":
            return self._parse_first_valid(text, **kwargs)
        elif self.strategy == "merge_all":
            return self._parse_merge_all(text, **kwargs)
        else:
            # Default to first parser
            return self.parsers[0].parse(text, **kwargs)
    
    def _parse_first_valid(self,
                           text: Union[str, GenerationResult],
                           **kwargs) -> ParsedOutput:
        """Use the first parser that produces valid output."""
        all_errors = []
        
        for i, parser in enumerate(self.parsers):
            result = parser.parse(text, **kwargs)
            if result.is_valid:
                result.metadata["parser_index"] = i
                result.metadata["parser_class"] = parser.__class__.__name__
                return result
            
            all_errors.extend([f"{parser.__class__.__name__}: {e}" for e in result.validation_errors])
        
        # None succeeded
        return ParsedOutput(
            data={},
            is_valid=False,
            validation_errors=all_errors,
            raw_text=self._extract_text(text)
        )
    
    def _parse_merge_all(self,
                         text: Union[str, GenerationResult],
                         **kwargs) -> ParsedOutput:
        """Merge results from all parsers."""
        merged_data = {}
        all_errors = []
        parser_results = []
        
        for parser in self.parsers:
            result = parser.parse(text, **kwargs)
            parser_results.append({
                "parser": parser.__class__.__name__,
                "valid": result.is_valid,
                "data_keys": list(result.data.keys())
            })
            
            # Merge data (later parsers override earlier ones)
            merged_data.update(result.data)
            
            # Collect errors
            if not result.is_valid:
                all_errors.extend([f"{parser.__class__.__name__}: {e}" for e in result.validation_errors])
        
        return ParsedOutput(
            data=merged_data,
            is_valid=len(merged_data) > 0,
            validation_errors=all_errors,
            raw_text=self._extract_text(text),
            metadata={"parser_results": parser_results}
        )