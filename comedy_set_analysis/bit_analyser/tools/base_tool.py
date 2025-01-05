from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import ClassVar, Any, List, Literal, Union
import os
import logging

logger = logging.getLogger(__name__)

class SimpleBaseTool(BaseModel, ABC):
    """Base class for tools that don't require vector processing"""
    _shared_state: ClassVar[Any] = None
    _caller_agent: Any = None
    _event_handler: Any = None

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

    @abstractmethod
    def run(self):
        pass

class BaseTool(BaseModel, ABC):
    """
    Custom BaseTool
    Implements basic shared state handling and configuration defaults.
    """
    _shared_state: ClassVar[Any] = None
    _caller_agent: Any = None
    _event_handler: Any = None

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

    bits_file: str = Field(description="Path to bits JSON file")
    metadata_file: str = Field(description="Path to metadata JSON file")
    transcript_file: str = Field(description="Path to transcript JSON file")
    vectors_dir: str = Field(description="Directory to store bit vectors")
    regenerate: bool = Field(default=False, description="Force regeneration of vectors")
    nlp: Any = Field(default=None, description="spaCy language model")
    sentence_model: Any = Field(default=None, description="Sentence transformer model")
    dimension: int = Field(default=384, description="Vector dimension")  # Default to SBERT dimension

    def __init__(self, **kwargs):
        if not self.__class__._shared_state:
            self.__class__._shared_state = {}
        super().__init__(**kwargs)

        # Validate paths
        for path in [self.bits_file, self.metadata_file, self.transcript_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # Create vectors directory
        os.makedirs(self.vectors_dir, exist_ok=True)

    @abstractmethod
    def run(self):
        """
        Execute the tool's main functionality.
        Must be implemented by all tool classes.
        """
        pass
