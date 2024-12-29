from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import ClassVar, Any, List, Literal, Union
import os
import logging

logger = logging.getLogger(__name__)

class BaseTool(BaseModel, ABC):
    """
    Custom BaseTool to replace dependency on agency_swarm.
    Implements basic shared state handling and configuration defaults.
    """
    _shared_state: ClassVar[Any] = None
    _caller_agent: Any = None
    _event_handler: Any = None

    class Config:
        arbitrary_types_allowed = True

    class ToolConfig:
        strict: bool = False
        one_call_at_a_time: bool = False
        output_as_result: bool = False
        async_mode: Union[Literal["threading"], None] = None

    def __init__(self, **kwargs):
        if not self.__class__._shared_state:
            self.__class__._shared_state = {}
        super().__init__(**kwargs)

    @abstractmethod
    def run(self):
        """
        Execute the tool's main functionality.
        Must be implemented by all tool classes.
        """
        pass
