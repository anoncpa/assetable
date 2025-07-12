"""
AI processing module for Assetable.

This module provides AI-powered document analysis capabilities
using Ollama for complete local processing.
"""

from .ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError
)
from .vision_processor import VisionProcessor, VisionProcessorError

__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelError",
    "OllamaResponseError",
    "OllamaTimeoutError",
    "VisionProcessor",
    "VisionProcessorError",
]
