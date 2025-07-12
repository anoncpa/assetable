"""
Ollama client for Assetable.

This module provides a client for interacting with Ollama API
for Vision-based AI processing. It handles model communication,
structured output, and error recovery.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import ollama
from pydantic import BaseModel, ValidationError

from ..config import AssetableConfig, get_config

T = TypeVar('T', bound=BaseModel)


class OllamaError(Exception):
    """Base exception for Ollama operations."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""
    pass


class OllamaModelError(OllamaError):
    """Raised when specified model is not available."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama response is invalid or malformed."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request times out."""
    pass


class OllamaClient:
    """
    Client for interacting with Ollama API.

    This class provides a high-level interface for Vision-based AI processing
    using Ollama. It handles model management, structured output, error recovery,
    and provides convenient methods for the assetable pipeline.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """
        Initialize Ollama client.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()

        # Configure ollama client
        self._client = ollama.Client(host=self.config.ai.ollama_host)

        # Track available models
        self._available_models: Optional[List[str]] = None
        self._model_check_time: Optional[datetime] = None

        # Request tracking for debugging
        self._request_count = 0
        self._total_processing_time = 0.0

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            models = self._client.list()
            return True
        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Ollama connection check failed: {e}")
            return False

    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of available models from Ollama.

        Args:
            force_refresh: If True, refresh the model list from server.

        Returns:
            List of available model names.

        Raises:
            OllamaConnectionError: If unable to connect to Ollama server.
        """
        # Check if we need to refresh the model list
        now = datetime.now()
        if (force_refresh or
            self._available_models is None or
            self._model_check_time is None or
            (now - self._model_check_time).total_seconds() > 300):  # 5 minutes cache

            try:
                response = self._client.list()
                self._available_models = [model['name'] for model in response['models']]
                self._model_check_time = now

                if self.config.processing.debug_mode:
                    print(f"Available models: {self._available_models}")

            except Exception as e:
                raise OllamaConnectionError(f"Failed to get model list: {e}")

        return self._available_models or []

    def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure that the specified model is available.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if model is available, False otherwise.

        Raises:
            OllamaModelError: If model is not available and cannot be pulled.
        """
        available_models = self.get_available_models()

        if model_name in available_models:
            return True

        # Try to pull the model
        if self.config.processing.debug_mode:
            print(f"Model {model_name} not found locally. Attempting to pull...")

        try:
            self._client.pull(model_name)
            # Refresh model list
            self.get_available_models(force_refresh=True)
            return model_name in (self._available_models or []) # Add check for None
        except Exception as e:
            raise OllamaModelError(f"Failed to pull model {model_name}: {e}")

    def chat_with_vision(
        self,
        model: str,
        prompt: str,
        image_path: Path,
        response_format: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Union[str, T]:
        """
        Send a chat request with image to Ollama with optional structured output.

        Args:
            model: Name of the model to use.
            prompt: The user prompt.
            image_path: Path to the image file.
            response_format: Pydantic model class for structured output.
            system_prompt: Optional system prompt.
            max_retries: Maximum number of retry attempts.
            timeout: Request timeout in seconds.

        Returns:
            Raw string response or structured Pydantic object.

        Raises:
            OllamaError: If request fails after all retries.
        """
        if not image_path.exists():
            raise OllamaError(f"Image file not found: {image_path}")

        if not image_path.is_file():
            raise OllamaError(f"Image path is not a file: {image_path}")

        # Use config defaults if not specified
        max_retries = max_retries or self.config.ai.max_retries
        timeout = timeout or self.config.ai.timeout_seconds

        # Ensure model is available
        self.ensure_model_available(model)

        # Prepare messages
        messages = []

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # Prepare user message with image
        user_message = {
            'role': 'user',
            'content': prompt,
            'images': [str(image_path.absolute())]
        }
        messages.append(user_message)

        # Prepare request options
        options: Dict[str, Any] = { # Add type hint for options
            'temperature': self.config.ai.temperature,
            'top_p': self.config.ai.top_p,
        }

        # Add structured output format if specified
        if response_format:
            options['format'] = "json" # Instruct Ollama to output JSON

        # Execute request with retries
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()

                if self.config.processing.debug_mode:
                    print(f"Ollama request attempt {attempt + 1}/{max_retries + 1}")
                    print(f"Model: {model}")
                    print(f"Image: {image_path.name}")
                    print(f"Prompt length: {len(prompt)} characters")

                # Make the request
                response = self._client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    # timeout=timeout # Timeout is not a valid parameter for self._client.chat
                )

                processing_time = time.time() - start_time
                self._request_count += 1
                self._total_processing_time += processing_time

                if self.config.processing.debug_mode:
                    print(f"Request completed in {processing_time:.2f} seconds")

                # Extract response content
                if not response or 'message' not in response:
                    raise OllamaResponseError("Invalid response format from Ollama")

                content = response['message'].get('content', '')
                if not content:
                    raise OllamaResponseError("Empty response from Ollama")

                # Parse structured output if requested
                if response_format:
                    try:
                        # Attempt to repair JSON if necessary
                        # content = self._repair_json(content) # Add repair_json method if needed
                        parsed_content = json.loads(content)
                        return response_format(**parsed_content)
                    except (json.JSONDecodeError, ValidationError) as e:
                        raise OllamaResponseError(f"Failed to parse structured output: {e}\nContent: {content}") # Include content in error

                return content

            except Exception as e:
                last_exception = e

                if self.config.processing.debug_mode:
                    print(f"Attempt {attempt + 1} failed: {e}")

                # Don't retry for certain types of errors
                if isinstance(e, (OllamaModelError, OllamaConnectionError, OllamaResponseError)): # Add OllamaResponseError
                    break

                # Wait before retry (exponential backoff)
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + (time.time() % 1) # Add jitter
                    if self.config.processing.debug_mode:
                        print(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

        if last_exception:
            raise OllamaError(f"Request failed after {max_retries + 1} attempts: {last_exception}")
        else:
            # This case should ideally not be reached if logic is correct
            raise OllamaError("Request failed due to an unknown error after all retries.")

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics.
        """
        avg_processing_time = (
            self._total_processing_time / self._request_count
            if self._request_count > 0 else 0
        )

        return {
            'total_requests': self._request_count,
            'total_processing_time': self._total_processing_time,
            'average_processing_time': avg_processing_time,
            'available_models': self._available_models or [],
            'ollama_host': self.config.ai.ollama_host,
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._request_count = 0
        self._total_processing_time = 0.0
