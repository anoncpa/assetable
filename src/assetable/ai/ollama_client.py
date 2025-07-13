# src/assetable/ai/ollama_client.py
"""
Ollama client for Assetable.

This module provides a client for interacting with Ollama API
for Vision-based AI processing. It handles model communication,
structured output, and error recovery.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import ollama
from ollama import ChatResponse
from pydantic import BaseModel, ValidationError

from ..config import AssetableConfig, get_config

T = TypeVar("T", bound=BaseModel)


# Custom exceptions



class OllamaError(Exception):
    """Base exception for Ollama operations."""


class OllamaConnectionError(OllamaError):
    """Raised when connection to Ollama fails."""


class OllamaModelError(OllamaError):
    """Raised when specified model is not available."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama response is invalid or malformed."""


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request times out."""



# Ollama client wrapper



class OllamaClient:
    """
    Client for interacting with Ollama API.

    This class provides a high-level interface for Vision-based AI processing
    using Ollama. It handles model management, structured output, error
    recovery, and provides convenient methods for the assetable pipeline.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        self.config: AssetableConfig = config or get_config()

        # Low-level Ollama client
        self._client: ollama.Client = ollama.Client(host=self.config.ai.ollama_host)

        # Cached model list
        self._available_models: Optional[List[str]] = None
        self._model_check_time: Optional[datetime] = None

        # Stats
        self._request_count: int = 0
        self._total_processing_time: float = 0.0

    # ---------------------------------------------------------------------
    # Connection / model helpers
    # ---------------------------------------------------------------------

    def check_connection(self) -> bool:
        """
        Check if Ollama server is reachable.

        Returns:
            True if reachable, otherwise False.
        """
        try:
            _ = self._client.list()  # 接続確認のみ
            return True
        except Exception as exc:  # pylint: disable=broad-except
            if self.config.processing.debug_mode:
                print(f"Ollama connection check failed: {exc}")
            return False

    def get_available_models(self, *, force_refresh: bool = False) -> List[str]:
        """
        Return a cached list of available model names.
        Cache is refreshed every 5 minutes or when ``force_refresh`` is True.
        """
        now = datetime.now()
        needs_refresh = (
            force_refresh
            or self._available_models is None
            or self._model_check_time is None
            or (now - self._model_check_time).total_seconds() > 300
        )
        if needs_refresh:
            try:
                # ListResponse オブジェクトを正しく処理
                list_response = self._client.list()

                # レスポンスが辞書の場合とオブジェクトの場合を両方処理
                if isinstance(list_response, dict):
                    models_data = list_response.get("models", [])
                else:
                    # ListResponse オブジェクトから models を取得
                    models_data = getattr(list_response, "models", [])

                # モデル名を正しく取得（'model'キーを使用）
                self._available_models = []
                for model in models_data:
                    if isinstance(model, dict):
                        # 'model'キーを最初に試し、なければ'name'を試す
                        model_name = model.get("model") or model.get("name")
                        if model_name:
                            self._available_models.append(model_name)
                    else:
                        # オブジェクトの場合は属性でアクセス
                        model_name = getattr(model, "model", None) or getattr(model, "name", None)
                        if model_name:
                            self._available_models.append(model_name)

                self._model_check_time = now
                if self.config.processing.debug_mode:
                    print(f"Available models: {self._available_models}")
                    print(f"Raw response structure: {type(list_response)}")
                    if models_data:
                        print(f"Sample model data: {models_data[0] if models_data else 'No models'}")

            except Exception as exc:  # pylint: disable=broad-except
                raise OllamaConnectionError(f"Failed to get model list: {exc}") from exc
        return self._available_models or []

    def ensure_model_available(self, model_name: str) -> bool:
        """
        Ensure that the requested model is present locally; pull if necessary.
        """
        if model_name in self.get_available_models():
            return True

        if self.config.processing.debug_mode:
            print(f"Model {model_name} not found locally. Attempting to pull…")

        try:
            self._client.pull(model_name)
            self.get_available_models(force_refresh=True)
            return model_name in (self._available_models or [])
        except Exception as exc:  # pylint: disable=broad-except
            raise OllamaModelError(f"Failed to pull model {model_name}: {exc}") from exc

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def chat_with_vision(
        self,
        *,
        model: str,
        prompt: str,
        image_path: Path,
        response_format: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Union[str, T]:
        """
        Call Ollama-Vision with an image and prompt.

        Args:
            model: Ollama model name.
            prompt: User prompt.
            image_path: Path to the input image file.
            response_format: Pydantic model class for structured output.
            system_prompt: Optional system prompt.
            max_retries: Override default retry count.
            timeout: **Not currently supported by ollama-python**.

        Returns:
            Raw string or instantiated ``response_format`` model.
        """
        if not image_path.is_file():
            raise OllamaError(f"Image file not found: {image_path}")

        # Defaults
        max_retries = max_retries or self.config.ai.max_retries
        timeout = timeout or self.config.ai.timeout_seconds  # noqa: F841 — reserved for future use

        # Ensure model is ready
        self.ensure_model_available(model)

        # Build message list with exact types from Ollama definition
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message: Dict[str, Any] = {
            "role": "user",
            "content": prompt,
            "images": [str(image_path.resolve())],
        }
        messages.append(user_message)

        # Ollama options with exact types
        options: Dict[str, Any] = {
            "temperature": self.config.ai.temperature,
            "top_p": self.config.ai.top_p,
        }

        # Format parameter with exact Literal types
        format_param: Optional[Union[Literal["", "json"]]] = None
        if response_format:
            format_param = "json"

        # Retry loop
        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                if self.config.processing.debug_mode:
                    print(
                        f"Ollama request attempt {attempt + 1}/{max_retries + 1} "
                        f"(model={model}, image={image_path.name})"
                    )

                # 型安全な chat メソッド呼び出し
                # 正確な型キャストでオーバーロード解決を確実にする
                chat_response: ChatResponse = self._client.chat(
                    model=model,
                    messages=cast(Sequence[Mapping[str, Any]], messages),
                    format=format_param,
                    options=cast(Mapping[str, Any], options),
                    stream=cast(Literal[False], False),  # 明示的にLiteral[False]としてキャスト
                )

                # Stats
                elapsed = time.time() - start
                self._request_count += 1
                self._total_processing_time += elapsed

                # ChatResponse オブジェクトからの安全なデータ取得
                if not hasattr(chat_response, "message") or not chat_response.message:
                    raise OllamaResponseError("Invalid response format from Ollama")

                message = chat_response.message
                if not message:
                    raise OllamaResponseError("No message in response from Ollama")

                # 型安全な content 取得
                content: str = ""
                if hasattr(message, "content") and message.content is not None:
                    content = str(message.content)
                else:
                    content = str(message)

                if not content:
                    raise OllamaResponseError("Empty response from Ollama")

                # Structured output
                if response_format:
                    try:
                        parsed = json.loads(content)
                        return response_format(**parsed)
                    except (json.JSONDecodeError, ValidationError) as exc:
                        raise OllamaResponseError(
                            f"Failed to parse structured output: {exc}\nContent: {content}"
                        ) from exc

                return content

            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                if self.config.processing.debug_mode:
                    print(f"Attempt {attempt + 1} failed: {exc}")

                # Fatal errors: do not retry
                if isinstance(exc, (OllamaModelError, OllamaConnectionError, OllamaResponseError)):
                    break

                # Back-off
                if attempt < max_retries:
                    wait = 2 ** attempt + (time.time() % 1)  # jitter
                    time.sleep(wait)

        raise OllamaError(f"Request failed after {max_retries + 1} attempts: {last_error}") from last_error

    # ---------------------------------------------------------------------
    # Stats helpers
    # ---------------------------------------------------------------------

    def get_processing_stats(self) -> Dict[str, Any]:
        avg = self._total_processing_time / self._request_count if self._request_count else 0.0
        return {
            "total_requests": self._request_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg,
            "available_models": self._available_models or [],
            "ollama_host": self.config.ai.ollama_host,
        }

    def reset_stats(self) -> None:
        self._request_count = 0
        self._total_processing_time = 0.0
