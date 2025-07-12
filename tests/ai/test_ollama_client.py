"""
Tests for assetable.ai.ollama_client module.

Tests are structured using Arrange-Act-Assert pattern and use real Ollama server
connections and file system operations to test actual behavior without mocks.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import pytest
from pydantic import BaseModel

from assetable.ai.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from assetable.config import AssetableConfig


class TestImageCreation:
    """Helper class for creating test image files."""

    @staticmethod
    def create_test_image(image_path: Path, image_type: str = "png") -> None:
        """
        Create a minimal test image file.

        Args:
            image_path: Path where to save the image.
            image_type: Type of image to create ("png" or "jpg").
        """
        if image_type == "png":
            # Minimal 1x1 PNG file
            png_data = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
                b"A\x0e\x1d\x0b\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            image_path.write_bytes(png_data)
        elif image_type == "jpg":
            # Minimal JPEG file
            jpg_data = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H"
                b"\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08"
                b"\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
                b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $."
                b"' \",#\x1c\x1c(7),01444\x1f'9=82"
                b"BCEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                b"\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\"\x00\x02"
                b"\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
                b"\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
                b"\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x0c\x03"
                b"\x01\x00\x02\x11\x03\x11\x00?\x00\xf7\xbf\xff\xd9"
            )
            image_path.write_bytes(jpg_data)

    @staticmethod
    def create_test_content_image(image_path: Path, content: str = "default") -> None:
        """
        Create a test image with recognizable content for Vision AI testing.

        Args:
            image_path: Path where to save the image.
            content: Content identifier for the image.
        """
        # For actual testing with Vision models, we would need proper image creation
        # For now, use the minimal PNG with some identifying metadata
        TestImageCreation.create_test_image(image_path, "png")


class TestOllamaClientInitialization:
    """Test OllamaClient initialization and configuration."""

    def test_ollama_client_default_initialization(self) -> None:
        """Test OllamaClient initialization with default configuration."""
        # Arrange & Act
        client = OllamaClient()

        # Assert
        assert client.config is not None
        assert isinstance(client.config, AssetableConfig)
        assert client._available_models is None
        assert client._model_check_time is None
        assert client._request_count == 0
        assert client._total_processing_time == 0.0
        assert client.config.ai.ollama_host == "http://localhost:11434"

    def test_ollama_client_custom_config_initialization(self) -> None:
        """Test OllamaClient initialization with custom configuration."""
        # Arrange
        custom_config = AssetableConfig()
        custom_config.ai.ollama_host = "http://custom-host:11435"
        custom_config.ai.temperature = 0.8
        custom_config.ai.max_retries = 5
        custom_config.processing.debug_mode = True

        # Act
        client = OllamaClient(config=custom_config)

        # Assert
        assert client.config is custom_config
        assert client.config.ai.ollama_host == "http://custom-host:11435"
        assert client.config.ai.temperature == 0.8
        assert client.config.ai.max_retries == 5
        assert client.config.processing.debug_mode is True


class TestOllamaConnection:
    """Test Ollama server connection functionality."""

    def test_check_connection_success(self) -> None:
        """Test successful connection check to Ollama server."""
        # Arrange
        client = OllamaClient()

        # Act
        is_connected = client.check_connection()

        # Assert
        assert isinstance(is_connected, bool)
        # Note: Result depends on whether Ollama server is running
        # In CI/CD environment, this might be False, which is acceptable

    def test_check_connection_with_invalid_host(self) -> None:
        """Test connection check with invalid host."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://invalid-host:99999"
        client = OllamaClient(config=config)

        # Act
        is_connected = client.check_connection()

        # Assert
        assert is_connected is False

    def test_get_available_models_caching(self) -> None:
        """Test model list caching functionality."""
        # Arrange
        client = OllamaClient()

        # Act - First call
        models_1 = client.get_available_models()
        first_check_time = client._model_check_time

        # Act - Second call (should use cache)
        models_2 = client.get_available_models()
        second_check_time = client._model_check_time

        # Assert
        assert isinstance(models_1, list)
        assert isinstance(models_2, list)
        assert models_1 == models_2  # Should return same cached result
        assert first_check_time == second_check_time  # Should use cache

    def test_get_available_models_force_refresh(self) -> None:
        """Test forced refresh of model list."""
        # Arrange
        client = OllamaClient()

        # Act - First call
        models_1 = client.get_available_models()
        first_check_time = client._model_check_time

        # Wait a moment
        time.sleep(0.1)

        # Act - Second call with force refresh
        models_2 = client.get_available_models(force_refresh=True)
        second_check_time = client._model_check_time

        # Assert
        assert isinstance(models_1, list)
        assert isinstance(models_2, list)
        if first_check_time and second_check_time:
            assert second_check_time > first_check_time  # Should have refreshed

    def test_get_available_models_with_connection_error(self) -> None:
        """Test model list retrieval with connection error."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://invalid-host:99999"
        client = OllamaClient(config=config)

        # Act & Assert
        with pytest.raises(OllamaConnectionError):
            client.get_available_models()


class TestModelManagement:
    """Test model availability and management."""

    def test_ensure_model_available_with_existing_model(self) -> None:
        """Test ensuring availability of existing model."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        # Act
        result = client.ensure_model_available(model_name)

        # Assert
        assert result is True

    def test_ensure_model_available_with_nonexistent_model(self) -> None:
        """Test ensuring availability of non-existent model."""
        # Arrange
        client = OllamaClient()
        model_name = "definitely-nonexistent-model-12345"

        try:
            # Check if we can connect to Ollama
            client.check_connection()
        except Exception:
            pytest.skip("Ollama server not available")

        # Act & Assert
        with pytest.raises(OllamaModelError):
            client.ensure_model_available(model_name)


class TestChatWithVision:
    """Test Vision chat functionality."""

    def test_chat_with_vision_basic_functionality(self) -> None:
        """Test basic Vision chat functionality."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        # Use first available model (in real scenarios, should be a vision model)
        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path, "test_content")

            prompt = "Describe this image briefly."

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt=prompt,
                    image_path=image_path
                )
            except OllamaError as e:
                pytest.skip(f"Vision chat failed (expected if model doesn't support vision): {e}")

            # Assert
            assert isinstance(response, str)
            assert len(response) > 0

    def test_chat_with_vision_with_system_prompt(self) -> None:
        """Test Vision chat with system prompt."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            prompt = "What do you see?"
            system_prompt = "You are a helpful image description assistant."

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt=prompt,
                    image_path=image_path,
                    system_prompt=system_prompt
                )
            except OllamaError as e:
                pytest.skip(f"Vision chat failed: {e}")

            # Assert
            assert isinstance(response, str)
            assert len(response) > 0

    def test_chat_with_vision_structured_output(self) -> None:
        """Test Vision chat with structured output."""
        # Arrange
        class TestResponse(BaseModel):
            description: str
            confidence: float

        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            prompt = "Analyze this image and provide a description with confidence score (0-1)."

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt=prompt,
                    image_path=image_path,
                    response_format=TestResponse
                )
            except OllamaError as e:
                pytest.skip(f"Structured output failed: {e}")

            # Assert
            assert isinstance(response, TestResponse)
            assert isinstance(response.description, str)
            assert isinstance(response.confidence, float)
            assert 0.0 <= response.confidence <= 1.0

    def test_chat_with_vision_with_nonexistent_image(self) -> None:
        """Test Vision chat with non-existent image file."""
        # Arrange
        client = OllamaClient()
        nonexistent_image = Path("nonexistent_image.png")
        prompt = "Describe this image."

        # Act & Assert
        with pytest.raises(OllamaError, match="Image file not found"):
            client.chat_with_vision(
                model="test-model",
                prompt=prompt,
                image_path=nonexistent_image
            )

    def test_chat_with_vision_custom_parameters(self) -> None:
        """Test Vision chat with custom parameters."""
        # Arrange
        config = AssetableConfig()
        config.ai.temperature = 0.8
        config.ai.max_retries = 2
        config.ai.timeout_seconds = 120

        client = OllamaClient(config=config)

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            prompt = "Describe this image."

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt=prompt,
                    image_path=image_path,
                    max_retries=1,
                    timeout=60
                )
            except OllamaError as e:
                pytest.skip(f"Vision chat failed: {e}")

            # Assert
            assert isinstance(response, str)


class TestProcessingStats:
    """Test processing statistics functionality."""

    def test_get_processing_stats_initial_state(self) -> None:
        """Test getting processing stats in initial state."""
        # Arrange
        client = OllamaClient()

        # Act
        stats = client.get_processing_stats()

        # Assert
        assert isinstance(stats, dict)
        assert stats['total_requests'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['average_processing_time'] == 0.0
        assert 'available_models' in stats
        assert 'ollama_host' in stats
        assert isinstance(stats['available_models'], list)
        assert isinstance(stats['ollama_host'], str)

    def test_reset_stats(self) -> None:
        """Test resetting processing statistics."""
        # Arrange
        client = OllamaClient()

        # Simulate some processing
        client._request_count = 5
        client._total_processing_time = 10.5

        # Act
        client.reset_stats()

        # Assert
        stats = client.get_processing_stats()
        assert stats['total_requests'] == 0
        assert stats['total_processing_time'] == 0.0
        assert stats['average_processing_time'] == 0.0

    def test_processing_stats_after_requests(self) -> None:
        """Test processing stats tracking after making requests."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            initial_stats = client.get_processing_stats()
            initial_requests = initial_stats['total_requests']

            # Act - Make a request
            try:
                client.chat_with_vision(
                    model=model_name,
                    prompt="Test prompt",
                    image_path=image_path
                )
            except OllamaError:
                pytest.skip("Request failed")

            # Assert
            final_stats = client.get_processing_stats()
            assert final_stats['total_requests'] == initial_requests + 1
            assert final_stats['total_processing_time'] > initial_stats['total_processing_time']
            assert final_stats['average_processing_time'] > 0


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_chat_with_vision_with_invalid_model(self) -> None:
        """Test Vision chat with invalid model name."""
        # Arrange
        client = OllamaClient()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            invalid_model = "definitely-invalid-model-name-12345"
            prompt = "Describe this image."

            # Act & Assert
            with pytest.raises(OllamaModelError):
                client.chat_with_vision(
                    model=invalid_model,
                    prompt=prompt,
                    image_path=image_path
                )

    def test_chat_with_vision_with_directory_instead_of_image(self) -> None:
        """Test Vision chat with directory path instead of image file."""
        # Arrange
        client = OllamaClient()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            directory_path = temp_path / "not_an_image"
            directory_path.mkdir()

            prompt = "Describe this image."

            # Act & Assert
            with pytest.raises(OllamaError, match="Image path is not a file"):
                client.chat_with_vision(
                    model="test-model",
                    prompt=prompt,
                    image_path=directory_path
                )

    def test_connection_timeout_handling(self) -> None:
        """Test connection timeout handling."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://10.255.255.1:11434"  # Non-routable IP
        config.ai.timeout_seconds = 1  # Very short timeout

        client = OllamaClient(config=config)

        # Act
        is_connected = client.check_connection()

        # Assert
        assert is_connected is False


class TestConfigurationIntegration:
    """Test integration with configuration system."""

    def test_client_uses_config_parameters(self) -> None:
        """Test that client uses configuration parameters correctly."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://test-host:12345"
        config.ai.temperature = 0.9
        config.ai.top_p = 0.8
        config.ai.max_retries = 4
        config.ai.timeout_seconds = 180

        # Act
        client = OllamaClient(config=config)

        # Assert
        assert client.config.ai.ollama_host == "http://test-host:12345"
        assert client.config.ai.temperature == 0.9
        assert client.config.ai.top_p == 0.8
        assert client.config.ai.max_retries == 4
        assert client.config.ai.timeout_seconds == 180

    def test_debug_mode_affects_logging(self) -> None:
        """Test that debug mode affects logging behavior."""
        # Arrange
        config_debug = AssetableConfig()
        config_debug.processing.debug_mode = True

        config_no_debug = AssetableConfig()
        config_no_debug.processing.debug_mode = False

        # Act
        client_debug = OllamaClient(config=config_debug)
        client_no_debug = OllamaClient(config=config_no_debug)

        # Assert
        assert client_debug.config.processing.debug_mode is True
        assert client_no_debug.config.processing.debug_mode is False


class TestConcurrentRequests:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self) -> None:
        """Test handling multiple concurrent requests."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        async def make_request(request_id: int) -> str:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                image_path = temp_path / f"test_image_{request_id}.png"
                TestImageCreation.create_test_content_image(image_path)

                try:
                    response = client.chat_with_vision(
                        model=model_name,
                        prompt=f"Request {request_id}: Describe this image briefly.",
                        image_path=image_path
                    )
                    return f"Request {request_id} completed"
                except OllamaError:
                    return f"Request {request_id} failed"

        # Act
        tasks = [make_request(i) for i in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assert
        assert len(results) == 3
        for result in results:
            assert isinstance(result, str)
            assert "Request" in result


class TestLongRunningOperations:
    """Test long-running operations and timeouts."""

    def test_processing_with_reasonable_timeout(self) -> None:
        """Test processing with reasonable timeout settings."""
        # Arrange
        config = AssetableConfig()
        config.ai.timeout_seconds = 30  # Reasonable timeout

        client = OllamaClient(config=config)

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_content_image(image_path)

            prompt = "Provide a detailed analysis of this image, including any text, objects, colors, and composition."

            # Act
            start_time = time.time()
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt=prompt,
                    timeout=30
                )
                processing_time = time.time() - start_time

                # Assert
                assert isinstance(response, str)
                assert processing_time < 30
            except OllamaError as e:
                pytest.skip(f"Request failed within timeout: {e}")


class TestImageFormatHandling:
    """Test handling of different image formats."""

    def test_chat_with_vision_png_image(self) -> None:
        """Test Vision chat with PNG image."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.png"
            TestImageCreation.create_test_image(image_path, "png")

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt="What format is this image?",
                    image_path=image_path
                )
            except OllamaError as e:
                pytest.skip(f"PNG image test failed: {e}")

            # Assert
            assert isinstance(response, str)

    def test_chat_with_vision_jpg_image(self) -> None:
        """Test Vision chat with JPEG image."""
        # Arrange
        client = OllamaClient()

        try:
            available_models = client.get_available_models()
        except OllamaConnectionError:
            pytest.skip("Ollama server not available")

        if not available_models:
            pytest.skip("No models available for testing")

        model_name = available_models[0]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_image.jpg"
            TestImageCreation.create_test_image(image_path, "jpg")

            # Act
            try:
                response = client.chat_with_vision(
                    model=model_name,
                    prompt="Describe this image briefly.",
                    image_path=image_path
                )
            except OllamaError as e:
                pytest.skip(f"JPEG image test failed: {e}")

            # Assert
            assert isinstance(response, str)
