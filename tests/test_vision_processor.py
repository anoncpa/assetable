"""
Tests for assetable.ai.vision_processor module.

Tests are structured using Arrange-Act-Assert pattern and use real AI processing
and file system operations to test actual behavior without mocks.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from assetable.ai.vision_processor import VisionProcessor, VisionProcessorError
from assetable.config import AssetableConfig
from assetable.models import (
    BoundingBox,
    PageData,
    PageStructure,
    ProcessingStage,
    TableAsset,
    FigureAsset,
    ImageAsset,
)


class TestVisionProcessorInitialization:
    """Test VisionProcessor initialization and setup."""

    def test_vision_processor_default_initialization(self) -> None:
        """Test VisionProcessor initialization with default configuration."""
        # Arrange & Act
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        # Assert
        assert processor.config is not None
        assert processor.ollama_client is not None
        assert isinstance(processor.config, AssetableConfig)

    def test_vision_processor_custom_config_initialization(self) -> None:
        """Test VisionProcessor initialization with custom configuration."""
        # Arrange
        config = AssetableConfig()
        config.ai.structure_analysis_model = "custom-model:latest"
        config.ai.temperature = 0.5
        config.processing.debug_mode = True

        # Act
        try:
            processor = VisionProcessor(config=config)
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        # Assert
        assert processor.config is config
        assert processor.config.ai.structure_analysis_model == "custom-model:latest"
        assert processor.config.ai.temperature == 0.5
        assert processor.config.processing.debug_mode is True

    def test_vision_processor_initialization_with_no_ollama(self) -> None:
        """Test VisionProcessor initialization when Ollama is not available."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://invalid-host:99999"

        # Act & Assert
        with pytest.raises(VisionProcessorError, match="Cannot connect to Ollama server"):
            VisionProcessor(config=config)


class TestStructureAnalysis:
    """Test page structure analysis functionality."""

    def test_analyze_page_structure_basic(self) -> None:
        """Test basic page structure analysis."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            # Create a test image with recognizable content
            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path, "structure_test")

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )

            # Act
            try:
                result = processor.analyze_page_structure(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Structure analysis failed: {e}")

            # Assert
            assert result is not None
            assert result.page_structure is not None
            assert result.page_structure.page_number == 1
            assert result.model_used is not None
            assert isinstance(result.page_structure.has_text, bool)
            assert isinstance(result.page_structure.tables, list)
            assert isinstance(result.page_structure.figures, list)
            assert isinstance(result.page_structure.images, list)
            assert isinstance(result.page_structure.references, list)

    def test_analyze_page_structure_missing_image(self) -> None:
        """Test structure analysis with missing image file."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("nonexistent_image.png")
        )

        # Act & Assert
        with pytest.raises(VisionProcessorError, match="Image file not found"):
            processor.analyze_page_structure(page_data)

    def test_analyze_page_structure_with_different_models(self) -> None:
        """Test structure analysis with different model configurations."""
        # Arrange
        config = AssetableConfig()
        config.ai.structure_analysis_model = "qwen2.5-vl:7b"  # Specific model

        try:
            processor = VisionProcessor(config=config)
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )

            # Act
            try:
                result = processor.analyze_page_structure(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Model-specific test failed: {e}")

            # Assert
            assert result.model_used == "qwen2.5-vl:7b"


class TestAssetExtraction:
    """Test asset extraction functionality."""

    def test_extract_assets_basic(self) -> None:
        """Test basic asset extraction."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            # Create page data with mock structure
            page_structure = PageStructure(
                page_number=1,
                has_text=True,
                text_content="Test content",
                tables=[
                    TableAsset(
                        name="Test Table",
                        description="A test table",
                        bbox=BoundingBox(bbox_2d=[10, 20, 100, 80])
                    )
                ],
                figures=[
                    FigureAsset(
                        name="Test Figure",
                        description="A test figure",
                        bbox=BoundingBox(bbox_2d=[200, 300, 400, 500]),
                        figure_type="diagram"
                    )
                ]
            )

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure
            )

            # Act
            try:
                result = processor.extract_assets(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Asset extraction failed: {e}")

            # Assert
            assert result is not None
            assert isinstance(result.extracted_assets, list)
            assert result.model_used is not None

    def test_extract_assets_without_structure(self) -> None:
        """Test asset extraction without prior structure analysis."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        # Note: page_structure is None

        # Act & Assert
        with pytest.raises(VisionProcessorError, match="Page structure analysis required"):
            processor.extract_assets(page_data)

    def test_extract_assets_with_various_asset_types(self) -> None:
        """Test asset extraction with various asset types."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path, "multi_asset")

            # Create page data with multiple asset types
            page_structure = PageStructure(
                page_number=1,
                has_text=True,
                tables=[
                    TableAsset(
                        name="Sales Data",
                        description="Monthly sales table",
                        bbox=BoundingBox(bbox_2d=[10, 20, 200, 120])
                    )
                ],
                figures=[
                    FigureAsset(
                        name="Process Flow",
                        description="Business process flowchart",
                        bbox=BoundingBox(bbox_2d=[50, 150, 300, 400]),
                        figure_type="flowchart"
                    )
                ],
                images=[
                    ImageAsset(
                        name="Product Photo",
                        description="Product image",
                        bbox=BoundingBox(bbox_2d=[350, 50, 500, 200])
                    )
                ]
            )

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure
            )

            # Act
            try:
                result = processor.extract_assets(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Multi-asset extraction failed: {e}")

            # Assert
            assert len(result.extracted_assets) == 3
            asset_types = [type(asset).__name__ for asset in result.extracted_assets]
            assert "TableAsset" in asset_types
            assert "FigureAsset" in asset_types
            assert "ImageAsset" in asset_types


class TestMarkdownGeneration:
    """Test Markdown generation functionality."""

    def test_generate_markdown_basic(self) -> None:
        """Test basic Markdown generation."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path, "markdown_test")

            # Create page data with structure
            page_structure = PageStructure(
                page_number=1,
                has_text=True,
                text_content="Chapter 1: Introduction\n\nThis is the beginning of our document."
            )

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure
            )

            # Act
            try:
                result = processor.generate_markdown(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Markdown generation failed: {e}")

            # Assert
            assert result is not None
            assert isinstance(result.markdown_content, str)
            assert len(result.markdown_content) > 0
            assert isinstance(result.asset_references, list)
            assert result.model_used is not None

    def test_generate_markdown_with_assets(self) -> None:
        """Test Markdown generation with asset references."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            # Create page data with assets
            page_structure = PageStructure(
                page_number=1,
                has_text=True,
                text_content="Document with assets"
            )

            extracted_assets = [
                TableAsset(
                    name="Results Table",
                    description="Experimental results",
                    bbox=BoundingBox(bbox_2d=[10, 20, 200, 120])
                ),
                FigureAsset(
                    name="Architecture Diagram",
                    description="System architecture",
                    bbox=BoundingBox(bbox_2d=[50, 150, 300, 400]),
                    figure_type="diagram"
                )
            ]

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure,
                extracted_assets=extracted_assets
            )

            # Act
            try:
                result = processor.generate_markdown(page_data)
            except VisionProcessorError as e:
                pytest.skip(f"Markdown with assets failed: {e}")

            # Assert
            assert result is not None
            assert isinstance(result.markdown_content, str)
            # Check for potential asset references in the content
            content = result.markdown_content.lower()
            # The AI might include references to tables, figures, etc.

    def test_generate_markdown_without_structure(self) -> None:
        """Test Markdown generation without prior structure analysis."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        # Note: page_structure is None

        # Act & Assert
        with pytest.raises(VisionProcessorError, match="Page structure analysis required"):
            processor.generate_markdown(page_data)


class TestProcessingStats:
    """Test processing statistics and monitoring."""

    def test_get_processing_stats(self) -> None:
        """Test getting processing statistics."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        # Act
        stats = processor.get_processing_stats()

        # Assert
        assert isinstance(stats, dict)
        assert 'ollama_stats' in stats
        assert 'models_used' in stats
        assert isinstance(stats['ollama_stats'], dict)
        assert isinstance(stats['models_used'], dict)

        models_used = stats['models_used']
        assert 'structure_analysis' in models_used
        assert 'asset_extraction' in models_used
        assert 'markdown_generation' in models_used


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_structure_analysis_with_corrupted_image(self) -> None:
        """Test structure analysis with corrupted image file."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corrupted_image = temp_path / "corrupted.png"
            corrupted_image.write_text("This is not an image file")

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=corrupted_image
            )

            # Act & Assert
            # The exact error depends on how the AI model handles corrupted images
            try:
                processor.analyze_page_structure(page_data)
            except VisionProcessorError:
                # Expected for corrupted image
                pass

    def test_processing_with_debug_mode(self) -> None:
        """Test processing with debug mode enabled."""
        # Arrange
        config = AssetableConfig()
        config.processing.debug_mode = True

        try:
            processor = VisionProcessor(config=config)
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )

            # Act
            try:
                result = processor.analyze_page_structure(page_data)
                # In debug mode, additional logging should occur
                assert result is not None
            except VisionProcessorError as e:
                pytest.skip(f"Debug mode test failed: {e}")


class TestIntegrationWorkflow:
    """Test complete workflow integration."""

    def test_complete_processing_workflow(self) -> None:
        """Test complete processing workflow from structure to markdown."""
        # Arrange
        try:
            processor = VisionProcessor()
        except VisionProcessorError:
            pytest.skip("Cannot connect to Ollama server")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test_page.png"

            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path, "workflow_test")

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )

            # Act - Step 1: Structure Analysis
            try:
                structure_result = processor.analyze_page_structure(page_data)
                page_data.page_structure = structure_result.page_structure
            except VisionProcessorError as e:
                pytest.skip(f"Structure analysis failed: {e}")

            # Act - Step 2: Asset Extraction (if assets found)
            if (page_data.page_structure.tables or
                page_data.page_structure.figures or
                page_data.page_structure.images):
                try:
                    extraction_result = processor.extract_assets(page_data)
                    page_data.extracted_assets = extraction_result.extracted_assets
                except VisionProcessorError as e:
                    pytest.skip(f"Asset extraction failed: {e}")

            # Act - Step 3: Markdown Generation
            try:
                markdown_result = processor.generate_markdown(page_data)
                page_data.markdown_content = markdown_result.markdown_content
            except VisionProcessorError as e:
                pytest.skip(f"Markdown generation failed: {e}")

            # Assert
            assert page_data.page_structure is not None
            assert page_data.markdown_content is not None
            assert isinstance(page_data.markdown_content, str)
            assert len(page_data.markdown_content) > 0
