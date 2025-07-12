"""
Tests for assetable.pipeline.ai_steps module.

Tests are structured using Arrange-Act-Assert pattern and test the integration
of AI processing with the pipeline execution framework.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from assetable.ai.vision_processor import VisionProcessorError
from assetable.config import AssetableConfig
from assetable.models import (
    BoundingBox,
    PageData,
    PageStructure,
    ProcessingStage,
    TableAsset,
)
from assetable.pipeline.ai_steps import (
    AIAssetExtractionStep,
    AIMarkdownGenerationStep,
    AIStructureAnalysisStep,
)
from assetable.pipeline.engine import PipelineStepError


class TestAIStructureAnalysisStep:
    """Test AI-powered structure analysis pipeline step."""

    def test_ai_structure_analysis_step_properties(self) -> None:
        """Test AIStructureAnalysisStep properties."""
        # Arrange
        config = AssetableConfig()

        # Act
        step = AIStructureAnalysisStep(config)

        # Assert
        assert step.step_name == "AI Structure Analysis"
        assert step.processing_stage == ProcessingStage.STRUCTURE_ANALYSIS
        assert step.dependencies == [ProcessingStage.PDF_SPLIT]

    @pytest.mark.asyncio
    async def test_execute_page_success(self) -> None:
        """Test successful execution of structure analysis step."""
        # Arrange
        config = AssetableConfig()
        config.output.output_directory = Path("/tmp/test_output")

        step = AIStructureAnalysisStep(config)

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
            # Mark dependency as completed
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

            # Act
            try:
                updated_page = await step.execute_page(page_data)
            except PipelineStepError as e:
                if "Cannot connect to Ollama" in str(e):
                    pytest.skip("Ollama server not available")
                raise

            # Assert
            assert updated_page.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            assert updated_page.page_structure is not None
            assert len(updated_page.processing_log) >= 2  # Dependency + analysis completion

    @pytest.mark.asyncio
    async def test_execute_page_missing_dependency(self) -> None:
        """Test execution with missing dependency."""
        # Arrange
        config = AssetableConfig()
        step = AIStructureAnalysisStep(config)

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        # Don't mark dependency as completed

        # Act & Assert
        with pytest.raises(PipelineStepError, match="missing dependencies"):
            await step.execute_page(page_data)

    @pytest.mark.asyncio
    async def test_execute_page_vision_processor_error(self) -> None:
        """Test execution when VisionProcessor raises an error."""
        # Arrange
        config = AssetableConfig()
        config.ai.ollama_host = "http://invalid-host:99999"

        step = AIStructureAnalysisStep(config)

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("nonexistent.png")
        )
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

        # Act & Assert
        with pytest.raises(PipelineStepError):
            await step.execute_page(page_data)


class TestAIAssetExtractionStep:
    """Test AI-powered asset extraction pipeline step."""

    def test_ai_asset_extraction_step_properties(self) -> None:
        """Test AIAssetExtractionStep properties."""
        # Arrange
        config = AssetableConfig()

        # Act
        step = AIAssetExtractionStep(config)

        # Assert
        assert step.step_name == "AI Asset Extraction"
        assert step.processing_stage == ProcessingStage.ASSET_EXTRACTION
        assert step.dependencies == [ProcessingStage.STRUCTURE_ANALYSIS]

    @pytest.mark.asyncio
    async def test_execute_page_success(self) -> None:
        """Test successful execution of asset extraction step."""
        # Arrange
        config = AssetableConfig()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config.output.output_directory = temp_path

            step = AIAssetExtractionStep(config)

            image_path = temp_path / "test_page.png"
            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            # Create page data with structure analysis completed
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
                ]
            )

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
            page_data.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)

            # Act
            try:
                updated_page = await step.execute_page(page_data)
            except PipelineStepError as e:
                if "Cannot connect to Ollama" in str(e):
                    pytest.skip("Ollama server not available")
                raise

            # Assert
            assert updated_page.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
            assert isinstance(updated_page.extracted_assets, list)

    @pytest.mark.asyncio
    async def test_execute_page_missing_structure(self) -> None:
        """Test execution without prior structure analysis."""
        # Arrange
        config = AssetableConfig()
        step = AIAssetExtractionStep(config)

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        # Don't mark structure analysis as completed

        # Act & Assert
        with pytest.raises(PipelineStepError, match="missing dependencies"):
            await step.execute_page(page_data)


class TestAIMarkdownGenerationStep:
    """Test AI-powered Markdown generation pipeline step."""

    def test_ai_markdown_generation_step_properties(self) -> None:
        """Test AIMarkdownGenerationStep properties."""
        # Arrange
        config = AssetableConfig()

        # Act
        step = AIMarkdownGenerationStep(config)

        # Assert
        assert step.step_name == "AI Markdown Generation"
        assert step.processing_stage == ProcessingStage.MARKDOWN_GENERATION
        assert step.dependencies == [ProcessingStage.ASSET_EXTRACTION]

    @pytest.mark.asyncio
    async def test_execute_page_success(self) -> None:
        """Test successful execution of Markdown generation step."""
        # Arrange
        config = AssetableConfig()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config.output.output_directory = temp_path

            step = AIMarkdownGenerationStep(config)

            image_path = temp_path / "test_page.png"
            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            # Create page data with completed dependencies
            page_structure = PageStructure(
                page_number=1,
                has_text=True,
                text_content="Chapter 1: Introduction\n\nThis is test content."
            )

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path,
                page_structure=page_structure,
                extracted_assets=[]
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
            page_data.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            page_data.mark_stage_completed(ProcessingStage.ASSET_EXTRACTION)

            # Act
            try:
                updated_page = await step.execute_page(page_data)
            except PipelineStepError as e:
                if "Cannot connect to Ollama" in str(e):
                    pytest.skip("Ollama server not available")
                raise

            # Assert
            assert updated_page.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)
            assert updated_page.markdown_content is not None
            assert isinstance(updated_page.markdown_content, str)
            assert len(updated_page.markdown_content) > 0

    @pytest.mark.asyncio
    async def test_execute_page_missing_dependencies(self) -> None:
        """Test execution with missing dependencies."""
        # Arrange
        config = AssetableConfig()
        step = AIMarkdownGenerationStep(config)

        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        # Don't mark asset extraction as completed

        # Act & Assert
        with pytest.raises(PipelineStepError, match="missing dependencies"):
            await step.execute_page(page_data)


class TestAIStepsIntegration:
    """Test integration between AI pipeline steps."""

    @pytest.mark.asyncio
    async def test_complete_ai_pipeline_workflow(self) -> None:
        """Test complete workflow through all AI pipeline steps."""
        # Arrange
        config = AssetableConfig()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config.output.output_directory = temp_path

            structure_step = AIStructureAnalysisStep(config)
            extraction_step = AIAssetExtractionStep(config)
            markdown_step = AIMarkdownGenerationStep(config)

            image_path = temp_path / "test_page.png"
            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path, "complete_workflow")

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

            # Act - Execute steps sequentially
            try:
                # Step 1: Structure Analysis
                page_data = await structure_step.execute_page(page_data)

                # Step 2: Asset Extraction
                page_data = await extraction_step.execute_page(page_data)

                # Step 3: Markdown Generation
                page_data = await markdown_step.execute_page(page_data)

            except PipelineStepError as e:
                if "Cannot connect to Ollama" in str(e):
                    pytest.skip("Ollama server not available")
                raise

            # Assert
            assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            assert page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
            assert page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)
            assert page_data.page_structure is not None
            assert page_data.markdown_content is not None
            assert len(page_data.processing_log) >= 6  # Each step adds 2 log entries minimum

    @pytest.mark.asyncio
    async def test_ai_steps_with_file_persistence(self) -> None:
        """Test AI steps with file persistence integration."""
        # Arrange
        config = AssetableConfig()

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config.output.output_directory = temp_path

            structure_step = AIStructureAnalysisStep(config)

            image_path = temp_path / "test_page.png"
            from tests.test_ollama_client import TestImageCreation
            TestImageCreation.create_test_content_image(image_path)

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=image_path
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

            # Act
            try:
                updated_page = await structure_step.execute_page(page_data)
            except PipelineStepError as e:
                if "Cannot connect to Ollama" in str(e):
                    pytest.skip("Ollama server not available")
                raise

            # Assert - Check that files were created
            assert updated_page.structure_json_path is not None
            assert updated_page.structure_json_path.exists()

            # Verify file content
            import json
            with open(updated_page.structure_json_path, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)

            assert structure_data["page_number"] == 1
            assert "has_text" in structure_data

    def test_ai_steps_configuration_usage(self) -> None:
        """Test that AI steps use configuration correctly."""
        # Arrange
        config = AssetableConfig()
        config.ai.structure_analysis_model = "custom-structure-model"
        config.ai.asset_extraction_model = "custom-extraction-model"
        config.ai.markdown_generation_model = "custom-markdown-model"
        config.processing.debug_mode = True

        # Act
        structure_step = AIStructureAnalysisStep(config)
        extraction_step = AIAssetExtractionStep(config)
        markdown_step = AIMarkdownGenerationStep(config)

        # Assert
        assert structure_step.config.ai.structure_analysis_model == "custom-structure-model"
        assert extraction_step.config.ai.asset_extraction_model == "custom-extraction-model"
        assert markdown_step.config.ai.markdown_generation_model == "custom-markdown-model"
        assert structure_step.config.processing.debug_mode is True
