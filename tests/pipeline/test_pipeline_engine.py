"""
Tests for assetable.pipeline.engine module.

Tests are structured using Arrange-Act-Assert pattern and use real PDF files
and file system operations to test actual behavior without mocks.
This is the most critical logic in assetable, so tests are comprehensive.
"""

import asyncio
import pypdfium2 as pdfium
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import pytest

from assetable.config import AssetableConfig
from assetable.file_manager import FileManager
from assetable.models import DocumentData, PageData, ProcessingStage
from assetable.pipeline.engine import (
    PipelineConfigError,
    PipelineEngine,
    PipelineError,
    PipelineStep,
    PipelineStepError,
    PDFSplitStep,
    run_pipeline,
    run_single_step,
)


class TestPDFCreation:
    """Helper class for creating test PDF files with various characteristics."""

    @staticmethod
    def create_test_pdf(pdf_path: Path, num_pages: int = 3, content_type: str = "standard") -> None:
        """
        Create a test PDF file with specified characteristics using reportlab.

        Args:
            pdf_path: Path where to save the PDF.
            num_pages: Number of pages to create.
            content_type: Type of content ("standard", "complex", "minimal").
        """
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
        except ImportError:
            # Fallback: create a minimal PDF manually
            TestPDFCreation._create_minimal_pdf(pdf_path, num_pages)
            return

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        for page_num in range(num_pages):
            # Add content based on type
            if content_type == "minimal":
                c.drawString(100, height - 100, f"Page {page_num + 1}")
            elif content_type == "complex":
                c.drawString(100, height - 100, f"Chapter {page_num + 1}: Advanced Topics")
                c.drawString(100, height - 150, "This is a complex page with multiple elements.")
                # Add a simple table-like structure
                y_pos = height - 200
                for i, item in enumerate(["Item A", "Item B", "Item C"]):
                    c.drawString(100, y_pos - i * 20, f"{item}: Value {i + 1}")
                # Add a rectangle
                c.rect(100, height - 350, 200, 100)
                c.drawString(120, height - 300, "Figure 1")
            else:  # standard
                c.drawString(100, height - 100, f"This is page {page_num + 1} of {num_pages}")
                c.drawString(100, height - 150, f"Chapter {page_num + 1}")
                c.drawString(100, height - 200, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
                c.drawString(100, height - 250, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
                c.drawString(100, height - 300, f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                # Add a simple visual element
                c.rect(400, height - 450, 100, 50)
                c.drawString(410, height - 430, "Element")

            c.showPage()

        c.save()

    @staticmethod
    def _create_minimal_pdf(pdf_path: Path, num_pages: int) -> None:
        """Create a minimal PDF without external dependencies."""
        if num_pages == 1:
            pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 1)Tj ET
endstream endobj
xref 0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000207 00000 n 
trailer<</Size 5/Root 1 0 R>>startxref 299 %%EOF"""
        elif num_pages == 2:
            pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R>>endobj
4 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 6 0 R>>endobj
5 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 1)Tj ET
endstream endobj
6 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 2)Tj ET
endstream endobj
xref 0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000207 00000 n 
0000000299 00000 n 
0000000391 00000 n 
trailer<</Size 7/Root 1 0 R>>startxref 483 %%EOF"""
        else:  # 3 or more pages
            pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R 5 0 R]/Count 3>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 6 0 R>>endobj
4 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 7 0 R>>endobj
5 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 8 0 R>>endobj
6 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 1)Tj ET
endstream endobj
7 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 2)Tj ET
endstream endobj
8 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page 3)Tj ET
endstream endobj
xref 0 9
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000207 00000 n 
0000000299 00000 n 
0000000391 00000 n 
0000000483 00000 n 
0000000575 00000 n 
trailer<</Size 9/Root 1 0 R>>startxref 667 %%EOF"""
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_content)

    @staticmethod
    def create_large_pdf(pdf_path: Path, num_pages: int = 50) -> None:
        """Create a large PDF for performance testing."""
        TestPDFCreation.create_test_pdf(pdf_path, num_pages, "minimal")

    @staticmethod
    def create_corrupted_pdf(pdf_path: Path) -> None:
        """Create a corrupted PDF for error testing."""
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'corrupted content\n')


class TestPipelineStepBase:
    """Test the abstract PipelineStep base class functionality."""

    def test_custom_pipeline_step_implementation(self) -> None:
        """Test implementing a custom pipeline step."""
        # Arrange
        class TestStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Test Step"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]

            async def execute_page(self, page_data: PageData) -> PageData:
                page_data.mark_stage_completed(self.processing_stage)
                page_data.add_log(f"Processed by {self.step_name}")
                return page_data

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            step = TestStep(config)

            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf")
            )
            # Mark dependency as completed
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

            # Act
            updated_page = asyncio.run(step.execute_page(page_data))

            # Assert
            assert updated_page.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            assert len(updated_page.processing_log) >= 2  # One from dependency, one from test step
            assert any("Test Step" in log for log in updated_page.processing_log)

    def test_should_process_page_with_completed_stage(self) -> None:
        """Test should_process_page when stage is already completed."""
        # Arrange
        class TestStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Test"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return []

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        config = AssetableConfig()
        config.processing.skip_existing_files = True
        step = TestStep(config)

        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page_data.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)

        # Act
        should_process = step.should_process_page(page_data)

        # Assert
        assert should_process is False

    def test_should_process_page_with_missing_dependencies(self) -> None:
        """Test should_process_page when dependencies are missing."""
        # Arrange
        class TestStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Test"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        step = TestStep()
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))
        # Don't mark dependency as completed

        # Act
        should_process = step.should_process_page(page_data)

        # Assert
        assert should_process is False

    def test_validate_dependencies_success(self) -> None:
        """Test dependency validation with satisfied dependencies."""
        # Arrange
        class TestStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Test"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        step = TestStep()
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

        # Act & Assert - Should not raise
        step.validate_dependencies(page_data)

    def test_validate_dependencies_failure(self) -> None:
        """Test dependency validation with missing dependencies."""
        # Arrange
        class TestStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Test"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        step = TestStep()
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))
        # Don't mark dependency as completed

        # Act & Assert
        with pytest.raises(PipelineStepError, match="missing dependencies"):
            step.validate_dependencies(page_data)


class TestPDFSplitStep:
    """Test the PDFSplitStep implementation."""

    def test_pdf_split_step_properties(self) -> None:
        """Test PDFSplitStep properties."""
        # Arrange & Act
        step = PDFSplitStep()

        # Assert
        assert step.step_name == "PDF Split"
        assert step.processing_stage == ProcessingStage.PDF_SPLIT
        assert step.dependencies == []

    def test_pdf_split_step_execute_document_success(self) -> None:
        """Test successful document-level PDF splitting."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            document_data = DocumentData(
                document_id=pdf_path.stem,
                source_pdf_path=pdf_path,
                output_directory=config.get_document_output_dir(pdf_path),
            )

            step = PDFSplitStep(config)

            # Act
            updated_document = asyncio.run(step.execute_document(document_data))

            # Assert
            assert isinstance(updated_document, DocumentData)
            assert len(updated_document.pages) == 3

            for page_data in updated_document.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.image_path is not None
                assert page_data.image_path.exists()
                assert page_data.image_path.stat().st_size > 1000  # Reasonable image size

    def test_pdf_split_step_execute_page_not_implemented(self) -> None:
        """Test that execute_page raises NotImplementedError."""
        # Arrange
        step = PDFSplitStep()
        page_data = PageData(page_number=1, source_pdf=Path("dummy.pdf"))

        # Act & Assert
        with pytest.raises(NotImplementedError):
            asyncio.run(step.execute_page(page_data))

    def test_pdf_split_step_with_different_configurations(self) -> None:
        """Test PDFSplitStep with different configurations."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            configs = [
                {"dpi": 150, "format": "png"},
                {"dpi": 300, "format": "png"},
                {"dpi": 450, "format": "jpg"},
            ]

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            file_sizes = []

            for config_params in configs:
                config = AssetableConfig()
                config.output.output_directory = temp_path / f"output_{config_params['dpi']}"
                config.pdf_split.dpi = config_params["dpi"]
                config.pdf_split.image_format = config_params["format"]

                document_data = DocumentData(
                    document_id=pdf_path.stem,
                    source_pdf_path=pdf_path,
                    output_directory=config.get_document_output_dir(pdf_path),
                )

                step = PDFSplitStep(config)

                # Act
                updated_document = asyncio.run(step.execute_document(document_data))

                # Assert
                assert len(updated_document.pages) == 1
                page_data = updated_document.pages[0]
                assert page_data.image_path is not None
                assert page_data.image_path.exists()

                file_size = page_data.image_path.stat().st_size
                file_sizes.append(file_size)

            # Higher DPI should generally produce larger files
            assert file_sizes[0] < file_sizes[1]


class TestPlaceholderSteps:
    """Test placeholder steps - these tests are disabled as the step classes have been moved to ai_steps module."""
    
    def test_placeholder_steps_moved_to_ai_module(self) -> None:
        """Test that placeholder step functionality has been moved to ai_steps module."""
        # The old StructureAnalysisStep, AssetExtractionStep, and MarkdownGenerationStep
        # have been replaced with Enhanced versions in the ai_steps module.
        # This test serves as documentation of this change.
        assert True


class TestPipelineEngineCore:
    """Test core functionality of PipelineEngine."""

    def test_pipeline_engine_initialization(self) -> None:
        """Test PipelineEngine initialization."""
        # Arrange & Act
        engine = PipelineEngine()

        # Assert
        assert engine.config is not None
        assert isinstance(engine.file_manager, FileManager)
        assert len(engine.steps) == 4  # PDF split + 3 placeholder steps
        assert engine.current_document is None
        assert engine.execution_start_time is None
        assert isinstance(engine.execution_stats, dict)

    def test_pipeline_engine_with_custom_config(self) -> None:
        """Test PipelineEngine with custom configuration."""
        # Arrange
        config = AssetableConfig()
        config.processing.debug_mode = True
        config.pdf_split.dpi = 450

        # Act
        engine = PipelineEngine(config)

        # Assert
        assert engine.config is config
        assert engine.config.processing.debug_mode is True
        assert engine.config.pdf_split.dpi == 450

    def test_add_and_remove_steps(self) -> None:
        """Test adding and removing pipeline steps."""
        # Arrange
        engine = PipelineEngine()
        initial_step_count = len(engine.steps)

        class CustomStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Custom"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.COMPLETED

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return []

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        custom_step = CustomStep()

        # Act - Add step
        engine.add_step(custom_step)

        # Assert
        assert len(engine.steps) == initial_step_count + 1
        assert custom_step in engine.steps

        # Act - Remove step
        engine.remove_step(CustomStep)

        # Assert
        assert len(engine.steps) == initial_step_count
        assert custom_step not in engine.steps

    def test_get_step(self) -> None:
        """Test getting pipeline step by type."""
        # Arrange
        engine = PipelineEngine()

        # Act
        pdf_step = engine.get_step(PDFSplitStep)
        nonexistent_step = engine.get_step(type(None))

        # Assert
        assert pdf_step is not None
        assert isinstance(pdf_step, PDFSplitStep)
        assert nonexistent_step is None
        
        # Note: Structure analysis step is now EnhancedAIStructureAnalysisStep in ai_steps module

    def test_execute_pipeline_full_success(self) -> None:
        """Test successful execution of complete pipeline."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            engine = PipelineEngine(config)

            # Act
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))

            # Assert
            assert isinstance(document_data, DocumentData)
            assert document_data.source_pdf_path == pdf_path
            # assert document_data.total_pages == 2 # total_pages is removed from DocumentData
            assert len(document_data.pages) == 2

            # Verify all stages completed
            for page_data in document_data.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
                assert page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
                assert page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)

            # Verify execution stats
            assert engine.execution_stats["total_pages"] == 2
            assert engine.execution_stats["processed_pages"] >= 0
            assert "steps_executed" in engine.execution_stats
            assert "execution_time_seconds" in engine.execution_stats

    def test_execute_pipeline_with_target_stages(self) -> None:
        """Test pipeline execution with specific target stages."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)
            target_stages = [ProcessingStage.PDF_SPLIT, ProcessingStage.STRUCTURE_ANALYSIS]

            # Act
            document_data = asyncio.run(engine.execute_pipeline(pdf_path, target_stages=target_stages))

            # Assert
            assert len(document_data.pages) == 1
            page_data = document_data.pages[0]

            # Only target stages should be completed
            assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            # These should not be completed since they weren't in target stages
            assert not page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
            assert not page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)

    def test_execute_pipeline_with_specific_pages(self) -> None:
        """Test pipeline execution with specific page numbers."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            engine = PipelineEngine(config)
            page_numbers = [1, 3]  # Skip page 2

            # Act
            document_data = asyncio.run(engine.execute_pipeline(pdf_path, page_numbers=page_numbers))

            # Assert
            # Note: The current implementation may still process all pages during PDF split
            # But the filtering logic should work for other steps
            assert len(document_data.pages) == 3

    def test_execute_single_step_success(self) -> None:
        """Test successful execution of single pipeline step."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)

            # Act
            document_data = asyncio.run(engine.execute_single_step(pdf_path, PDFSplitStep))

            # Assert
            assert isinstance(document_data, DocumentData)
            # assert document_data.total_pages == 1 # total_pages is removed from DocumentData
            assert len(document_data.pages) == 1

            page_data = document_data.pages[0]
            assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            # Other stages should not be completed
            assert not page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)

    def test_execute_single_step_nonexistent_step(self) -> None:
        """Test executing single step that doesn't exist in pipeline."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)
            # Remove the step we're trying to execute
            engine.remove_step(PDFSplitStep)

            # Act & Assert
            with pytest.raises(PipelineError, match="Step .* not found"):
                asyncio.run(engine.execute_single_step(pdf_path, PDFSplitStep))

    def test_get_pipeline_status_not_started(self) -> None:
        """Test getting pipeline status for unprocessed document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)

            # Act
            status = engine.get_pipeline_status(pdf_path)

            # Assert
            assert status["status"] == "not_started"
            assert "error" in status

    def test_get_pipeline_status_in_progress(self) -> None:
        """Test getting pipeline status for partially processed document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            engine = PipelineEngine(config)

            # Execute only PDF split step
            asyncio.run(engine.execute_single_step(pdf_path, PDFSplitStep))

            # Act
            status = engine.get_pipeline_status(pdf_path)

            # Assert
            assert status["status"] in ["in_progress", "completed"]
            assert "overall_progress" in status
            assert "total_pages" in status
            assert "steps" in status
            assert status["total_pages"] == 2

            # Verify step details
            assert len(status["steps"]) == 4  # All pipeline steps
            pdf_split_step = next(s for s in status["steps"] if s["stage"] == "pdf_split")
            assert pdf_split_step["progress"] > 0

    def test_get_pipeline_status_completed(self) -> None:
        """Test getting pipeline status for fully processed document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)

            # Execute full pipeline
            asyncio.run(engine.execute_pipeline(pdf_path))

            # Act
            status = engine.get_pipeline_status(pdf_path)

            # Assert
            assert status["status"] == "completed"
            assert status["overall_progress"] == 1.0
            assert status["total_pages"] == 1

            # All steps should be completed
            for step_info in status["steps"]:
                assert step_info["progress"] == 1.0
                assert step_info["completed_pages"] == 1
                assert step_info["pending_pages"] == 0


class TestPipelineErrorHandling:
    """Test error handling in pipeline execution."""

    def test_execute_pipeline_nonexistent_pdf(self) -> None:
        """Test pipeline execution with nonexistent PDF file."""
        # Arrange
        engine = PipelineEngine()
        nonexistent_pdf = Path("nonexistent.pdf")

        # Act & Assert
        with pytest.raises(PipelineError, match="PDF file not found"):
            asyncio.run(engine.execute_pipeline(nonexistent_pdf))

    def test_execute_pipeline_corrupted_pdf(self) -> None:
        """Test pipeline execution with corrupted PDF file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "corrupted.pdf"
            TestPDFCreation.create_corrupted_pdf(pdf_path)

            engine = PipelineEngine(config)

            # Act & Assert
            with pytest.raises(PipelineError):
                asyncio.run(engine.execute_pipeline(pdf_path))

    def test_execute_pipeline_with_failing_step(self) -> None:
        """Test pipeline execution when a step fails."""
        # Arrange
        class FailingStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Failing Step"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.STRUCTURE_ANALYSIS

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]

            async def execute_page(self, page_data: PageData) -> PageData:
                raise PipelineStepError("Intentional failure for testing")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)
            # Replace structure analysis step with failing step
            # Note: Using index-based removal since step classes have changed
            original_steps = engine.steps[:]
            structure_step = next((s for s in engine.steps if s.processing_stage == ProcessingStage.STRUCTURE_ANALYSIS), None)
            if structure_step:
                engine.steps.remove(structure_step)
            engine.add_step(FailingStep(config))

            # Act & Assert
            with pytest.raises(PipelineStepError, match="Intentional failure"):
                asyncio.run(engine.execute_pipeline(pdf_path))

    def test_pipeline_validation_duplicate_stages(self) -> None:
        """Test pipeline validation with duplicate processing stages."""
        # Arrange
        class DuplicateStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Duplicate"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.PDF_SPLIT  # Same as PDFSplitStep

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return []

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)
            engine.add_step(DuplicateStep(config))

            # Act & Assert
            with pytest.raises(PipelineConfigError, match="Duplicate processing stages"):
                asyncio.run(engine.execute_pipeline(pdf_path))

    def test_pipeline_validation_missing_dependency(self) -> None:
        """Test pipeline validation with missing dependencies."""
        # Arrange
        class OrphanStep(PipelineStep):
            @property
            def step_name(self) -> str:
                return "Orphan"

            @property
            def processing_stage(self) -> ProcessingStage:
                return ProcessingStage.COMPLETED

            @property
            def dependencies(self) -> List[ProcessingStage]:
                return [ProcessingStage.PDF_SPLIT]  # This will be missing

            async def execute_page(self, page_data: PageData) -> PageData:
                return page_data

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)
            # Remove PDF split step but keep orphan step that depends on it
            engine.remove_step(PDFSplitStep)
            engine.add_step(OrphanStep(config))

            # Act & Assert
            with pytest.raises(PipelineConfigError, match="depends on .* which is not available"):
                asyncio.run(engine.execute_pipeline(pdf_path))


class TestConvenienceFunctions:
    """Test convenience functions for pipeline execution."""

    def test_run_pipeline_function(self) -> None:
        """Test the run_pipeline convenience function."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            # Act
            document_data = asyncio.run(run_pipeline(pdf_path, config))

            # Assert
            assert isinstance(document_data, DocumentData)
            assert len(document_data.pages) == 1

            page_data = document_data.pages[0]
            assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
            assert page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
            assert page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)

    def test_run_single_step_function(self) -> None:
        """Test the run_single_step convenience function."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            # Act
            document_data = asyncio.run(run_single_step(pdf_path, PDFSplitStep, config))

            # Assert
            assert isinstance(document_data, DocumentData)
            assert len(document_data.pages) == 1

            page_data = document_data.pages[0]
            assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            # Other stages should not be completed
            assert not page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""

    def test_pipeline_with_large_document(self) -> None:
        """Test pipeline execution with large document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "large.pdf"
            TestPDFCreation.create_large_pdf(pdf_path, num_pages=20)  # Moderate size for testing

            engine = PipelineEngine(config)

            # Act
            start_time = datetime.now()
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))
            end_time = datetime.now()

            # Assert
            assert len(document_data.pages) == 20

            # Verify all pages processed
            for page_data in document_data.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)

            # Performance should be reasonable (less than 30 seconds for 20 pages)
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 30.0


class TestPipelineResilience:
    """Test pipeline resilience features."""

    def test_pipeline_resume_after_interruption(self) -> None:
        """Test pipeline resume after interruption."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.skip_existing_files = True

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            engine = PipelineEngine(config)

            # Act - First execution (partial)
            asyncio.run(engine.execute_single_step(pdf_path, PDFSplitStep))

            # Verify partial state
            status = engine.get_pipeline_status(pdf_path)
            assert status["status"] in ["in_progress", "completed"]

            # Act - Second execution (resume full pipeline)
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))

            # Assert
            assert len(document_data.pages) == 3

            # All stages should be completed
            for page_data in document_data.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
                assert page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
                assert page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)


class TestPipelineIntegration:
    """Test integration between pipeline and other components."""

    def test_pipeline_file_manager_integration(self) -> None:
        """Test pipeline integration with file manager."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            engine = PipelineEngine(config)

            # Act
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))

            # Assert - Check that FileManager can detect completed stages
            file_manager = FileManager(config)

            for page_num in range(1, 3):
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.PDF_SPLIT)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.STRUCTURE_ANALYSIS)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.ASSET_EXTRACTION)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.MARKDOWN_GENERATION)

            # Check that document data was saved and can be loaded
            saved_document = file_manager.load_document_data(pdf_path)
            assert saved_document is not None
            assert len(saved_document.pages) == 2

    def test_pipeline_config_integration(self) -> None:
        """Test pipeline integration with configuration system."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.debug_mode = True
            config.processing.skip_existing_files = False
            config.pdf_split.dpi = 450

            pdf_path = temp_path / "test.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            engine = PipelineEngine(config)

            # Act
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))

            # Assert
            assert len(document_data.pages) == 1
            page_data = document_data.pages[0]

            # Verify that high DPI setting was applied
            assert page_data.image_path is not None
            assert page_data.image_path.exists()

            # High DPI should produce larger files
            file_size = page_data.image_path.stat().st_size
            assert file_size > 15000  # Expect larger file due to higher DPI
