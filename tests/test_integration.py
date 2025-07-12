"""
Integration tests for Assetable system.

Tests the complete pipeline from PDF processing to final Markdown generation,
ensuring all components work together correctly. Uses Arrange-Act-Assert pattern
and real file operations without mocks.
"""

import asyncio
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import fitz  # PyMuPDF
import pytest

from assetable.ai.ollama_client import OllamaClient, OllamaConnectionError
from assetable.ai.vision_processor import VisionProcessor, VisionProcessorError
from assetable.config import AssetableConfig
from assetable.file_manager import FileManager
from assetable.models import DocumentData, PageData, ProcessingStage
from assetable.pipeline.engine import PipelineEngine, run_pipeline
from assetable.pipeline.pdf_splitter import PDFSplitter


class TestDocumentCreation:
    """Helper class for creating comprehensive test documents."""

    @staticmethod
    def create_complex_test_pdf(pdf_path: Path, num_pages: int = 5) -> None:
        """
        Create a complex test PDF with various content types.

        Args:
            pdf_path: Path where to save the PDF.
            num_pages: Number of pages to create.
        """
        doc = fitz.open()  # Create new document

        for page_num in range(num_pages):
            page = doc.new_page(width=595, height=842)  # A4 size

            if page_num == 0:
                # Title page
                page.insert_text((100, 100), "Test Document", fontsize=24, color=(0, 0, 0))
                page.insert_text((100, 150), "Integration Test Sample", fontsize=16, color=(0.2, 0.2, 0.2))
                page.insert_text((100, 200), f"Created: {datetime.now().strftime('%Y-%m-%d')}", fontsize=12)

                # Add a simple figure representation
                rect = fitz.Rect(100, 300, 400, 500)
                page.draw_rect(rect, color=(0.8, 0.8, 0.8), width=2)
                page.insert_text((120, 350), "Figure 1.1: System Overview", fontsize=12, color=(0, 0, 0))
                page.insert_text((120, 380), "Input → Processing → Output", fontsize=10, color=(0.5, 0.5, 0.5))
                page.insert_text((120, 410), "PDF → Analysis → Markdown", fontsize=10, color=(0.5, 0.5, 0.5))

            elif page_num == 1:
                # Content page with table
                page.insert_text((100, 100), "Chapter 1: Data Analysis", fontsize=18, color=(0, 0, 0))
                page.insert_text((100, 140), "This chapter covers data processing methods.", fontsize=12)

                # Create table-like content
                page.insert_text((100, 200), "Table 1.1: Processing Results", fontsize=14, color=(0, 0, 0))
                table_data = [
                    ["Method", "Accuracy", "Speed", "Notes"],
                    ["AI Analysis", "95%", "Fast", "Recommended"],
                    ["Manual Review", "99%", "Slow", "Backup method"],
                    ["Hybrid Approach", "98%", "Medium", "Best practice"],
                ]

                y_pos = 230
                for row in table_data:
                    x_pos = 100
                    for cell in row:
                        page.insert_text((x_pos, y_pos), cell, fontsize=10)
                        x_pos += 120
                    y_pos += 25

                # Draw table borders
                table_rect = fitz.Rect(95, 225, 575, 330)
                page.draw_rect(table_rect, color=(0.5, 0.5, 0.5), width=1)

                # Add reference to next page
                page.insert_text((100, 400), "See Figure 2.1 on page 3 for detailed workflow.", fontsize=12)

            elif page_num == 2:
                # Figure page
                page.insert_text((100, 100), "Chapter 2: Workflow", fontsize=18, color=(0, 0, 0))
                page.insert_text((100, 140), "The following diagram shows the complete process flow.", fontsize=12)

                # Create complex figure
                page.insert_text((100, 180), "Figure 2.1: Complete Workflow", fontsize=14, color=(0, 0, 0))

                # Draw workflow boxes
                boxes = [
                    (150, 220, 200, 250, "Input"),
                    (250, 220, 300, 250, "Parse"),
                    (350, 220, 400, 250, "Analyze"),
                    (450, 220, 500, 250, "Output"),
                ]

                for x1, y1, x2, y2, label in boxes:
                    rect = fitz.Rect(x1, y1, x2, y2)
                    page.draw_rect(rect, color=(0.7, 0.7, 0.7), width=2)
                    # Center text in box
                    text_x = x1 + (x2 - x1) / 2 - len(label) * 3
                    text_y = y1 + (y2 - y1) / 2 + 5
                    page.insert_text((text_x, text_y), label, fontsize=9)

                # Draw arrows between boxes
                arrow_y = 235
                for i in range(len(boxes) - 1):
                    start_x = boxes[i][2]
                    end_x = boxes[i + 1][0]
                    page.draw_line((start_x, arrow_y), (end_x, arrow_y), color=(0, 0, 0), width=2)
                    # Simple arrow head
                    page.draw_line((end_x - 5, arrow_y - 3), (end_x, arrow_y), color=(0, 0, 0), width=2)
                    page.draw_line((end_x - 5, arrow_y + 3), (end_x, arrow_y), color=(0, 0, 0), width=2)

                page.insert_text((100, 300), "This workflow ensures comprehensive document processing.", fontsize=12)
                page.insert_text((100, 330), "Each stage builds upon the previous results.", fontsize=12)

            elif page_num == 3:
                # Results page with image placeholder
                page.insert_text((100, 100), "Chapter 3: Results", fontsize=18, color=(0, 0, 0))
                page.insert_text((100, 140), "The system produces high-quality outputs.", fontsize=12)

                # Image placeholder
                page.insert_text((100, 180), "Image 3.1: Sample Output", fontsize=14, color=(0, 0, 0))
                image_rect = fitz.Rect(100, 200, 400, 350)
                page.draw_rect(image_rect, color=(0.9, 0.9, 0.9), width=1, fill=True)
                page.insert_text((200, 270), "[Sample Screenshot]", fontsize=12, color=(0.5, 0.5, 0.5))

                # Results text
                page.insert_text((100, 380), "Key achievements:", fontsize=12, color=(0, 0, 0))
                page.insert_text((120, 410), "• Accurate text recognition", fontsize=11)
                page.insert_text((120, 430), "• Structured data extraction", fontsize=11)
                page.insert_text((120, 450), "• Clean Markdown output", fontsize=11)

            else:
                # Bibliography/References page
                page.insert_text((100, 100), "References", fontsize=18, color=(0, 0, 0))
                page.insert_text((100, 140), "This document references the following sources:", fontsize=12)

                refs = [
                    "1. AI Document Processing Handbook",
                    "2. PDF Analysis Techniques, 2024",
                    "3. Markdown Best Practices Guide",
                ]

                y_pos = 180
                for ref in refs:
                    page.insert_text((100, y_pos), ref, fontsize=11)
                    y_pos += 25

        doc.save(pdf_path)
        doc.close()

    @staticmethod
    def create_simple_test_pdf(pdf_path: Path, pages: int = 3) -> None:
        """Create a simple test PDF for basic testing."""
        doc = fitz.open()

        for page_num in range(pages):
            page = doc.new_page(width=595, height=842)
            page.insert_text((100, 100), f"Simple Test Page {page_num + 1}", fontsize=16, color=(0, 0, 0))
            page.insert_text((100, 150), "This is a basic test page with minimal content.", fontsize=12)
            page.insert_text((100, 200), f"Page number: {page_num + 1}", fontsize=12)

        doc.save(pdf_path)
        doc.close()


class TestSystemIntegration:
    """Test complete system integration."""

    def test_pdf_splitter_integration(self) -> None:
        """Test PDF splitter integration."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "simple_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=2)

            splitter = PDFSplitter(config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert isinstance(document_data, DocumentData)
            assert len(document_data.pages) == 2

            for page_data in document_data.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.image_path.exists()
                assert page_data.image_path.stat().st_size > 1000

    @pytest.mark.slow
    def test_complete_pipeline_execution_complex_document(self) -> None:
        """Test complete pipeline execution with a complex document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.debug_mode = True

            # Create complex test PDF
            pdf_path = temp_path / "complex_test.pdf"
            TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=3)

            # Act
            start_time = time.time()
            try:
                document_data = asyncio.run(run_pipeline(pdf_path, config))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            processing_time = time.time() - start_time

            # Assert
            assert isinstance(document_data, DocumentData)
            assert document_data.total_pages == 3
            assert len(document_data.pages) == 3

            # Verify processing completed in reasonable time
            assert processing_time < 300  # Example: 5 minutes limit

    @pytest.mark.skip(reason="Asyncio test is not working in the environment.")
    @pytest.mark.asyncio
    async def test_pipeline_resume_from_partial_completion(self) -> None:
        """Test pipeline resume after partial completion."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.skip_existing_files = True

            pdf_path = temp_path / "resume_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=3)

            engine = PipelineEngine(config)

            # Act - First run (partial)
            try:
                from assetable.pipeline.engine import PDFSplitStep
                await engine.execute_single_step(pdf_path, PDFSplitStep)
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Verify partial completion
            status_partial = engine.get_pipeline_status(pdf_path)
            assert status_partial["status"] in ["in_progress", "completed"]

            # Act - Second run (complete)
            try:
                document_data = await engine.execute_pipeline(pdf_path)
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert
            assert document_data.total_pages == 3

            # All stages should be completed
            for page_data in document_data.pages:
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
                assert page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
                assert page_data.is_stage_completed(ProcessingStage.MARKDOWN_GENERATION)

            # Verify final status
            status_final = engine.get_pipeline_status(pdf_path)
            assert status_final["status"] == "completed"
            assert status_final["overall_progress"] == 1.0


@pytest.mark.skip(reason="Ollama connection not available in test environment")
class TestComponentIntegration:
    """Test integration between specific components."""

    def test_file_manager_pipeline_integration(self) -> None:
        """Test integration between FileManager and Pipeline."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "integration_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=2)

            file_manager = FileManager(config)
            engine = PipelineEngine(config)

            # Act - Run pipeline
            try:
                document_data = asyncio.run(engine.execute_pipeline(pdf_path))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert - FileManager can detect all completed stages
            for page_num in range(1, 3):
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.PDF_SPLIT)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.STRUCTURE_ANALYSIS)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.ASSET_EXTRACTION)
                assert file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.MARKDOWN_GENERATION)

            # FileManager can load saved data
            saved_document = file_manager.load_document_data(pdf_path)
            assert saved_document is not None
            assert saved_document.total_pages == 2

            # Individual page data can be loaded
            for page_num in range(1, 3):
                page_data = file_manager.load_page_data(pdf_path, page_num)
                assert page_data is not None
                assert page_data.page_number == page_num

    def test_ai_processing_file_persistence_integration(self) -> None:
        """Test integration between AI processing and file persistence."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "ai_file_test.pdf"
            TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=1)

            # Setup components
            splitter = PDFSplitter(config)
            file_manager = FileManager(config)

            try:
                vision_processor = VisionProcessor(config)
            except VisionProcessorError:
                pytest.skip("AI services not available")

            # Act - Execute each stage manually
            document_data = splitter.split_pdf(pdf_path)
            page_data = document_data.pages[0]

            # Structure analysis
            structure_result = vision_processor.analyze_page_structure(page_data)
            page_data.page_structure = structure_result.page_structure

            # Save and verify structure
            structure_path = file_manager.save_page_structure(pdf_path, 1, page_data.page_structure)
            assert structure_path.exists()

            # Load and verify structure
            loaded_structure = file_manager.load_page_structure(pdf_path, 1)
            assert loaded_structure is not None
            assert loaded_structure.page_number == page_data.page_structure.page_number

            # Asset extraction (if assets found)
            if (page_data.page_structure.tables or
                page_data.page_structure.figures or
                page_data.page_structure.images):

                extraction_result = vision_processor.extract_assets(page_data)
                page_data.extracted_assets = extraction_result.extracted_assets

                # Save assets
                for asset in page_data.extracted_assets:
                    asset_path = file_manager.save_asset_file(pdf_path, 1, asset)
                    assert asset_path.exists()

            # Markdown generation
            markdown_result = vision_processor.generate_markdown(page_data)
            page_data.markdown_content = markdown_result.markdown_content

            # Save and verify markdown
            markdown_path = file_manager.save_markdown_content(pdf_path, 1, page_data.markdown_content)
            assert markdown_path.exists()

            loaded_markdown = file_manager.load_markdown_content(pdf_path, 1)
            assert loaded_markdown == page_data.markdown_content

    def test_configuration_system_integration(self) -> None:
        """Test integration with configuration system across components."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.pdf_split.dpi = 450  # High DPI
            config.processing.debug_mode = True
            config.ai.temperature = 0.2  # Low temperature for consistency

            pdf_path = temp_path / "config_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=1)

            # Act - Initialize all components with config
            splitter = PDFSplitter(config)
            file_manager = FileManager(config)
            engine = PipelineEngine(config)

            try:
                ollama_client = OllamaClient(config)
                vision_processor = VisionProcessor(config)
            except (VisionProcessorError, OllamaConnectionError):
                pytest.skip("AI services not available")

            # Verify configuration propagation
            assert splitter.config.pdf_split.dpi == 450
            assert file_manager.config.processing.debug_mode is True
            assert engine.config.ai.temperature == 0.2
            assert ollama_client.config.ai.temperature == 0.2
            assert vision_processor.config.pdf_split.dpi == 450

            # Execute pipeline and verify configuration effects
            document_data = asyncio.run(engine.execute_pipeline(pdf_path))

            # High DPI should produce larger image files
            page_data = document_data.pages[0]
            assert page_data.image_path is not None
            image_size = page_data.image_path.stat().st_size
            assert image_size > 15000  # Larger due to high DPI


@pytest.mark.skip(reason="Ollama connection not available in test environment")
class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_pipeline_resilience_to_ai_failures(self) -> None:
        """Test pipeline resilience when AI processing fails."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.ai.ollama_host = "http://invalid-host:99999"  # Invalid host

            pdf_path = temp_path / "failure_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=1)

            engine = PipelineEngine(config)

            # Act & Assert
            with pytest.raises(Exception):  # Should fail due to invalid Ollama host
                asyncio.run(engine.execute_pipeline(pdf_path))

    @pytest.mark.asyncio
    async def test_partial_pipeline_recovery(self) -> None:
        """Test recovery from partial pipeline failures."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.skip_existing_files = True

            pdf_path = temp_path / "recovery_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=2)

            engine = PipelineEngine(config)

            # Act - First run PDF split only
            try:
                from assetable.pipeline.engine import PDFSplitStep
                await engine.execute_single_step(pdf_path, PDFSplitStep)
            except Exception as e:
                pytest.skip(f"PDF split failed: {e}")

            # Verify partial completion
            for page_num in range(1, 3):
                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()

            # Second run with valid AI config (if available)
            try:
                # Reset config for AI processing
                config.ai.ollama_host = "http://localhost:11434"
                engine = PipelineEngine(config)

                document_data = await engine.execute_pipeline(pdf_path)

                # All stages should be completed
                for page_data in document_data.pages:
                    assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                    # AI stages may or may not complete depending on Ollama availability

            except (VisionProcessorError, OllamaConnectionError):
                pytest.skip("AI services not available for recovery test")


@pytest.mark.slow
@pytest.mark.skip(reason="Ollama connection not available in test environment")
class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""

    def test_processing_performance_metrics(self) -> None:
        """Test performance metrics across the integrated system."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            config.processing.debug_mode = True

            pdf_path = temp_path / "performance_test.pdf"
            TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=3)

            engine = PipelineEngine(config)

            # Act
            start_time = time.time()
            try:
                document_data = asyncio.run(engine.execute_pipeline(pdf_path))
                total_time = time.time() - start_time
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert
            # Performance benchmarks (adjust based on hardware)
            assert total_time > 0

            # Calculate pages per minute
            pages_per_minute = (3 / total_time) * 60
            assert pages_per_minute > 1  # At least 1 page per minute

    def test_memory_usage_integration(self) -> None:
        """Test memory usage patterns during integration processing."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "memory_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=5)

            engine = PipelineEngine(config)

            # Act
            try:
                document_data = asyncio.run(engine.execute_pipeline(pdf_path))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert
            # Memory should be reasonably managed
            assert len(document_data.pages) == 5

            # Each page should have released unnecessary data
            for page_data in document_data.pages:
                # Image paths should be stored as paths, not loaded data
                assert isinstance(page_data.image_path, Path)
                # Processing logs should be reasonable in size
                if page_data.processing_log:
                    total_log_size = sum(len(log) for log in page_data.processing_log)
                    assert total_log_size < 10000  # Limit log size


@pytest.mark.slow
class TestDataQualityIntegration:
    """Test data quality across integrated components."""

    def test_data_quality_from_input_to_output(self) -> None:
        """Test data quality from PDF input to final Markdown output."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "quality_test.pdf"
            TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=2)

            # Act
            try:
                document_data = asyncio.run(run_pipeline(pdf_path, config))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert - Data Quality Checks
            assert document_data.total_pages == 2

            for page_data in document_data.pages:
                # Image quality
                assert page_data.image_path is not None
                assert page_data.image_path.exists()
                assert page_data.image_path.stat().st_size > 5000  # Reasonable image size

                # Structure analysis quality
                if page_data.page_structure:
                    structure = page_data.page_structure
                    assert structure.page_number == page_data.page_number
                    assert isinstance(structure.has_text, bool)

                    # Validate bounding boxes if assets detected
                    for table in structure.tables:
                        bbox = table.bbox.bbox_2d
                        assert len(bbox) == 4
                        assert bbox[0] < bbox[2]  # x1 < x2
                        assert bbox[1] < bbox[3]  # y1 < y2

                # Markdown quality
                if page_data.markdown_content:
                    markdown = page_data.markdown_content
                    assert isinstance(markdown, str)
                    assert len(markdown) > 0

                    # Should contain some recognizable markdown elements
                    # (This is basic check - actual content depends on AI processing)
                    assert len(markdown) > 20  # Minimum reasonable content length

    def test_file_structure_consistency(self) -> None:
        """Test consistency of generated file structure."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = temp_path / "structure_test.pdf"
            TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=3)

            # Act
            try:
                document_data = asyncio.run(run_pipeline(pdf_path, config))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert - File Structure Consistency
            doc_dir = config.get_document_output_dir(pdf_path)

            # Check directory structure
            required_dirs = [
                "pdfSplitted",
                "pageStructure",
                "markdown",
                "markdown/csv",
                "markdown/images",
                "markdown/figures"
            ]

            for dir_name in required_dirs:
                dir_path = doc_dir / dir_name
                assert dir_path.exists()
                assert dir_path.is_dir()

            # Check file naming consistency
            for page_num in range(1, 4):
                # Image files
                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()
                assert image_path.name == f"page_{page_num:04d}.png"

                # Structure files
                structure_path = config.get_structure_json_path(pdf_path, page_num)
                assert structure_path.exists()
                assert structure_path.name == f"page_{page_num:04d}.json"

                # Markdown files
                markdown_path = config.get_markdown_path(pdf_path, page_num)
                assert markdown_path.exists()
                assert markdown_path.name == f"page_{page_num:04d}.md"

            # Check document-level files
            doc_data_path = doc_dir / "document_data.json"
            assert doc_data_path.exists()

            # Verify document data content
            with open(doc_data_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

            assert doc_data["total_pages"] == 3
            assert "source_pdf" in doc_data
            assert "pages" in doc_data


@pytest.mark.slow
class TestCLIIntegration:
    """Test CLI integration with the complete system."""

    def test_cli_pipeline_command_integration(self) -> None:
        """Test CLI pipeline command with real execution."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "cli_test.pdf"
            TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=2)

            # This would typically be tested with subprocess calls to the CLI
            # For integration testing, we test the underlying functions
            from assetable.config import AssetableConfig

            config = AssetableConfig()
            config.output.output_directory = temp_path

            # Act
            try:
                document_data = asyncio.run(run_pipeline(pdf_path, config))
            except (VisionProcessorError, OllamaConnectionError) as e:
                pytest.skip(f"AI services not available: {e}")

            # Assert
            # CLI equivalent functionality should work
            assert document_data.total_pages == 2

            # Verify the same file structure CLI would create
            doc_dir = config.get_document_output_dir(pdf_path)
            assert doc_dir.exists()

            # Should have created the same files CLI would create
            for page_num in range(1, 3):
                assert config.get_page_image_path(pdf_path, page_num).exists()
                assert config.get_structure_json_path(pdf_path, page_num).exists()
                assert config.get_markdown_path(pdf_path, page_num).exists()


# Pytest configuration for integration tests
pytestmark = [
    pytest.mark.integration,
]
