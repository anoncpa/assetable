"""
Tests for assetable.pipeline.pdf_splitter module.

Tests are structured using Arrange-Act-Assert pattern and use real PDF files
and file system operations to test actual behavior without mocks.
"""

import pypdfium2 as pdfium
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import pytest

from assetable.config import AssetableConfig
from assetable.file_manager import FileManager
from assetable.models import DocumentData, PageData, ProcessingStage
from assetable.pdf.pdf_splitter import (
    ImageConversionError,
    PDFCorruptedError,
    PDFNotFoundError,
    PDFSplitter,
    PDFSplitterError,
    split_pdf_cli,
)


class TestPDFCreation:
    """Helper class for creating test PDF files."""

    @staticmethod
    def create_test_pdf(pdf_path: Path, num_pages: int = 3) -> None:
        """
        Create a test PDF file with specified number of pages.

        Args:
            pdf_path: Path where to save the PDF.
            num_pages: Number of pages to create.
        """
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
        except ImportError:
            # Fallback: create a minimal PDF manually
            pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT/F1 12 Tf 100 700 Td(Test Page)Tj ET
endstream endobj
xref 0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000207 00000 n 
trailer<</Size 5/Root 1 0 R>>startxref 295 %%EOF"""
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
            return

        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        for page_num in range(num_pages):
            # Add content to the page
            text = f"This is page {page_num + 1} of {num_pages}"
            c.drawString(100, height - 100, text)
            
            # Add some additional content to make it realistic
            content_lines = [
                f"Chapter {page_num + 1}",
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco.",
                f"Page created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]

            for i, line in enumerate(content_lines):
                c.drawString(100, height - 150 - i * 20, line)

            # Add a simple rectangle as visual element
            c.rect(400, height - 450, 100, 50)
            c.drawString(410, height - 430, "Box")

            c.showPage()

        c.save()

    @staticmethod
    def create_corrupted_pdf(pdf_path: Path) -> None:
        """
        Create a corrupted PDF file for testing error handling.

        Args:
            pdf_path: Path where to save the corrupted PDF.
        """
        # Create a file with PDF header but corrupted content
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n')  # Valid PDF header
            f.write(b'corrupted content that is not valid PDF\n')
            f.write(b'more corrupted data\n')

    @staticmethod
    def create_empty_pdf(pdf_path: Path) -> None:
        """
        Create an empty PDF file with no pages.

        Args:
            pdf_path: Path where to save the empty PDF.
        """
        # Create a file that pypdfium2 will likely interpret as empty or corrupted
        # in a way that leads to 0 pages, or fails to open (caught as PDFCorruptedError).
        with open(pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%EOF\n') # Minimal, but likely invalid for page count
        # Previous version:
        # doc = fitz.open()  # Create new document with no pages
        # doc.save(str(pdf_path)) # This fails as PyMuPDF prevents saving 0-page docs
        # doc.close()


class TestPDFSplitterInitialization:
    """Test PDFSplitter initialization and configuration."""

    def test_pdf_splitter_default_initialization(self) -> None:
        """Test PDFSplitter initialization with default configuration."""
        # Arrange & Act
        splitter = PDFSplitter()

        # Assert
        assert splitter.config is not None
        assert isinstance(splitter.config, AssetableConfig)
        assert isinstance(splitter.file_manager, FileManager)
        assert splitter.config.pdf_split.dpi == 300  # Default DPI
        assert splitter.config.pdf_split.image_format == "png"  # Default format

    def test_pdf_splitter_custom_config_initialization(self) -> None:
        """Test PDFSplitter initialization with custom configuration."""
        # Arrange
        custom_config = AssetableConfig()
        custom_config.pdf_split.dpi = 450
        custom_config.pdf_split.image_format = "jpg"
        custom_config.processing.debug_mode = True

        # Act
        splitter = PDFSplitter(config=custom_config)

        # Assert
        assert splitter.config is custom_config
        assert splitter.config.pdf_split.dpi == 450
        assert splitter.config.pdf_split.image_format == "jpg"
        assert splitter.config.processing.debug_mode is True


class TestPDFInfoRetrieval:
    """Test PDF information retrieval functionality."""

    def test_get_pdf_info_valid_file(self) -> None:
        """Test getting information from a valid PDF file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "test.pdf"

            TestPDFCreation.create_test_pdf(pdf_path, num_pages=5)

            splitter = PDFSplitter()

            # Act
            info = splitter.get_pdf_info(pdf_path)

            # Assert
            assert info["filename"] == "test.pdf"
            assert info["total_pages"] == 5
            assert info["path"] == str(pdf_path)
            assert info["file_size"] > 0
            assert info["is_encrypted"] is False
            assert isinstance(info["creation_date"], datetime)
            assert isinstance(info["modification_date"], datetime)
            assert "page_dimensions" in info
            assert info["page_dimensions"]["width"] == 595  # A4 width
            assert info["page_dimensions"]["height"] == 842  # A4 height

    def test_get_pdf_info_nonexistent_file(self) -> None:
        """Test getting information from a non-existent PDF file."""
        # Arrange
        nonexistent_path = Path("nonexistent.pdf")
        splitter = PDFSplitter()

        # Act & Assert
        with pytest.raises(PDFNotFoundError, match="PDF file not found"):
            splitter.get_pdf_info(nonexistent_path)

    def test_get_pdf_info_corrupted_file(self) -> None:
        """Test getting information from a corrupted PDF file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "corrupted.pdf"

            TestPDFCreation.create_corrupted_pdf(pdf_path)

            splitter = PDFSplitter()

            # Act & Assert
            with pytest.raises(PDFCorruptedError, match="PDF file is corrupted"):
                splitter.get_pdf_info(pdf_path)


class TestPDFSplitting:
    """Test PDF splitting functionality."""

    def test_split_pdf_basic_functionality(self) -> None:
        """Test basic PDF splitting functionality."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            # Ensure output directory is within the temp_dir for cleanup
            config.output.output_directory = temp_path / "output_data"

            pdf_path = temp_path / "input.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert isinstance(document_data, DocumentData)
            assert document_data.source_pdf_path == pdf_path
            assert len(document_data.pages) == 3

            # Check that image files were created
            split_dir = config.get_pdf_split_dir(pdf_path)
            assert split_dir.exists()

            for page_num in range(1, 4):
                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()
                assert image_path.is_file()
                assert image_path.suffix == ".png" # Default format

                # Check that the image has reasonable file size
                assert image_path.stat().st_size > 1000  # At least 1KB

            # Check page data
            for page_data in document_data.pages:
                assert isinstance(page_data, PageData)
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
                assert page_data.image_path is not None
                assert page_data.image_path.exists()

    def test_split_pdf_with_different_dpi(self) -> None:
        """Test PDF splitting with different DPI settings."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test with different DPI values
            dpi_values = [150, 300, 450]
            file_sizes = []

            for dpi in dpi_values:
                config = AssetableConfig()
                config.output.output_directory = temp_path / f"dpi_output_{dpi}"
                config.pdf_split.dpi = dpi

                pdf_path = temp_path / "test_dpi.pdf" # Use a consistent name
                if not pdf_path.exists():
                    TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

                splitter = PDFSplitter(config=config)

                # Act
                document_data = splitter.split_pdf(pdf_path)

                # Assert
                assert len(document_data.pages) == 1
                page_data = document_data.pages[0]
                assert page_data.image_path is not None
                assert page_data.image_path.exists()

                # Higher DPI should produce larger files
                file_size = page_data.image_path.stat().st_size
                file_sizes.append(file_size)

            # Assert that higher DPI produces larger files
            assert file_sizes[0] < file_sizes[1] < file_sizes[2]

    def test_split_pdf_jpeg_format(self) -> None:
        """Test PDF splitting with JPEG format."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "output_jpeg"
            config.pdf_split.image_format = "jpg"

            pdf_path = temp_path / "test_jpeg.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert len(document_data.pages) == 2

            for page_data in document_data.pages:
                assert page_data.image_path is not None
                assert page_data.image_path.exists()
                # Check if PyMuPDF respects the format or defaults to PNG
                # Based on current pdf_splitter.py, it should save as JPG if not PNG
                assert page_data.image_path.suffix.lower() == ".jpg"

    def test_split_pdf_skip_existing_files(self) -> None:
        """Test PDF splitting with skip existing files option."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "output_skip"
            config.processing.skip_existing_files = True

            pdf_path = temp_path / "test_skip.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            splitter = PDFSplitter(config=config)

            # Act - First split
            document_data_1 = splitter.split_pdf(pdf_path)

            # Get modification time of first image
            first_image_path = document_data_1.pages[0].image_path
            assert first_image_path is not None
            first_mtime = first_image_path.stat().st_mtime

            # Act - Second split (should skip existing)
            document_data_2 = splitter.split_pdf(pdf_path)

            # Assert
            assert len(document_data_2.pages) == 2

            # File should not have been modified
            second_mtime = first_image_path.stat().st_mtime
            assert first_mtime == second_mtime

    def test_split_pdf_force_regenerate(self) -> None:
        """Test PDF splitting with force regenerate option."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "output_force"
            config.processing.skip_existing_files = True # Should be overridden by force

            pdf_path = temp_path / "test_force.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            splitter = PDFSplitter(config=config)

            # Act - First split
            document_data_1 = splitter.split_pdf(pdf_path)

            # Get modification time of first image
            first_image_path = document_data_1.pages[0].image_path
            assert first_image_path is not None
            first_mtime = first_image_path.stat().st_mtime

            # Wait a bit to ensure different timestamp if regenerated
            import time
            time.sleep(0.01) # Small delay

            # Act - Second split with force regenerate
            document_data_2 = splitter.split_pdf(pdf_path, force_regenerate=True)

            # Assert
            assert len(document_data_2.pages) == 1

            # File should have been regenerated
            second_mtime = first_image_path.stat().st_mtime
            assert second_mtime > first_mtime


class TestPDFSplittingErrorHandling:
    """Test error handling in PDF splitting."""

    def test_split_pdf_nonexistent_file(self) -> None:
        """Test splitting a non-existent PDF file."""
        # Arrange
        nonexistent_path = Path("nonexistent_error.pdf")
        splitter = PDFSplitter()

        # Act & Assert
        with pytest.raises(PDFNotFoundError, match="PDF file not found"):
            splitter.split_pdf(nonexistent_path)

    def test_split_pdf_directory_instead_of_file(self) -> None:
        """Test splitting when path points to a directory."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dir_path = temp_path / "not_a_file_dir"
            dir_path.mkdir()

            splitter = PDFSplitter()

            # Act & Assert
            with pytest.raises(PDFNotFoundError, match="Path is not a file"):
                splitter.split_pdf(dir_path)

    def test_split_pdf_corrupted_file(self) -> None:
        """Test splitting a corrupted PDF file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "corrupted_error.pdf"

            TestPDFCreation.create_corrupted_pdf(pdf_path)

            config = AssetableConfig()
            config.output.output_directory = temp_path / "output_corrupted"
            splitter = PDFSplitter(config=config) # Config needed for file_manager

            # Act & Assert
            with pytest.raises(PDFCorruptedError, match="PDF file is corrupted"):
                splitter.split_pdf(pdf_path)

    def test_split_pdf_empty_file(self) -> None:
        """Test splitting an empty PDF file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "empty_error.pdf"

            TestPDFCreation.create_empty_pdf(pdf_path)

            config = AssetableConfig()
            config.output.output_directory = temp_path / "output_empty"
            splitter = PDFSplitter(config=config) # Config needed for file_manager

            # Act & Assert
            # For the minimal PDF created by create_empty_pdf, pypdfium2 fails directly.
            # So we expect a general PDFCorruptedError related to opening the file.
            with pytest.raises(PDFCorruptedError, match="PDF file is corrupted: Failed to open file"):
                splitter.split_pdf(pdf_path)


class TestProcessingStatus:
    """Test processing status functionality."""

    def test_get_processing_status_fresh_pdf(self) -> None:
        """Test getting processing status for a fresh PDF."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "status_fresh"

            pdf_path = temp_path / "test_status_fresh.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            splitter = PDFSplitter(config=config)

            # Act
            status = splitter.get_processing_status(pdf_path)

            # Assert
            assert "pdf_info" in status
            assert "processing_summary" in status
            assert "split_status" in status

            pdf_info = status["pdf_info"]
            assert pdf_info["total_pages"] == 3
            assert pdf_info["filename"] == "test_status_fresh.pdf"

            split_status = status["split_status"]
            assert split_status["completed"] == 0
            # Pending pages can be tricky if they are not yet in DocumentData
            # The current implementation of get_processing_summary might need adjustment
            # For now, we check progress.
            assert split_status["progress"] == 0.0
            # If DocumentData is not created yet, pending might be 0.
            # Let's check total pages vs completed.
            assert (pdf_info["total_pages"] - split_status["completed"]) == 3


    def test_get_processing_status_partially_processed(self) -> None:
        """Test getting processing status for a partially processed PDF."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "status_partial"

            pdf_path = temp_path / "test_status_partial.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=4)

            splitter = PDFSplitter(config=config)

            # Process first 2 pages manually by creating page data
            splitter.file_manager.setup_document_structure(pdf_path)
            doc_data = splitter._create_document_data(pdf_path, 4) # Create doc data

            for page_num in [1, 2]:
                image_path = config.get_page_image_path(pdf_path, page_num)
                # Create dummy image file for PageData to be valid
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_text("dummy image data")

                page_data = PageData(
                    page_number=page_num,
                    source_pdf=pdf_path,
                    image_path=image_path
                )
                page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
                doc_data.add_page(page_data)
                splitter.file_manager.save_page_data(page_data)
            splitter.file_manager.save_document_data(doc_data)

            # Act
            status = splitter.get_processing_status(pdf_path)

            # Assert
            split_status = status["split_status"]
            assert split_status["completed"] == 2
            assert split_status["pending"] == 2 # Based on DocumentData
            assert split_status["progress"] == 0.5

    def test_get_processing_status_completed(self) -> None:
        """Test getting processing status for a completed PDF."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "status_completed"

            pdf_path = temp_path / "test_status_completed.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            splitter = PDFSplitter(config=config)

            # Act - Process the PDF
            splitter.split_pdf(pdf_path)

            # Get status after processing
            status = splitter.get_processing_status(pdf_path)

            # Assert
            split_status = status["split_status"]
            assert split_status["completed"] == 2
            assert split_status["pending"] == 0
            assert split_status["progress"] == 1.0


class TestFileCleanup:
    """Test file cleanup functionality."""

    def test_cleanup_split_files_all(self) -> None:
        """Test cleaning up all split files."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "cleanup_all"

            pdf_path = temp_path / "test_cleanup_all.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            splitter = PDFSplitter(config=config)

            # Process the PDF
            splitter.split_pdf(pdf_path)

            # Verify files exist
            split_dir = config.get_pdf_split_dir(pdf_path)
            image_files = list(split_dir.glob(f"page_*.{config.pdf_split.image_format}"))
            assert len(image_files) == 3

            # Act
            splitter.cleanup_split_files(pdf_path)

            # Assert
            remaining_files = list(split_dir.glob(f"page_*.{config.pdf_split.image_format}"))
            assert len(remaining_files) == 0

    def test_cleanup_split_files_specific_pages(self) -> None:
        """Test cleaning up specific page files."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "cleanup_specific"

            pdf_path = temp_path / "test_cleanup_specific.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=4)

            splitter = PDFSplitter(config=config)

            # Process the PDF
            splitter.split_pdf(pdf_path)

            # Verify files exist
            split_dir = config.get_pdf_split_dir(pdf_path)
            image_files = list(split_dir.glob(f"page_*.{config.pdf_split.image_format}"))
            assert len(image_files) == 4

            # Act - Clean up pages 2 and 3
            splitter.cleanup_split_files(pdf_path, page_numbers=[2, 3])

            # Assert
            remaining_files = list(split_dir.glob(f"page_*.{config.pdf_split.image_format}"))
            assert len(remaining_files) == 2

            # Check that correct files remain
            page_1_path = config.get_page_image_path(pdf_path, 1)
            page_4_path = config.get_page_image_path(pdf_path, 4)
            assert page_1_path.exists()
            assert page_4_path.exists()

            # Check that correct files were removed
            page_2_path = config.get_page_image_path(pdf_path, 2)
            page_3_path = config.get_page_image_path(pdf_path, 3)
            assert not page_2_path.exists()
            assert not page_3_path.exists()

    def test_cleanup_split_files_nonexistent_directory(self) -> None:
        """Test cleaning up files when directory doesn't exist."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "cleanup_nonexistent"

            pdf_path = temp_path / "nonexistent_cleanup.pdf" # PDF itself doesn't need to exist for this test
            splitter = PDFSplitter(config=config)

            # Act - Should not raise an error
            splitter.cleanup_split_files(pdf_path)

            # Assert - No exception should have been raised
            assert True


class TestCLIWrapper:
    """Test CLI wrapper functionality."""

    def test_split_pdf_cli_basic(self) -> None:
        """Test basic CLI wrapper functionality."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "cli_basic"

            pdf_path = temp_path / "test_cli_basic.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            # Act
            document_data = split_pdf_cli(pdf_path, force_regenerate=False, config=config)

            # Assert
            assert isinstance(document_data, DocumentData)
            assert len(document_data.pages) == 2

            # Verify images were created
            for page_num in range(1, 3):
                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()

    def test_split_pdf_cli_force_regenerate(self) -> None:
        """Test CLI wrapper with force regenerate option."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "cli_force"

            pdf_path = temp_path / "test_cli_force.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            # First processing
            document_data_1 = split_pdf_cli(pdf_path, force_regenerate=False, config=config)
            image_path = document_data_1.pages[0].image_path
            assert image_path is not None
            first_mtime = image_path.stat().st_mtime

            # Wait a bit
            import time
            time.sleep(0.01)

            # Act - Force regenerate
            document_data_2 = split_pdf_cli(pdf_path, force_regenerate=True, config=config)

            # Assert
            assert len(document_data_2.pages) == 1
            second_mtime = image_path.stat().st_mtime
            assert second_mtime > first_mtime

    def test_split_pdf_cli_with_custom_config(self) -> None:
        """Test CLI wrapper with custom configuration."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig() # Create a new config instance
            config.output.output_directory = temp_path / "cli_custom"
            config.pdf_split.dpi = 450
            config.processing.debug_mode = True # Example of another custom setting

            pdf_path = temp_path / "test_cli_custom.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            # Act
            document_data = split_pdf_cli(pdf_path, config=config) # Pass the custom config

            # Assert
            assert len(document_data.pages) == 1
            page_data = document_data.pages[0]
            assert page_data.image_path is not None
            assert page_data.image_path.exists()

            # Image should be larger due to higher DPI
            # This is an indirect check of DPI, actual DPI check is harder without image analysis
            file_size = page_data.image_path.stat().st_size
            # Check relative to a baseline if possible, or ensure it's larger than a typical low-DPI image
            assert file_size > 5000 # Adjusted expectation based on content


class TestIntegrationWithFileManager:
    """Test integration between PDFSplitter and FileManager."""

    def test_pdf_splitter_file_manager_integration(self) -> None:
        """Test that PDFSplitter properly integrates with FileManager."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "integration_fm"

            pdf_path = temp_path / "test_integration.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=3)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            # Check that FileManager can detect completed stages from DocumentData
            for page_data in document_data.pages:
                 assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)

            # Check that page data was saved by FileManager and can be loaded
            for page_num in range(1, 4):
                saved_page_data = splitter.file_manager.load_page_data(pdf_path, page_num)
                assert saved_page_data is not None
                assert saved_page_data.page_number == page_num
                assert saved_page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)

            # Check that document data was saved by FileManager and can be loaded
            saved_document_data = splitter.file_manager.load_document_data(pdf_path)
            assert saved_document_data is not None
            assert len(saved_document_data.pages) == 3 # Pages are added to DocumentData

    def test_pdf_splitter_resume_processing(self) -> None:
        """Test that PDFSplitter can resume interrupted processing."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "integration_resume"
            config.processing.skip_existing_files = True

            pdf_path = temp_path / "test_resume.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=4)

            splitter = PDFSplitter(config=config)

            # Simulate partial processing: process pages 1 and 3
            # Create document data and save it first
            initial_doc_data = splitter._create_document_data(pdf_path, 4)
            splitter.file_manager.save_document_data(initial_doc_data)

            for page_num_to_process in [1, 3]:
                # Manually process a page (simplified for test setup)
                pdf_doc = pdfium.PdfDocument(pdf_path)
                page_data = splitter._process_page(pdf_doc, pdf_path, page_num_to_process, force_regenerate=False)
                pdf_doc.close()
                assert page_data is not None
                initial_doc_data.add_page(page_data)
                splitter.file_manager.save_page_data(page_data)
            splitter.file_manager.save_document_data(initial_doc_data) # Save updated doc data

            # Get modification times for already processed files
            path_page1 = config.get_page_image_path(pdf_path, 1)
            path_page3 = config.get_page_image_path(pdf_path, 3)
            mtime_page1_initial = path_page1.stat().st_mtime
            mtime_page3_initial = path_page3.stat().st_mtime
            import time
            time.sleep(0.01)

            # Act: Run split_pdf, which should resume and process pages 2 and 4
            resumed_document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert len(resumed_document_data.pages) == 4 # All pages should be in the final DocumentData

            # Check that all pages are now marked as completed
            for page_data_final in resumed_document_data.pages:
                assert page_data_final.is_stage_completed(ProcessingStage.PDF_SPLIT)

            # Check that previously created files were skipped (not modified)
            assert path_page1.stat().st_mtime == mtime_page1_initial
            assert path_page3.stat().st_mtime == mtime_page3_initial

            # Check that new files were created for missing pages (2 and 4)
            path_page2 = config.get_page_image_path(pdf_path, 2)
            path_page4 = config.get_page_image_path(pdf_path, 4)
            assert path_page2.exists()
            assert path_page4.exists()
            assert path_page2.stat().st_size > 1000  # Real image file
            assert path_page4.stat().st_size > 1000  # Real image file


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_split_pdf_single_page(self) -> None:
        """Test splitting a single-page PDF."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "edge_single"

            pdf_path = temp_path / "single_page.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=1)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert len(document_data.pages) == 1

            page_data = document_data.pages[0]
            assert page_data.page_number == 1
            assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            assert page_data.image_path is not None
            assert page_data.image_path.exists()

    @pytest.mark.slow  # Mark as slow if it takes significant time
    def test_split_pdf_large_number_of_pages(self) -> None:
        """Test splitting a PDF with many pages."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "edge_large"

            pdf_path = temp_path / "large.pdf"
            num_pages = 20 # Reduced from 50 for faster test execution, adjust if needed
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=num_pages)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert len(document_data.pages) == num_pages

            # Check that all pages were processed
            for page_num in range(1, num_pages + 1):
                page_data = document_data.get_page_by_number(page_num)
                assert page_data is not None
                assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)

                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()

    def test_split_pdf_with_unicode_filename(self) -> None:
        """Test splitting a PDF with Unicode characters in filename."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "edge_unicode"

            pdf_path = temp_path / "テスト文書_日本語.pdf"
            TestPDFCreation.create_test_pdf(pdf_path, num_pages=2)

            splitter = PDFSplitter(config=config)

            # Act
            document_data = splitter.split_pdf(pdf_path)

            # Assert
            assert len(document_data.pages) == 2

            # Check that files were created with proper naming
            split_dir = config.get_pdf_split_dir(pdf_path)
            assert split_dir.exists()
            # Ensure the directory name itself handles Unicode if it's part of the path derivation
            assert "テスト文書_日本語" in str(document_data.output_directory)

            for page_num in range(1, 3):
                image_path = config.get_page_image_path(pdf_path, page_num)
                assert image_path.exists()
