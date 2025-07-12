"""
PDF splitting functionality for Assetable.

This module provides PDF splitting capabilities using PyMuPDF (fitz).
It converts PDF pages into high-quality images for AI processing.
"""

import fitz  # PyMuPDF
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import AssetableConfig, get_config
from ..file_manager import FileManager
from ..models import DocumentData, PageData, ProcessingStage


class PDFSplitterError(Exception):
    """Base exception for PDF splitter operations."""
    pass


class PDFNotFoundError(PDFSplitterError):
    """Raised when PDF file is not found."""
    pass


class PDFCorruptedError(PDFSplitterError):
    """Raised when PDF file is corrupted or cannot be opened."""
    pass


class ImageConversionError(PDFSplitterError):
    """Raised when image conversion fails."""
    pass


class PDFSplitter:
    """
    PDF splitter for converting PDF pages to images.

    This class handles the conversion of PDF pages to high-quality images
    suitable for AI processing. It uses PyMuPDF for reliable PDF processing
    and supports configurable DPI settings.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """
        Initialize PDF splitter.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.file_manager = FileManager(self.config)

    def split_pdf(self, pdf_path: Path, force_regenerate: bool = False) -> DocumentData:
        """
        Split PDF into individual page images.

        Args:
            pdf_path: Path to the PDF file.
            force_regenerate: If True, regenerate images even if they exist.

        Returns:
            DocumentData object containing processing results.

        Raises:
            PDFNotFoundError: If PDF file doesn't exist.
            PDFCorruptedError: If PDF file is corrupted.
            ImageConversionError: If image conversion fails.
        """
        if not pdf_path.exists():
            raise PDFNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise PDFNotFoundError(f"Path is not a file: {pdf_path}")

        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            if total_pages == 0:
                raise PDFCorruptedError(f"PDF has no pages: {pdf_path}")

            # Setup output directory structure
            self.file_manager.setup_document_structure(pdf_path)

            # Create or load document data
            document_data = self._create_document_data(pdf_path, total_pages)

            # Process each page
            processed_pages = []
            for page_num in range(1, total_pages + 1):
                try:
                    page_data = self._process_page(doc, pdf_path, page_num, force_regenerate)
                    if page_data:
                        processed_pages.append(page_data)
                        document_data.add_page(page_data)

                        # Save progress
                        self.file_manager.save_page_data(page_data)

                        if self.config.processing.debug_mode:
                            print(f"Processed page {page_num}/{total_pages}")

                except Exception as e:
                    error_msg = f"Failed to process page {page_num}: {e}"
                    if self.config.processing.debug_mode:
                        print(f"Warning: {error_msg}")
                    continue

            # Save document data
            self.file_manager.save_document_data(document_data)

            # Close document
            doc.close()

            return document_data

        except fitz.FileDataError as e:
            raise PDFCorruptedError(f"PDF file is corrupted: {e}")
        except Exception as e:
            raise PDFSplitterError(f"Unexpected error during PDF splitting: {e}")

    def _create_document_data(self, pdf_path: Path, total_pages: int) -> DocumentData:
        """Create or load document data."""
        # Try to load existing document data
        existing_data = self.file_manager.load_document_data(pdf_path)

        if existing_data:
            # total_pages is not a direct attribute of DocumentData anymore
            # It's used for processing summary, not stored directly in the model
            return existing_data

        # Create new document data
        document_id = pdf_path.stem # Use filename without extension as ID
        return DocumentData(
            document_id=document_id,
            source_pdf_path=pdf_path,
            output_directory=self.config.get_document_output_dir(pdf_path)
            # total_pages is no longer part of DocumentData constructor
        )

    def _process_page(
        self,
        doc: fitz.Document,
        pdf_path: Path,
        page_num: int,
        force_regenerate: bool
    ) -> Optional[PageData]:
        """
        Process a single PDF page.

        Args:
            doc: PyMuPDF document object.
            pdf_path: Path to the PDF file.
            page_num: Page number (1-based).
            force_regenerate: If True, regenerate even if file exists.

        Returns:
            PageData object if processing was successful, None otherwise.
        """
        try:
            # Check if already processed and skip if configured
            if not force_regenerate and self.config.processing.skip_existing_files:
                if self.file_manager.is_stage_completed(pdf_path, page_num, ProcessingStage.PDF_SPLIT):
                    existing_data = self.file_manager.load_page_data(pdf_path, page_num)
                    if existing_data:
                        return existing_data
                    # If file exists but data is corrupted, continue with processing

            # Get page from document (0-based indexing)
            page = doc[page_num - 1]

            # Convert page to image
            image_path = self._convert_page_to_image(page, pdf_path, page_num)

            # Create page data
            page_data = PageData(
                page_number=page_num,
                source_pdf=pdf_path,
                image_path=image_path
            )

            # Mark PDF split stage as completed
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
            page_data.add_log(f"Successfully split page {page_num}")

            return page_data

        except Exception as e:
            raise ImageConversionError(f"Failed to convert page {page_num} to image: {e}")

    def _convert_page_to_image(self, page: fitz.Page, pdf_path: Path, page_num: int) -> Path:
        """
        Convert a PDF page to an image file.

        Args:
            page: PyMuPDF page object.
            pdf_path: Path to the PDF file.
            page_num: Page number (1-based).

        Returns:
            Path to the generated image file.
        """
        # Get output path
        image_path = self.config.get_page_image_path(pdf_path, page_num)
        image_path.parent.mkdir(parents=True, exist_ok=True)

        # Create transformation matrix for DPI scaling
        # PyMuPDF uses 72 DPI by default, so we need to scale
        scale_factor = self.config.pdf_split.dpi / 72.0
        mat = fitz.Matrix(scale_factor, scale_factor)

        # Convert page to pixmap
        pix = page.get_pixmap(matrix=mat)

        # Save image
        image_format = self.config.pdf_split.image_format.lower()
        if image_format == 'png':
            pix.save(image_path)
        elif image_format == 'jpg' or image_format == 'jpeg':
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)  # remove alpha for JPEG
            pix.save(image_path, "jpeg")
        else:
            # For other formats, just save with the path
            pix.save(image_path)

        # Clean up
        pix = None

        return image_path

    def get_pdf_info(self, pdf_path: Path) -> dict:
        """
        Get basic information about a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary containing PDF information.

        Raises:
            PDFNotFoundError: If PDF file doesn't exist.
            PDFCorruptedError: If PDF file is corrupted.
        """
        if not pdf_path.exists():
            raise PDFNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)

            info = {
                "path": str(pdf_path),
                "filename": pdf_path.name,
                "total_pages": len(doc),
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted,
                "file_size": pdf_path.stat().st_size,
                "creation_date": datetime.fromtimestamp(pdf_path.stat().st_ctime),
                "modification_date": datetime.fromtimestamp(pdf_path.stat().st_mtime),
            }

            # Get page dimensions for first page
            if len(doc) > 0:
                first_page = doc[0]
                rect = first_page.rect
                info["page_dimensions"] = {
                    "width": rect.width,
                    "height": rect.height,
                    "width_inches": rect.width / 72,
                    "height_inches": rect.height / 72,
                }

            doc.close()
            return info

        except fitz.FileDataError as e:
            raise PDFCorruptedError(f"PDF file is corrupted: {e}")
        except Exception as e:
            raise PDFSplitterError(f"Failed to get PDF info: {e}")

    def get_processing_status(self, pdf_path: Path) -> dict:
        """
        Get processing status for a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary containing processing status.
        """
        try:
            # Get PDF info
            pdf_info = self.get_pdf_info(pdf_path)
            total_pages = pdf_info["total_pages"]

            # Get processing summary
            summary = self.file_manager.get_processing_summary(pdf_path, total_pages)

            return {
                "pdf_info": pdf_info,
                "processing_summary": summary,
                "split_status": {
                    "completed": len(summary["stages"]["pdf_split"]["completed_pages"]),
                    "pending": len(summary["stages"]["pdf_split"]["pending_pages"]),
                    "progress": summary["stages"]["pdf_split"]["progress"],
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "pdf_path": str(pdf_path),
            }

    def cleanup_split_files(self, pdf_path: Path, page_numbers: Optional[List[int]] = None) -> None:
        """
        Clean up generated split files.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: Specific page numbers to clean up. If None, cleans all.
        """
        try:
            split_dir = self.config.get_pdf_split_dir(pdf_path)

            if not split_dir.exists():
                return

            if page_numbers is None:
                # Clean up all files
                for file_path in split_dir.glob("page_*.png"): # Assuming PNG, adjust if format changes
                    file_path.unlink()
                    if self.config.processing.debug_mode:
                        print(f"Removed: {file_path}")
            else:
                # Clean up specific pages
                for page_num in page_numbers:
                    image_path = self.config.get_page_image_path(pdf_path, page_num)
                    if image_path.exists():
                        image_path.unlink()
                        if self.config.processing.debug_mode:
                            print(f"Removed: {image_path}")

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Error during cleanup: {e}")


def split_pdf_cli(
    pdf_path: Path,
    force_regenerate: bool = False,
    config: Optional[AssetableConfig] = None
) -> DocumentData:
    """
    CLI wrapper for PDF splitting functionality.

    Args:
        pdf_path: Path to the PDF file.
        force_regenerate: If True, regenerate images even if they exist.
        config: Configuration object. If None, uses global config.

    Returns:
        DocumentData object containing processing results.
    """
    splitter = PDFSplitter(config)
    return splitter.split_pdf(pdf_path, force_regenerate)
