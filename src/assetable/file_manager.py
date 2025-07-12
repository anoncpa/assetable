"""
File management for Assetable.

This module provides file-based state management where the existence of files
indicates completion of processing stages. It handles directory structure,
file persistence, and state tracking for the entire pipeline.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from pydantic import ValidationError

from .config import AssetableConfig, get_config
from .models import (
    AssetType,
    DocumentData,
    PageData,
    PageStructure,
    ProcessingStage,
    TableAsset,
    FigureAsset,
    ImageAsset,
)


class FileManagerError(Exception):
    """Base exception for file manager operations."""
    pass


class FileNotFoundError(FileManagerError):
    """Raised when a required file is not found."""
    pass


class DirectoryCreationError(FileManagerError):
    """Raised when directory creation fails."""
    pass


class FileOperationError(FileManagerError):
    """Raised when file operations fail."""
    pass


class FileManager:
    """
    File manager for Assetable processing pipeline.

    This class implements file-based state management where the existence
    of files indicates completion of processing stages. It provides methods
    for creating directories, managing file paths, and persisting processing
    results.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """
        Initialize file manager.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self._created_directories: Set[Path] = set()

    def setup_document_structure(self, pdf_path: Path) -> None:
        """
        Set up directory structure for a document.

        Args:
            pdf_path: Path to the PDF file.

        Raises:
            DirectoryCreationError: If directory creation fails.
        """
        try:
            self.config.create_output_directories(pdf_path)

            # Track created directories
            doc_dir = self.config.get_document_output_dir(pdf_path)
            self._created_directories.add(doc_dir)

            # Log directory creation
            self._log_operation(f"Created directory structure for {pdf_path.name}")

        except Exception as e:
            raise DirectoryCreationError(f"Failed to create directories for {pdf_path}: {e}")

    def is_stage_completed(self, pdf_path: Path, page_number: int, stage: ProcessingStage) -> bool:
        """
        Check if a processing stage is completed for a page.

        File-based state management: if the output file exists, the stage is complete.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).
            stage: Processing stage to check.

        Returns:
            True if the stage is completed, False otherwise.
        """
        try:
            if stage == ProcessingStage.PDF_SPLIT:
                image_path = self.config.get_page_image_path(pdf_path, page_number)
                return image_path.exists()

            elif stage == ProcessingStage.STRUCTURE_ANALYSIS:
                structure_path = self.config.get_structure_json_path(pdf_path, page_number)
                return structure_path.exists()

            elif stage == ProcessingStage.ASSET_EXTRACTION:
                # For asset extraction, check if structure file exists and has asset data
                structure_path = self.config.get_structure_json_path(pdf_path, page_number)
                if not structure_path.exists():
                    return False

                try:
                    page_structure = self.load_page_structure(pdf_path, page_number)
                    # Consider completed if structure analysis is done
                    # (assets will be extracted based on structure)
                    return page_structure is not None
                except Exception:
                    return False

            elif stage == ProcessingStage.MARKDOWN_GENERATION:
                markdown_path = self.config.get_markdown_path(pdf_path, page_number)
                return markdown_path.exists()

            elif stage == ProcessingStage.COMPLETED:
                # All stages must be completed
                return all(
                    self.is_stage_completed(pdf_path, page_number, s)
                    for s in [
                        ProcessingStage.PDF_SPLIT,
                        ProcessingStage.STRUCTURE_ANALYSIS,
                        ProcessingStage.ASSET_EXTRACTION,
                        ProcessingStage.MARKDOWN_GENERATION,
                    ]
                )

            return False

        except Exception:
            return False

    def get_completed_pages(self, pdf_path: Path, stage: ProcessingStage) -> List[int]:
        """
        Get list of page numbers that have completed a specific stage.

        Args:
            pdf_path: Path to the PDF file.
            stage: Processing stage to check.

        Returns:
            List of page numbers that have completed the stage.
        """
        completed_pages = []

        # Check each possible page (assume reasonable maximum)
        for page_num in range(1, 1000):  # Check up to 999 pages
            if self.is_stage_completed(pdf_path, page_num, stage):
                completed_pages.append(page_num)
            else:
                # If consecutive pages are missing, assume we've reached the end
                if page_num > 1 and not self.is_stage_completed(pdf_path, page_num - 1, stage):
                    break

        return completed_pages

    def get_pending_pages(self, pdf_path: Path, stage: ProcessingStage, total_pages: int) -> List[int]:
        """
        Get list of page numbers that need to complete a specific stage.

        Args:
            pdf_path: Path to the PDF file.
            stage: Processing stage to check.
            total_pages: Total number of pages in the document.

        Returns:
            List of page numbers that need to complete the stage.
        """
        pending_pages = []

        for page_num in range(1, total_pages + 1):
            if not self.is_stage_completed(pdf_path, page_num, stage):
                pending_pages.append(page_num)

        return pending_pages

    def save_page_structure(self, pdf_path: Path, page_number: int, page_structure: PageStructure) -> Path:
        """
        Save page structure to JSON file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).
            page_structure: Page structure to save.

        Returns:
            Path to the saved JSON file.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            structure_path = self.config.get_structure_json_path(pdf_path, page_number)
            structure_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict with proper serialization
            data = page_structure.model_dump()

            # Save with proper encoding
            with open(structure_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._log_operation(f"Saved page structure for page {page_number} to {structure_path}")
            return structure_path

        except Exception as e:
            raise FileOperationError(f"Failed to save page structure: {e}")

    def load_page_structure(self, pdf_path: Path, page_number: int) -> Optional[PageStructure]:
        """
        Load page structure from JSON file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).

        Returns:
            PageStructure object if file exists and is valid, None otherwise.

        Raises:
            FileOperationError: If load operation fails with invalid data.
        """
        try:
            structure_path = self.config.get_structure_json_path(pdf_path, page_number)

            if not structure_path.exists():
                return None

            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return PageStructure(**data)

        except ValidationError as e:
            raise FileOperationError(f"Invalid page structure data in {structure_path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to load page structure: {e}")

    def save_page_data(self, page_data: PageData) -> Path:
        """
        Save complete page data to JSON file.

        Args:
            page_data: Page data to save.

        Returns:
            Path to the saved JSON file.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            # Use a dedicated page data file
            doc_dir = self.config.get_document_output_dir(page_data.source_pdf)
            page_data_path = doc_dir / f"page_{page_data.page_number:04d}_data.json"

            # Convert to dict with proper serialization
            data = page_data.model_dump()

            # Save with proper encoding
            with open(page_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._log_operation(f"Saved page data for page {page_data.page_number}")
            return page_data_path

        except Exception as e:
            raise FileOperationError(f"Failed to save page data: {e}")

    def load_page_data(self, pdf_path: Path, page_number: int) -> Optional[PageData]:
        """
        Load complete page data from JSON file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).

        Returns:
            PageData object if file exists and is valid, None otherwise.

        Raises:
            FileOperationError: If load operation fails with invalid data.
        """
        try:
            doc_dir = self.config.get_document_output_dir(pdf_path)
            page_data_path = doc_dir / f"page_{page_number:04d}_data.json"

            if not page_data_path.exists():
                return None

            with open(page_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return PageData(**data)

        except ValidationError as e:
            raise FileOperationError(f"Invalid page data in {page_data_path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to load page data: {e}")

    def save_document_data(self, document_data: DocumentData) -> Path:
        """
        Save complete document data to JSON file.

        Args:
            document_data: Document data to save.

        Returns:
            Path to the saved JSON file.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            doc_dir = self.config.get_document_output_dir(document_data.source_pdf_path)
            doc_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            doc_data_path = doc_dir / "document_data.json"

            # Convert to dict with proper serialization
            data = document_data.model_dump()

            # Save with proper encoding
            with open(doc_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._log_operation(f"Saved document data for {document_data.source_pdf_path.name}")
            return doc_data_path

        except Exception as e:
            raise FileOperationError(f"Failed to save document data: {e}")

    def load_document_data(self, pdf_path: Path) -> Optional[DocumentData]:
        """
        Load complete document data from JSON file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            DocumentData object if file exists and is valid, None otherwise.

        Raises:
            FileOperationError: If load operation fails with invalid data.
        """
        try:
            doc_dir = self.config.get_document_output_dir(pdf_path)
            doc_data_path = doc_dir / "document_data.json"

            if not doc_data_path.exists():
                return None

            with open(doc_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return DocumentData(**data)

        except ValidationError as e:
            raise FileOperationError(f"Invalid document data in {doc_data_path}: {e}")
        except Exception as e:
            raise FileOperationError(f"Failed to load document data: {e}")

    def save_asset_file(self, pdf_path: Path, page_number: int, asset: Union[TableAsset, FigureAsset, ImageAsset]) -> Path:
        """
        Save asset data to appropriate file format.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).
            asset: Asset to save.

        Returns:
            Path to the saved asset file.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            if isinstance(asset, TableAsset):
                return self._save_table_asset(pdf_path, page_number, asset)
            elif isinstance(asset, FigureAsset):
                return self._save_figure_asset(pdf_path, page_number, asset)
            elif isinstance(asset, ImageAsset):
                return self._save_image_asset(pdf_path, page_number, asset)
            else:
                raise FileOperationError(f"Unknown asset type: {type(asset)}")

        except Exception as e:
            raise FileOperationError(f"Failed to save asset {asset.name}: {e}")

    def _save_table_asset(self, pdf_path: Path, page_number: int, table: TableAsset) -> Path:
        """Save table asset as CSV file."""
        asset_path = self.config.get_asset_path(pdf_path, page_number, "table", table.name)
        asset_path.parent.mkdir(parents=True, exist_ok=True)

        # Save CSV data
        if table.csv_data:
            with open(asset_path, 'w', encoding='utf-8') as f:
                f.write(table.csv_data)
        else:
            # Generate CSV from columns and rows
            import csv
            with open(asset_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if table.columns:
                    writer.writerow(table.columns)
                if table.rows:
                    writer.writerows(table.rows)

        self._log_operation(f"Saved table asset {table.name} to {asset_path}")
        return asset_path

    def _save_figure_asset(self, pdf_path: Path, page_number: int, figure: FigureAsset) -> Path:
        """Save figure asset as JSON file."""
        asset_path = self.config.get_asset_path(pdf_path, page_number, "figure", figure.name)
        asset_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare figure data
        figure_data = {
            "name": figure.name,
            "description": figure.description,
            "figure_type": figure.figure_type,
            "bbox": figure.bbox.model_dump(),
            "structure": [node.model_dump() for node in figure.structure] if figure.structure else None,
            "raw_json": figure.raw_json,
            "created_at": datetime.now().isoformat(),
        }

        # Save as JSON
        with open(asset_path, 'w', encoding='utf-8') as f:
            json.dump(figure_data, f, ensure_ascii=False, indent=2, default=str)

        self._log_operation(f"Saved figure asset {figure.name} to {asset_path}")
        return asset_path

    def _save_image_asset(self, pdf_path: Path, page_number: int, image: ImageAsset) -> Path:
        """Save image asset metadata (actual image handled elsewhere)."""
        asset_path = self.config.get_asset_path(pdf_path, page_number, "image", image.name)
        asset_path.parent.mkdir(parents=True, exist_ok=True)

        # For image assets, we save metadata as JSON
        # The actual image file is handled by the image processing pipeline
        metadata_path = asset_path.with_suffix('.json')

        image_data = {
            "name": image.name,
            "description": image.description,
            "image_type": image.image_type,
            "bbox": image.bbox.model_dump(),
            "image_path": str(image.image_path) if image.image_path else None,
            "created_at": datetime.now().isoformat(),
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(image_data, f, ensure_ascii=False, indent=2, default=str)

        self._log_operation(f"Saved image asset metadata {image.name} to {metadata_path}")
        return metadata_path

    def save_markdown_content(self, pdf_path: Path, page_number: int, markdown_content: str) -> Path:
        """
        Save markdown content to file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).
            markdown_content: Markdown content to save.

        Returns:
            Path to the saved markdown file.

        Raises:
            FileOperationError: If save operation fails.
        """
        try:
            markdown_path = self.config.get_markdown_path(pdf_path, page_number)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)

            # Save markdown content
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self._log_operation(f"Saved markdown for page {page_number} to {markdown_path}")
            return markdown_path

        except Exception as e:
            raise FileOperationError(f"Failed to save markdown content: {e}")

    def load_markdown_content(self, pdf_path: Path, page_number: int) -> Optional[str]:
        """
        Load markdown content from file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).

        Returns:
            Markdown content if file exists, None otherwise.

        Raises:
            FileOperationError: If load operation fails.
        """
        try:
            markdown_path = self.config.get_markdown_path(pdf_path, page_number)

            if not markdown_path.exists():
                return None

            with open(markdown_path, 'r', encoding='utf-8') as f:
                return f.read()

        except Exception as e:
            raise FileOperationError(f"Failed to load markdown content: {e}")

    def cleanup_incomplete_files(self, pdf_path: Path, page_number: int) -> None:
        """
        Clean up incomplete or corrupted files for a page.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-based).
        """
        try:
            files_to_check = [
                self.config.get_structure_json_path(pdf_path, page_number),
                self.config.get_markdown_path(pdf_path, page_number),
                self.config.get_document_output_dir(pdf_path) / f"page_{page_number:04d}_data.json",
            ]

            for file_path in files_to_check:
                if file_path.exists():
                    try:
                        # Try to load and validate the file
                        if file_path.suffix == '.json':
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                    except Exception:
                        # File is corrupted, remove it
                        file_path.unlink()
                        self._log_operation(f"Removed corrupted file: {file_path}")

        except Exception as e:
            self._log_operation(f"Error during cleanup: {e}")

    def get_processing_summary(self, pdf_path: Path, total_pages: int) -> Dict[str, any]:
        """
        Get processing summary for a document.

        Args:
            pdf_path: Path to the PDF file.
            total_pages: Total number of pages in the document.

        Returns:
            Dictionary containing processing summary.
        """
        summary = {
            "document": pdf_path.name,
            "total_pages": total_pages,
            "stages": {},
            "overall_progress": 0.0,
        }

        stages = [
            ProcessingStage.PDF_SPLIT,
            ProcessingStage.STRUCTURE_ANALYSIS,
            ProcessingStage.ASSET_EXTRACTION,
            ProcessingStage.MARKDOWN_GENERATION,
        ]

        total_progress = 0
        for stage in stages:
            completed = self.get_completed_pages(pdf_path, stage)
            pending = self.get_pending_pages(pdf_path, stage, total_pages)

            stage_progress = len(completed) / total_pages if total_pages > 0 else 0
            total_progress += stage_progress

            summary["stages"][stage.value] = {
                "completed_pages": completed,
                "pending_pages": pending,
                "progress": stage_progress,
                "completed_count": len(completed),
                "pending_count": len(pending),
            }

        summary["overall_progress"] = total_progress / len(stages) if stages else 0
        return summary

    def _log_operation(self, message: str) -> None:
        """Log file operation with timestamp."""
        if self.config.processing.debug_mode:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] FileManager: {message}")

    def __enter__(self) -> "FileManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Cleanup or logging can be performed here if needed
        pass
