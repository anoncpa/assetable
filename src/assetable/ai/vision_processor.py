"""
Vision-based AI processing for Assetable.

This module provides enhanced Vision AI processing capabilities using Ollama
with detailed prompt engineering and robust error handling.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..config import AssetableConfig, get_config
from ..models import (
    FigureAsset,
    ImageAsset,
    PageData,
    PageStructure,
    TableAsset,
)
from .ollama_client import OllamaClient
from .prompts import (
    AssetExtractionPrompts,
    MarkdownGenerationPrompts,
    StructureAnalysisPrompts,
)


class VisionProcessorError(Exception):
    """Base exception for vision processor operations."""
    pass


class ProcessingResult(BaseModel):
    """Base result class for processing operations."""

    success: bool
    processing_time: float
    error_message: Optional[str] = None
    retry_count: int = 0


class StructureAnalysisResult(ProcessingResult):
    """Result from structure analysis processing."""

    page_structure: Optional[PageStructure] = None
    model_used: Optional[str] = None


class AssetExtractionResult(ProcessingResult):
    """Result from asset extraction processing."""

    extracted_assets: List[Union[TableAsset, FigureAsset, ImageAsset]] = []
    model_used: Optional[str] = None


class MarkdownGenerationResult(ProcessingResult):
    """Result from Markdown generation processing."""

    markdown_content: Optional[str] = None
    asset_references: List[str] = []
    model_used: Optional[str] = None


class EnhancedVisionProcessor:
    """
    Enhanced vision processor with detailed prompt engineering and robust processing.

    This class implements the three-stage processing approach with:
    - Detailed prompt templates
    - Robust error handling and retry logic
    - Performance monitoring
    - Quality validation
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """
        Initialize enhanced vision processor.

        Args:
            config: Configuration object. If None, uses global config.

        Raises:
            VisionProcessorError: If initialization fails.
        """
        self.config = config or get_config()

        try:
            self.ollama_client = OllamaClient(self.config)
        except Exception as e:
            raise VisionProcessorError(f"Failed to initialize Ollama client: {e}")

        # Performance tracking
        self._processing_stats = {
            'structure_analysis': {'count': 0, 'total_time': 0.0, 'success_count': 0},
            'asset_extraction': {'count': 0, 'total_time': 0.0, 'success_count': 0},
            'markdown_generation': {'count': 0, 'total_time': 0.0, 'success_count': 0},
        }

        # Verify connection
        if not self.ollama_client.check_connection():
            raise VisionProcessorError("Cannot connect to Ollama server")

    def analyze_page_structure(
        self,
        page_data: PageData,
        document_type: str = "technical book"
    ) -> StructureAnalysisResult:
        """
        Perform comprehensive structure analysis on a page image.

        Args:
            page_data: Page data containing image path.
            document_type: Type of document being processed.

        Returns:
            StructureAnalysisResult with detailed analysis.

        Raises:
            VisionProcessorError: If analysis fails.
        """
        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        start_time = time.time()
        self._processing_stats['structure_analysis']['count'] += 1

        try:
            # Create prompts
            system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(
                page_data.page_number,
                document_type
            )

            if self.config.processing.debug_mode:
                print(f"Starting structure analysis for page {page_data.page_number}")
                print(f"Image path: {page_data.image_path}")

            # Execute AI request with structured output
            page_structure = self.ollama_client.chat_with_vision(
                model=self.config.ai.structure_analysis_model,
                prompt=user_prompt,
                image_path=page_data.image_path,
                response_format=PageStructure,
                system_prompt=system_prompt
            )

            if not isinstance(page_structure, PageStructure):
                raise VisionProcessorError("Invalid response format from structure analysis")

            # Validate and update page structure
            page_structure.page_number = page_data.page_number
            page_structure.ai_model_used = self.config.ai.structure_analysis_model

            # Quality validation
            self._validate_page_structure(page_structure)

            processing_time = time.time() - start_time
            self._processing_stats['structure_analysis']['total_time'] += processing_time
            self._processing_stats['structure_analysis']['success_count'] += 1

            if self.config.processing.debug_mode:
                print(f"Structure analysis completed in {processing_time:.2f}s")
                print(f"Found: {len(page_structure.tables)} tables, "
                      f"{len(page_structure.figures)} figures, "
                      f"{len(page_structure.images)} images")

            return StructureAnalysisResult(
                success=True,
                processing_time=processing_time,
                page_structure=page_structure,
                model_used=self.config.ai.structure_analysis_model
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._processing_stats['structure_analysis']['total_time'] += processing_time

            error_msg = f"Structure analysis failed for page {page_data.page_number}: {e}"
            if self.config.processing.debug_mode:
                print(f"Error: {error_msg}")

            return StructureAnalysisResult(
                success=False,
                processing_time=processing_time,
                error_message=error_msg
            )

    def extract_assets(self, page_data: PageData) -> AssetExtractionResult:
        """
        Extract and structure assets from a page with detailed processing.

        Args:
            page_data: Page data with structure analysis results.

        Returns:
            AssetExtractionResult with extracted assets.

        Raises:
            VisionProcessorError: If extraction fails.
        """
        if not page_data.page_structure:
            raise VisionProcessorError("Page structure analysis required before asset extraction")

        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        start_time = time.time()
        self._processing_stats['asset_extraction']['count'] += 1

        try:
            extracted_assets: List[Union[TableAsset, FigureAsset, ImageAsset]] = []

            if self.config.processing.debug_mode:
                print(f"Starting asset extraction for page {page_data.page_number}")

            # Extract tables
            for table in page_data.page_structure.tables:
                try:
                    extracted_table = self._extract_table_data(page_data.image_path, table, page_data.page_number)
                    extracted_assets.append(extracted_table)

                    if self.config.processing.debug_mode:
                        print(f"Extracted table: {table.name}")

                except Exception as e:
                    if self.config.processing.debug_mode:
                        print(f"Failed to extract table {table.name}: {e}")
                    # Continue with other assets
                    extracted_assets.append(table)  # Keep original without extraction

            # Extract figures
            for figure in page_data.page_structure.figures:
                try:
                    extracted_figure = self._extract_figure_data(page_data.image_path, figure, page_data.page_number)
                    extracted_assets.append(extracted_figure)

                    if self.config.processing.debug_mode:
                        print(f"Extracted figure: {figure.name}")

                except Exception as e:
                    if self.config.processing.debug_mode:
                        print(f"Failed to extract figure {figure.name}: {e}")
                    # Continue with other assets
                    extracted_assets.append(figure)  # Keep original without extraction

            # Extract images
            for image in page_data.page_structure.images:
                try:
                    extracted_image = self._extract_image_data(page_data.image_path, image, page_data.page_number)
                    extracted_assets.append(extracted_image)

                    if self.config.processing.debug_mode:
                        print(f"Extracted image: {image.name}")

                except Exception as e:
                    if self.config.processing.debug_mode:
                        print(f"Failed to extract image {image.name}: {e}")
                    # Continue with other assets
                    extracted_assets.append(image)  # Keep original without extraction

            processing_time = time.time() - start_time
            self._processing_stats['asset_extraction']['total_time'] += processing_time
            self._processing_stats['asset_extraction']['success_count'] += 1

            if self.config.processing.debug_mode:
                print(f"Asset extraction completed in {processing_time:.2f}s")
                print(f"Extracted {len(extracted_assets)} assets")

            return AssetExtractionResult(
                success=True,
                processing_time=processing_time,
                extracted_assets=extracted_assets,
                model_used=self.config.ai.asset_extraction_model
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._processing_stats['asset_extraction']['total_time'] += processing_time

            error_msg = f"Asset extraction failed for page {page_data.page_number}: {e}"
            if self.config.processing.debug_mode:
                print(f"Error: {error_msg}")

            return AssetExtractionResult(
                success=False,
                processing_time=processing_time,
                error_message=error_msg
            )

    def generate_markdown(
        self,
        page_data: PageData,
        document_type: str = "technical document"
    ) -> MarkdownGenerationResult:
        """
        Generate comprehensive Markdown content for a page.

        Args:
            page_data: Page data with structure and asset extraction results.
            document_type: Type of document being processed.

        Returns:
            MarkdownGenerationResult with generated content.

        Raises:
            VisionProcessorError: If generation fails.
        """
        if not page_data.page_structure:
            raise VisionProcessorError("Page structure analysis required before Markdown generation")

        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        start_time = time.time()
        self._processing_stats['markdown_generation']['count'] += 1

        try:
            if self.config.processing.debug_mode:
                print(f"Starting Markdown generation for page {page_data.page_number}")

            # Create prompts with context
            system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(
                page_data,
                document_type
            )

            # Execute AI request
            markdown_content = self.ollama_client.chat_with_vision(
                model=self.config.ai.markdown_generation_model,
                prompt=user_prompt,
                image_path=page_data.image_path,
                response_format=None,  # Free-form text output
                system_prompt=system_prompt
            )

            if not markdown_content or not markdown_content.strip():
                raise VisionProcessorError("Empty or invalid Markdown content generated")

            # Clean up and validate Markdown content
            markdown_content = self._clean_markdown_content(markdown_content)

            # Extract asset references
            asset_references = self._extract_asset_references(markdown_content)

            processing_time = time.time() - start_time
            self._processing_stats['markdown_generation']['total_time'] += processing_time
            self._processing_stats['markdown_generation']['success_count'] += 1

            if self.config.processing.debug_mode:
                print(f"Markdown generation completed in {processing_time:.2f}s")
                print(f"Generated {len(markdown_content)} characters")
                print(f"Found {len(asset_references)} asset references")

            return MarkdownGenerationResult(
                success=True,
                processing_time=processing_time,
                markdown_content=markdown_content,
                asset_references=asset_references,
                model_used=self.config.ai.markdown_generation_model
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self._processing_stats['markdown_generation']['total_time'] += processing_time

            error_msg = f"Markdown generation failed for page {page_data.page_number}: {e}"
            if self.config.processing.debug_mode:
                print(f"Error: {error_msg}")

            return MarkdownGenerationResult(
                success=False,
                processing_time=processing_time,
                error_message=error_msg
            )

    def _extract_table_data(self, image_path: Path, table: TableAsset, page_number: int) -> TableAsset:
        """Extract detailed table data with enhanced prompts."""
        try:
            system_prompt, user_prompt = AssetExtractionPrompts.create_table_prompt(table, page_number)

            csv_content = self.ollama_client.chat_with_vision(
                model=self.config.ai.asset_extraction_model,
                prompt=user_prompt,
                image_path=image_path,
                response_format=None,
                system_prompt=system_prompt
            )

            if csv_content and csv_content.strip():
                # Clean and validate CSV content
                csv_content = self._clean_csv_content(csv_content)

                # Parse CSV to extract columns and rows
                lines = csv_content.strip().split('\n')
                if lines:
                    import csv
                    import io

                    csv_reader = csv.reader(io.StringIO(csv_content))
                    rows = list(csv_reader)

                    if rows:
                        columns = rows[0] if rows else []
                        data_rows = rows[1:] if len(rows) > 1 else []

                        table.csv_data = csv_content
                        table.columns = columns
                        table.rows = data_rows

            return table

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Table extraction failed for {table.name}: {e}")
            return table

    def _extract_figure_data(self, image_path: Path, figure: FigureAsset, page_number: int) -> FigureAsset:
        """Extract structured figure data with enhanced prompts."""
        try:
            system_prompt, user_prompt = AssetExtractionPrompts.create_figure_prompt(figure, page_number)

            # Define response schema for structured figure data
            class FigureStructureResponse(BaseModel):
                elements: List[Dict[str, Any]]
                relationships: List[Dict[str, Any]]
                layout: Dict[str, Any]
                metadata: Dict[str, Any]

            figure_data = self.ollama_client.chat_with_vision(
                model=self.config.ai.asset_extraction_model,
                prompt=user_prompt,
                image_path=image_path,
                response_format=FigureStructureResponse,
                system_prompt=system_prompt
            )

            if isinstance(figure_data, FigureStructureResponse):
                figure.raw_json = figure_data.model_dump()

                # TODO: Convert to FigureNode structure in future iterations
                # For now, store in raw_json format

            return figure

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Figure extraction failed for {figure.name}: {e}")
            return figure

    def _extract_image_data(self, image_path: Path, image: ImageAsset, page_number: int) -> ImageAsset:
        """Extract detailed image metadata and description."""
        try:
            system_prompt, user_prompt = AssetExtractionPrompts.create_image_prompt(image, page_number)

            description = self.ollama_client.chat_with_vision(
                model=self.config.ai.asset_extraction_model,
                prompt=user_prompt,
                image_path=image_path,
                response_format=None,
                system_prompt=system_prompt
            )

            if description and description.strip():
                # Update image with enhanced description
                image.description = description.strip()
                image.image_type = image.image_type or "analyzed"

            return image

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Image extraction failed for {image.name}: {e}")
            return image

    def _validate_page_structure(self, page_structure: PageStructure) -> None:
        """Validate page structure quality."""
        # Basic validation
        if page_structure.page_number < 1:
            raise VisionProcessorError("Invalid page number in structure")

        all_assets = page_structure.tables + page_structure.figures + page_structure.images
        for asset in all_assets:
            if not asset.name:
                raise VisionProcessorError("Asset name is missing")

            if not asset.bbox or len(asset.bbox.bbox_2d) != 4:
                raise VisionProcessorError(f"Invalid bounding box for {asset.name}")

            bbox = asset.bbox.bbox_2d
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise VisionProcessorError(f"Invalid bounding box coordinates for {asset.name}")

    def _clean_csv_content(self, csv_content: str) -> str:
        """Clean and validate CSV content."""
        # Remove any markdown formatting
        lines: List[str] = []
        for line in csv_content.split('\n'):
            line = line.strip()
            # Skip empty lines and markdown formatting
            if line and not line.startswith('```'):
                lines.append(line)

        return '\n'.join(lines)

    def _clean_markdown_content(self, markdown_content: str) -> str:
        """Clean and validate Markdown content."""
        # Remove any extra formatting or artifacts
        lines: List[str] = []
        for line in markdown_content.split('\n'):
            # Skip lines that look like AI response artifacts
            if line.strip() and not line.strip().startswith('```'):
                lines.append(line)

        return '\n'.join(lines).strip()

    def _extract_asset_references(self, markdown_content: str) -> List[str]:
        """Extract asset file references from Markdown content."""
        import re

        references: List[str] = []

        # Find Markdown links and images
        link_pattern = r'$$([^$$]+)$$$$([^)]+)$$'
        image_pattern = r'!$$([^$$]*)$$$$([^)]+)$$'

        for match in re.finditer(link_pattern, markdown_content):
            ref_path = match.group(2)
            if any(ref_path.startswith(f'./{folder}/') for folder in ['csv', 'figures', 'images']):
                references.append(ref_path)

        for match in re.finditer(image_pattern, markdown_content):
            ref_path = match.group(2)
            if any(ref_path.startswith(f'./{folder}/') for folder in ['csv', 'figures', 'images']):
                references.append(ref_path)

        return list(set(references))  # Remove duplicates

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats: Dict[str, Any] = {'processing_stats': self._processing_stats.copy()}

        # Calculate averages
        for stage_stats in stats['processing_stats'].values():
            if stage_stats['count'] > 0:
                stage_stats['average_time'] = stage_stats['total_time'] / stage_stats['count']
                stage_stats['success_rate'] = stage_stats['success_count'] / stage_stats['count']
            else:
                stage_stats['average_time'] = 0.0
                stage_stats['success_rate'] = 0.0

        # Add Ollama client stats
        stats['ollama_stats'] = self.ollama_client.get_processing_stats()

        # Add model configuration as a separate key to avoid type conflicts
        stats['model_configuration'] = {
            'structure_analysis': self.config.ai.structure_analysis_model,
            'asset_extraction': self.config.ai.asset_extraction_model,
            'markdown_generation': self.config.ai.markdown_generation_model,
        }

        return stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        for stage_stats in self._processing_stats.values():
            stage_stats.update({'count': 0, 'total_time': 0.0, 'success_count': 0})

        self.ollama_client.reset_stats()


# Maintain backward compatibility
VisionProcessor = EnhancedVisionProcessor
