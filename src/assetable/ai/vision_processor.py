"""
Vision-based AI processing for Assetable.

This module provides Vision AI processing capabilities using Ollama.
It implements the three-stage processing approach for page analysis.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..config import AssetableConfig, get_config
from ..models import (
    AIInput,
    AssetExtractionOutput,
    BoundingBox,
    CrossPageReference,
    FigureAsset,
    ImageAsset,
    MarkdownGenerationOutput,
    PageData,
    PageStructure,
    ProcessingStage,
    ReferenceType,
    StructureAnalysisOutput,
    TableAsset,
)
from .ollama_client import OllamaClient, OllamaError


class VisionProcessorError(Exception):
    """Base exception for vision processor operations."""
    pass


class VisionProcessor:
    """
    Vision-based AI processor for document analysis.

    This class implements the three-stage processing approach:
    1. Structure Analysis: Detect text, images, figures, tables, and cross-page references
    2. Asset Extraction: Extract and structure figures, tables, and images
    3. Markdown Generation: Generate complete Markdown with asset links
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """
        Initialize vision processor.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.ollama_client = OllamaClient(self.config)

        # Verify Ollama connection
        if not self.ollama_client.check_connection():
            raise VisionProcessorError("Cannot connect to Ollama server")

    def analyze_page_structure(self, page_data: PageData) -> StructureAnalysisOutput:
        """
        Perform structure analysis on a page image.

        This is the first stage of processing that identifies:
        - Text content
        - Tables (with positions)
        - Figures (with positions)
        - Images (with positions)
        - Cross-page references

        Args:
            page_data: Page data containing image path.

        Returns:
            StructureAnalysisOutput containing page structure.

        Raises:
            VisionProcessorError: If analysis fails.
        """
        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        try:
            # Prepare prompt for structure analysis
            prompt = self._create_structure_analysis_prompt(page_data.page_number)

            # System prompt for structure analysis
            system_prompt = """You are a document analysis expert. Analyze the provided page image and identify:
1. All text content on the page
2. Tables with their positions and content structure
3. Figures/diagrams with their positions and types
4. Images with their positions and content types
5. Any references to other pages, headings, tables, figures, or images

Provide accurate bounding box coordinates using absolute pixel coordinates.
Be thorough but precise in your analysis.
Your output must be a valid JSON that conforms to the PageStructure schema.
"""

            # Execute AI request with structured output
            result = self.ollama_client.chat_with_vision(
                model=self.config.ai.structure_analysis_model,
                prompt=prompt,
                image_path=page_data.image_path,
                response_format=PageStructure,
                system_prompt=system_prompt
            )

            # Validate that result is PageStructure
            if not isinstance(result, PageStructure):
                raise VisionProcessorError("Invalid response format from structure analysis")

            # Ensure page number matches
            result.page_number = page_data.page_number
            result.ai_model_used = self.config.ai.structure_analysis_model

            return StructureAnalysisOutput(
                page_structure=result,
                model_used=self.config.ai.structure_analysis_model
            )

        except Exception as e:
            if isinstance(e, VisionProcessorError):
                raise
            raise VisionProcessorError(f"Structure analysis failed for page {page_data.page_number}: {e}")

    def extract_assets(self, page_data: PageData) -> AssetExtractionOutput:
        """
        Extract and structure assets from a page.

        This is the second stage that processes:
        - Tables → CSV format
        - Figures → JSON structure
        - Images → Metadata and potential cropping

        Args:
            page_data: Page data with structure analysis results.

        Returns:
            AssetExtractionOutput containing extracted assets.

        Raises:
            VisionProcessorError: If extraction fails.
        """
        if not page_data.page_structure:
            raise VisionProcessorError("Page structure analysis required before asset extraction")

        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        try:
            extracted_assets: List[Union[TableAsset, FigureAsset, ImageAsset]] = [] # Add type hint

            # Extract tables
            if page_data.page_structure.tables:
                for table in page_data.page_structure.tables:
                    extracted_table = self._extract_table_data(page_data.image_path, table)
                    extracted_assets.append(extracted_table)

            # Extract figures
            if page_data.page_structure.figures:
                for figure in page_data.page_structure.figures:
                    extracted_figure = self._extract_figure_data(page_data.image_path, figure)
                    extracted_assets.append(extracted_figure)

            # Extract images
            if page_data.page_structure.images:
                for image in page_data.page_structure.images:
                    extracted_image = self._extract_image_data(page_data.image_path, image)
                    extracted_assets.append(extracted_image)

            return AssetExtractionOutput(
                extracted_assets=extracted_assets,
                model_used=self.config.ai.asset_extraction_model
            )

        except Exception as e:
            if isinstance(e, VisionProcessorError):
                raise
            raise VisionProcessorError(f"Asset extraction failed for page {page_data.page_number}: {e}")

    def generate_markdown(self, page_data: PageData) -> MarkdownGenerationOutput:
        """
        Generate complete Markdown for a page.

        This is the third stage that creates:
        - Complete Markdown text
        - Proper asset references and links
        - Structured headings and content

        Args:
            page_data: Page data with structure and asset extraction results.

        Returns:
            MarkdownGenerationOutput containing Markdown content.

        Raises:
            VisionProcessorError: If generation fails.
        """
        if not page_data.page_structure:
            raise VisionProcessorError("Page structure analysis required before Markdown generation")

        if not page_data.image_path or not page_data.image_path.exists():
            raise VisionProcessorError(f"Image file not found for page {page_data.page_number}")

        try:
            # Prepare prompt for Markdown generation
            prompt = self._create_markdown_generation_prompt(page_data)

            # System prompt for Markdown generation
            system_prompt = """You are a document conversion expert. Generate clean, well-structured Markdown content from the page image.

Requirements:
1. Create proper headings and structure
2. Convert all text content to Markdown
3. Reference tables, figures, and images using the provided asset information
4. Use relative links for assets (e.g., ./csv/table_name.csv, ./figures/figure_name.json)
5. Maintain reading flow and logical structure
6. Include any cross-page references found

Output clean, readable Markdown that accurately represents the page content."""

            # Execute AI request
            markdown_content = self.ollama_client.chat_with_vision(
                model=self.config.ai.markdown_generation_model,
                prompt=prompt,
                image_path=page_data.image_path,
                response_format=None,  # Free-form text output
                system_prompt=system_prompt
            )

            if not isinstance(markdown_content, str):
                raise VisionProcessorError("Invalid response format from Markdown generation")

            # Extract asset references from the generated content
            asset_references = self._extract_asset_references(markdown_content)

            return MarkdownGenerationOutput(
                markdown_content=markdown_content,
                asset_references=asset_references,
                model_used=self.config.ai.markdown_generation_model
            )

        except Exception as e:
            if isinstance(e, VisionProcessorError):
                raise
            raise VisionProcessorError(f"Markdown generation failed for page {page_data.page_number}: {e}")

    def _create_structure_analysis_prompt(self, page_number: int) -> str:
        """Create prompt for structure analysis."""
        return f"""Analyze this page image (page {page_number}) and identify all structural elements.

Please identify and provide precise information about:

1. **Text Content**: All readable text on the page
2. **Tables**: Any tabular data with rows and columns
   - Provide name, description, and precise bounding box coordinates
3. **Figures**: Diagrams, charts, flowcharts, or conceptual illustrations
   - Provide name, description, type, and precise bounding box coordinates
4. **Images**: Photographs, graphics, or other visual elements
   - Provide name, description, type, and precise bounding box coordinates
5. **Cross-page References**: Any mentions of other pages, chapters, figures, tables, or headings
   - Identify reference type (page, heading, table, figure, image)
   - Provide target page number if mentioned
   - Include the exact reference text

Use absolute pixel coordinates for all bounding boxes in the format [x1, y1, x2, y2] where:
- (x1, y1) is the top-left corner
- (x2, y2) is the bottom-right corner

Your output must be a valid JSON object that conforms to the PageStructure schema.
Be thorough and accurate in your analysis."""

    def _create_markdown_generation_prompt(self, page_data: PageData) -> str:
        """Create prompt for Markdown generation."""
        # Include context from structure analysis
        context_info = ""
        if page_data.page_structure:
            tables_info = f"Tables found: {len(page_data.page_structure.tables) if page_data.page_structure.tables else 0}"
            figures_info = f"Figures found: {len(page_data.page_structure.figures) if page_data.page_structure.figures else 0}"
            images_info = f"Images found: {len(page_data.page_structure.images) if page_data.page_structure.images else 0}"
            context_info = f"\n\nPage structure context:\n- {tables_info}\n- {figures_info}\n- {images_info}"

        # Include asset information if available
        asset_info = ""
        if page_data.extracted_assets:
            asset_names = [asset.name for asset in page_data.extracted_assets]
            asset_info = f"\n\nAvailable assets: {', '.join(asset_names)}"

        return f"""Convert this page image (page {page_data.page_number}) into clean, well-structured Markdown.

Create complete Markdown content that includes:
1. Proper headings and text structure
2. All readable text content
3. References to tables using: [Table Name](./csv/page_{page_data.page_number:04d}_table_name.csv)
4. References to figures using: [Figure Name](./figures/page_{page_data.page_number:04d}_figure_name.json)
5. References to images using: ![Image Description](./images/page_{page_data.page_number:04d}_image_name.jpg)
6. Any cross-page references found in the content

Focus on readability and maintaining the logical flow of the document.{context_info}{asset_info}"""

    def _extract_table_data(self, image_path: Path, table: TableAsset) -> TableAsset:
        """Extract detailed table data from image region."""
        try:
            # Prepare prompt for table extraction
            prompt = f"""Extract the complete data from this table: "{table.name}"

Location: {table.bbox.bbox_2d if table.bbox else "Unknown"}
Description: {table.description}

Please provide:
1. All table data in CSV format
2. Clear column headers
3. All row data accurately transcribed

Output the data as clean CSV that can be directly saved to a file.
The output should only contain the CSV data, with no extra text or explanations.
"""

            # Extract table data
            csv_content = self.ollama_client.chat_with_vision(
                model=self.config.ai.asset_extraction_model,
                prompt=prompt,
                image_path=image_path,
                response_format=None  # Free-form text for CSV
            )

            if isinstance(csv_content, str) and csv_content.strip():
                # Clean up potential markdown code blocks
                csv_content = csv_content.strip()
                if csv_content.startswith("```csv"):
                    csv_content = csv_content[len("```csv"):].strip()
                if csv_content.startswith("```"):
                    csv_content = csv_content[len("```"):].strip()
                if csv_content.endswith("```"):
                    csv_content = csv_content[:-len("```")].strip()

                # Parse CSV to extract columns and rows
                lines = csv_content.strip().split('\n')
                if lines:
                    # First line as headers
                    columns = [col.strip().strip('"') for col in lines[0].split(',')]
                    # Remaining lines as rows
                    rows = []
                    for line in lines[1:]:
                        if line.strip():
                            row = [cell.strip().strip('"') for cell in line.split(',')]
                            rows.append(row)

                    # Update table with extracted data
                    table.csv_data = csv_content.strip()
                    table.columns = columns
                    table.rows = rows

            return table

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Table extraction failed for {table.name}: {e}")
            return table

    def _extract_figure_data(self, image_path: Path, figure: FigureAsset) -> FigureAsset:
        """Extract structured figure data from image region."""
        try:
            # Prepare prompt for figure extraction
            prompt = f"""Analyze and structure this figure: "{figure.name}"

Location: {figure.bbox.bbox_2d if figure.bbox else "Unknown"}
Description: {figure.description}
Type: {figure.figure_type}

Please provide a structured JSON representation of this figure that includes:
1. All text elements with their positions
2. All graphical elements (boxes, arrows, lines, etc.)
3. Relationships between elements
4. Overall structure and layout

Create a hierarchical JSON structure that captures the essence and details of this figure.
Your output must be a valid JSON object that conforms to the FigureStructureResponse schema.
"""

            # Create response model for figure structure
            class FigureStructureResponse(BaseModel):
                elements: List[Dict[str, Any]] = Field(description="List of all elements in the figure")
                relationships: List[Dict[str, Any]] = Field(description="List of relationships between elements")
                metadata: Dict[str, Any] = Field(description="Metadata about the figure")

            # Extract figure structure
            figure_data = self.ollama_client.chat_with_vision(
                model=self.config.ai.asset_extraction_model,
                prompt=prompt,
                image_path=image_path,
                response_format=FigureStructureResponse
            )

            if isinstance(figure_data, FigureStructureResponse):
                # Convert to our format
                figure.raw_json = figure_data.dict()

                # TODO: Convert to FigureNode structure
                # For now, store in raw_json format

            return figure

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Figure extraction failed for {figure.name}: {e}")
            return figure

    def _extract_image_data(self, image_path: Path, image: ImageAsset) -> ImageAsset:
        """Extract image metadata and prepare for cropping if needed."""
        try:
            # For now, just update metadata
            # Future implementation could include actual image cropping
            image.image_type = image.image_type or "extracted"
            return image

        except Exception as e:
            if self.config.processing.debug_mode:
                print(f"Image extraction failed for {image.name}: {e}")
            return image

    def _extract_asset_references(self, markdown_content: str) -> List[str]:
        """Extract asset file references from Markdown content."""
        import re

        references = []

        # Find Markdown links and images
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

        for match in re.finditer(link_pattern, markdown_content):
            references.append(match.group(2))

        for match in re.finditer(image_pattern, markdown_content):
            references.append(match.group(2))

        return references

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        ollama_stats = self.ollama_client.get_processing_stats()

        return {
            'ollama_stats': ollama_stats,
            'models_used': {
                'structure_analysis': self.config.ai.structure_analysis_model,
                'asset_extraction': self.config.ai.asset_extraction_model,
                'markdown_generation': self.config.ai.markdown_generation_model,
            }
        }
