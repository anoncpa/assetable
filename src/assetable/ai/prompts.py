"""
AI processing prompts for Assetable.

This module provides structured prompts for the three-stage AI processing:
1. Structure Analysis
2. Asset Extraction
3. Markdown Generation
"""

from typing import Any

from pydantic import BaseModel

from ..models import PageData, TableAsset, FigureAsset, ImageAsset


class PromptTemplate(BaseModel):
    """Base class for prompt templates."""

    system_prompt: str
    user_prompt_template: str

    def format_user_prompt(self, **kwargs: Any) -> str:
        """Format the user prompt with provided parameters."""
        return self.user_prompt_template.format(**kwargs)


class StructureAnalysisPrompts:
    """Prompts for structure analysis stage."""

    SYSTEM_PROMPT = """You are an expert document analyzer specializing in Japanese and English text recognition.
Your task is to analyze page images from scanned books and identify structural elements with high precision.

Requirements:
- Identify all text content on the page
- Detect tables, figures, diagrams, and images with accurate bounding boxes
- Use absolute pixel coordinates [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
- Identify cross-page references (page numbers, figure/table references, etc.)
- Provide descriptive names for all detected elements
- Be thorough but accurate in your analysis

Always respond with valid JSON matching the specified schema."""

    USER_PROMPT_TEMPLATE = """Analyze this page image (page {page_number}) and provide a complete structural analysis.

Please identify:

1. **Text Content**: All readable text on the page
2. **Tables**: Tabular data with rows and columns
   - Provide descriptive name, detailed description, and precise bounding box
3. **Figures**: Diagrams, charts, flowcharts, or conceptual illustrations
   - Provide descriptive name, detailed description, figure type, and precise bounding box
4. **Images**: Photographs, graphics, or other visual elements
   - Provide descriptive name, detailed description, image type, and precise bounding box
5. **Cross-page References**: Any references to other pages, chapters, figures, tables, or headings
   - Identify reference type (page, heading, table, figure, image)
   - Extract target page number if mentioned
   - Include exact reference text

Use absolute pixel coordinates for bounding boxes in format [x1, y1, x2, y2].
Be precise and thorough in your analysis.

Context: This is page {page_number} from a {document_type} document."""

    @classmethod
    def create_prompt(
        cls,
        page_number: int,
        document_type: str = "technical book"
    ) -> tuple[str, str]:
        """Create system and user prompts for structure analysis."""
        user_prompt = cls.USER_PROMPT_TEMPLATE.format(
            page_number=page_number,
            document_type=document_type
        )
        return cls.SYSTEM_PROMPT, user_prompt


class AssetExtractionPrompts:
    """Prompts for asset extraction stage."""

    SYSTEM_PROMPT = """You are an expert in extracting and structuring data from document images.
Your task is to extract detailed content from identified tables, figures, and images.

For tables:
- Extract all data in CSV format with proper headers
- Maintain original structure and relationships
- Handle merged cells appropriately

For figures:
- Create structured JSON representation
- Identify all elements (boxes, arrows, text, connections)
- Capture hierarchical relationships
- Preserve spatial layout information

For images:
- Provide detailed descriptions
- Identify key visual elements
- Note any text within images

Always provide clean, structured output that can be saved directly to files."""

    TABLE_EXTRACTION_TEMPLATE = """Extract complete data from this table: "{table_name}"

Location: {bbox}
Description: {description}
Context: Page {page_number} from the document

Please provide:
1. All table data in clean CSV format
2. Proper column headers
3. All row data accurately transcribed
4. Handle any merged cells or complex structures

Output clean CSV that can be saved directly to a file.
Include only the CSV data in your response, no additional formatting or explanation."""

    FIGURE_EXTRACTION_TEMPLATE = """Analyze and structure this figure: "{figure_name}"

Location: {bbox}
Description: {description}
Figure Type: {figure_type}
Context: Page {page_number} from the document

Create a detailed JSON structure that includes:
1. All text elements with their positions and content
2. All graphical elements (boxes, arrows, lines, shapes)
3. Relationships and connections between elements
4. Overall layout and structure
5. Any flow or process information

Provide a hierarchical JSON that captures the essence and details of this figure.
Make it suitable for programmatic reconstruction or analysis."""

    IMAGE_EXTRACTION_TEMPLATE = """Analyze this image: "{image_name}"

Location: {bbox}
Description: {description}
Image Type: {image_type}
Context: Page {page_number} from the document

Provide:
1. Detailed description of visual content
2. Any text visible in the image
3. Key visual elements (objects, people, scenes)
4. Color scheme and composition
5. Technical details if applicable (charts, graphs, etc.)

Focus on extracting all meaningful information that could be referenced in text."""

    @classmethod
    def create_table_prompt(cls, table_asset: TableAsset, page_number: int) -> tuple[str, str]:
        """Create prompts for table extraction."""
        user_prompt = cls.TABLE_EXTRACTION_TEMPLATE.format(
            table_name=table_asset.name,
            bbox=table_asset.bbox.bbox_2d,
            description=table_asset.description,
            page_number=page_number
        )
        return cls.SYSTEM_PROMPT, user_prompt

    @classmethod
    def create_figure_prompt(cls, figure_asset: FigureAsset, page_number: int) -> tuple[str, str]:
        """Create prompts for figure extraction."""
        user_prompt = cls.FIGURE_EXTRACTION_TEMPLATE.format(
            figure_name=figure_asset.name,
            bbox=figure_asset.bbox.bbox_2d,
            description=figure_asset.description,
            figure_type=figure_asset.figure_type,
            page_number=page_number
        )
        return cls.SYSTEM_PROMPT, user_prompt

    @classmethod
    def create_image_prompt(cls, image_asset: ImageAsset, page_number: int) -> tuple[str, str]:
        """Create prompts for image extraction."""
        user_prompt = cls.IMAGE_EXTRACTION_TEMPLATE.format(
            image_name=image_asset.name,
            bbox=image_asset.bbox.bbox_2d,
            description=image_asset.description,
            image_type=image_asset.image_type or "general",
            page_number=page_number
        )
        return cls.SYSTEM_PROMPT, user_prompt


class MarkdownGenerationPrompts:
    """Prompts for Markdown generation stage."""

    SYSTEM_PROMPT = """You are an expert technical writer specializing in converting document images to clean, well-structured Markdown.

Your task is to create comprehensive Markdown content that:
- Maintains the original document structure and flow
- Uses proper Markdown formatting (headings, lists, emphasis)
- References external assets using relative links
- Preserves the logical reading order
- Includes all textual content from the page

For asset references:
- Tables: [Table Name](./csv/page_XXXX_table_name.csv)
- Figures: [Figure Name](./figures/page_XXXX_figure_name.json)
- Images: ![Image Description](./images/page_XXXX_image_name.jpg)

Create clean, readable Markdown that accurately represents the page content."""

    USER_PROMPT_TEMPLATE = """Convert this page image (page {page_number}) into clean, well-structured Markdown content.

Page Context:
- Page Number: {page_number}
- Document Type: {document_type}
{structure_context}
{asset_context}

Requirements:
1. Create proper heading hierarchy and document structure
2. Convert all readable text to appropriate Markdown formatting
3. Reference assets using these specific patterns:
   - Tables: [Table Name](./csv/page_{page_number:04d}_table_name.csv)
   - Figures: [Figure Name](./figures/page_{page_number:04d}_figure_name.json)
   - Images: ![Image Description](./images/page_{page_number:04d}_image_name.jpg)
4. Maintain logical reading flow and document structure
5. Include any cross-page references found in the content
6. Use appropriate Markdown syntax for emphasis, lists, code blocks, etc.

Focus on creating comprehensive, readable content that serves as a complete textual representation of this page.

Output only the Markdown content, no additional formatting or explanation."""

    @classmethod
    def create_prompt(
        cls,
        page_data: PageData,
        document_type: str = "technical document"
    ) -> tuple[str, str]:
        """Create system and user prompts for Markdown generation."""
        # Build structure context
        structure_context = ""
        if page_data.page_structure:
            structure = page_data.page_structure
            structure_parts: list[str] = []
            if structure.tables:
                structure_parts.append(f"Tables: {len(structure.tables)}")
            if structure.figures:
                structure_parts.append(f"Figures: {len(structure.figures)}")
            if structure.images:
                structure_parts.append(f"Images: {len(structure.images)}")
            if structure.references:
                structure_parts.append(f"Cross-references: {len(structure.references)}")

            if structure_parts:
                structure_context = f"\n- Detected elements: {', '.join(structure_parts)}"

        # Build asset context
        asset_context = ""
        if page_data.extracted_assets:
            asset_names = [asset.name for asset in page_data.extracted_assets]
            asset_context = f"\n- Available assets: {', '.join(asset_names)}"

        user_prompt = cls.USER_PROMPT_TEMPLATE.format(
            page_number=page_data.page_number,
            document_type=document_type,
            structure_context=structure_context,
            asset_context=asset_context
        )

        return cls.SYSTEM_PROMPT, user_prompt
