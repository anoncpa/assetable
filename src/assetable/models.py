"""
Assetable data models for PDF processing pipeline.

This module defines Pydantic models for the entire pipeline from PDF splitting
to final Markdown generation. The main pattern is gradual data expansion -
a single PageData object is passed through the pipeline and extended at each stage.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, field_validator


class ProcessingStage(str, Enum):
    """Processing stages in the pipeline."""

    PDF_SPLIT = "pdf_split"
    STRUCTURE_ANALYSIS = "structure_analysis"
    ASSET_EXTRACTION = "asset_extraction"
    MARKDOWN_GENERATION = "markdown_generation"
    COMPLETED = "completed"


class BoundingBox(BaseModel):
    """
    Bounding box coordinates using Qwen2.5-VL format.
    Uses absolute coordinates [x1, y1, x2, y2] where (x1, y1) is top-left
    and (x2, y2) is bottom-right.
    """

    bbox_2d: List[int] = Field(min_length=4, max_length=4)

    @field_validator("bbox_2d")
    def validate_bbox(cls, v: List[int]) -> List[int]:
        """Validate bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("bbox_2d must have exactly 4 coordinates")
        x1, y1, x2, y2 = v
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid bounding box: top-left must be before bottom-right")
        return v

    @property
    def x1(self) -> int:
        """Top-left x coordinate."""
        return self.bbox_2d[0]

    @property
    def y1(self) -> int:
        """Top-left y coordinate."""
        return self.bbox_2d[1]

    @property
    def x2(self) -> int:
        """Bottom-right x coordinate."""
        return self.bbox_2d[2]

    @property
    def y2(self) -> int:
        """Bottom-right y coordinate."""
        return self.bbox_2d[3]

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1


class AssetType(str, Enum):
    """Types of assets that can be extracted from pages."""

    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"


class ReferenceType(str, Enum):
    """Types of cross-page references."""

    PAGE = "page"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"


class AssetBase(BaseModel):
    """Base class for all page assets."""

    # 基底クラスでは型アノテーションのみを定義し、デフォルト値は設定しない
    name: str = Field(description="Descriptive name for the asset")
    description: str = Field(description="Detailed description of the asset content")
    bbox: BoundingBox = Field(description="Location of the asset on the page")

    @field_validator("name")
    def validate_name(cls, v: str) -> str:
        """Validate asset name format."""
        if not v.strip():
            raise ValueError("Asset name cannot be empty")
        # Remove characters that are problematic for filenames
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '|']
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Asset name cannot contain '{char}'")
        return v.strip()


class TableAsset(AssetBase):
    """Table asset with CSV data."""

    type: Literal[AssetType.TABLE] = Field(default=AssetType.TABLE)
    csv_data: Optional[str] = Field(default=None, description="CSV format table data")
    columns: Optional[List[str]] = Field(default=None, description="Column headers")
    rows: Optional[List[List[str]]] = Field(default=None, description="Table rows")


class FigureNode(BaseModel):
    """Node in a figure tree structure."""

    id: str = Field(description="Unique identifier for this node")
    type: str = Field(description="Type of the node (e.g., 'box', 'arrow', 'text')")
    label: Optional[str] = Field(default=None, description="Text label for the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    children: List["FigureNode"] = Field(default_factory=list, description="Child nodes")
    position: Optional[BoundingBox] = Field(default=None, description="Position within the figure")


# Enable forward references for recursive model
FigureNode.model_rebuild()


class FigureAsset(AssetBase):
    """Figure asset with structured JSON data."""

    type: Literal[AssetType.FIGURE] = Field(default=AssetType.FIGURE)
    figure_type: str = Field(description="Type of figure (e.g., 'flowchart', 'diagram', 'chart')")
    structure: Optional[List[FigureNode]] = Field(default=None, description="Structured figure data")
    raw_json: Optional[Dict[str, Any]] = Field(default=None, description="Raw JSON structure")


class ImageAsset(AssetBase):
    """Image asset extracted from page."""

    type: Literal[AssetType.IMAGE] = Field(default=AssetType.IMAGE)
    image_path: Optional[Path] = Field(default=None, description="Path to extracted image file")
    image_type: Optional[str] = Field(default=None, description="Type of image content")


class CrossPageReference(BaseModel):
    """Reference to another page in the document."""

    target_page: int = Field(description="Page number being referenced (1-based)")
    reference_text: str = Field(description="Text of the reference")
    reference_type: ReferenceType = Field(description="Type of reference")


class PageStructure(BaseModel):
    """
    Page structure analysis result from AI.
    This is the output of the first AI processing stage.
    """

    page_number: int = Field(description="Page number (1-based)")
    text_content: Optional[str] = Field(default=None, description="all the recognized text content on the page of the book. no need to contain coordinates.")

    # Assets detected on the page
    tables: List[TableAsset] = Field(default_factory=list, description="Tables found on the page")
    figures: List[FigureAsset] = Field(default_factory=list, description="Figures found on the page")
    images: List[ImageAsset] = Field(default_factory=list, description="Images found on the page")

    # Cross-page references
    references: List[CrossPageReference] = Field(default_factory=list, description="References to other pages")

    # Processing metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    ai_model_used: Optional[str] = Field(default=None, description="AI model used for analysis")


class AIInput(BaseModel):
    """Input data for AI processing."""

    prompt: str = Field(description="Prompt for the AI model")
    image_path: Path = Field(description="Path to the image file")
    ocr_text: Optional[str] = Field(default=None, description="Pre-recognized text data")
    page_structure: Optional[PageStructure] = Field(default=None, description="Page structure from previous stage")

    @field_validator("image_path")
    def validate_image_path(cls, v: Path) -> Path:
        """Validate image path exists."""
        if not v.exists():
            raise ValueError(f"Image file does not exist: {v}")
        if not v.is_file():
            raise ValueError(f"Image path is not a file: {v}")
        return v


class AIOutput(BaseModel):
    """Base class for AI processing outputs."""

    processing_timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(description="AI model used for processing")
    processing_time_seconds: Optional[float] = Field(default=None, description="Processing time in seconds")


class StructureAnalysisOutput(AIOutput):
    """Output from structure analysis AI processing."""

    page_structure: PageStructure = Field(description="Analyzed page structure")


class AssetExtractionOutput(AIOutput):
    """Output from asset extraction AI processing."""

    extracted_assets: List[Union[TableAsset, FigureAsset, ImageAsset]] = Field(
        description="Assets extracted from the page"
    )


class MarkdownGenerationOutput(AIOutput):
    """Output from markdown generation AI processing."""

    markdown_content: str = Field(description="Generated Markdown content")
    asset_references: List[str] = Field(description="List of asset files referenced in the Markdown")


class PageData(BaseModel):
    """
    Main data object that flows through the entire pipeline.
    This object is gradually extended at each processing stage.
    """

    # Basic page information
    page_number: int = Field(description="Page number (1-based)")
    source_pdf: Path = Field(description="Path to the source PDF file")

    # Processing status
    current_stage: ProcessingStage = Field(default=ProcessingStage.PDF_SPLIT)
    completed_stages: List[ProcessingStage] = Field(default_factory=list)

    # File paths for each stage
    image_path: Optional[Path] = Field(default=None, description="Path to the page image")
    structure_json_path: Optional[Path] = Field(default=None, description="Path to structure analysis JSON")
    markdown_path: Optional[Path] = Field(default=None, description="Path to generated Markdown file")

    # Processing results (gradually populated)
    page_structure: Optional[PageStructure] = Field(default=None, description="Page structure analysis")
    extracted_assets: List[Union[TableAsset, FigureAsset, ImageAsset]] = Field(
        default_factory=list, description="Extracted assets"
    )
    markdown_content: Optional[str] = Field(default=None, description="Generated Markdown content")

    # Asset file paths
    asset_files: Dict[str, Path] = Field(default_factory=dict, description="Mapping of asset names to file paths")

    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    processing_log: List[str] = Field(default_factory=list, description="Processing log messages")

    def add_log(self, message: str) -> None:
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.processing_log.append(f"[{timestamp}] {message}")
        self.last_updated = datetime.now()

    def mark_stage_completed(self, stage: ProcessingStage) -> None:
        """Mark a processing stage as completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.current_stage = stage
        self.last_updated = datetime.now()
        self.add_log(f"Completed stage: {stage.value}")

    def is_stage_completed(self, stage: ProcessingStage) -> bool:
        """Check if a processing stage is completed."""
        return stage in self.completed_stages

    def get_next_stage(self) -> Optional[ProcessingStage]:
        """Get the next processing stage."""
        stages = list(ProcessingStage)
        try:
            current_index = stages.index(self.current_stage)
            if current_index + 1 < len(stages):
                return stages[current_index + 1]
        except ValueError:
            # Current stage not in defined stages (should not happen)
            return None
        return None


class DocumentData(BaseModel):
    """
    Represents an entire document being processed.
    Contains data for all pages and overall document metadata.
    """

    document_id: str = Field(description="Unique identifier for the document")
    source_pdf_path: Path = Field(description="Path to the source PDF file")
    output_directory: Path = Field(description="Directory for processed output files")
    pages: List[PageData] = Field(default_factory=list, description="List of page data objects")

    # Document-level metadata
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    publication_date: Optional[datetime] = Field(default=None, description="Publication date")

    # Processing status
    overall_status: ProcessingStage = Field(default=ProcessingStage.PDF_SPLIT)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_page_by_number(self, page_number: int) -> Optional[PageData]:
        """Get page data by page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def add_page(self, page_data: PageData) -> None:
        """Add page data to the document."""
        # Remove existing page with same number
        self.pages = [p for p in self.pages if p.page_number != page_data.page_number]
        self.pages.append(page_data)
        self.pages.sort(key=lambda p: p.page_number)
        self.last_updated = datetime.now()

    def get_completed_pages(self, stage: ProcessingStage) -> List[PageData]:
        """Get all pages that have completed a specific stage."""
        return [page for page in self.pages if page.is_stage_completed(stage)]

    def get_pending_pages(self, stage: ProcessingStage) -> List[PageData]:
        """Get all pages that need to complete a specific stage."""
        return [page for page in self.pages if not page.is_stage_completed(stage)]
