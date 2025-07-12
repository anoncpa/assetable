"""
Configuration management for Assetable.

This module provides centralized configuration management using Pydantic.
All configuration is type-safe and validated.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PDFSplitConfig(BaseModel):
    """Configuration for PDF splitting."""

    dpi: int = Field(default=300, description="DPI for image conversion")
    image_format: str = Field(default="png", description="Output image format")

    @validator("dpi")
    def validate_dpi(cls, v: int) -> int:
        """Validate DPI value."""
        if v < 72:
            raise ValueError("DPI must be at least 72")
        if v > 600:
            raise ValueError("DPI should not exceed 600 for performance reasons")
        return v

    @validator("image_format")
    def validate_image_format(cls, v: str) -> str:
        """Validate image format."""
        allowed_formats = ["png", "jpg", "jpeg"]
        if v.lower() not in allowed_formats:
            raise ValueError(f"Image format must be one of: {allowed_formats}")
        return v.lower()


class AIConfig(BaseModel):
    """Configuration for AI processing."""

    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")

    # Model settings
    structure_analysis_model: str = Field(
        default="qwen2.5-vl:7b",
        description="Model for structure analysis"
    )
    asset_extraction_model: str = Field(
        default="qwen2.5-vl:7b",
        description="Model for asset extraction"
    )
    markdown_generation_model: str = Field(
        default="qwen2.5-vl:7b",
        description="Model for markdown generation"
    )

    # Processing settings
    max_retries: int = Field(default=3, description="Maximum retry attempts for AI calls")
    timeout_seconds: int = Field(default=300, description="Timeout for AI processing in seconds")

    # Temperature and other model parameters
    temperature: float = Field(default=0.1, description="Temperature for AI model")
    top_p: float = Field(default=0.9, description="Top-p for AI model")

    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature value."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator("top_p")
    def validate_top_p(cls, v: float) -> float:
        """Validate top-p value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")
        return v


class OutputConfig(BaseModel):
    """Configuration for output directories and file patterns."""

    # Base directories
    input_directory: Path = Field(default=Path("input"), description="Base directory for input PDFs")
    output_directory: Path = Field(default=Path("output"), description="Base directory for processed output")

    # Subdirectory names
    pdf_split_subdir: str = Field(default="pdf_split", description="Subdirectory for split page images")
    structure_subdir: str = Field(default="structure", description="Subdirectory for page structure JSONs")
    markdown_subdir: str = Field(default="markdown", description="Subdirectory for generated markdown files")
    csv_subdir: str = Field(default="csv", description="Subdirectory for extracted CSV tables")
    images_subdir: str = Field(default="images", description="Subdirectory for extracted images")
    figures_subdir: str = Field(default="figures", description="Subdirectory for extracted figures data")

    # File naming patterns
    page_image_pattern: str = Field(default="page_{page:04d}.png", description="Filename pattern for page images")
    structure_json_pattern: str = Field(default="page_{page:04d}_structure.json", description="Filename pattern for structure JSONs")
    markdown_pattern: str = Field(default="page_{page:04d}.md", description="Filename pattern for markdown files")

    @validator("input_directory", "output_directory", pre=True, always=True)
    def validate_dirs(cls, v: Path) -> Path:
        """Validate directory paths."""
        # Convert to absolute path
        absolute_path = Path(v).resolve()
        return absolute_path


class ProcessingConfig(BaseModel):
    """Configuration for processing behavior."""

    # Skip settings
    skip_existing_files: bool = Field(
        default=True,
        description="Skip processing if output files already exist"
    )

    # Parallel processing
    max_parallel_pages: int = Field(
        default=1,
        description="Maximum number of pages to process in parallel"
    )

    # Debug settings
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    save_intermediate_results: bool = Field(
        default=True,
        description="Save intermediate processing results"
    )

    # Asset extraction settings
    min_table_rows: int = Field(default=2, description="Minimum rows to consider as table")
    min_figure_elements: int = Field(default=1, description="Minimum elements to consider as figure")

    @validator("max_parallel_pages")
    def validate_parallel_pages(cls, v: int) -> int:
        """Validate parallel processing settings."""
        if v < 1:
            raise ValueError("max_parallel_pages must be at least 1")
        if v > 10:
            raise ValueError("max_parallel_pages should not exceed 10 for stability")
        return v


class AssetableConfig(BaseModel):
    """Main configuration class for Assetable."""

    # Sub-configurations
    pdf_split: PDFSplitConfig = Field(default_factory=PDFSplitConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Version info
    version: str = Field(default="0.1.0", description="Assetable version")

    @classmethod
    def from_env(cls) -> "AssetableConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if they exist
        if ollama_host := os.getenv("ASSETABLE_OLLAMA_HOST"):
            config.ai.ollama_host = ollama_host

        if structure_model := os.getenv("ASSETABLE_STRUCTURE_MODEL"):
            config.ai.structure_analysis_model = structure_model

        if asset_model := os.getenv("ASSETABLE_ASSET_MODEL"):
            config.ai.asset_extraction_model = asset_model

        if markdown_model := os.getenv("ASSETABLE_MARKDOWN_MODEL"):
            config.ai.markdown_generation_model = markdown_model

        if dpi := os.getenv("ASSETABLE_DPI"):
            config.pdf_split.dpi = int(dpi)

        if input_dir := os.getenv("ASSETABLE_INPUT_DIR"):
            config.output.input_directory = Path(input_dir)

        if output_dir := os.getenv("ASSETABLE_OUTPUT_DIR"):
            config.output.output_directory = Path(output_dir)

        if debug := os.getenv("ASSETABLE_DEBUG"):
            config.processing.debug_mode = debug.lower() in ("true", "1", "yes")

        return config

    def get_document_output_dir(self, pdf_path: Path) -> Path:
        """Get output directory for a specific PDF document."""
        pdf_name = pdf_path.stem
        return self.output.output_directory / pdf_name

    def get_pdf_split_dir(self, pdf_path: Path) -> Path:
        """Get PDF split directory for a specific document."""
        return self.get_document_output_dir(pdf_path) / self.output.pdf_split_subdir

    def get_structure_dir(self, pdf_path: Path) -> Path:
        """Get page structure directory for a specific document."""
        return self.get_document_output_dir(pdf_path) / self.output.structure_subdir

    def get_markdown_dir(self, pdf_path: Path) -> Path:
        """Get markdown directory for a specific document."""
        return self.get_document_output_dir(pdf_path) / self.output.markdown_subdir

    def get_csv_dir(self, pdf_path: Path) -> Path:
        """Get CSV directory for a specific document."""
        return self.get_markdown_dir(pdf_path) / self.output.csv_subdir

    def get_images_dir(self, pdf_path: Path) -> Path:
        """Get images directory for a specific document."""
        return self.get_markdown_dir(pdf_path) / self.output.images_subdir

    def get_figures_dir(self, pdf_path: Path) -> Path:
        """Get figures directory for a specific document."""
        return self.get_markdown_dir(pdf_path) / self.output.figures_subdir

    def get_page_image_path(self, pdf_path: Path, page_number: int) -> Path:
        """Get path for a specific page image."""
        filename = self.output.page_image_pattern.format(page=page_number)
        return self.get_pdf_split_dir(pdf_path) / filename

    def get_structure_json_path(self, pdf_path: Path, page_number: int) -> Path:
        """Get path for a specific page structure JSON."""
        filename = self.output.structure_json_pattern.format(page=page_number)
        return self.get_structure_dir(pdf_path) / filename

    def get_markdown_path(self, pdf_path: Path, page_number: int) -> Path:
        """Get path for a specific page markdown file."""
        filename = self.output.markdown_pattern.format(page=page_number)
        return self.get_markdown_dir(pdf_path) / filename

    def get_asset_path(self, pdf_path: Path, page_number: int, asset_type: str, asset_name: str) -> Path:
        """Get path for a specific asset file."""
        # Clean asset name for filename
        clean_name = "".join(c for c in asset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_name = clean_name.replace(' ', '_')

        if asset_type == "table":
            filename = f"page_{page_number:04d}_{clean_name}.csv"
            return self.get_csv_dir(pdf_path) / filename
        elif asset_type == "figure":
            filename = f"page_{page_number:04d}_{clean_name}.json"
            return self.get_figures_dir(pdf_path) / filename
        elif asset_type == "image":
            # Assuming common image extension, can be made more robust
            filename = f"page_{page_number:04d}_{clean_name}.jpg"
            return self.get_images_dir(pdf_path) / filename
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

    def create_output_directories(self, pdf_path: Path) -> None:
        """Create all necessary output directories for a document."""
        directories = [
            self.get_document_output_dir(pdf_path),
            self.get_pdf_split_dir(pdf_path),
            self.get_structure_dir(pdf_path),
            self.get_markdown_dir(pdf_path),
            self.get_csv_dir(pdf_path),
            self.get_images_dir(pdf_path),
            self.get_figures_dir(pdf_path),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        # Pydantic v2 uses model_dump, v1 uses dict
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        import json

        with open(file_path, 'w', encoding='utf-8') as f:
            # Pydantic v2 uses model_dump_json
            if hasattr(self, "model_dump_json"):
                f.write(self.model_dump_json(indent=2))
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "AssetableConfig":
        """Load configuration from JSON file."""
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(**data)


# Global configuration instance
_config: Optional[AssetableConfig] = None


def get_config() -> AssetableConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AssetableConfig.from_env()
    return _config


def set_config(config: AssetableConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
