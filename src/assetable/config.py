"""
Configuration management for Assetable.

This module provides centralized configuration handling using Pydantic.
All configuration is type-safe and validated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, cast
from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv



# PDF splitting settings


class PDFSplitConfig(BaseModel):
    """Configuration for PDF splitting."""

    dpi: int = Field(300, description="DPI for image conversion")
    image_format: str = Field("png", description="Output image format")

    @field_validator("dpi")
    def _validate_dpi(cls, v: int) -> int:
        if v < 72:
            raise ValueError("DPI must be at least 72")
        if v > 600:
            raise ValueError("DPI should not exceed 600 for performance reasons")
        return v

    @field_validator("image_format")
    def _validate_image_format(cls, v: str) -> str:
        allowed = {"png", "jpg", "jpeg"}
        if v.lower() not in allowed:
            raise ValueError(f"Image format must be one of: {sorted(allowed)}")
        return v.lower()



# AI processing settings


class AIConfig(BaseModel):
    """Configuration for AI processing."""

    # Ollama
    ollama_host: str = Field("http://localhost:11434", description="Ollama server URL")

    # Models
    structure_analysis_model: str = Field("mistral-small3.2:latest", description="Model for structure analysis")
    asset_extraction_model: str = Field("mistral-small3.2:latest", description="Model for asset extraction")
    markdown_generation_model: str = Field("mistral-small3.2:latest", description="Model for markdown generation")

    # Retry / timeout
    max_retries: int = Field(3, description="Maximum retry attempts for AI calls")
    timeout_seconds: int = Field(300, description="Timeout for AI processing in seconds")

    # Sampling parameters
    temperature: float = Field(0.1, description="Temperature for AI model")
    top_p: float = Field(0.9, description="Top-p for AI model")

    @field_validator("temperature")
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("top_p")
    def _validate_top_p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")
        return v



# Output directory & filename settings


class OutputConfig(BaseModel):
    """Configuration for output directories and file patterns."""

    # Base directories
    input_directory: Path = Field(Path("input"), description="Base directory for input PDFs")
    output_directory: Path = Field(Path("output"), description="Base directory for processed output")

    # Sub-directories
    pdf_split_subdir: str = Field("pdfSplitted", description="Subdirectory for split page images")
    structure_subdir: str = Field("pageStructure", description="Subdirectory for page structure JSONs")
    markdown_subdir: str = Field("markdown", description="Subdirectory for generated markdown files")
    csv_subdir: str = Field("csv", description="Subdirectory for extracted CSV tables")
    images_subdir: str = Field("images", description="Subdirectory for extracted images")
    figures_subdir: str = Field("figures", description="Subdirectory for extracted figure data")

    # Filename patterns
    page_image_pattern: str = Field("page_{page:04d}.png", description="Filename pattern for page images")
    structure_json_pattern: str = Field("page_{page:04d}_structure.json",
                                        description="Filename pattern for structure JSONs")
    markdown_pattern: str = Field("page_{page:04d}.md", description="Filename pattern for markdown files")

    @field_validator("input_directory", "output_directory", mode="before")
    def _abs_paths(cls, v: Any) -> Path:
        return Path(v).resolve()



# Processing behaviour settings


class ProcessingConfig(BaseModel):
    """Configuration for processing behaviour."""

    skip_existing_files: bool = Field(True, description="Skip processing if output files already exist")
    max_parallel_pages: int = Field(1, description="Maximum number of pages to process in parallel")

    debug_mode: bool = Field(False, description="Enable debug logging")
    save_intermediate_results: bool = Field(True, description="Save intermediate results")

    # Asset extraction thresholds
    min_table_rows: int = Field(2, description="Minimum rows to consider as table")
    min_figure_elements: int = Field(1, description="Minimum elements to consider as figure")

    # --- validators --------------------------------------------------------

    @field_validator("max_parallel_pages")
    def _validate_parallel(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_pages must be at least 1")
        if v > 10:
            raise ValueError("max_parallel_pages should not exceed 10 for stability")
        return v



# Master configuration

class AssetableConfig(BaseModel):
    """Main configuration class for Assetable."""

# default_factory は 「引数の要らない callable」 を受け取りますが，PDFSplitConfig など Pydantic BaseModel 派生クラス は
# def __init__(self, **data: Any) -> None: ...
# という 可変長キーワード引数 を持つコンストラクタと解釈されるため，Pyright は
# パラメーター "dpi", "image_format" に引数がありませんのように 必須実引数を渡していない と誤検知します。
# 最小限で型安全な解決策： typing.cast で 「引数なし callable」の型に変換してから渡す と Pyright の型判定が通ります。
# ランタイム挙動は一切変わりません。

    pdf_split: PDFSplitConfig = Field(
        default_factory=cast(Callable[[], PDFSplitConfig], PDFSplitConfig)
    )
    ai: AIConfig = Field(
        default_factory=cast(Callable[[], AIConfig], AIConfig)
    )
    output: OutputConfig = Field(
        default_factory=cast(Callable[[], OutputConfig], OutputConfig)
    )
    processing: ProcessingConfig = Field(
        default_factory=cast(Callable[[], ProcessingConfig], ProcessingConfig)
    )

    version: str = Field(default="0.1.0", description="Assetable version")


    # Environment overrides

    @classmethod
    def from_env(cls) -> "AssetableConfig":
        cfg = cls()

        if h := os.getenv("ASSETABLE_OLLAMA_HOST"):
            cfg.ai.ollama_host = h

        if m := os.getenv("ASSETABLE_STRUCTURE_MODEL"):
            cfg.ai.structure_analysis_model = m

        if m := os.getenv("ASSETABLE_ASSET_MODEL"):
            cfg.ai.asset_extraction_model = m

        if m := os.getenv("ASSETABLE_MARKDOWN_MODEL"):
            cfg.ai.markdown_generation_model = m

        if dpi := os.getenv("ASSETABLE_DPI"):
            cfg.pdf_split.dpi = int(dpi)

        if d := os.getenv("ASSETABLE_INPUT_DIR"):
            cfg.output.input_directory = Path(d)

        if d := os.getenv("ASSETABLE_OUTPUT_DIR"):
            cfg.output.output_directory = Path(d)

        if dbg := os.getenv("ASSETABLE_DEBUG"):
            cfg.processing.debug_mode = dbg.lower() in {"true", "1", "yes"}

        return cfg


    # Helper path getters

    def get_document_output_dir(self, pdf_path: Path) -> Path:
        return self.output.output_directory / pdf_path.stem

    def get_pdf_split_dir(self, pdf_path: Path) -> Path:
        return self.get_document_output_dir(pdf_path) / self.output.pdf_split_subdir

    def get_structure_dir(self, pdf_path: Path) -> Path:
        return self.get_document_output_dir(pdf_path) / self.output.structure_subdir

    def get_markdown_dir(self, pdf_path: Path) -> Path:
        return self.get_document_output_dir(pdf_path) / self.output.markdown_subdir

    def get_csv_dir(self, pdf_path: Path) -> Path:
        return self.get_markdown_dir(pdf_path) / self.output.csv_subdir

    def get_images_dir(self, pdf_path: Path) -> Path:
        return self.get_markdown_dir(pdf_path) / self.output.images_subdir

    def get_figures_dir(self, pdf_path: Path) -> Path:
        return self.get_markdown_dir(pdf_path) / self.output.figures_subdir

    def get_page_image_path(self, pdf_path: Path, page: int) -> Path:
        ext = self.pdf_split.image_format.lower().replace("jpeg", "jpg")
        base = self.output.page_image_pattern.rsplit(".", 1)[0]
        filename = f"{base.format(page=page)}.{ext}"
        return self.get_pdf_split_dir(pdf_path) / filename

    def get_structure_json_path(self, pdf_path: Path, page: int) -> Path:
        filename = self.output.structure_json_pattern.format(page=page)
        return self.get_structure_dir(pdf_path) / filename

    def get_markdown_path(self, pdf_path: Path, page: int) -> Path:
        filename = self.output.markdown_pattern.format(page=page)
        return self.get_markdown_dir(pdf_path) / filename

    def get_asset_path(self, pdf_path: Path, page: int, asset_type: str, asset_name: str) -> Path:
        clean = "".join(c for c in asset_name if c.isalnum() or c in {" ", "-", "_"}).strip().replace(" ", "_")

        if asset_type == "table":
            return self.get_csv_dir(pdf_path) / f"page_{page:04d}_{clean}.csv"
        if asset_type == "figure":
            return self.get_figures_dir(pdf_path) / f"page_{page:04d}_{clean}.json"
        if asset_type == "image":
            return self.get_images_dir(pdf_path) / f"page_{page:04d}_{clean}.jpg"
        raise ValueError(f"Unknown asset type: {asset_type}")


    # I/O helpers

    def create_output_directories(self, pdf_path: Path) -> None:
        for d in (
            self.get_document_output_dir(pdf_path),
            self.get_pdf_split_dir(pdf_path),
            self.get_structure_dir(pdf_path),
            self.get_markdown_dir(pdf_path),
            self.get_csv_dir(pdf_path),
            self.get_images_dir(pdf_path),
            self.get_figures_dir(pdf_path),
        ):
            d.mkdir(parents=True, exist_ok=True)


    # Serialization helpers

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a plain Python dictionary.

        Uses `model_dump()` for Pydantic v2 and falls back to `dict()` for v1.
        """
        if hasattr(self, "model_dump"):
            return self.model_dump()  # type: ignore[attr-defined]
        return self.dict()  # type: ignore[deprecated]

    def save_to_file(self, file_path: Path) -> None:
        import json

        with open(file_path, "w", encoding="utf-8") as f:
            if hasattr(self, "model_dump_json"):
                f.write(self.model_dump_json(indent=2))  # type: ignore[attr-defined]
            else:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "AssetableConfig":
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))




_config: Optional[AssetableConfig] = None


def get_config() -> AssetableConfig:
    global _config
    if _config is None:
        # Load .env file if it exists
        load_dotenv()
        _config = AssetableConfig.from_env()
    return _config


def set_config(config: AssetableConfig) -> None:
    global _config
    _config = config


def reset_config() -> None:
    global _config
    _config = None
