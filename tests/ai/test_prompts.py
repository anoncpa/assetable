"""
Tests for assetable.ai.prompts module.

Tests the prompt generation and template functionality used for AI processing.
"""

from pathlib import Path

import pytest

from assetable.ai.prompts import (
    AssetExtractionPrompts,
    MarkdownGenerationPrompts,
    PromptTemplate,
    StructureAnalysisPrompts,
)
from assetable.models import (
    BoundingBox,
    FigureAsset,
    ImageAsset,
    PageData,
    PageStructure,
    TableAsset,
)


class TestPromptTemplate:
    """Test base PromptTemplate functionality."""

    def test_prompt_template_creation(self) -> None:
        """Test creating and using a basic prompt template."""
        # Arrange
        template = PromptTemplate(
            system_prompt="You are a helpful assistant.",
            user_prompt_template="Analyze this {item_type} with {detail_level} detail."
        )

        # Act
        formatted_prompt = template.format_user_prompt(
            item_type="document",
            detail_level="high"
        )

        # Assert
        assert template.system_prompt == "You are a helpful assistant."
        assert formatted_prompt == "Analyze this document with high detail."


class TestStructureAnalysisPrompts:
    """Test structure analysis prompt generation."""

    def test_structure_analysis_prompt_creation(self) -> None:
        """Test creating structure analysis prompts."""
        # Arrange
        page_number = 5
        document_type = "academic paper"

        # Act
        system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(
            page_number, document_type
        )

        # Assert
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

        # Check content includes provided parameters
        assert str(page_number) in user_prompt
        assert document_type in user_prompt

        # Check for required analysis elements
        assert "text content" in user_prompt.lower()
        assert "tables" in user_prompt.lower()
        assert "figures" in user_prompt.lower()
        assert "images" in user_prompt.lower()
        assert "bounding box" in user_prompt.lower()

    def test_structure_analysis_prompt_default_document_type(self) -> None:
        """Test structure analysis prompt with default document type."""
        # Arrange
        page_number = 1

        # Act
        system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(page_number)

        # Assert
        assert "technical book" in user_prompt
        assert str(page_number) in user_prompt

    def test_structure_analysis_prompt_various_document_types(self) -> None:
        """Test structure analysis prompts with various document types."""
        # Arrange
        document_types = ["manual", "research paper", "novel", "technical specification"]

        for doc_type in document_types:
            # Act
            system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(1, doc_type)

            # Assert
            assert doc_type in user_prompt
            assert "page 1" in user_prompt


class TestAssetExtractionPrompts:
    """Test asset extraction prompt generation."""

    def test_table_extraction_prompt(self) -> None:
        """Test table extraction prompt generation."""
        # Arrange
        table_asset = TableAsset(
            name="Performance Metrics",
            description="System performance measurement data",
            bbox=BoundingBox(bbox_2d=[100, 150, 400, 300])
        )
        page_number = 3

        # Act
        system_prompt, user_prompt = AssetExtractionPrompts.create_table_prompt(
            table_asset, page_number
        )

        # Assert
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

        # Check content includes asset information
        assert table_asset.name in user_prompt
        assert table_asset.description in user_prompt
        assert str(table_asset.bbox.bbox_2d) in user_prompt
        assert str(page_number) in user_prompt

        # Check for CSV-specific requirements
        assert "csv" in user_prompt.lower()
        assert "header" in user_prompt.lower()
        assert "row" in user_prompt.lower()

    def test_figure_extraction_prompt(self) -> None:
        """Test figure extraction prompt generation."""
        # Arrange
        figure_asset = FigureAsset(
            name="System Architecture",
            description="Overall system design and component relationships",
            bbox=BoundingBox(bbox_2d=[50, 200, 500, 600]),
            figure_type="architecture diagram"
        )
        page_number = 7

        # Act
        system_prompt, user_prompt = AssetExtractionPrompts.create_figure_prompt(
            figure_asset, page_number
        )

        # Assert
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

        # Check content includes figure information
        assert figure_asset.name in user_prompt
        assert figure_asset.description in user_prompt
        assert figure_asset.figure_type in user_prompt
        assert str(figure_asset.bbox.bbox_2d) in user_prompt
        assert str(page_number) in user_prompt

        # Check for JSON-specific requirements
        assert "json" in user_prompt.lower()
        assert "element" in user_prompt.lower()
        assert "relationship" in user_prompt.lower()

    def test_image_extraction_prompt(self) -> None:
        """Test image extraction prompt generation."""
        # Arrange
        image_asset = ImageAsset(
            name="User Interface Screenshot",
            description="Application main interface showing key features",
            bbox=BoundingBox(bbox_2d=[20, 50, 600, 400]),
            image_type="screenshot"
        )
        page_number = 12

        # Act
        system_prompt, user_prompt = AssetExtractionPrompts.create_image_prompt(
            image_asset, page_number
        )

        # Assert
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

        # Check content includes image information
        assert image_asset.name in user_prompt
        assert image_asset.description in user_prompt
        assert image_asset.image_type in user_prompt
        assert str(image_asset.bbox.bbox_2d) in user_prompt
        assert str(page_number) in user_prompt

        # Check for image analysis requirements
        assert "description" in user_prompt.lower()
        assert "visual" in user_prompt.lower()

    def test_image_extraction_prompt_without_image_type(self) -> None:
        """Test image extraction prompt when image_type is None."""
        # Arrange
        image_asset = ImageAsset(
            name="Diagram",
            description="Technical diagram",
            bbox=BoundingBox(bbox_2d=[10, 20, 300, 200]),
            image_type=None
        )
        page_number = 1

        # Act
        system_prompt, user_prompt = AssetExtractionPrompts.create_image_prompt(
            image_asset, page_number
        )

        # Assert
        assert "general" in user_prompt  # Should use default value


class TestMarkdownGenerationPrompts:
    """Test Markdown generation prompt functionality."""

    def test_markdown_generation_prompt_basic(self) -> None:
        """Test basic Markdown generation prompt creation."""
        # Arrange
        page_data = PageData(
            page_number=8,
            source_pdf=Path("test_document.pdf"),
            image_path=Path("page_008.png")
        )
        document_type = "user manual"

        # Act
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(
            page_data, document_type
        )

        # Assert
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

        # Check content includes page information
        assert str(page_data.page_number) in user_prompt
        assert document_type in user_prompt

        # Check for Markdown requirements
        assert "markdown" in user_prompt.lower()
        assert "heading" in user_prompt.lower()
        assert "page_0008" in user_prompt  # Page number formatting

    def test_markdown_generation_prompt_with_structure(self) -> None:
        """Test Markdown generation prompt with page structure context."""
        # Arrange
        page_structure = PageStructure(
            page_number=3,
            has_text=True,
            text_content="Sample content",
            tables=[
                TableAsset(
                    name="Data Table",
                    description="Sample data",
                    bbox=BoundingBox(bbox_2d=[10, 20, 200, 100])
                )
            ],
            figures=[
                FigureAsset(
                    name="Flow Chart",
                    description="Process flow",
                    bbox=BoundingBox(bbox_2d=[250, 50, 500, 300]),
                    figure_type="flowchart"
                )
            ]
        )

        page_data = PageData(
            page_number=3,
            source_pdf=Path("test.pdf"),
            image_path=Path("page_003.png"),
            page_structure=page_structure
        )

        # Act
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)

        # Assert
        assert "Tables: 1" in user_prompt
        assert "Figures: 1" in user_prompt
        assert "Detected elements:" in user_prompt

    def test_markdown_generation_prompt_with_assets(self) -> None:
        """Test Markdown generation prompt with extracted assets context."""
        # Arrange
        extracted_assets = [
            TableAsset(
                name="Results Table",
                description="Experimental results",
                bbox=BoundingBox(bbox_2d=[30, 40, 350, 180])
            ),
            FigureAsset(
                name="Architecture Diagram",
                description="System architecture",
                bbox=BoundingBox(bbox_2d=[400, 50, 700, 350]),
                figure_type="diagram"
            ),
            ImageAsset(
                name="Screenshot",
                description="Application screenshot",
                bbox=BoundingBox(bbox_2d=[100, 400, 600, 650])
            )
        ]

        page_data = PageData(
            page_number=15,
            source_pdf=Path("comprehensive.pdf"),
            image_path=Path("page_015.png"),
            extracted_assets=extracted_assets
        )

        # Act
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)

        # Assert
        assert "Available assets:" in user_prompt
        assert "Results Table" in user_prompt
        assert "Architecture Diagram" in user_prompt
        assert "Screenshot" in user_prompt

    def test_markdown_generation_prompt_empty_page(self) -> None:
        """Test Markdown generation prompt with minimal page data."""
        # Arrange
        page_data = PageData(
            page_number=1,
            source_pdf=Path("simple.pdf"),
            image_path=Path("page_001.png")
        )

        # Act
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)

        # Assert
        # Should still generate valid prompts even with minimal data
        assert str(page_data.page_number) in user_prompt
        assert "page_0001" in user_prompt  # Proper formatting

    def test_markdown_generation_asset_reference_patterns(self) -> None:
        """Test that Markdown generation prompts include proper asset reference patterns."""
        # Arrange
        page_data = PageData(
            page_number=42,
            source_pdf=Path("test.pdf"),
            image_path=Path("page_042.png")
        )

        # Act
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)

        # Assert
        # Check for proper asset reference patterns
        assert "./csv/page_0042" in user_prompt
        assert "./figures/page_0042" in user_prompt
        assert "./images/page_0042" in user_prompt

        # Check for Markdown link patterns
        assert "[Table Name]" in user_prompt
        assert "[Figure Name]" in user_prompt
        assert "![Image Description]" in user_prompt


class TestPromptQuality:
    """Test prompt quality and completeness."""

    def test_all_prompts_contain_required_elements(self) -> None:
        """Test that all prompts contain required elements for effective AI processing."""
        # Arrange
        test_assets = {
            'table': TableAsset(
                name="Test Table",
                description="Test description",
                bbox=BoundingBox(bbox_2d=[10, 20, 100, 80])
            ),
            'figure': FigureAsset(
                name="Test Figure",
                description="Test description",
                bbox=BoundingBox(bbox_2d=[200, 300, 400, 500]),
                figure_type="diagram"
            ),
            'image': ImageAsset(
                name="Test Image",
                description="Test description",
                bbox=BoundingBox(bbox_2d=[50, 100, 300, 250])
            )
        }

        # Act & Assert - Structure Analysis
        system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(1)
        assert "analyze" in system_prompt.lower()
        assert "bounding box" in user_prompt.lower()
        assert "absolute" in user_prompt.lower()

        # Act & Assert - Asset Extraction
        for asset_type, asset in test_assets.items():
            if asset_type == 'table':
                system_prompt, user_prompt = AssetExtractionPrompts.create_table_prompt(asset, 1)
                assert "csv" in user_prompt.lower()
            elif asset_type == 'figure':
                system_prompt, user_prompt = AssetExtractionPrompts.create_figure_prompt(asset, 1)
                assert "json" in user_prompt.lower()
            elif asset_type == 'image':
                system_prompt, user_prompt = AssetExtractionPrompts.create_image_prompt(asset, 1)
                assert "description" in user_prompt.lower()

            # Common requirements
            assert len(system_prompt) > 50  # Substantial system prompt
            assert len(user_prompt) > 100  # Detailed user prompt
            assert asset.name in user_prompt
            assert asset.description in user_prompt

        # Act & Assert - Markdown Generation
        page_data = PageData(
            page_number=1,
            source_pdf=Path("test.pdf"),
            image_path=Path("test.png")
        )
        system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)
        assert "markdown" in system_prompt.lower()
        assert "heading" in user_prompt.lower()
        assert "links" in system_prompt.lower()

    def test_prompt_consistency_across_pages(self) -> None:
        """Test that prompts maintain consistency across different page numbers."""
        # Arrange
        page_numbers = [1, 5, 10, 50, 100]

        for page_num in page_numbers:
            # Act - Structure Analysis
            system_prompt, user_prompt = StructureAnalysisPrompts.create_prompt(page_num)

            # Assert
            assert f"page {page_num}" in user_prompt
            assert len(system_prompt) > 0
            assert len(user_prompt) > 0

            # Check formatting consistency
            page_data = PageData(
                page_number=page_num,
                source_pdf=Path("test.pdf"),
                image_path=Path("test.png")
            )
            system_prompt, user_prompt = MarkdownGenerationPrompts.create_prompt(page_data)
            expected_format = f"page_{page_num:04d}"
            assert expected_format in user_prompt
