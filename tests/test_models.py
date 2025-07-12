"""
Tests for assetable.models module.

Tests are structured using Arrange-Act-Assert pattern and focus on
actual behavior rather than mocked dependencies.
"""

import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import pytest

from assetable.models import (
    AssetType,
    BoundingBox,
    CrossPageReference,
    DocumentData,
    FigureAsset,
    FigureNode,
    ImageAsset,
    PageData,
    PageStructure,
    ProcessingStage,
    ReferenceType,
    TableAsset,
)


class TestBoundingBox:
    """Test BoundingBox model validation and properties."""

    def test_valid_bounding_box_creation(self) -> None:
        """Test creating a valid bounding box."""
        # Arrange
        coordinates = [100, 200, 300, 400]

        # Act
        bbox = BoundingBox(bbox_2d=coordinates)

        # Assert
        assert bbox.bbox_2d == coordinates
        assert bbox.x1 == 100
        assert bbox.y1 == 200
        assert bbox.x2 == 300
        assert bbox.y2 == 400
        assert bbox.width == 200
        assert bbox.height == 200

    def test_invalid_bounding_box_coordinates(self) -> None:
        """Test bounding box validation with invalid coordinates."""
        # Arrange
        invalid_coordinates = [300, 200, 100, 400]  # x1 > x2

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid bounding box"):
            BoundingBox(bbox_2d=invalid_coordinates)

    def test_invalid_bounding_box_length(self) -> None:
        """Test bounding box validation with wrong number of coordinates."""
        # Arrange
        invalid_coordinates = [100, 200, 300]  # Only 3 coordinates

        # Act & Assert
        # Pydantic V2 raises pydantic_core.ValidationError and the message is different
        from pydantic_core import ValidationError
        with pytest.raises(ValidationError, match="List should have at least 4 items after validation, not 3"):
            BoundingBox(bbox_2d=invalid_coordinates)


class TestAssetModels:
    """Test asset model validation and behavior."""

    def test_table_asset_creation(self) -> None:
        """Test creating a valid table asset."""
        # Arrange
        bbox = BoundingBox(bbox_2d=[10, 20, 100, 80])
        csv_data = "Name,Age\nJohn,25\nJane,30"
        columns = ["Name", "Age"]
        rows = [["John", "25"], ["Jane", "30"]]

        # Act
        table = TableAsset(
            name="売上データ",
            description="月別売上データの表",
            bbox=bbox,
            csv_data=csv_data,
            columns=columns,
            rows=rows
        )

        # Assert
        assert table.type == AssetType.TABLE
        assert table.name == "売上データ"
        assert table.csv_data == csv_data
        assert table.columns == columns
        assert table.rows == rows

    def test_figure_asset_with_structure(self) -> None:
        """Test creating a figure asset with structured data."""
        # Arrange
        bbox = BoundingBox(bbox_2d=[50, 100, 200, 300])
        node1 = FigureNode(
            id="node1",
            type="box",
            label="開始",
            properties={"color": "blue"},
            position=BoundingBox(bbox_2d=[60, 110, 120, 140])
        )
        node2 = FigureNode(
            id="node2",
            type="arrow",
            label="→",
            properties={"direction": "right"}
        )
        structure = [node1, node2]

        # Act
        figure = FigureAsset(
            name="フローチャート",
            description="処理フローを示すチャート",
            bbox=bbox,
            figure_type="flowchart",
            structure=structure
        )

        # Assert
        assert figure.type == AssetType.FIGURE
        assert figure.figure_type == "flowchart"
        assert figure.structure is not None  # Add this assertion
        assert len(figure.structure) == 2
        assert figure.structure[0].label == "開始"

    def test_image_asset_creation(self) -> None:
        """Test creating an image asset."""
        # Arrange
        bbox = BoundingBox(bbox_2d=[0, 0, 640, 480])
        image_path = Path("test_image.jpg")

        # Act
        image = ImageAsset(
            name="グラフ",
            description="売上推移グラフ",
            bbox=bbox,
            image_path=image_path,
            image_type="chart"
        )

        # Assert
        assert image.type == AssetType.IMAGE
        assert image.image_path == image_path
        assert image.image_type == "chart"

    def test_asset_name_validation(self) -> None:
        """Test asset name validation for filename safety."""
        # Arrange
        bbox = BoundingBox(bbox_2d=[0, 0, 100, 100])
        invalid_name = "file/with:invalid*chars"

        # Act & Assert
        with pytest.raises(ValueError, match="Asset name cannot contain"):
            TableAsset(
                name=invalid_name,
                description="Test table",
                bbox=bbox
            )


class TestCrossPageReference:
    """Test cross-page reference model."""

    def test_valid_reference_creation(self) -> None:
        """Test creating valid cross-page references."""
        # Arrange
        reference_data = [
            (5, "5ページ参照", ReferenceType.PAGE),
            (3, "表2.1参照", ReferenceType.TABLE),
            (7, "図1.5参照", ReferenceType.FIGURE),
            (2, "第3章参照", ReferenceType.HEADING),
            (9, "写真3参照", ReferenceType.IMAGE),
        ]

        for target_page, text, ref_type in reference_data:
            # Act
            reference = CrossPageReference(
                target_page=target_page,
                reference_text=text,
                reference_type=ref_type
            )

            # Assert
            assert reference.target_page == target_page
            assert reference.reference_text == text
            assert reference.reference_type == ref_type


class TestPageStructure:
    """Test page structure model."""

    def test_page_structure_creation(self) -> None:
        """Test creating a complete page structure."""
        # Arrange
        page_number = 1
        text_content = "これはテストページです。"

        table = TableAsset(
            name="テストテーブル",
            description="テスト用の表",
            bbox=BoundingBox(bbox_2d=[10, 20, 100, 80])
        )

        figure = FigureAsset(
            name="テスト図",
            description="テスト用の図",
            bbox=BoundingBox(bbox_2d=[200, 300, 400, 500]),
            figure_type="diagram"
        )

        reference = CrossPageReference(
            target_page=2,
            reference_text="次ページ参照",
            reference_type=ReferenceType.PAGE
        )

        # Act
        structure = PageStructure(
            page_number=page_number,
            has_text=True,
            text_content=text_content,
            tables=[table],
            figures=[figure],
            references=[reference],
            ai_model_used="qwen2.5-vl:7b"
        )

        # Assert
        assert structure.page_number == page_number
        assert structure.has_text is True
        assert structure.text_content == text_content
        assert len(structure.tables) == 1
        assert len(structure.figures) == 1
        assert len(structure.references) == 1
        assert structure.ai_model_used == "qwen2.5-vl:7b"
        assert isinstance(structure.analysis_timestamp, datetime)


class TestPageData:
    """Test page data model and its methods."""

    def test_page_data_creation(self) -> None:
        """Test creating page data with basic information."""
        # Arrange
        page_number = 1
        source_pdf = Path("test.pdf")

        # Act
        page_data = PageData(
            page_number=page_number,
            source_pdf=source_pdf
        )

        # Assert
        assert page_data.page_number == page_number
        assert page_data.source_pdf == source_pdf
        assert page_data.current_stage == ProcessingStage.PDF_SPLIT
        assert len(page_data.completed_stages) == 0
        assert isinstance(page_data.created_at, datetime)

    def test_stage_completion_tracking(self) -> None:
        """Test stage completion tracking functionality."""
        # Arrange
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))

        # Act
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        page_data.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)

        # Assert
        assert page_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
        assert page_data.is_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
        assert not page_data.is_stage_completed(ProcessingStage.ASSET_EXTRACTION)
        assert page_data.current_stage == ProcessingStage.STRUCTURE_ANALYSIS
        assert len(page_data.completed_stages) == 2

    def test_next_stage_calculation(self) -> None:
        """Test next stage calculation."""
        # Arrange
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))

        # Act & Assert
        assert page_data.get_next_stage() == ProcessingStage.STRUCTURE_ANALYSIS

        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        assert page_data.get_next_stage() == ProcessingStage.STRUCTURE_ANALYSIS

        page_data.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)
        assert page_data.get_next_stage() == ProcessingStage.ASSET_EXTRACTION

        page_data.mark_stage_completed(ProcessingStage.COMPLETED)
        assert page_data.get_next_stage() is None


    def test_logging_functionality(self) -> None:
        """Test logging functionality."""
        # Arrange
        page_data = PageData(page_number=1, source_pdf=Path("test.pdf"))
        message = "Test log message"

        # Act
        page_data.add_log(message)

        # Assert
        assert len(page_data.processing_log) == 1
        assert message in page_data.processing_log[0]
        assert isinstance(page_data.last_updated, datetime)


class TestDocumentData:
    """Test document data model and its methods."""

    def test_document_data_creation(self) -> None:
        """Test creating document data."""
        # Arrange
        document_id = "test_doc_123"
        source_pdf = Path("test_document.pdf")
        output_dir = Path("output/test_document")

        # Act
        doc_data = DocumentData(
            document_id=document_id,
            source_pdf_path=source_pdf,
            output_directory=output_dir,
        )

        # Assert
        assert doc_data.document_id == document_id
        assert doc_data.source_pdf_path == source_pdf
        assert doc_data.output_directory == output_dir
        assert len(doc_data.pages) == 0

    def test_page_management(self) -> None:
        """Test adding and retrieving pages."""
        # Arrange
        doc_data = DocumentData(
            document_id="test_doc",
            source_pdf_path=Path("test.pdf"),
            output_directory=Path("output"),
        )

        page1 = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page2 = PageData(page_number=2, source_pdf=Path("test.pdf"))
        page3 = PageData(page_number=3, source_pdf=Path("test.pdf"))

        # Act
        doc_data.add_page(page1)
        doc_data.add_page(page3) # Add out of order
        doc_data.add_page(page2)

        # Assert
        assert len(doc_data.pages) == 3
        assert doc_data.get_page_by_number(1) == page1
        assert doc_data.get_page_by_number(2) == page2
        assert doc_data.get_page_by_number(3) == page3
        assert doc_data.get_page_by_number(4) is None

        # Pages should be sorted by page number
        assert doc_data.pages[0].page_number == 1
        assert doc_data.pages[1].page_number == 2
        assert doc_data.pages[2].page_number == 3

    def test_page_replacement(self) -> None:
        """Test replacing existing page data."""
        # Arrange
        doc_data = DocumentData(
            document_id="test_doc",
            source_pdf_path=Path("test.pdf"),
            output_directory=Path("output"),
        )

        page1_old = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page1_new = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page1_new.add_log("Updated page")

        # Act
        doc_data.add_page(page1_old)
        doc_data.add_page(page1_new) # This should replace page1_old

        # Assert
        assert len(doc_data.pages) == 1
        retrieved_page = doc_data.get_page_by_number(1)
        assert retrieved_page is not None
        assert len(retrieved_page.processing_log) == 1
        assert "Updated page" in retrieved_page.processing_log[0]

    def test_stage_filtering(self) -> None:
        """Test filtering pages by completion stage."""
        # Arrange
        doc_data = DocumentData(
            document_id="test_doc",
            source_pdf_path=Path("test.pdf"),
            output_directory=Path("output"),
        )

        page1 = PageData(page_number=1, source_pdf=Path("test.pdf"))
        page2 = PageData(page_number=2, source_pdf=Path("test.pdf"))
        page3 = PageData(page_number=3, source_pdf=Path("test.pdf"))

        page1.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        page2.mark_stage_completed(ProcessingStage.PDF_SPLIT)
        page2.mark_stage_completed(ProcessingStage.STRUCTURE_ANALYSIS)

        doc_data.add_page(page1)
        doc_data.add_page(page2)
        doc_data.add_page(page3)

        # Act
        completed_split = doc_data.get_completed_pages(ProcessingStage.PDF_SPLIT)
        completed_analysis = doc_data.get_completed_pages(ProcessingStage.STRUCTURE_ANALYSIS)
        pending_split = doc_data.get_pending_pages(ProcessingStage.PDF_SPLIT)

        # Assert
        assert len(completed_split) == 2
        assert page1 in completed_split
        assert page2 in completed_split

        assert len(completed_analysis) == 1
        assert page2 in completed_analysis

        assert len(pending_split) == 1
        assert page3 in pending_split


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_page_structure_json_serialization(self) -> None:
        """Test PageStructure JSON serialization."""
        # Arrange
        table = TableAsset(
            name="テストテーブル",
            description="テスト用の表",
            bbox=BoundingBox(bbox_2d=[10, 20, 100, 80]),
            csv_data="A,B\n1,2"
        )

        structure = PageStructure(
            page_number=1,
            has_text=True,
            text_content="テストテキスト",
            tables=[table],
            ai_model_used="test-model"
        )

        # Act
        # Pydantic v1 uses .dict(), v2 uses .model_dump()
        if hasattr(structure, "model_dump"):
            json_data = structure.model_dump()
        else:
            json_data = structure.dict()
        reconstructed = PageStructure(**json_data)

        # Assert
        assert reconstructed.page_number == structure.page_number
        assert reconstructed.text_content == structure.text_content
        assert len(reconstructed.tables) == 1
        assert reconstructed.tables[0].name == "テストテーブル"

    def test_complex_model_serialization(self) -> None:
        """Test complex nested model serialization."""
        # Arrange
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = Path(f.name)

        try:
            page_data = PageData(
                page_number=1,
                source_pdf=Path("test.pdf"),
                image_path=Path("page_001.png")
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
            page_data.add_log("Test log entry")

            # Act
            # Pydantic v1 uses .dict(), v2 uses .model_dump()
            if hasattr(page_data, "model_dump"):
                json_data = page_data.model_dump()
            else:
                json_data = page_data.dict()

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, default=str, ensure_ascii=False, indent=2)

            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            reconstructed = PageData(**loaded_data)

            # Assert
            assert reconstructed.page_number == page_data.page_number
            assert reconstructed.is_stage_completed(ProcessingStage.PDF_SPLIT)
            # Log for stage completion + manual log
            assert len(reconstructed.processing_log) == 2

        finally:
            if temp_file.exists():
                temp_file.unlink()
