"""
Tests for assetable.file_manager module.

Tests are structured using Arrange-Act-Assert pattern and use real file system
operations to test actual behavior without mocks.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import pytest

from assetable.config import AssetableConfig
from assetable.file_manager import (
    DirectoryCreationError,
    FileManager,
    FileManagerError,
    FileNotFoundError,
    FileOperationError,
)
from assetable.models import (
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


class TestFileManagerInitialization:
    """Test FileManager initialization and configuration."""

    def test_file_manager_default_initialization(self) -> None:
        """Test FileManager initialization with default configuration."""
        # Arrange & Act
        file_manager = FileManager()

        # Assert
        assert file_manager.config is not None
        assert isinstance(file_manager.config, AssetableConfig)
        assert len(file_manager._created_directories) == 0

    def test_file_manager_custom_config_initialization(self) -> None:
        """Test FileManager initialization with custom configuration."""
        # Arrange
        custom_config = AssetableConfig()
        custom_config.pdf_split.dpi = 450
        custom_config.processing.debug_mode = True

        # Act
        file_manager = FileManager(config=custom_config)

        # Assert
        assert file_manager.config is custom_config
        assert file_manager.config.pdf_split.dpi == 450
        assert file_manager.config.processing.debug_mode is True


class TestDirectoryManagement:
    """Test directory creation and structure management."""

    def test_setup_document_structure(self) -> None:
        """Test document directory structure creation."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test_document.pdf")

            # Act
            file_manager.setup_document_structure(pdf_path)

            # Assert
            expected_dirs = [
                temp_path / "test_document",
                temp_path / "test_document" / "pdfSplitted",
                temp_path / "test_document" / "pageStructure",
                temp_path / "test_document" / "markdown",
                temp_path / "test_document" / "markdown" / "csv",
                temp_path / "test_document" / "markdown" / "images",
                temp_path / "test_document" / "markdown" / "figures",
            ]

            for expected_dir in expected_dirs:
                assert expected_dir.exists()
                assert expected_dir.is_dir()

            # Check that directories are tracked
            doc_dir = temp_path / "test_document"
            assert doc_dir in file_manager._created_directories

    def test_setup_document_structure_with_existing_directories(self) -> None:
        """Test document structure creation when directories already exist."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("existing_document.pdf")

            # Create some directories beforehand
            doc_dir = temp_path / "existing_document"
            doc_dir.mkdir()
            (doc_dir / "pdfSplitted").mkdir()

            # Act
            file_manager.setup_document_structure(pdf_path)

            # Assert
            # All directories should still exist
            expected_dirs = [
                temp_path / "existing_document",
                temp_path / "existing_document" / "pdfSplitted",
                temp_path / "existing_document" / "pageStructure",
                temp_path / "existing_document" / "markdown",
                temp_path / "existing_document" / "markdown" / "csv",
                temp_path / "existing_document" / "markdown" / "images",
                temp_path / "existing_document" / "markdown" / "figures",
            ]

            for expected_dir in expected_dirs:
                assert expected_dir.exists()
                assert expected_dir.is_dir()


class TestStageCompletion:
    """Test processing stage completion tracking."""

    def test_is_stage_completed_pdf_split(self) -> None:
        """Test PDF split stage completion check."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Act & Assert - Initially not completed
            assert not file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.PDF_SPLIT)

            # Create the image file
            image_path = config.get_page_image_path(pdf_path, page_number)
            image_path.write_text("dummy image content")

            # Act & Assert - Now completed
            assert file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.PDF_SPLIT)

    def test_is_stage_completed_structure_analysis(self) -> None:
        """Test structure analysis stage completion check."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Act & Assert - Initially not completed
            assert not file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.STRUCTURE_ANALYSIS)

            # Create structure file
            page_structure = PageStructure(
                page_number=page_number,
                has_text=True,
                text_content="Test content",
                ai_model_used="test-model"
            )
            file_manager.save_page_structure(pdf_path, page_number, page_structure)

            # Act & Assert - Now completed
            assert file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.STRUCTURE_ANALYSIS)

    def test_is_stage_completed_markdown_generation(self) -> None:
        """Test markdown generation stage completion check."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Act & Assert - Initially not completed
            assert not file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.MARKDOWN_GENERATION)

            # Create markdown file
            markdown_content = "# Test Page\n\nThis is test content."
            file_manager.save_markdown_content(pdf_path, page_number, markdown_content)

            # Act & Assert - Now completed
            assert file_manager.is_stage_completed(pdf_path, page_number, ProcessingStage.MARKDOWN_GENERATION)

    def test_get_completed_pages(self) -> None:
        """Test getting list of completed pages."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")

            file_manager.setup_document_structure(pdf_path)

            # Create image files for pages 1, 3, 5
            for page_num in [1, 3, 5]:
                image_path = config.get_page_image_path(pdf_path, page_num)
                image_path.write_text("dummy image content")

            # Act
            completed_pages = file_manager.get_completed_pages(pdf_path, ProcessingStage.PDF_SPLIT)

            # Assert
            assert completed_pages == [1, 3, 5]

    def test_get_pending_pages(self) -> None:
        """Test getting list of pending pages."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            total_pages = 5

            file_manager.setup_document_structure(pdf_path)

            # Complete pages 1 and 3
            for page_num in [1, 3]:
                image_path = config.get_page_image_path(pdf_path, page_num)
                image_path.write_text("dummy image content")

            # Act
            pending_pages = file_manager.get_pending_pages(pdf_path, ProcessingStage.PDF_SPLIT, total_pages)

            # Assert
            assert pending_pages == [2, 4, 5]


class TestPageStructurePersistence:
    """Test page structure save and load operations."""

    def test_save_and_load_page_structure(self) -> None:
        """Test saving and loading page structure."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Create test data
            table = TableAsset(
                name="売上テーブル",
                description="月別売上データ",
                bbox=BoundingBox(bbox_2d=[10, 20, 200, 100]),
                csv_data="月,売上\n1月,100\n2月,150"
            )

            reference = CrossPageReference(
                target_page=2,
                reference_text="次ページ参照",
                reference_type=ReferenceType.PAGE
            )

            page_structure = PageStructure(
                page_number=page_number,
                text_content="これはテストページです。",
                tables=[table],
                references=[reference],
                ai_model_used="mistral-small3.2:latest"
            )

            # Act - Save
            saved_path = file_manager.save_page_structure(pdf_path, page_number, page_structure)

            # Assert - File exists
            assert saved_path.exists()
            assert saved_path.is_file()

            # Act - Load
            loaded_structure = file_manager.load_page_structure(pdf_path, page_number)

            # Assert - Data matches
            assert loaded_structure is not None
            assert loaded_structure.page_number == page_number
            # has_text attribute has been removed from PageStructure
            assert loaded_structure.text_content == "これはテストページです。"
            assert len(loaded_structure.tables) == 1
            assert loaded_structure.tables[0].name == "売上テーブル"
            assert len(loaded_structure.references) == 1
            assert loaded_structure.references[0].target_page == 2
            assert loaded_structure.ai_model_used == "mistral-small3.2:latest"

    def test_load_nonexistent_page_structure(self) -> None:
        """Test loading page structure that doesn't exist."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 999

            # Act
            loaded_structure = file_manager.load_page_structure(pdf_path, page_number)

            # Assert
            assert loaded_structure is None

    def test_load_corrupted_page_structure(self) -> None:
        """Test loading corrupted page structure file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Create corrupted JSON file
            structure_path = config.get_structure_json_path(pdf_path, page_number)
            structure_path.write_text("{ invalid json content")

            # Act & Assert
            with pytest.raises(FileOperationError):
                file_manager.load_page_structure(pdf_path, page_number)


class TestPageDataPersistence:
    """Test page data save and load operations."""

    def test_save_and_load_page_data(self) -> None:
        """Test saving and loading complete page data."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Create test page data
            page_data = PageData(
                page_number=page_number,
                source_pdf=pdf_path,
                image_path=Path("page_0001.png"),
                markdown_content="# Test Page\n\nTest content."
            )
            page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)
            page_data.add_log("Test log entry")

            # Act - Save
            saved_path = file_manager.save_page_data(page_data)

            # Assert - File exists
            assert saved_path.exists()
            assert saved_path.is_file()

            # Act - Load
            loaded_data = file_manager.load_page_data(pdf_path, page_number)

            # Assert - Data matches
            assert loaded_data is not None
            assert loaded_data.page_number == page_number
            assert loaded_data.source_pdf == pdf_path
            assert loaded_data.is_stage_completed(ProcessingStage.PDF_SPLIT)
            assert loaded_data.markdown_content == "# Test Page\n\nTest content."
            assert len(loaded_data.processing_log) == 2  # One from stage completion, one manual

    def test_load_nonexistent_page_data(self) -> None:
        """Test loading page data that doesn't exist."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 999

            # Act
            loaded_data = file_manager.load_page_data(pdf_path, page_number)

            # Assert
            assert loaded_data is None


class TestDocumentDataPersistence:
    """Test document data save and load operations."""

    def test_save_and_load_document_data(self) -> None:
        """Test saving and loading complete document data."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test_document.pdf")
            output_dir = temp_path / "test_document"

            file_manager.setup_document_structure(pdf_path)

            # Create test document data
            document_data = DocumentData(
                document_id="test_doc_id",
                source_pdf_path=pdf_path, # Corrected: Was source_pdf
                output_directory=output_dir,
                # total_pages is not a direct field of DocumentData, it's derived or managed internally
            )
            # Set total_pages if it's a property or method, or adjust model
            # For now, assuming it might be inferred or not strictly needed for this test part
            # If DocumentData should store total_pages, the model or test setup needs adjustment.
            # Let's assume DocumentData infers total_pages from added pages or it's set elsewhere.
            # If direct assignment like `document_data.total_pages = 3` is needed, it implies model structure.

            # Add some pages
            for page_num in range(1, 4):
                page_data = PageData(
                    page_number=page_num,
                    source_pdf=pdf_path
                )
                document_data.add_page(page_data)

            # Act - Save
            saved_path = file_manager.save_document_data(document_data)

            # Assert - File exists
            assert saved_path.exists()
            assert saved_path.is_file()

            # Act - Load
            loaded_data = file_manager.load_document_data(pdf_path)

            # Assert - Data matches
            assert loaded_data is not None
            assert loaded_data.source_pdf_path == pdf_path
            assert len(loaded_data.pages) == 3
            # get_page_by_number is the correct method name
            assert loaded_data.get_page_by_number(1) is not None
            assert loaded_data.get_page_by_number(2) is not None
            assert loaded_data.get_page_by_number(3) is not None


class TestAssetFilePersistence:
    """Test asset file save operations."""

    def test_save_table_asset(self) -> None:
        """Test saving table asset as CSV file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            table = TableAsset(
                name="売上データ",
                description="月別売上データ",
                bbox=BoundingBox(bbox_2d=[10, 20, 200, 100]),
                csv_data="月,売上\n1月,100\n2月,150\n3月,200"
            )

            # Act
            saved_path = file_manager.save_asset_file(pdf_path, page_number, table)

            # Assert
            assert saved_path.exists()
            assert saved_path.suffix == ".csv"
            assert "売上データ" in saved_path.name

            # Check CSV content
            content = saved_path.read_text(encoding='utf-8')
            assert "月,売上" in content
            assert "1月,100" in content
            assert "2月,150" in content
            assert "3月,200" in content

    def test_save_figure_asset(self) -> None:
        """Test saving figure asset as JSON file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            node = FigureNode(
                id="node1",
                type="box",
                label="開始",
                properties={"color": "blue"}
            )

            figure = FigureAsset(
                name="フローチャート",
                description="処理フローチャート",
                bbox=BoundingBox(bbox_2d=[50, 100, 300, 400]),
                figure_type="flowchart",
                structure=[node]
            )

            # Act
            saved_path = file_manager.save_asset_file(pdf_path, page_number, figure)

            # Assert
            assert saved_path.exists()
            assert saved_path.suffix == ".json"
            assert "フローチャート" in saved_path.name

            # Check JSON content
            with open(saved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data["name"] == "フローチャート"
            assert data["figure_type"] == "flowchart"
            assert len(data["structure"]) == 1
            assert data["structure"][0]["label"] == "開始"

    def test_save_image_asset(self) -> None:
        """Test saving image asset metadata."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            image = ImageAsset(
                name="グラフ画像",
                description="売上グラフ",
                bbox=BoundingBox(bbox_2d=[0, 0, 640, 480]),
                image_path=Path("graph.jpg"),
                image_type="chart"
            )

            # Act
            saved_path = file_manager.save_asset_file(pdf_path, page_number, image)

            # Assert
            assert saved_path.exists()
            assert saved_path.suffix == ".json"
            assert "グラフ画像" in saved_path.name

            # Check JSON content
            with open(saved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data["name"] == "グラフ画像"
            assert data["image_type"] == "chart"
            assert "graph.jpg" in data["image_path"]


class TestMarkdownPersistence:
    """Test markdown content save and load operations."""

    def test_save_and_load_markdown_content(self) -> None:
        """Test saving and loading markdown content."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            markdown_content = """# 第1章 はじめに

これはテストページです。

## 1.1 概要

本書では以下の内容を扱います：

- データ分析の基礎
- 機械学習の応用
- 実装例

[売上データ](./csv/page_0001_売上データ.csv)を参照してください。

![グラフ](./images/page_0001_グラフ.jpg)
"""

            # Act - Save
            saved_path = file_manager.save_markdown_content(pdf_path, page_number, markdown_content)

            # Assert - File exists
            assert saved_path.exists()
            assert saved_path.suffix == ".md"

            # Act - Load
            loaded_content = file_manager.load_markdown_content(pdf_path, page_number)

            # Assert - Content matches
            assert loaded_content is not None
            assert loaded_content == markdown_content
            assert "第1章 はじめに" in loaded_content
            assert "売上データ" in loaded_content
            assert "グラフ" in loaded_content

    def test_load_nonexistent_markdown_content(self) -> None:
        """Test loading markdown content that doesn't exist."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 999

            # Act
            loaded_content = file_manager.load_markdown_content(pdf_path, page_number)

            # Assert
            assert loaded_content is None


class TestProcessingSummary:
    """Test processing summary generation."""

    def test_get_processing_summary(self) -> None:
        """Test getting processing summary for a document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            total_pages = 5

            file_manager.setup_document_structure(pdf_path)

            # Complete different stages for different pages
            # Page 1: All stages completed
            for stage_file in [
                config.get_page_image_path(pdf_path, 1),
                config.get_structure_json_path(pdf_path, 1),
                config.get_markdown_path(pdf_path, 1),
            ]:
                stage_file.parent.mkdir(parents=True, exist_ok=True)
                if stage_file.suffix == ".json":
                    stage_file.write_text('{"page_number": 1, "has_text": true}')
                else:
                    stage_file.write_text("dummy content")

            # Page 2: Only PDF split completed
            image_path = config.get_page_image_path(pdf_path, 2)
            image_path.write_text("dummy image")

            # Page 3: PDF split and structure analysis completed
            image_path = config.get_page_image_path(pdf_path, 3)
            image_path.write_text("dummy image")
            structure_path = config.get_structure_json_path(pdf_path, 3)
            structure_path.write_text('{"page_number": 3, "has_text": true}')

            # Act
            summary = file_manager.get_processing_summary(pdf_path, total_pages)

            # Assert
            assert summary["document"] == "test.pdf"
            assert summary["total_pages"] == 5
            assert "stages" in summary
            assert "overall_progress" in summary

            # Check individual stage progress
            pdf_split_stage = summary["stages"]["pdf_split"]
            assert pdf_split_stage["completed_count"] == 3
            assert pdf_split_stage["pending_count"] == 2
            assert pdf_split_stage["progress"] == 0.6

            structure_stage = summary["stages"]["structure_analysis"]
            assert structure_stage["completed_count"] == 2
            assert structure_stage["pending_count"] == 3
            assert structure_stage["progress"] == 0.4

            markdown_stage = summary["stages"]["markdown_generation"]
            assert markdown_stage["completed_count"] == 1
            assert markdown_stage["pending_count"] == 4
            assert markdown_stage["progress"] == 0.2

    def test_get_processing_summary_empty_document(self) -> None:
        """Test getting processing summary for empty document."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("empty.pdf")
            total_pages = 0

            # Act
            summary = file_manager.get_processing_summary(pdf_path, total_pages)

            # Assert
            assert summary["document"] == "empty.pdf"
            assert summary["total_pages"] == 0
            assert summary["overall_progress"] == 0.0

            for stage_info in summary["stages"].values():
                assert stage_info["completed_count"] == 0
                assert stage_info["pending_count"] == 0
                assert stage_info["progress"] == 0.0


class TestErrorHandling:
    """Test error handling and exception scenarios."""

    def test_save_page_structure_with_invalid_directory(self) -> None:
        """Test saving page structure with invalid directory permissions."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path / "nonexistent" / "path"

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            page_structure = PageStructure(
                page_number=page_number,
                has_text=True,
                ai_model_used="test-model"
            )

            # Act & Assert
            # This should not raise an error as directories are created automatically
            saved_path = file_manager.save_page_structure(pdf_path, page_number, page_structure)
            assert saved_path.exists()

    def test_cleanup_incomplete_files(self) -> None:
        """Test cleanup of incomplete or corrupted files."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            file_manager = FileManager(config=config)
            pdf_path = Path("test.pdf")
            page_number = 1

            file_manager.setup_document_structure(pdf_path)

            # Create corrupted files
            structure_path = config.get_structure_json_path(pdf_path, page_number)
            structure_path.write_text("{ corrupted json")

            markdown_path = config.get_markdown_path(pdf_path, page_number)
            markdown_path.write_text("valid markdown content")

            # Act
            file_manager.cleanup_incomplete_files(pdf_path, page_number)

            # Assert
            assert not structure_path.exists()  # Corrupted file should be removed
            assert markdown_path.exists()  # Valid file should remain


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_usage(self) -> None:
        """Test FileManager as context manager."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            pdf_path = Path("test.pdf")

            # Act
            with FileManager(config=config) as file_manager:
                file_manager.setup_document_structure(pdf_path)

                # Assert
                assert isinstance(file_manager, FileManager)
                doc_dir = temp_path / pdf_path.stem
                assert doc_dir.exists()

    def test_context_manager_with_exception(self) -> None:
        """Test context manager behavior when exception occurs."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path

            # Act & Assert
            try:
                with FileManager(config=config) as file_manager:
                    file_manager.setup_document_structure(Path("test.pdf"))
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected exception

            # File manager should still work properly after exception
            assert temp_path.exists()
