"""
Tests for assetable.config module.

Tests configuration management functionality using real file system
operations and environment variable manipulation.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any

import pytest

from assetable.config import (
    AIConfig,
    AssetableConfig,
    OutputConfig,
    PDFSplitConfig,
    ProcessingConfig,
    get_config,
    reset_config,
    set_config,
)


class TestPDFSplitConfig:
    """Test PDF split configuration validation."""

    def test_default_pdf_split_config(self) -> None:
        """Test default PDF split configuration."""
        # Arrange & Act
        config = PDFSplitConfig()

        # Assert
        assert config.dpi == 300
        assert config.image_format == "png"

    def test_valid_dpi_values(self) -> None:
        """Test valid DPI value validation."""
        # Arrange
        valid_dpi_values = [72, 150, 300, 600]

        for dpi in valid_dpi_values:
            # Act
            config = PDFSplitConfig(dpi=dpi)

            # Assert
            assert config.dpi == dpi

    def test_invalid_dpi_values(self) -> None:
        """Test invalid DPI value validation."""
        # Arrange
        invalid_dpi_values = [50, 700] # 71 is also invalid based on current validator

        for dpi in invalid_dpi_values:
            # Act & Assert
            with pytest.raises(ValueError):
                PDFSplitConfig(dpi=dpi)

    def test_image_format_validation(self) -> None:
        """Test image format validation."""
        # Arrange
        valid_formats = ["png", "jpg", "jpeg", "PNG", "JPG"]
        invalid_formats = ["gif", "bmp", "tiff"]

        # Act & Assert - Valid formats
        for fmt in valid_formats:
            config = PDFSplitConfig(image_format=fmt)
            assert config.image_format == fmt.lower()

        # Act & Assert - Invalid formats
        for fmt in invalid_formats:
            with pytest.raises(ValueError):
                PDFSplitConfig(image_format=fmt)


class TestAIConfig:
    """Test AI configuration validation."""

    def test_default_ai_config(self) -> None:
        """Test default AI configuration."""
        # Arrange & Act
        config = AIConfig()

        # Assert
        assert config.ollama_host == "http://localhost:11434"
        assert config.structure_analysis_model == "qwen2.5-vl:7b"
        assert config.temperature == 0.1
        assert config.top_p == 0.9
        assert config.max_retries == 3
        assert config.timeout_seconds == 300

    def test_temperature_validation(self) -> None:
        """Test temperature value validation."""
        # Arrange
        valid_temperatures = [0.0, 0.5, 1.0, 2.0]
        invalid_temperatures = [-0.1, 2.1]

        # Act & Assert - Valid temperatures
        for temp in valid_temperatures:
            config = AIConfig(temperature=temp)
            assert config.temperature == temp

        # Act & Assert - Invalid temperatures
        for temp in invalid_temperatures:
            with pytest.raises(ValueError):
                AIConfig(temperature=temp)

    def test_top_p_validation(self) -> None:
        """Test top-p value validation."""
        # Arrange
        valid_top_p_values = [0.0, 0.5, 0.9, 1.0]
        invalid_top_p_values = [-0.1, 1.1]

        # Act & Assert - Valid values
        for top_p in valid_top_p_values:
            config = AIConfig(top_p=top_p)
            assert config.top_p == top_p

        # Act & Assert - Invalid values
        for top_p in invalid_top_p_values:
            with pytest.raises(ValueError):
                AIConfig(top_p=top_p)


class TestOutputConfig:
    """Test output configuration and path handling."""

    def test_default_output_config(self) -> None:
        """Test default output configuration."""
        # Arrange & Act
        config = OutputConfig()

        # Assert
        assert config.input_directory == Path("input").resolve()
        assert config.output_directory == Path("output").resolve()
        assert config.pdf_split_subdir == "pdf_split" # Corrected default
        assert config.markdown_subdir == "markdown"
        assert config.page_image_pattern == "page_{page:04d}.png"

    def test_path_resolution(self) -> None:
        """Test path resolution to absolute paths."""
        # Arrange
        relative_input = "relative/input" # Test with string
        relative_output = Path("relative/output")

        # Act
        config = OutputConfig(
            input_directory=relative_input, # type: ignore
            output_directory=relative_output
        )

        # Assert
        assert config.input_directory.is_absolute()
        assert config.output_directory.is_absolute()
        assert config.input_directory.name == "input"
        assert config.output_directory.name == "output"


class TestProcessingConfig:
    """Test processing configuration validation."""

    def test_default_processing_config(self) -> None:
        """Test default processing configuration."""
        # Arrange & Act
        config = ProcessingConfig()

        # Assert
        assert config.skip_existing_files is True
        assert config.max_parallel_pages == 1
        assert config.debug_mode is False
        assert config.min_table_rows == 2
        assert config.min_figure_elements == 1

    def test_parallel_pages_validation(self) -> None:
        """Test parallel pages validation."""
        # Arrange
        valid_values = [1, 5, 10]
        invalid_values = [0, 11] # 0 is invalid based on current validator

        # Act & Assert - Valid values
        for value in valid_values:
            config = ProcessingConfig(max_parallel_pages=value)
            assert config.max_parallel_pages == value

        # Act & Assert - Invalid values
        for value in invalid_values:
            with pytest.raises(ValueError):
                ProcessingConfig(max_parallel_pages=value)


class TestAssetableConfig:
    """Test main configuration class and its methods."""

    def test_default_assetable_config(self) -> None:
        """Test default configuration creation."""
        # Arrange & Act
        config = AssetableConfig()

        # Assert
        assert isinstance(config.pdf_split, PDFSplitConfig)
        assert isinstance(config.ai, AIConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert config.version == "0.1.0"

    def test_path_generation_methods(self) -> None:
        """Test path generation methods."""
        # Arrange
        config = AssetableConfig()
        # Ensure output subdirs match defaults in config.py
        config.output.pdf_split_subdir = "pdf_split"
        config.output.structure_subdir = "structure"
        config.output.markdown_subdir = "markdown"
        config.output.csv_subdir = "csv"
        config.output.images_subdir = "images"
        config.output.figures_subdir = "figures"
        config.output.page_image_pattern = "page_{page:04d}.png"
        config.output.structure_json_pattern = "page_{page:04d}_structure.json"
        config.output.markdown_pattern = "page_{page:04d}.md"


        pdf_path = Path("test_book.pdf")
        page_number = 1

        # Act
        doc_dir = config.get_document_output_dir(pdf_path)
        split_dir = config.get_pdf_split_dir(pdf_path)
        structure_dir = config.get_structure_dir(pdf_path)
        markdown_dir = config.get_markdown_dir(pdf_path)
        csv_dir = config.get_csv_dir(pdf_path)
        images_dir = config.get_images_dir(pdf_path)
        figures_dir = config.get_figures_dir(pdf_path)

        image_path = config.get_page_image_path(pdf_path, page_number)
        structure_path = config.get_structure_json_path(pdf_path, page_number)
        markdown_path = config.get_markdown_path(pdf_path, page_number)

        # Assert
        assert doc_dir.name == "test_book"
        assert split_dir.name == "pdf_split"
        assert structure_dir.name == "structure"
        assert markdown_dir.name == "markdown"
        assert csv_dir.name == "csv"
        assert images_dir.name == "images"
        assert figures_dir.name == "figures"

        assert image_path.name == "page_0001.png"
        assert structure_path.name == "page_0001_structure.json" # Corrected pattern
        assert markdown_path.name == "page_0001.md"
        assert image_path.parent == split_dir
        assert structure_path.parent == structure_dir
        assert markdown_path.parent == markdown_dir

    def test_asset_path_generation(self) -> None:
        """Test asset file path generation."""
        # Arrange
        config = AssetableConfig()
        pdf_path = Path("test_book.pdf")
        page_number = 1

        # Act
        table_path = config.get_asset_path(pdf_path, page_number, "table", "売上データ")
        figure_path = config.get_asset_path(pdf_path, page_number, "figure", "フローチャート")
        image_path = config.get_asset_path(pdf_path, page_number, "image", "グラフ画像")

        # Assert
        assert table_path.name == "page_0001_売上データ.csv"
        assert figure_path.name == "page_0001_フローチャート.json"
        assert image_path.name == "page_0001_グラフ画像.jpg"
        assert table_path.parent.name == "csv"
        assert figure_path.parent.name == "figures"
        assert image_path.parent.name == "images"

    def test_asset_name_cleaning(self) -> None:
        """Test asset name cleaning for file paths."""
        # Arrange
        config = AssetableConfig()
        pdf_path = Path("test.pdf")
        page_number = 1
        dirty_name = "図表 1-2: データ/分析"

        # Act
        clean_path = config.get_asset_path(pdf_path, page_number, "table", dirty_name)

        # Assert
        assert "図表_1-2_データ_分析" in clean_path.name # Corrected cleaning
        assert "/" not in clean_path.name
        assert ":" not in clean_path.name

    def test_invalid_asset_type(self) -> None:
        """Test error handling for invalid asset types."""
        # Arrange
        config = AssetableConfig()
        pdf_path = Path("test.pdf")

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown asset type"):
            config.get_asset_path(pdf_path, 1, "invalid_type", "test")

    def test_directory_creation(self) -> None:
        """Test output directory creation."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AssetableConfig()
            config.output.output_directory = temp_path
            # Ensure subdirs match defaults in config.py for this test
            config.output.pdf_split_subdir = "pdf_split"
            config.output.structure_subdir = "structure"
            config.output.markdown_subdir = "markdown"
            config.output.csv_subdir = "csv"
            config.output.images_subdir = "images"
            config.output.figures_subdir = "figures"

            pdf_path = Path("test_book.pdf")

            # Act
            config.create_output_directories(pdf_path)

            # Assert
            expected_dirs = [
                temp_path / "test_book",
                temp_path / "test_book" / "pdf_split",
                temp_path / "test_book" / "structure",
                temp_path / "test_book" / "markdown",
                temp_path / "test_book" / "markdown" / "csv",
                temp_path / "test_book" / "markdown" / "images",
                temp_path / "test_book" / "markdown" / "figures",
            ]

            for expected_dir in expected_dirs:
                assert expected_dir.exists()
                assert expected_dir.is_dir()


class TestConfigSerialization:
    """Test configuration serialization and file operations."""

    def test_config_to_dict(self) -> None:
        """Test configuration conversion to dictionary."""
        # Arrange
        config = AssetableConfig()
        config.ai.temperature = 0.5
        config.pdf_split.dpi = 400

        # Act
        config_dict = config.to_dict() # Relies on Pydantic's .dict() or .model_dump()

        # Assert
        assert isinstance(config_dict, dict)
        assert config_dict["ai"]["temperature"] == 0.5
        assert config_dict["pdf_split"]["dpi"] == 400
        assert config_dict["version"] == "0.1.0"

    def test_config_file_save_and_load(self) -> None:
        """Test saving and loading configuration from file."""
        # Arrange
        with TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"

            original_config = AssetableConfig()
            original_config.ai.temperature = 0.7
            original_config.pdf_split.dpi = 450
            original_config.processing.debug_mode = True

            # Act - Save
            original_config.save_to_file(config_file)

            # Assert - File exists and is valid JSON
            assert config_file.exists()
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert saved_data["ai"]["temperature"] == 0.7

            # Act - Load
            loaded_config = AssetableConfig.load_from_file(config_file)

            # Assert - Configuration matches
            assert loaded_config.ai.temperature == 0.7
            assert loaded_config.pdf_split.dpi == 450
            assert loaded_config.processing.debug_mode is True


class TestEnvironmentVariables:
    """Test environment variable configuration loading."""

    def test_config_from_environment_variables(self) -> None:
        """Test loading configuration from environment variables."""
        # Arrange
        env_vars = {
            "ASSETABLE_OLLAMA_HOST": "http://test-host:11434",
            "ASSETABLE_STRUCTURE_MODEL": "qwen2.5-vl:14b",
            "ASSETABLE_DPI": "400",
            "ASSETABLE_DEBUG": "true",
            "ASSETABLE_INPUT_DIR": "/custom/input",
            "ASSETABLE_OUTPUT_DIR": "/custom/output",
        }

        original_env = os.environ.copy()
        # Act - Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            reset_config() # Ensure from_env is called
            config = get_config()

            # Assert
            assert config.ai.ollama_host == "http://test-host:11434"
            assert config.ai.structure_analysis_model == "qwen2.5-vl:14b"
            assert config.pdf_split.dpi == 400
            assert config.processing.debug_mode is True
            assert config.output.input_directory == Path("/custom/input").resolve()
            assert config.output.output_directory == Path("/custom/output").resolve()

        finally:
            # Cleanup - Restore environment variables
            os.environ.clear()
            os.environ.update(original_env)
            reset_config() # Clean up global config

    def test_partial_environment_override(self) -> None:
        """Test partial configuration override via environment variables."""
        # Arrange
        original_env = os.environ.copy()
        os.environ["ASSETABLE_OLLAMA_HOST"] = "http://partial-test:11434"

        try:
            reset_config() # Ensure from_env is called
            # Act
            config = get_config()

            # Assert
            assert config.ai.ollama_host == "http://partial-test:11434"
            # Other values should remain defaults
            assert config.ai.structure_analysis_model == "qwen2.5-vl:7b" # Default
            assert config.pdf_split.dpi == 300 # Default

        finally:
            # Cleanup
            os.environ.clear()
            os.environ.update(original_env)
            reset_config() # Clean up global config


class TestGlobalConfigManagement:
    """Test global configuration instance management."""

    def test_get_global_config(self) -> None:
        """Test getting global configuration instance."""
        # Arrange
        reset_config()  # Ensure clean state

        # Act
        config1 = get_config()
        config2 = get_config()

        # Assert
        assert config1 is config2  # Same instance
        assert isinstance(config1, AssetableConfig)

    def test_set_global_config(self) -> None:
        """Test setting global configuration instance."""
        # Arrange
        reset_config()
        custom_config = AssetableConfig()
        custom_config.ai.temperature = 0.8

        # Act
        set_config(custom_config)
        retrieved_config = get_config()

        # Assert
        assert retrieved_config is custom_config
        assert retrieved_config.ai.temperature == 0.8

    def test_reset_global_config(self) -> None:
        """Test resetting global configuration instance."""
        # Arrange
        reset_config() # Start clean
        config1 = get_config()
        config1.ai.temperature = 0.9 # Modify the instance
        set_config(config1) # Ensure this modified instance is global

        # Act
        reset_config() # This should create a new default instance
        config2 = get_config()

        # Assert
        assert config2 is not config1
        assert config2.ai.temperature == 0.1  # Default value
        # Ensure original config1 was not changed by reset if it's a different object
        if config1 is not config2:
             assert config1.ai.temperature == 0.9
