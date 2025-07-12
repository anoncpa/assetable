"""
AI-powered pipeline steps for Assetable.

This module implements the pipeline steps that use AI processing
for structure analysis, asset extraction, and Markdown generation.
"""

from pathlib import Path
from typing import List

from ..ai.vision_processor import VisionProcessor, VisionProcessorError
from ..config import AssetableConfig
from ..file_manager import FileManager
from ..models import PageData, ProcessingStage
from .engine import PipelineStep, PipelineStepError


class AIStructureAnalysisStep(PipelineStep):
    """Pipeline step for AI-powered structure analysis."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = VisionProcessor(config)

    @property
    def step_name(self) -> str:
        return "AI Structure Analysis"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.STRUCTURE_ANALYSIS

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.PDF_SPLIT]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute AI structure analysis for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            # Perform structure analysis
            analysis_result = self.vision_processor.analyze_page_structure(page_data)

            # Update page data with results
            page_data.page_structure = analysis_result.page_structure

            # Save structure to file
            if page_data.page_structure:
                structure_path = self.file_manager.save_page_structure(
                    page_data.source_pdf,
                    page_data.page_number,
                    page_data.page_structure
                )
                page_data.structure_json_path = structure_path

            # Mark stage as completed
            page_data.mark_stage_completed(self.processing_stage)
            page_data.add_log(f"AI structure analysis completed using {analysis_result.model_used}")

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI structure analysis failed for page {page_data.page_number}: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI structure analysis for page {page_data.page_number}: {e}")


class AIAssetExtractionStep(PipelineStep):
    """Pipeline step for AI-powered asset extraction."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = VisionProcessor(config)

    @property
    def step_name(self) -> str:
        return "AI Asset Extraction"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.ASSET_EXTRACTION

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.STRUCTURE_ANALYSIS]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute AI asset extraction for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            # Perform asset extraction
            extraction_result = self.vision_processor.extract_assets(page_data)

            # Update page data with extracted assets
            page_data.extracted_assets = extraction_result.extracted_assets

            # Save individual asset files
            if page_data.extracted_assets:
                for asset in page_data.extracted_assets:
                    asset_path = self.file_manager.save_asset_file(
                        page_data.source_pdf,
                        page_data.page_number,
                        asset
                    )
                    if asset.name not in page_data.asset_files:
                        page_data.asset_files[asset.name] = asset_path

            # Mark stage as completed
            page_data.mark_stage_completed(self.processing_stage)
            page_data.add_log(f"AI asset extraction completed: {len(page_data.extracted_assets) if page_data.extracted_assets else 0} assets extracted")

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI asset extraction failed for page {page_data.page_number}: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI asset extraction for page {page_data.page_number}: {e}")


class AIMarkdownGenerationStep(PipelineStep):
    """Pipeline step for AI-powered Markdown generation."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = VisionProcessor(config)

    @property
    def step_name(self) -> str:
        return "AI Markdown Generation"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.MARKDOWN_GENERATION

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.ASSET_EXTRACTION]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute AI Markdown generation for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            # Generate Markdown
            markdown_result = self.vision_processor.generate_markdown(page_data)

            # Update page data with Markdown content
            page_data.markdown_content = markdown_result.markdown_content

            # Save Markdown to file
            if page_data.markdown_content:
                markdown_path = self.file_manager.save_markdown_content(
                    page_data.source_pdf,
                    page_data.page_number,
                    page_data.markdown_content
                )
                page_data.markdown_path = markdown_path

            # Mark stage as completed
            page_data.mark_stage_completed(self.processing_stage)
            page_data.add_log(f"AI Markdown generation completed using {markdown_result.model_used}")

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI Markdown generation failed for page {page_data.page_number}: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI Markdown generation for page {page_data.page_number}: {e}")
