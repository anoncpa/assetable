"""
AI-powered pipeline steps for Assetable.

This module implements comprehensive AI processing pipeline steps
with enhanced error handling, retry logic, and performance monitoring.
"""

from typing import List

from ..ai.vision_processor import EnhancedVisionProcessor, VisionProcessorError
from ..config import AssetableConfig
from ..models import PageData, ProcessingStage
from ..pipeline.engine import PipelineStep, PipelineStepError


class EnhancedAIStructureAnalysisStep(PipelineStep):
    """Enhanced AI-powered structure analysis pipeline step."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = EnhancedVisionProcessor(config)
        self.document_type = "technical book"  # Can be configured

    @property
    def step_name(self) -> str:
        return "Enhanced AI Structure Analysis"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.STRUCTURE_ANALYSIS

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.PDF_SPLIT]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute enhanced AI structure analysis for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            if self.config.processing.debug_mode:
                print(f"Starting enhanced structure analysis for page {page_data.page_number}")

            # Check if already processed and should skip
            if (self.config.processing.skip_existing_files and
                self.file_manager.is_stage_completed(
                    page_data.source_pdf,
                    page_data.page_number,
                    self.processing_stage
                )):

                # Load existing structure
                existing_structure = self.file_manager.load_page_structure(
                    page_data.source_pdf,
                    page_data.page_number
                )

                if existing_structure:
                    page_data.page_structure = existing_structure
                    page_data.add_log("Loaded existing structure analysis")
                    return page_data

            # Perform structure analysis
            result = self.vision_processor.analyze_page_structure(page_data, self.document_type)

            if not result.success:
                raise PipelineStepError(f"Structure analysis failed: {result.error_message}")

            # Update page data with results
            page_data.page_structure = result.page_structure

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
            page_data.add_log(
                f"Enhanced AI structure analysis completed using {result.model_used} "
                f"in {result.processing_time:.2f}s"
            )

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI structure analysis failed: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI structure analysis: {e}")


class EnhancedAIAssetExtractionStep(PipelineStep):
    """Enhanced AI-powered asset extraction pipeline step."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = EnhancedVisionProcessor(config)

    @property
    def step_name(self) -> str:
        return "Enhanced AI Asset Extraction"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.ASSET_EXTRACTION

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.STRUCTURE_ANALYSIS]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute enhanced AI asset extraction for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            if self.config.processing.debug_mode:
                print(f"Starting enhanced asset extraction for page {page_data.page_number}")

            # Load page structure if not already loaded
            if not page_data.page_structure:
                page_data.page_structure = self.file_manager.load_page_structure(
                    page_data.source_pdf, page_data.page_number
                )

                if not page_data.page_structure:
                    raise PipelineStepError("Page structure not found for asset extraction")

            # Check if already processed
            if (self.config.processing.skip_existing_files and
                self.file_manager.is_stage_completed(
                    page_data.source_pdf,
                    page_data.page_number,
                    self.processing_stage
                )):

                page_data.add_log("Asset extraction already completed")
                return page_data

            # Perform asset extraction
            result = self.vision_processor.extract_assets(page_data)

            if not result.success:
                raise PipelineStepError(f"Asset extraction failed: {result.error_message}")

            # Update page data with extracted assets
            page_data.extracted_assets = result.extracted_assets

            # Save individual asset files
            saved_assets = 0
            for asset in page_data.extracted_assets:
                try:
                    asset_path = self.file_manager.save_asset_file(
                        page_data.source_pdf,
                        page_data.page_number,
                        asset
                    )
                    page_data.asset_files[asset.name] = asset_path
                    saved_assets += 1
                except Exception as e:
                    if self.config.processing.debug_mode:
                        print(f"Failed to save asset {asset.name}: {e}")
                    # Continue with other assets

            # Mark stage as completed
            page_data.mark_stage_completed(self.processing_stage)
            page_data.add_log(
                f"Enhanced AI asset extraction completed: {len(page_data.extracted_assets)} assets "
                f"extracted, {saved_assets} saved in {result.processing_time:.2f}s"
            )

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI asset extraction failed: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI asset extraction: {e}")


class EnhancedAIMarkdownGenerationStep(PipelineStep):
    """Enhanced AI-powered Markdown generation pipeline step."""

    def __init__(self, config: AssetableConfig) -> None:
        super().__init__(config)
        self.vision_processor = EnhancedVisionProcessor(config)
        self.document_type = "technical document"  # Can be configured

    @property
    def step_name(self) -> str:
        return "Enhanced AI Markdown Generation"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.MARKDOWN_GENERATION

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return [ProcessingStage.ASSET_EXTRACTION]

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute enhanced AI Markdown generation for a single page."""
        try:
            # Validate dependencies
            self.validate_dependencies(page_data)

            if self.config.processing.debug_mode:
                print(f"Starting enhanced Markdown generation for page {page_data.page_number}")

            # Load dependencies if not already loaded
            if not page_data.page_structure:
                page_data.page_structure = self.file_manager.load_page_structure(
                    page_data.source_pdf, page_data.page_number
                )

            if not page_data.extracted_assets:
                # Load extracted assets from previous stage if available
                page_data.extracted_assets = []  # Will be populated if assets exist

            # Check if already processed
            if (self.config.processing.skip_existing_files and
                self.file_manager.is_stage_completed(
                    page_data.source_pdf,
                    page_data.page_number,
                    self.processing_stage
                )):

                existing_markdown = self.file_manager.load_markdown_content(
                    page_data.source_pdf, page_data.page_number
                )

                if existing_markdown:
                    page_data.markdown_content = existing_markdown
                    page_data.add_log("Loaded existing Markdown content")
                    return page_data

            # Generate Markdown
            result = self.vision_processor.generate_markdown(page_data, self.document_type)

            if not result.success:
                raise PipelineStepError(f"Markdown generation failed: {result.error_message}")

            # Update page data with Markdown content
            page_data.markdown_content = result.markdown_content

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
            page_data.add_log(
                f"Enhanced AI Markdown generation completed using {result.model_used} "
                f"in {result.processing_time:.2f}s, {len(result.asset_references)} asset references"
            )

            return page_data

        except VisionProcessorError as e:
            raise PipelineStepError(f"AI Markdown generation failed: {e}")
        except Exception as e:
            raise PipelineStepError(f"Unexpected error in AI Markdown generation: {e}")


# Maintain backward compatibility while encouraging use of enhanced versions
AIStructureAnalysisStep = EnhancedAIStructureAnalysisStep
AIAssetExtractionStep = EnhancedAIAssetExtractionStep
AIMarkdownGenerationStep = EnhancedAIMarkdownGenerationStep
