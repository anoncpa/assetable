"""
Pipeline execution engine for Assetable.

This module provides the core pipeline execution functionality that orchestrates
all processing steps from PDF splitting to final Markdown generation.
The engine implements a step-based architecture with proper error handling
and state management.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from ..config import AssetableConfig, get_config
from ..file_manager import FileManager
from ..models import DocumentData, PageData, ProcessingStage, PageStructure
from .pdf_splitter import PDFSplitter

# These are imported later to avoid circular dependencies
# from .ai_steps import (
#     AIStructureAnalysisStep,
#     AIAssetExtractionStep,
#     AIMarkdownGenerationStep,
# )


class PipelineError(Exception):
    """Base exception for pipeline operations."""
    pass


class PipelineStepError(PipelineError):
    """Raised when a pipeline step fails."""
    pass


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Each step in the processing pipeline should inherit from this class
    and implement the execute method. Steps are responsible for:
    - Processing a single page or document
    - Updating the PageData/DocumentData objects
    - Handling their own errors appropriately
    - Logging progress and status
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """Initialize pipeline step with configuration."""
        self.config = config or get_config()
        self.file_manager = FileManager(self.config)

    @property
    @abstractmethod
    def step_name(self) -> str:
        """Return the name of this pipeline step."""
        pass

    @property
    @abstractmethod
    def processing_stage(self) -> ProcessingStage:
        """Return the processing stage this step handles."""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> List[ProcessingStage]:
        """Return list of processing stages this step depends on."""
        pass

    @abstractmethod
    async def execute_page(self, page_data: PageData) -> PageData:
        """
        Execute this step for a single page.

        Args:
            page_data: Page data to process.

        Returns:
            Updated page data.

        Raises:
            PipelineStepError: If step execution fails.
        """
        pass

    async def execute_document(self, document_data: DocumentData) -> DocumentData:
        """
        Execute this step for an entire document.

        Default implementation processes pages individually.
        Override for document-level processing.

        Args:
            document_data: Document data to process.

        Returns:
            Updated document data.
        """
        for page_data in document_data.pages:
            if self.should_process_page(page_data):
                try:
                    updated_page = await self.execute_page(page_data)
                    document_data.add_page(updated_page)

                    # Save progress
                    self.file_manager.save_page_data(updated_page)

                except Exception as e:
                    error_msg = f"Failed to process page {page_data.page_number} in {self.step_name}: {e}"
                    page_data.add_log(error_msg)
                    if self.config.processing.debug_mode:
                        print(f"Warning: {error_msg}")

                    if isinstance(e, PipelineStepError):
                        raise  # Re-raise critical step errors to stop this step's execution
                    else:
                        # Wrap non-PipelineStepError errors before re-raising
                        raise PipelineStepError(error_msg) from e

        return document_data

    def should_process_page(self, page_data: PageData) -> bool:
        """
        Determine if this step should process the given page.

        Args:
            page_data: Page data to check.

        Returns:
            True if page should be processed, False otherwise.
        """
        # Check if step is already completed
        if page_data.is_stage_completed(self.processing_stage):
            if self.config.processing.skip_existing_files:
                return False

        # Check dependencies
        for dependency in self.dependencies:
            if not page_data.is_stage_completed(dependency):
                return False

        return True

    def validate_dependencies(self, page_data: PageData) -> None:
        """
        Validate that all dependencies are satisfied.

        Args:
            page_data: Page data to validate.

        Raises:
            PipelineStepError: If dependencies are not satisfied.
        """
        missing_deps = []
        for dependency in self.dependencies:
            if not page_data.is_stage_completed(dependency):
                missing_deps.append(dependency.value)

        if missing_deps:
            raise PipelineStepError(
                f"Step {self.step_name} missing dependencies: {', '.join(missing_deps)}"
            )


class PDFSplitStep(PipelineStep):
    """Pipeline step for PDF splitting."""

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        super().__init__(config)
        self.pdf_splitter = PDFSplitter(self.config)

    @property
    def step_name(self) -> str:
        return "PDF Split"

    @property
    def processing_stage(self) -> ProcessingStage:
        return ProcessingStage.PDF_SPLIT

    @property
    def dependencies(self) -> List[ProcessingStage]:
        return []  # No dependencies

    async def execute_page(self, page_data: PageData) -> PageData:
        """Execute PDF splitting for a single page."""
        # PDF splitting is handled at document level, not page level
        raise NotImplementedError("PDF splitting must be done at document level")

    async def execute_document(self, document_data: DocumentData) -> DocumentData:
        """Execute PDF splitting for entire document."""
        try:
            # Use the existing PDF splitter, passing the document_data to be updated
            document_data = self.pdf_splitter.split_pdf(
                document_data.source_pdf_path,
                force_regenerate=not self.config.processing.skip_existing_files,
                document_data=document_data
            )
            return document_data

        except Exception as e:
            raise PipelineStepError(f"PDF splitting failed: {e}") from e


class PipelineEngine:
    """
    Main pipeline execution engine.

    This class orchestrates the execution of all pipeline steps,
    manages dependencies, handles errors, and provides progress tracking.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        """Initialize pipeline engine with configuration."""
        self.config = config or get_config()
        self.file_manager = FileManager(self.config)

        # Define default pipeline steps
        from .ai_steps import (
            AIStructureAnalysisStep,
            AIAssetExtractionStep,
            AIMarkdownGenerationStep,
        )
        self.steps: List[PipelineStep] = [
            PDFSplitStep(self.config),
            AIStructureAnalysisStep(self.config),
            AIAssetExtractionStep(self.config),
            AIMarkdownGenerationStep(self.config),
        ]

        # Execution state
        self.current_document: Optional[DocumentData] = None
        self.execution_start_time: Optional[datetime] = None
        self.execution_stats: Dict[str, Any] = {}

    def add_step(self, step: PipelineStep) -> None:
        """Add a custom pipeline step."""
        self.steps.append(step)

    def remove_step(self, step_type: Type[PipelineStep]) -> None:
        """Remove a pipeline step by type."""
        self.steps = [step for step in self.steps if not isinstance(step, step_type)]

    def get_step(self, step_type: Type[PipelineStep]) -> Optional[PipelineStep]:
        """Get a pipeline step by type."""
        for step in self.steps:
            if isinstance(step, step_type):
                return step
        return None

    async def execute_pipeline(
        self,
        pdf_path: Path,
        target_stages: Optional[List[ProcessingStage]] = None,
        page_numbers: Optional[List[int]] = None
    ) -> DocumentData:
        """
        Execute the complete processing pipeline.

        Args:
            pdf_path: Path to the PDF file to process.
            target_stages: List of stages to execute. If None, executes all stages.
            page_numbers: Specific page numbers to process. If None, processes all pages.

        Returns:
            DocumentData object with processing results.

        Raises:
            PipelineError: If pipeline execution fails.
        """
        try:
            # Initialize execution
            self.execution_start_time = datetime.now()
            self.execution_stats = {
                "total_pages": 0,
                "processed_pages": 0,
                "failed_pages": 0,
                "skipped_pages": 0,
                "steps_executed": [],
                "errors": [],
            }

            # Load or create document data
            document_data = await self._initialize_document(pdf_path)
            self.current_document = document_data

            # Filter steps based on target stages
            steps_to_execute = self._filter_steps(target_stages)

            # Validate pipeline configuration
            self._validate_pipeline(steps_to_execute)

            # Execute each step
            for step in steps_to_execute:
                if self.config.processing.debug_mode:
                    print(f"Executing step: {step.step_name}")

                try:
                    document_data = await step.execute_document(document_data)
                    self.execution_stats["steps_executed"].append(step.step_name)

                    # Save document progress
                    self.file_manager.save_document_data(document_data)

                except Exception as e:
                    error_msg = f"Step {step.step_name} failed: {e}"
                    self.execution_stats["errors"].append(error_msg)

                    if self.config.processing.debug_mode:
                        print(f"Error: {error_msg}")

                    raise

            # Update execution stats
            self._update_execution_stats(document_data)

            return document_data

        except Exception as e:
            if not isinstance(e, PipelineError):
                raise PipelineError(f"Pipeline execution failed: {e}") from e
            raise

    async def execute_single_step(
        self,
        pdf_path: Path,
        step_type: Type[PipelineStep],
        page_numbers: Optional[List[int]] = None
    ) -> DocumentData:
        """
        Execute a single pipeline step.

        Args:
            pdf_path: Path to the PDF file to process.
            step_type: Type of step to execute.
            page_numbers: Specific page numbers to process.

        Returns:
            Updated DocumentData object.

        Raises:
            PipelineError: If step execution fails.
        """
        step = self.get_step(step_type)
        if not step:
            raise PipelineError(f"Step {step_type.__name__} not found in pipeline")

        # Load document data
        document_data = await self._initialize_document(pdf_path)

        # Filter pages if specified
        if page_numbers:
            filtered_pages = [p for p in document_data.pages if p.page_number in page_numbers]
            document_data.pages = filtered_pages

        # Execute step
        document_data = await step.execute_document(document_data)

        # Save results
        self.file_manager.save_document_data(document_data)

        return document_data

    def get_pipeline_status(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get current pipeline execution status.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary containing pipeline status information.
        """
        try:
            # Load document data
            document_data = self.file_manager.load_document_data(pdf_path)
            if not document_data:
                return {
                    "status": "not_started",
                    "error": "No document data found"
                }

            total_pages = len(document_data.pages)

            # Get processing summary
            summary = self.file_manager.get_processing_summary(pdf_path, total_pages)

            # Calculate overall progress
            total_steps = len(self.steps)
            completed_steps = 0

            for step in self.steps:
                stage_summary = summary["stages"].get(step.processing_stage.value, {})
                if stage_summary.get("progress", 0) >= 1.0:
                    completed_steps += 1

            overall_progress = completed_steps / total_steps if total_steps > 0 else 0

            # Determine status
            if overall_progress >= 1.0:
                status = "completed"
            elif overall_progress > 0:
                status = "in_progress"
            else:
                status = "not_started"

            return {
                "status": status,
                "overall_progress": overall_progress,
                "total_pages": total_pages,
                "steps": [
                    {
                        "name": step.step_name,
                        "stage": step.processing_stage.value,
                        "progress": summary["stages"].get(step.processing_stage.value, {}).get("progress", 0),
                        "completed_pages": summary["stages"].get(step.processing_stage.value, {}).get("completed_count", 0),
                        "pending_pages": summary["stages"].get(step.processing_stage.value, {}).get("pending_count", 0),
                    }
                    for step in self.steps
                ],
                "execution_stats": self.execution_stats,
                "last_updated": document_data.last_updated.isoformat(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _initialize_document(self, pdf_path: Path) -> DocumentData:
        """Initialize or load document data."""
        if not pdf_path.exists():
            raise PipelineError(f"PDF file not found: {pdf_path}")

        # Try to load existing document data
        document_data = self.file_manager.load_document_data(pdf_path)

        if document_data:
            return document_data

        # Create new document data
        # We need to get the page count from the PDF
        from .pdf_splitter import PDFSplitter
        pdf_splitter = PDFSplitter(self.config)
        pdf_info = pdf_splitter.get_pdf_info(pdf_path)

        document_id = pdf_path.stem  # Use PDF filename (without extension) as document_id
        document_data = DocumentData(
            document_id=document_id,
            source_pdf_path=pdf_path,
            output_directory=self.config.get_document_output_dir(pdf_path),
            # total_pages is no longer a direct field of DocumentData based on models.py
        )

        # Create placeholder page data
        # Use pdf_info["total_pages"] as total_pages is removed from DocumentData model
        for page_num in range(1, pdf_info["total_pages"] + 1):
            page_data = PageData(
                page_number=page_num,
                source_pdf=pdf_path
            )
            document_data.add_page(page_data)

        # Setup directory structure
        self.file_manager.setup_document_structure(pdf_path)

        return document_data

    def _filter_steps(self, target_stages: Optional[List[ProcessingStage]]) -> List[PipelineStep]:
        """Filter steps based on target stages."""
        if target_stages is None:
            return self.steps

        target_stage_set = set(target_stages)
        return [step for step in self.steps if step.processing_stage in target_stage_set]

    def _validate_pipeline(self, steps: List[PipelineStep]) -> None:
        """Validate pipeline configuration."""
        # Check for duplicate stages
        stages = [step.processing_stage for step in steps]
        if len(stages) != len(set(stages)):
            raise PipelineConfigError("Duplicate processing stages found in pipeline")

        # Check dependencies
        available_stages = set(stages)
        for step in steps:
            for dependency in step.dependencies:
                if dependency not in available_stages:
                    raise PipelineConfigError(
                        f"Step {step.step_name} depends on {dependency.value} which is not available"
                    )

    def _update_execution_stats(self, document_data: DocumentData) -> None:
        """Update execution statistics."""
        total_pages = len(document_data.pages)
        self.execution_stats["total_pages"] = total_pages

        processed_count = 0
        failed_count = 0

        for page_data in document_data.pages:
            if any("Error" in log for log in page_data.processing_log):
                failed_count += 1
            elif len(page_data.completed_stages) > 0:
                processed_count += 1

        self.execution_stats["processed_pages"] = processed_count
        self.execution_stats["failed_pages"] = failed_count
        self.execution_stats["skipped_pages"] = (
            total_pages - processed_count - failed_count
        )

        if self.execution_start_time:
            execution_time = datetime.now() - self.execution_start_time
            self.execution_stats["execution_time_seconds"] = execution_time.total_seconds()


async def run_pipeline(
    pdf_path: Path,
    config: Optional[AssetableConfig] = None,
    target_stages: Optional[List[ProcessingStage]] = None,
    page_numbers: Optional[List[int]] = None
) -> DocumentData:
    """
    Convenience function to run the complete pipeline.

    Args:
        pdf_path: Path to the PDF file to process.
        config: Configuration object. If None, uses global config.
        target_stages: List of stages to execute. If None, executes all stages.
        page_numbers: Specific page numbers to process. If None, processes all pages.

    Returns:
        DocumentData object with processing results.
    """
    engine = PipelineEngine(config)
    return await engine.execute_pipeline(pdf_path, target_stages, page_numbers)


async def run_single_step(
    pdf_path: Path,
    step_type: Type[PipelineStep],
    config: Optional[AssetableConfig] = None,
    page_numbers: Optional[List[int]] = None
) -> DocumentData:
    """
    Convenience function to run a single pipeline step.

    Args:
        pdf_path: Path to the PDF file to process.
        step_type: Type of step to execute.
        config: Configuration object. If None, uses global config.
        page_numbers: Specific page numbers to process.

    Returns:
        Updated DocumentData object.
    """
    engine = PipelineEngine(config)
    return await engine.execute_single_step(pdf_path, step_type, page_numbers)
