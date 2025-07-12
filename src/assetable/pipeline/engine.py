"""
Pipeline execution engine for Assetable.

This module orchestrates all processing steps from PDF splitting to
Markdown generation.  Each step is executed with proper dependency
management and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..config import AssetableConfig, get_config
from ..file_manager import FileManager
from ..models import DocumentData, PageData, ProcessingStage
from ..pdf.pdf_splitter import PDFSplitter

# These are imported later to avoid circular dependencies
# from ..ai.ai_steps import (
#     EnhancedAIStructureAnalysisStep,
#     EnhancedAIAssetExtractionStep,
#     EnhancedAIMarkdownGenerationStep,
# )


class PipelineError(Exception):
    """Base exception for pipeline operations."""


class PipelineStepError(PipelineError):
    """Raised when a pipeline step fails."""


class PipelineConfigError(PipelineError):
    """Raised when pipeline configuration is invalid."""


class PipelineStep(ABC):
    """
    Abstract base class for pipeline steps.

    Each concrete step must implement:
      • step_name
      • processing_stage
      • dependencies
      • execute_page
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        self.config = config or get_config()
        self.file_manager = FileManager(self.config)

    @property
    @abstractmethod
    def step_name(self) -> str: ...

    @property
    @abstractmethod
    def processing_stage(self) -> ProcessingStage: ...

    @property
    @abstractmethod
    def dependencies(self) -> List[ProcessingStage]: ...

    @abstractmethod
    async def execute_page(self, page_data: PageData) -> PageData: ...

    async def execute_document(self, document_data: DocumentData) -> DocumentData:
        """
        Default implementation: iterate each page and call `execute_page`.
        Override when a step needs document-level context.
        """
        for page_data in document_data.pages:
            if self.should_process_page(page_data):
                try:
                    updated_page = await self.execute_page(page_data)
                    document_data.add_page(updated_page)
                    self.file_manager.save_page_data(updated_page)
                except Exception as exc:  # noqa: BLE001
                    msg = f"Failed to process page {page_data.page_number} in {self.step_name}: {exc}"
                    page_data.add_log(msg)
                    if self.config.processing.debug_mode:
                        print(f"Warning: {msg}")
                    raise

        return document_data

    def should_process_page(self, page_data: PageData) -> bool:
        """
        Return True when dependencies are met and this stage is not yet complete.
        """
        if page_data.is_stage_completed(self.processing_stage):
            return not self.config.processing.skip_existing_files

        for dependency in self.dependencies:
            if not page_data.is_stage_completed(dependency):
                return False

        return True

    def validate_dependencies(self, page_data: PageData) -> None:
        missing_deps: list[str] = [
            dependency.value
            for dependency in self.dependencies
            if not page_data.is_stage_completed(dependency)
        ]
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
        return []

    async def execute_page(self, page_data: PageData) -> PageData:  # pragma: no cover
        raise NotImplementedError("PDF splitting is handled at document level")

    async def execute_document(self, document_data: DocumentData) -> DocumentData:
        try:
            return self.pdf_splitter.split_pdf(
                document_data.source_pdf_path,
                force_regenerate=not self.config.processing.skip_existing_files,
                document_data=document_data,
            )
        except Exception as exc:  # noqa: BLE001
            raise PipelineStepError(f"PDF splitting failed: {exc}") from exc


class PipelineEngine:
    """
    Orchestrates all pipeline steps and tracks execution state.
    """

    def __init__(self, config: Optional[AssetableConfig] = None) -> None:
        self.config = config or get_config()
        self.file_manager = FileManager(self.config)

        from ..ai.ai_steps import (  # local import to avoid circular dependency
            EnhancedAIStructureAnalysisStep,
            EnhancedAIAssetExtractionStep,
            EnhancedAIMarkdownGenerationStep,
        )

        self.steps: List[PipelineStep] = [
            PDFSplitStep(self.config),
            EnhancedAIStructureAnalysisStep(self.config),
            EnhancedAIAssetExtractionStep(self.config),
            EnhancedAIMarkdownGenerationStep(self.config),
        ]

        self.current_document: Optional[DocumentData] = None
        self.execution_start_time: Optional[datetime] = None
        self.execution_stats: Dict[str, Any] = {}

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    async def execute_pipeline(
        self,
        pdf_path: Path,
        target_stages: Optional[List[ProcessingStage]] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> DocumentData:
        """
        Run all (or selected) stages of the pipeline.
        """
        self.execution_start_time = datetime.now()
        self.execution_stats = {
            "total_pages": 0,
            "processed_pages": 0,
            "failed_pages": 0,
            "skipped_pages": 0,
            "steps_executed": [],
            "errors": [],
        }

        document_data = await self._initialize_document(pdf_path)

        # ページ指定がある場合は先に絞り込む
        if page_numbers:
            document_data.pages = [
                p for p in document_data.pages if p.page_number in page_numbers
            ]

        steps_to_execute = self._filter_steps(target_stages)
        self._validate_pipeline(steps_to_execute)

        for step in steps_to_execute:
            if self.config.processing.debug_mode:
                print(f"Executing step: {step.step_name}")

            try:
                document_data = await step.execute_document(document_data)
                self.execution_stats["steps_executed"].append(step.step_name)
                self.file_manager.save_document_data(document_data)
            except Exception as exc:  # noqa: BLE001
                msg = f"Step {step.step_name} failed: {exc}"
                self.execution_stats["errors"].append(msg)
                if self.config.processing.debug_mode:
                    print(f"Error: {msg}")
                raise

        self._update_execution_stats(document_data)
        return document_data

    async def execute_single_step(
        self,
        pdf_path: Path,
        step_type: Type[PipelineStep],
        page_numbers: Optional[List[int]] = None,
    ) -> DocumentData:
        step = self.get_step(step_type)
        if not step:
            raise PipelineError(f"Step {step_type.__name__} not found in pipeline")

        document_data = await self._initialize_document(pdf_path)

        if page_numbers:
            document_data.pages = [
                p for p in document_data.pages if p.page_number in page_numbers
            ]

        document_data = await step.execute_document(document_data)
        self.file_manager.save_document_data(document_data)
        return document_data

    def get_pipeline_status(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get the current status of the pipeline for the given PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dictionary containing pipeline status with keys:
              - status: str (e.g., 'not_started', 'in_progress', 'completed')
              - overall_progress: float (0.0 to 1.0)
              - steps: List of step status dicts with keys:
                  - name: str
                  - progress: float
                  - completed_pages: int
                  - pending_pages: int
              - last_updated: ISO datetime string
        """
        # Load existing document data if available
        document_data: Optional[DocumentData] = self.file_manager.load_document_data(pdf_path)

        if document_data is None:
            # No data yet, pipeline not started
            return {
                "status": "not_started",
                "overall_progress": 0.0,
                "steps": [],
                "last_updated": datetime.now().isoformat(),
            }

        total_pages = len(document_data.pages)

        # Calculate progress per stage
        stages = [
            ProcessingStage.PDF_SPLIT,
            ProcessingStage.STRUCTURE_ANALYSIS,
            ProcessingStage.ASSET_EXTRACTION,
            ProcessingStage.MARKDOWN_GENERATION,
        ]

        steps_status: List[Dict[str, Any]] = []
        total_progress = 0.0

        for stage in stages:
            completed_pages = len(document_data.get_completed_pages(stage))
            pending_pages = total_pages - completed_pages
            progress = completed_pages / total_pages if total_pages > 0 else 0.0

            steps_status.append({
                "name": stage.value,
                "progress": progress,
                "completed_pages": completed_pages,
                "pending_pages": pending_pages,
            })

            total_progress += progress

        overall_progress = total_progress / len(stages) if stages else 0.0

        # Determine overall status
        if overall_progress >= 1.0:
            status = "completed"
        elif overall_progress > 0.0:
            status = "in_progress"
        else:
            status = "not_started"

        return {
            "status": status,
            "overall_progress": overall_progress,
            "steps": steps_status,
            "last_updated": datetime.now().isoformat(),
        }

    # --------------------------------------------------------------------- #
    # Helper methods                                                        #
    # --------------------------------------------------------------------- #
    def add_step(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def remove_step(self, step_type: Type[PipelineStep]) -> None:
        self.steps = [s for s in self.steps if not isinstance(s, step_type)]

    def get_step(self, step_type: Type[PipelineStep]) -> Optional[PipelineStep]:
        return next((s for s in self.steps if isinstance(s, step_type)), None)

    def _filter_steps(self, target_stages: Optional[List[ProcessingStage]]) -> List[PipelineStep]:
        if target_stages is None:
            return self.steps
        target_set = set(target_stages)
        return [s for s in self.steps if s.processing_stage in target_set]

    def _validate_pipeline(self, steps: List[PipelineStep]) -> None:
        stages = [s.processing_stage for s in steps]
        if len(stages) != len(set(stages)):
            raise PipelineConfigError("Duplicate processing stages in pipeline")
        available = set(stages)
        for step in steps:
            for dep in step.dependencies:
                if dep not in available:
                    raise PipelineConfigError(
                        f"Step {step.step_name} depends on {dep.value}, which is not scheduled"
                    )

    async def _initialize_document(self, pdf_path: Path) -> DocumentData:
        if not pdf_path.exists():
            raise PipelineError(f"PDF not found: {pdf_path}")

        existing = self.file_manager.load_document_data(pdf_path)
        if existing:
            return existing

        pdf_splitter = PDFSplitter(self.config)
        pdf_info = pdf_splitter.get_pdf_info(pdf_path)

        doc = DocumentData(
            document_id=pdf_path.stem,
            source_pdf_path=pdf_path,
            output_directory=self.config.get_document_output_dir(pdf_path),
        )

        for page_num in range(1, pdf_info["total_pages"] + 1):
            doc.add_page(
                PageData(
                    page_number=page_num,
                    source_pdf=pdf_path,
                )
            )

        self.file_manager.setup_document_structure(pdf_path)
        return doc

    def _update_execution_stats(self, document_data: DocumentData) -> None:
        total_pages = len(document_data.pages)
        processed = sum(
            1 for p in document_data.pages if p.completed_stages
        )
        failed = sum(
            1 for p in document_data.pages if any("Error" in log for log in p.processing_log)
        )
        self.execution_stats.update(
            {
                "total_pages": total_pages,
                "processed_pages": processed,
                "failed_pages": failed,
                "skipped_pages": total_pages - processed - failed,
                "execution_time_seconds": (
                    datetime.now() - self.execution_start_time
                ).total_seconds()
                if self.execution_start_time
                else None,
            }
        )


# ------------------------------------------------------------------------- #
# Convenience async helpers                                                 #
# ------------------------------------------------------------------------- #
async def run_pipeline(
    pdf_path: Path,
    config: Optional[AssetableConfig] = None,
    target_stages: Optional[List[ProcessingStage]] = None,
    page_numbers: Optional[List[int]] = None,
) -> DocumentData:
    engine = PipelineEngine(config)
    return await engine.execute_pipeline(pdf_path, target_stages, page_numbers)


async def run_single_step(
    pdf_path: Path,
    step_type: Type[PipelineStep],
    config: Optional[AssetableConfig] = None,
    page_numbers: Optional[List[int]] = None,
) -> DocumentData:
    engine = PipelineEngine(config)
    return await engine.execute_single_step(pdf_path, step_type, page_numbers)
