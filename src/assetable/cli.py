"""
Command-line interface for Assetable.

This module provides CLI commands for PDF processing pipeline using Typer.
"""

import typer
from pathlib import Path
from typing import Optional

from .config import AssetableConfig, get_config, ProcessingStage
from .pipeline.pdf_splitter import split_pdf_cli, PDFSplitter
from .file_manager import FileManager
from .pipeline.engine import (
    PipelineEngine,
    PDFSplitStep,
    StructureAnalysisStep,
    AssetExtractionStep,
    MarkdownGenerationStep,
    run_pipeline,
    run_single_step,
)
import asyncio

app = typer.Typer(
    name="assetable",
    help="Convert scanned books into AI- and human-readable digital assets",
    add_completion=False
)


@app.command()
def split(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to split"),
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration even if files exist"),
    dpi: Optional[int] = typer.Option(None, "--dpi", help="DPI for image conversion"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Split PDF into individual page images.

    This command converts each page of a PDF into a high-quality image
    suitable for AI processing. Images are saved in PNG format by default.

    Examples:
        assetable split input/book.pdf
        assetable split input/book.pdf --force --dpi 400
        assetable split input/book.pdf --output custom_output --debug
    """
    # Validate input
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    if not pdf_file.is_file():
        typer.echo(f"Error: Path is not a file: {pdf_path}", err=True)
        raise typer.Exit(1)

    # Setup configuration
    config = get_config()

    # Override configuration with CLI options
    if dpi is not None:
        config.pdf_split.dpi = dpi

    if output_dir is not None:
        config.output.output_directory = Path(output_dir)

    if debug:
        config.processing.debug_mode = True

    try:
        # Display PDF information
        splitter = PDFSplitter(config)
        pdf_info = splitter.get_pdf_info(pdf_file)

        typer.echo(f"PDF Information:")
        typer.echo(f"  File: {pdf_info['filename']}")
        typer.echo(f"  Pages: {pdf_info['total_pages']}")
        typer.echo(f"  Size: {pdf_info['file_size']:,} bytes")

        if pdf_info.get('page_dimensions'):
            dims = pdf_info['page_dimensions']
            typer.echo(f"  Page size: {dims['width']:.1f}x{dims['height']:.1f} points "
                      f"({dims['width_inches']:.1f}x{dims['height_inches']:.1f} inches)")

        typer.echo(f"  Output DPI: {config.pdf_split.dpi}")
        typer.echo(f"  Image format: {config.pdf_split.image_format.upper()}")
        typer.echo("")

        # Check processing status
        status = splitter.get_processing_status(pdf_file)
        if 'split_status' in status:
            split_status = status['split_status']
            if split_status['completed'] > 0 and not force:
                typer.echo(f"Found {split_status['completed']} existing pages. "
                          f"Use --force to regenerate.")

        # Perform splitting
        typer.echo("Starting PDF splitting...")

        with typer.progressbar(length=pdf_info['total_pages'],
                             label="Processing pages") as progress:

            document_data = split_pdf_cli(pdf_file, force, config)

            # This progress bar update is not ideal as it runs after the fact.
            # For a real-time progress bar, we would need to pass a callback
            # to the splitter. For now, we just fill the bar.
            progress.update(pdf_info['total_pages'])

        # Display results
        completed_pages = len([p for p in document_data.pages
                             if p.is_stage_completed(p.current_stage)])

        typer.echo(f"\nCompleted: {completed_pages}/{pdf_info['total_pages']} pages")

        output_dir_path = config.get_pdf_split_dir(pdf_file)
        typer.echo(f"Output directory: {output_dir_path}")

        if config.processing.debug_mode:
            typer.echo(f"Document data saved to: {config.get_document_output_dir(pdf_file)}")

        typer.echo("PDF splitting completed successfully!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
) -> None:
    """
    Display information about a PDF file.

    Shows basic information about the PDF including page count,
    dimensions, and processing status.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    try:
        splitter = PDFSplitter()
        info = splitter.get_pdf_info(pdf_file)
        status = splitter.get_processing_status(pdf_file)

        typer.echo(f"PDF Information:")
        typer.echo(f"  File: {info['filename']}")
        typer.echo(f"  Path: {info['path']}")
        typer.echo(f"  Total pages: {info['total_pages']}")
        typer.echo(f"  File size: {info['file_size']:,} bytes")
        typer.echo(f"  Encrypted: {info['is_encrypted']}")
        typer.echo(f"  Created: {info['creation_date']}")
        typer.echo(f"  Modified: {info['modification_date']}")

        if info.get('page_dimensions'):
            dims = info['page_dimensions']
            typer.echo(f"  Page dimensions: {dims['width']:.1f}x{dims['height']:.1f} points")
            typer.echo(f"  Page size: {dims['width_inches']:.1f}x{dims['height_inches']:.1f} inches")

        # Display metadata
        if info.get('metadata'):
            metadata = info['metadata']
            typer.echo(f"  Metadata:")
            for key, value in metadata.items():
                if value:
                    typer.echo(f"    {key}: {value}")

        # Display processing status
        if 'split_status' in status:
            split_status = status['split_status']
            typer.echo(f"\nProcessing Status:")
            typer.echo(f"  Split completed: {split_status['completed']}/{info['total_pages']} pages")
            typer.echo(f"  Progress: {split_status['progress']:.1%}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
) -> None:
    """
    Show processing status for a PDF file.

    Displays detailed information about processing progress
    for all pipeline stages.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    try:
        file_manager = FileManager()
        splitter = PDFSplitter()

        # Get PDF info
        pdf_info = splitter.get_pdf_info(pdf_file)
        total_pages = pdf_info['total_pages']

        # Get processing summary
        summary = file_manager.get_processing_summary(pdf_file, total_pages)

        typer.echo(f"Processing Status for {pdf_info['filename']}:")
        typer.echo(f"Total pages: {total_pages}")
        typer.echo(f"Overall progress: {summary['overall_progress']:.1%}")
        typer.echo("")

        # Display stage-by-stage status
        for stage_name, stage_info in summary['stages'].items():
            typer.echo(f"{stage_name.replace('_', ' ').title()}:")
            typer.echo(f"  Completed: {stage_info['completed_count']}/{total_pages} pages")
            typer.echo(f"  Progress: {stage_info['progress']:.1%}")

            if stage_info['pending_count'] > 0:
                pending_pages = stage_info['pending_pages'][:5]  # Show first 5
                remaining = stage_info['pending_count'] - len(pending_pages)
                typer.echo(f"  Pending pages: {pending_pages}")
                if remaining > 0:
                    typer.echo(f"  ... and {remaining} more")

            typer.echo("")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def cleanup(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file"),
    stage: str = typer.Option("split", "--stage", "-s",
                             help="Stage to clean up (split, structure, markdown, all)"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """
    Clean up generated files for a PDF.

    Removes generated files for the specified processing stage.
    Use with caution as this operation cannot be undone.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    if not confirm:
        typer.echo(f"This will remove {stage} files for {pdf_file.name}")
        if not typer.confirm("Are you sure?"):
            typer.echo("Operation cancelled.")
            raise typer.Exit(0)

    try:
        if stage == "split":
            splitter = PDFSplitter()
            splitter.cleanup_split_files(pdf_file)
            typer.echo("Split files cleaned up successfully.")

        elif stage == "all":
            # Clean up all generated files
            config = get_config()
            doc_dir = config.get_document_output_dir(pdf_file)

            if doc_dir.exists():
                import shutil
                shutil.rmtree(doc_dir)
                typer.echo(f"All files cleaned up: {doc_dir}")
            else:
                typer.echo("No files found to clean up.")

        else:
            typer.echo(f"Unknown stage: {stage}")
            typer.echo("Available stages: split, all")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error during cleanup: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def pipeline(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to process"),
    stages: Optional[str] = typer.Option(None, "--stages", "-s",
                                       help="Comma-separated list of stages to run (split,structure,extraction,markdown)"),
    pages: Optional[str] = typer.Option(None, "--pages", "-p",
                                      help="Comma-separated list of page numbers to process"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing of existing files"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Run the complete processing pipeline.

    This command executes the full pipeline from PDF splitting to Markdown generation.
    You can specify which stages to run and which pages to process.

    Examples:
        assetable pipeline input/book.pdf
        assetable pipeline input/book.pdf --stages split,structure
        assetable pipeline input/book.pdf --pages 1,2,3 --force
        assetable pipeline input/book.pdf --output custom_output --debug
    """
    # Validate input
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    # Setup configuration
    config = get_config()

    if output_dir is not None:
        config.output.output_directory = Path(output_dir)

    if debug:
        config.processing.debug_mode = True

    if force:
        config.processing.skip_existing_files = False

    # Parse stages
    target_stages = None
    if stages:
        stage_mapping = {
            "split": ProcessingStage.PDF_SPLIT,
            "structure": ProcessingStage.STRUCTURE_ANALYSIS,
            "extraction": ProcessingStage.ASSET_EXTRACTION,
            "markdown": ProcessingStage.MARKDOWN_GENERATION,
        }

        stage_names = [s.strip() for s in stages.split(",")]
        target_stages = []

        for stage_name in stage_names:
            if stage_name not in stage_mapping:
                typer.echo(f"Error: Unknown stage '{stage_name}'. Available: {', '.join(stage_mapping.keys())}", err=True)
                raise typer.Exit(1)
            target_stages.append(stage_mapping[stage_name])

    # Parse page numbers
    page_numbers = None
    if pages:
        try:
            page_numbers = [int(p.strip()) for p in pages.split(",")]
        except ValueError:
            typer.echo("Error: Invalid page numbers format. Use comma-separated integers.", err=True)
            raise typer.Exit(1)

    try:
        # Create pipeline engine
        engine = PipelineEngine(config)

        # Get initial status
        initial_status = engine.get_pipeline_status(pdf_file)

        if initial_status.get("status") == "not_started":
            typer.echo(f"Starting pipeline for {pdf_file.name}")
        else:
            typer.echo(f"Resuming pipeline for {pdf_file.name}")
            typer.echo(f"Current progress: {initial_status.get('overall_progress', 0):.1%}")

        typer.echo("")

        # Run pipeline
        async def run_async():
            return await engine.execute_pipeline(pdf_file, target_stages, page_numbers)

        # Execute pipeline
        with typer.progressbar(length=100, label="Processing") as progress:
            # Note: For now, we'll run the pipeline and update progress at the end
            # In a more sophisticated implementation, we could update progress in real-time
            document_data = asyncio.run(run_async())
            progress.update(100)

        # Display results
        final_status = engine.get_pipeline_status(pdf_file)

        typer.echo(f"\nPipeline completed!")
        typer.echo(f"Total pages: {document_data.total_pages}")
        typer.echo(f"Overall progress: {final_status.get('overall_progress', 0):.1%}")

        # Show step details
        if final_status.get("steps"):
            typer.echo("\nStep Details:")
            for step_info in final_status["steps"]:
                typer.echo(f"  {step_info['name']}: {step_info['progress']:.1%} "
                          f"({step_info['completed_pages']}/{step_info['completed_pages'] + step_info['pending_pages']} pages)")

        # Show execution stats
        if engine.execution_stats and debug:
            typer.echo(f"\nExecution Statistics:")
            stats = engine.execution_stats
            typer.echo(f"  Execution time: {stats.get('execution_time_seconds', 0):.1f} seconds")
            typer.echo(f"  Steps executed: {', '.join(stats.get('steps_executed', []))}")

            if stats.get('errors'):
                typer.echo(f"  Errors: {len(stats['errors'])}")
                for error in stats['errors'][:3]:  # Show first 3 errors
                    typer.echo(f"    - {error}")

        output_dir_path = config.get_document_output_dir(pdf_file)
        typer.echo(f"Output directory: {output_dir_path}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run_step(
    pdf_path: str = typer.Argument(..., help="Path to the PDF file to process"),
    step: str = typer.Argument(..., help="Step to run (split, structure, extraction, markdown)"),
    pages: Optional[str] = typer.Option(None, "--pages", "-p",
                                      help="Comma-separated list of page numbers to process"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing of existing files"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Run a single pipeline step.

    This command executes only the specified processing step.
    Useful for debugging or reprocessing specific stages.

    Examples:
        assetable run-step input/book.pdf split
        assetable run-step input/book.pdf structure --pages 1,2,3
        assetable run-step input/book.pdf markdown --force --debug
    """
    # Validate input
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        typer.echo(f"Error: PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)

    # Map step names to classes
    step_mapping = {
        "split": PDFSplitStep,
        "structure": StructureAnalysisStep,
        "extraction": AssetExtractionStep,
        "markdown": MarkdownGenerationStep,
    }

    if step not in step_mapping:
        typer.echo(f"Error: Unknown step '{step}'. Available: {', '.join(step_mapping.keys())}", err=True)
        raise typer.Exit(1)

    step_class = step_mapping[step]

    # Setup configuration
    config = get_config()

    if debug:
        config.processing.debug_mode = True

    if force:
        config.processing.skip_existing_files = False

    # Parse page numbers
    page_numbers = None
    if pages:
        try:
            page_numbers = [int(p.strip()) for p in pages.split(",")]
        except ValueError:
            typer.echo("Error: Invalid page numbers format. Use comma-separated integers.", err=True)
            raise typer.Exit(1)

    try:
        # Run single step
        typer.echo(f"Running {step} step for {pdf_file.name}")

        async def run_async():
            return await run_single_step(pdf_file, step_class, config, page_numbers)

        document_data = asyncio.run(run_async())

        typer.echo(f"Step '{step}' completed successfully!")
        typer.echo(f"Processed {len(document_data.pages)} pages")

        # Show step status
        engine = PipelineEngine(config)
        status = engine.get_pipeline_status(pdf_file)

        if status.get("steps"):
            for step_info in status["steps"]:
                if step_info["name"].lower().replace(" ", "") == step.replace("_", ""):
                    typer.echo(f"Progress: {step_info['progress']:.1%} "
                              f"({step_info['completed_pages']}/{step_info['completed_pages'] + step_info['pending_pages']} pages)")
                    break

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
