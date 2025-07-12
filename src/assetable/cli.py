"""
Command-line interface for Assetable.

This module provides CLI commands for PDF processing pipeline using Typer.
"""

import typer
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import get_config
from .pdf.pdf_splitter import split_pdf_cli, PDFSplitter
from .file_manager import FileManager
from .pipeline.engine import (
    PipelineEngine,
)
from .ai.ollama_client import OllamaClient
from .ai.vision_processor import EnhancedVisionProcessor
from .ai.ai_steps import (
    EnhancedAIStructureAnalysisStep,
    EnhancedAIAssetExtractionStep,
    EnhancedAIMarkdownGenerationStep,
)
import asyncio
import time
from .models import PageData, ProcessingStage, DocumentData

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
    target_stages: Optional[List[ProcessingStage]] = None
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
        initial_status: Dict[str, Any] = engine.get_pipeline_status(pdf_file)

        if initial_status.get("status") == "not_started":
            typer.echo(f"Starting pipeline for {pdf_file.name}")
        else:
            typer.echo(f"Resuming pipeline for {pdf_file.name}")
            typer.echo(f"Current progress: {initial_status.get('overall_progress', 0):.1%}")

        typer.echo("")

        # Run pipeline
        async def run_async() -> DocumentData:
            return await engine.execute_pipeline(pdf_file, target_stages, page_numbers)

        # Execute pipeline
        with typer.progressbar(length=100, label="Processing") as progress:
            # Note: For now, we'll run the pipeline and update progress at the end
            # In a more sophisticated implementation, we could update progress in real-time
            document_data = asyncio.run(run_async())
            progress.update(100)

        # Display results
        final_status: Dict[str, Any] = engine.get_pipeline_status(pdf_file)

        typer.echo(f"\nPipeline completed!")
        typer.echo(f"Total pages: {len(document_data.pages)}")
        typer.echo(f"Overall progress: {final_status.get('overall_progress', 0):.1%}")

        # Show step details
        if final_status.get("steps"):
            typer.echo("\nStep Details:")
            for step_info in final_status["steps"]:
                typer.echo(f"  {step_info['name']}: {step_info['progress']:.1%} "
                          f"({step_info['completed_pages']}/{step_info['completed_pages'] + step_info['pending_pages']} pages)")

        # Show execution stats
        if hasattr(engine, 'execution_stats') and engine.execution_stats and debug:
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
def analyze(
    pdf_path: str = typer.Argument(..., help="Path to PDF file for analysis"),
    page: int = typer.Option(1, "--page", "-p", help="Page number to analyze"),
    stage: str = typer.Option("structure", "--stage", "-s",
                             help="Analysis stage (structure, extraction, markdown, all)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    document_type: str = typer.Option("technical book", "--type", "-t",
                                    help="Document type (technical book, novel, manual, etc.)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Analyze a specific page using AI processing.

    This command runs detailed AI analysis on a single page to help with
    development and debugging of the AI processing pipeline.

    Examples:
        assetable analyze input/book.pdf --page 1 --stage structure
        assetable analyze input/book.pdf --page 5 --stage all --debug
        assetable analyze input/book.pdf --page 3 --type "technical manual"
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

    try:
        # Initialize vision processor
        typer.echo("Initializing AI system...")
        vision_processor = EnhancedVisionProcessor(config)

        # Ensure page image exists
        image_path = config.get_page_image_path(pdf_file, page)
        if not image_path.exists():
            typer.echo(f"Page image not found. Running PDF split first...")
            splitter = PDFSplitter(config)
            splitter.split_pdf(pdf_file)

        # Create page data
        page_data = PageData(
            page_number=page,
            source_pdf=pdf_file,
            image_path=image_path
        )
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

        typer.echo(f"Analyzing page {page} of {pdf_file.name}")
        typer.echo(f"Document type: {document_type}")
        typer.echo(f"Stage: {stage}")
        typer.echo("")

        # Run analysis stages
        if stage in ["structure", "all"]:
            typer.echo("🔍 Running structure analysis...")
            start_time = time.time()

            result = vision_processor.analyze_page_structure(page_data, document_type)

            if result.success:
                processing_time = time.time() - start_time
                page_data.page_structure = result.page_structure

                typer.echo(f"✅ Structure analysis completed in {processing_time:.2f} seconds")

                if page_data.page_structure:
                    structure = page_data.page_structure
                    typer.echo(f"   📊 Tables found: {len(structure.tables)}")
                    typer.echo(f"   📈 Figures found: {len(structure.figures)}")
                    typer.echo(f"   🖼️  Images found: {len(structure.images)}")
                    typer.echo(f"   🔗 References found: {len(structure.references)}")

                    if debug and structure.tables:
                        typer.echo("   Table details:")
                        for i, table in enumerate(structure.tables, 1):
                            typer.echo(f"     {i}. {table.name}: {table.description}")

                    if debug and structure.figures:
                        typer.echo("   Figure details:")
                        for i, figure in enumerate(structure.figures, 1):
                            typer.echo(f"     {i}. {figure.name} ({figure.figure_type}): {figure.description}")
            else:
                typer.echo(f"❌ Structure analysis failed: {result.error_message}")
                if stage == "structure":
                    raise typer.Exit(1)

        if stage in ["extraction", "all"] and page_data.page_structure:
            typer.echo("🔧 Running asset extraction...")
            start_time = time.time()

            result = vision_processor.extract_assets(page_data)

            if result.success:
                processing_time = time.time() - start_time
                page_data.extracted_assets = result.extracted_assets

                typer.echo(f"✅ Asset extraction completed in {processing_time:.2f} seconds")
                typer.echo(f"   📦 Assets extracted: {len(page_data.extracted_assets)}")

                if debug and page_data.extracted_assets:
                    typer.echo("   Asset details:")
                    for i, asset in enumerate(page_data.extracted_assets, 1):
                        asset_type = type(asset).__name__.replace("Asset", "")
                        typer.echo(f"     {i}. {asset_type}: {asset.name}")
            else:
                typer.echo(f"❌ Asset extraction failed: {result.error_message}")
                if stage == "extraction":
                    raise typer.Exit(1)

        if stage in ["markdown", "all"] and page_data.page_structure:
            typer.echo("📝 Running Markdown generation...")
            start_time = time.time()

            result = vision_processor.generate_markdown(page_data, document_type)

            if result.success:
                processing_time = time.time() - start_time
                page_data.markdown_content = result.markdown_content

                typer.echo(f"✅ Markdown generation completed in {processing_time:.2f} seconds")

                if page_data.markdown_content:
                    content_length = len(page_data.markdown_content)
                    lines = page_data.markdown_content.count('\n') + 1
                    typer.echo(f"   📄 Generated: {content_length} characters, {lines} lines")
                    typer.echo(f"   🔗 Asset references: {len(result.asset_references)}")

                    if debug:
                        typer.echo("\n   Markdown preview (first 300 characters):")
                        preview = page_data.markdown_content[:300]
                        if len(page_data.markdown_content) > 300:
                            preview += "..."
                        typer.echo(f"   {preview}")
            else:
                typer.echo(f"❌ Markdown generation failed: {result.error_message}")
                if stage == "markdown":
                    raise typer.Exit(1)

        # Show processing statistics
        if debug:
            stats = vision_processor.get_processing_stats()
            processing_stats = stats['processing_stats']

            typer.echo(f"\n📊 Processing Statistics:")
            for stage_name, stage_stats in processing_stats.items():
                if stage_stats['count'] > 0:
                    typer.echo(f"   {stage_name.replace('_', ' ').title()}:")
                    typer.echo(f"     Requests: {stage_stats['count']}")
                    typer.echo(f"     Success rate: {stage_stats['success_rate']:.1%}")
                    typer.echo(f"     Average time: {stage_stats['average_time']:.2f}s")

        # Save results if any processing was done
        file_manager = FileManager(config)

        if page_data.page_structure:
            structure_path = file_manager.save_page_structure(pdf_file, page, page_data.page_structure)
            typer.echo(f"\n💾 Saved structure to: {structure_path}")

        if page_data.extracted_assets:
            saved_count = 0
            for asset in page_data.extracted_assets:
                try:
                    file_manager.save_asset_file(pdf_file, page, asset)
                    saved_count += 1
                except Exception as e:
                    if debug:
                        typer.echo(f"   Failed to save {asset.name}: {e}")

            if saved_count > 0:
                typer.echo(f"💾 Saved {saved_count} assets")

        if page_data.markdown_content:
            markdown_path = file_manager.save_markdown_content(pdf_file, page, page_data.markdown_content)
            typer.echo(f"💾 Saved Markdown to: {markdown_path}")

        output_dir_path = config.get_document_output_dir(pdf_file)
        typer.echo(f"\n📁 Output directory: {output_dir_path}")
        typer.echo("✨ Analysis completed successfully!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if debug:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def process_single(
    pdf_path: str = typer.Argument(..., help="Path to PDF file"),
    page: int = typer.Argument(..., help="Page number to process"),
    stage: str = typer.Option("all", "--stage", "-s",
                             help="Processing stage (structure, extraction, markdown, all)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Process a single page through the AI pipeline.

    This command processes a single page through one or more stages
    of the AI processing pipeline using the actual pipeline steps.

    Examples:
        assetable process-single input/book.pdf 1
        assetable process-single input/book.pdf 5 --stage structure
        assetable process-single input/book.pdf 3 --force --debug
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

    try:
        # Setup file manager
        file_manager = FileManager(config)
        file_manager.setup_document_structure(pdf_file)

        # Ensure page image exists
        image_path = config.get_page_image_path(pdf_file, page)
        if not image_path.exists():
            typer.echo(f"Page image not found. Running PDF split first...")
            splitter = PDFSplitter(config)
            splitter.split_pdf(pdf_file)

        # Create initial page data
        page_data = PageData(
            page_number=page,
            source_pdf=pdf_file,
            image_path=image_path
        )
        page_data.mark_stage_completed(ProcessingStage.PDF_SPLIT)

        typer.echo(f"Processing page {page} of {pdf_file.name}")
        typer.echo(f"Stage: {stage}")

        # Initialize pipeline steps
        structure_step = EnhancedAIStructureAnalysisStep(config)
        extraction_step = EnhancedAIAssetExtractionStep(config)
        markdown_step = EnhancedAIMarkdownGenerationStep(config)

        # Execute stages
        if stage in ["structure", "all"]:
            typer.echo("\n🔍 Executing structure analysis step...")
            page_data = asyncio.run(structure_step.execute_page(page_data))
            typer.echo("✅ Structure analysis step completed")

        if stage in ["extraction", "all"]:
            typer.echo("\n🔧 Executing asset extraction step...")
            page_data = asyncio.run(extraction_step.execute_page(page_data))
            typer.echo("✅ Asset extraction step completed")

        if stage in ["markdown", "all"]:
            typer.echo("\n📝 Executing Markdown generation step...")
            page_data = asyncio.run(markdown_step.execute_page(page_data))
            typer.echo("✅ Markdown generation step completed")

        # Save final page data
        file_manager.save_page_data(page_data)

        # Show completion status
        typer.echo(f"\n📋 Processing Summary:")
        for stage_enum in [ProcessingStage.PDF_SPLIT, ProcessingStage.STRUCTURE_ANALYSIS,
                          ProcessingStage.ASSET_EXTRACTION, ProcessingStage.MARKDOWN_GENERATION]:
            status = "✅" if page_data.is_stage_completed(stage_enum) else "⏳"
            typer.echo(f"   {status} {stage_enum.value.replace('_', ' ').title()}")

        if debug and page_data.processing_log:
            typer.echo(f"\n📜 Processing Log:")
            for log_entry in page_data.processing_log[-5:]:  # Show last 5 entries
                typer.echo(f"   {log_entry}")

        output_dir_path = config.get_document_output_dir(pdf_file)
        typer.echo(f"\n📁 Output directory: {output_dir_path}")
        typer.echo("✨ Single page processing completed!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        if debug:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def check_ai(
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """
    Check AI system status and available models.

    This command verifies that Ollama is running and shows available models
    for document processing.
    """
    # Setup configuration
    config = get_config()
    if debug:
        config.processing.debug_mode = True

    try:
        # Check Ollama connection
        ollama_client = OllamaClient(config)

        typer.echo("Checking AI system status...")
        typer.echo(f"Ollama host: {config.ai.ollama_host}")

        # Test connection
        if ollama_client.check_connection():
            typer.echo("✓ Ollama connection: OK")
        else:
            typer.echo("✗ Ollama connection: FAILED")
            typer.echo("Please ensure Ollama is running and accessible.")
            raise typer.Exit(1)

        # Get available models
        try:
            models = ollama_client.get_available_models()
            typer.echo(f"✓ Available models: {len(models)}")

            if models:
                typer.echo("\nInstalled models:")
                for model in models:
                    typer.echo(f"  - {model}")
            else:
                typer.echo("No models found. You may need to pull vision models first.")

            # Check configured models
            typer.echo(f"\nConfigured models:")
            typer.echo(f"  Structure analysis: {config.ai.structure_analysis_model}")
            typer.echo(f"  Asset extraction: {config.ai.asset_extraction_model}")
            typer.echo(f"  Markdown generation: {config.ai.markdown_generation_model}")

            # Verify configured models are available
            missing_models: List[str] = []
            for model_name in [config.ai.structure_analysis_model,
                               config.ai.asset_extraction_model,
                               config.ai.markdown_generation_model]:
                if model_name not in models:
                    missing_models.append(model_name)

            if missing_models:
                typer.echo(f"\n⚠ Missing models: {', '.join(missing_models)}")
                typer.echo("Run 'ollama pull <model_name>' to install missing models.")
            else:
                typer.echo("✓ All configured models are available")

        except Exception as e:
            typer.echo(f"✗ Model check failed: {e}")
            raise typer.Exit(1)

        # Get processing stats if available
        stats = ollama_client.get_processing_stats()
        if stats['total_requests'] > 0:
            typer.echo(f"\nProcessing statistics:")
            typer.echo(f"  Total requests: {stats['total_requests']}")
            typer.echo(f"  Total processing time: {stats['total_processing_time']:.2f} seconds")
            typer.echo(f"  Average processing time: {stats['average_processing_time']:.2f} seconds")

        typer.echo("\nAI system check completed successfully!")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
