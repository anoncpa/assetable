import typer
from assetable.pipeline import pdf_splitter

app = typer.Typer(add_completion=False)

@app.command()
def split(pdf_path: str):
    """
    Split the input PDF into per-page PNG files.
    """
    pdf_splitter.split_pdf(pdf_path)

if __name__ == "__main__":
    app()
