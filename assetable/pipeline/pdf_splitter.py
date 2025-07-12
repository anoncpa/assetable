from pathlib import Path
import fitz  # PyMuPDF
from pydantic import BaseModel

class SplitConfig(BaseModel):
    dpi: int = 300
    output_dir: Path

def split_pdf(pdf_path: str, cfg: SplitConfig | None = None) -> None:
    cfg = cfg or SplitConfig(output_dir=Path("output") / Path(pdf_path).stem / "pdfSplitted")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=cfg.dpi)
        out_file = cfg.output_dir / f"page_{idx:04}.png"
        pix.save(out_file)
