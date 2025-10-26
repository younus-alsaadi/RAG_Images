from typing import List, Tuple, Optional
from ..interfaces import OCRModel
class OCRPipeline:
    """Thin wrapper so you can replace OCRModel via dependency injection."""
    def __init__(self, ocr: OCRModel):
        self.ocr = ocr

    def run_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) ->List[Tuple[int, str]]:
        return self.ocr.ocr_pdf(pdf_path, pages=pages)