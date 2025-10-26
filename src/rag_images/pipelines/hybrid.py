from typing import List, Tuple
from ..interfaces import OCRModel, Retriever

class HybridPipeline:
    """Retrieve relevant pages with VisRAG, then OCR only those pages with
    Qwen."""
    def __init__(self, retriever: Retriever, ocr: OCRModel, top_k: int = 5):
        self.retriever = retriever
        self.ocr = ocr
        self.top_k = top_k

    def build(self, pdf_path: str) -> None:
        self._pdf_path = pdf_path
        self.retriever.build_index(pdf_path)

    def answer(self, query: str) -> List[Tuple[int, str]]:
        # 1) Retrieve top‑k relevant pages
        hits = self.retriever.search (query, top_k=self.top_k) # [(page,score), ...]
        pages = pages = [p for p, _in hits]
        # 2) OCR those pages only
        texts = self.ocr.ocr_pdf(self._pdf_path, pages=pages)
        # texts is [(page, text), ...] — already aligned by page numbers
        return texts