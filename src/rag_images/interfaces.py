from typing import Protocol, List, Tuple, Optional
from PIL import Image


class OCRModel(Protocol):
    def ocr_image(self, image: Image.Image, lang_hint: Optional[str] = None) -> str:
        ...

    def ocr_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Tuple[int, str]]:
        ...


class Retriever(Protocol):
    def build_index(self, pdf_path: str) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        ...


class Generator(Protocol):
    def generate(self, query: str, context: str) -> str:
        ...
