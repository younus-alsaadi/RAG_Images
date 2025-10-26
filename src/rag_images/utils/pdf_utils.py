from typing import List, Optional
from pdf2image import convert_from_path
from PIL import Image
from ..config import PDFConfig

def pdf_to_images(pdf_path: str, cfg: PDFConfig) -> List[Image.Image]:
    images = convert_from_path(pdf_path, dpi=cfg.dpi)
    if cfg.max_pages is not None:
    images = images[: cfg.max_pages]
    return images