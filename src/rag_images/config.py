from dataclasses import dataclass
from typing import Optional

@dataclass
class DeviceConfig:
    device: str = "cuda" # "cuda" or "cpu"
    dtype: str = "auto" # "auto", "float16", "bfloat16", "float32"

@dataclass
class QwenConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    min_pixels: int = 128 * 16 * 16
    max_pixels: int = 1024 * 16 * 16
    cache_dir: Optional[str] = None

@dataclass
class ColPaliConfig:
    model_name: str = "vidore/colpali"
    index_dir: str = ".rag_index"
    top_k: int = 5
    
@dataclass
class PDFConfig:
    dpi: int = 200
    max_pages: Optional[int] = None  # None = all pages