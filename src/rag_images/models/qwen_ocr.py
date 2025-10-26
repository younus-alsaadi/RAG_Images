from typing import List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from ..config import QwenConfig, DeviceConfig
from ..utils.pdf_utils import pdf_to_images
from ..config import PDFConfig

class QwenOCR:
    """OCR using Qwen2.5‑VL. Focused and minimal.
    Methods return plain text for each page/image.
    """
    def __init__(self, qwen_cfg: QwenConfig, dev: DeviceConfig):
        self.qwen_cfg = qwen_cfg
        self.dev = dev
        torch_dtype = None
        if dev.dtype == "float16":
            torch_dtype = torch.float16
        elif dev.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dev.dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None # auto

        self.processor = AutoProcessor.from_pretrained(
            qwen_cfg.model_name, trust_remote_code=True,
            cache_dir=qwen_cfg.cache_dir
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_cfg.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if dev.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=qwen_cfg.cache_dir,
        )

    def _run(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor(
            messages,
            return_tensors="pt",
            max_pixels=self.qwen_cfg.max_pixels,
            min_pixels=self.qwen_cfg.min_pixels,
        ).to(self.model.device)

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=512)
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)
        return text.strip()

    def ocr_image(self, image: Image.Image, lang_hint: Optional[str] = None) -> str:
        prompt = "Extract all visible text. Preserve reading order."
        if lang_hint:
            prompt += f" Language hint: {lang_hint}."
        return self._run(image, prompt)

    def ocr_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Tuple[int, str]]:
        from ..config import PDFConfig
        images = pdf_to_images(pdf_path, PDFConfig())
        results: List[Tuple[int, str]] = []

        for i, img in enumerate(images, start=1):
            if pages and i not in pages:
                continue
            text = self.ocr_image(img)
            results.append((i, text))

        return results


