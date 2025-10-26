from src.rag_images.config import QwenConfig, DeviceConfig
from src.rag_images.models.qwen_ocr import QwenOCR
from src.rag_images.pipelines.ocr_only import OCRPipeline

if __name__ == "__main__":
    qwen = QwenOCR(QwenConfig(), DeviceConfig(device="cuda", dtype="auto"))
    pipe = OCRPipeline(qwen)
    # Example
    pdf_path = "..files/moodys-rating-report"
    results = pipe.run_pdf(pdf_path)
    for page, text in results:
        print(f"\n--- Page {page} ---\n{text[:1000]}\n")