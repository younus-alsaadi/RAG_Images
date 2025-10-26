from src.rag_images.config import QwenConfig, DeviceConfig, ColPaliConfig
from src.rag_images.models.qwen_ocr import QwenOCR
from src.rag_images.models.visrag import VisRAG
from src.rag_images.pipelines.hybrid import HybridPipeline


if __name__ == "__main__":
    ocr = QwenOCR(QwenConfig(), DeviceConfig(device="cuda", dtype="auto"))
    retriever = VisRAG(ColPaliConfig(top_k=3))
    pipe = HybridPipeline(retriever=retriever, ocr=ocr, top_k=3)
    pdf_path = "examples/sample.pdf"
    pipe.build(pdf_path)
    question = "What are the 2023 quarterly sales?"
    answers = pipe.answer(question)
    for page, text in answers:
        print(f"\n### Page {page}\n{text[:1200]}\n")