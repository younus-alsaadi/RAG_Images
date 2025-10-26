from src.rag_images.config import ColPaliConfig
from src.rag_images.models.visrag import VisRAG
from src.rag_images.pipelines.visrag_only import VisRAGPipeline

if __name__ == "__main__":
    retriever = VisRAG(ColPaliConfig(top_k=5))
    pipe = VisRAGPipeline(retriever)

    pdf_path = "..files/moodys-rating-report"

    pipe.build(pdf_path)
    query = "bar chart about revenue"

    hits = pipe.query(query, top_k=5)
    for page, score in hits:
        print(f"page={page} score={score:.4f}")