from typing import List, Tuple
from ..interfaces import Retriever

class VisRAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def build(self, pdf_path: str) -> None:
        self.retriever.build_index(pdf_path)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
         return self.retriever.search(text, top_k=top_k)