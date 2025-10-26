from typing import List, Tuple
import os
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    # ColPali official engine (preferred)
    from colpali_engine import ColPali, ColPaliProcessor
except Exception: # pragma: no cover
    ColPali = None
    ColPaliProcessor = None

from ..config import ColPaliConfig, PDFConfig
from ..utils.pdf_utils import pdf_to_images

class VisRAG:
    """Visual retriever over PDF pages using ColPali‑style embeddings.
    It indexes each page image, then allows top‑K search by a text query.
    """
    def __init__(self, cfg: ColPaliConfig):
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.index_dir / "index.pt"
        self._pages: List[int] = []
        self._embeddings = None
        self._processor = None
        self._model = None

    def _lazy_load(self):
        if ColPali is None:
            raise ImportError("colpali_engine not available. Please install colpali-engine and its deps."
        )
        if self._model is None:
            self._processor = ColPaliProcessor.from_pretrained(self.cfg.model_name)
            self._model = ColPali.from_pretrained(self.cfg.model_name)

    def build_index(self, pdf_path: str) -> None:
        self._lazy_load()
        images = pdf_to_images(pdf_path, PDFConfig())
        page_embs = []
        self._pages = list(range(1, len(images) + 1))

        for img in images:
            inputs = self._processor(images=[img], return_tensors="pt")
            with torch.inference_mode():
                emb = self._model.get_image_features(**inputs)  # shape: [1,D]
            page_embs.append(emb.cpu())
        self._embeddings = torch.cat(page_embs, dim=0)  # [N, D]
        torch.save({"pages": self._pages, "embeddings": self._embeddings},self._db_path)

    def load_index(self) -> bool:
        if not self._db_path.exists():
            return False
        blob = torch.load(self._db_path, map_location="cpu")
        self._embeddings = blob["embeddings"]
        return True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        self._lazy_load()
        if self._embeddings is None:
            if not self.load_index():
                raise RuntimeError("Index not built. Call build_index(pdf_path) first.")

        q_inputs = self._processor(text=[query], return_tensors="pt")
        with torch.inference_mode():
            q_emb = self._model.get_text_features(**q_inputs)  # [1, D]
        page_embs = self._embeddings  # [N, D]
        sims = F.cosine_similarity(q_emb, page_embs)  # broadcast to [N]
        scores, idxs = torch.topk(sims, k=min(top_k, sims.shape[0]))
        results = [(self._pages[int(i)], float(s)) for s, i in zip(scores,idxs)]
        return results