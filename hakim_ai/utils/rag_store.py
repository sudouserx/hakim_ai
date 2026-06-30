"""
Production RAG knowledge store for the histopathology pipeline.

Design: Vector store retrieval (FAISS, ChromaDB) and BM25-style keyword matching
over JSON-loaded knowledge bases (WHO criteria, guidelines, cases).
"""
from __future__ import annotations

import math
import re
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("rag_store")

@dataclass
class KnowledgeDocument:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    term_freq: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.term_freq:
            self.term_freq = _tokenize_freq(self.content)

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())

def _tokenize_freq(text: str) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for token in _tokenize(text):
        freq[token] = freq.get(token, 0) + 1
    return freq


class RAGStore:
    """
    Retrieves relevant clinical knowledge or past cases from a database.
    """
    def __init__(self, knowledge_base_path: str, use_vector_store: bool = True):
        if not knowledge_base_path or not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {knowledge_base_path}")
            
        self.documents = []
        self.similar_cases = []
        
        # Load from JSON database
        try:
            with open(knowledge_base_path, 'r') as f:
                data = json.load(f)
                docs = data.get("documents", [])
                self.similar_cases = data.get("cases", [])
                
                for d in docs:
                    self.documents.append(KnowledgeDocument(
                        doc_id=d.get("doc_id", ""),
                        content=d.get("content", ""),
                        metadata=d.get("metadata", {})
                    ))
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base from {knowledge_base_path}: {e}")

        self._idf: Dict[str, float] = {}
        self._build_idf()
        
        self.use_vector_store = use_vector_store
        self._faiss_index = None
        self._embedder = None
        
        if self.use_vector_store:
            self._try_init_vector_store()

    def _try_init_vector_store(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            if not self.documents:
                return
                
            texts = [doc.content for doc in self.documents]
            embeddings = self._embedder.encode(texts)
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)
        except ImportError:
            logger.warning("FAISS or sentence-transformers missing. Falling back to BM25.")
            self.use_vector_store = False
        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            self.use_vector_store = False

    def _build_idf(self) -> None:
        if not self.documents:
            return
        n = len(self.documents)
        term_doc_count: Dict[str, int] = {}
        for doc in self.documents:
            for term in set(doc.term_freq.keys()):
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        for term, df in term_doc_count.items():
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    @property
    def idf(self) -> Dict[str, float]:
        return self._idf

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeDocument]:
        if not self.documents:
            return []
            
        if self.use_vector_store and self._faiss_index is not None and self._embedder is not None:
            import faiss
            q_emb = self._embedder.encode([query])
            faiss.normalize_L2(q_emb)
            distances, indices = self._faiss_index.search(q_emb, min(top_k, len(self.documents)))
            return [self.documents[idx] for idx in indices[0] if idx != -1]
            
        query_terms = _tokenize(query)
        if not query_terms:
            return self.documents[:top_k]

        k1, b = 1.5, 0.75
        avg_dl = sum(sum(d.term_freq.values()) for d in self.documents) / max(len(self.documents), 1)

        scored: List[tuple] = []
        for doc in self.documents:
            dl = sum(doc.term_freq.values())
            score = 0.0
            for term in query_terms:
                tf = doc.term_freq.get(term, 0)
                idf_val = self._idf.get(term, 0.0)
                tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
                score += idf_val * tf_norm
            scored.append((score, doc))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def retrieve_by_category(self, category: str, top_k: int = 5, query: Optional[str] = None) -> List[KnowledgeDocument]:
        matches = [d for d in self.documents if d.metadata.get("category") == category]
        if not matches:
            return []
            
        if query:
            query_terms = _tokenize(query)
            scored = []
            for doc in matches:
                score = 0.0
                for term in query_terms:
                    tf = doc.term_freq.get(term, 0)
                    idf_val = self._idf.get(term, 0.0)
                    score += idf_val * tf
                scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored[:top_k]]
            
        return matches[:top_k]

    def get_similar_cases(self, query_embedding: Optional[List[float]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.similar_cases:
            return []
            
        # In a real implementation, we would use FAISS on the query_embedding against case embeddings.
        # Here we just return top_k cases if embedding similarity is not implemented,
        # but WITHOUT deterministic random shuffling (which is testing behavior).
        return self.similar_cases[:top_k]