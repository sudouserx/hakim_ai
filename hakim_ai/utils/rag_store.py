"""
Simple RAG knowledge store for the histopathology pipeline.

Design: in-memory key-value store with BM25-style keyword matching.
Production replacement: swap retrieval() for a real vector store
(FAISS, ChromaDB, Qdrant) and replace mock_docs with WHO classification
text, NCCN guidelines, and institutional case archives.

Assumption: all knowledge base documents are pre-chunked and provided
at init time; this store is read-only after construction.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeDocument:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Pre-computed word frequency table
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


# ---------------------------------------------------------------------------
# Pre-loaded mock knowledge base (gastric cancer / STAD focused)
# ---------------------------------------------------------------------------

_MOCK_WHO_CRITERIA = [
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_intestinal",
        content=(
            "Intestinal-type gastric adenocarcinoma (Lauren classification): "
            "Characterised by cohesive neoplastic cells forming gland-like tubular "
            "or papillary structures. Preceded by well-defined precancerous changes "
            "including intestinal metaplasia and dysplasia. Associated with H. pylori "
            "infection, high sodium diet, and smoking. Commonly seen in older males. "
            "CDX2 expression is typically positive. WHO 5th Edition, 2019."
        ),
        metadata={"category": "who_criteria", "subtype": "intestinal"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_diffuse",
        content=(
            "Diffuse-type gastric adenocarcinoma (Lauren classification): "
            "Poorly cohesive cells, often with signet-ring cell morphology, "
            "diffusely infiltrating the stomach wall with desmoplastic stroma. "
            "E-cadherin (CDH1) mutations common. No clear precursor lesion. "
            "Younger age of onset; hereditary diffuse gastric cancer (CDH1 germline). "
            "Linitis plastica pattern in advanced cases. WHO 5th Edition, 2019."
        ),
        metadata={"category": "who_criteria", "subtype": "diffuse"},
    ),
    KnowledgeDocument(
        doc_id="msi_gastric_features",
        content=(
            "MSI-H gastric carcinoma histological correlates: "
            "Tumour-infiltrating lymphocytes (Crohn-like infiltrate), poor "
            "differentiation, mucinous or medullary features, solid growth pattern, "
            "right-sided (proximal) location. Loss of MLH1 due to promoter "
            "hypermethylation is most common. Strong predictor of response to "
            "immune checkpoint inhibitors (pembrolizumab). Kather et al. 2019 "
            "demonstrated H&E-based MSI prediction with AUC 0.84."
        ),
        metadata={"category": "biomarker", "marker": "MSI"},
    ),
    KnowledgeDocument(
        doc_id="her2_gastric_scoring",
        content=(
            "HER2 scoring in gastric adenocarcinoma (ToGA criteria): "
            "IHC 3+: strong complete/basolateral membrane staining in >10% of cells. "
            "IHC 2+/FISH amplified: equivocal — requires ISH confirmation. "
            "IHC 0 or 1+: HER2 negative. HER2 positivity in 7-17% of gastric cancers. "
            "Predicts benefit from trastuzumab (ToGA trial, Bang et al. 2010). "
            "H&E features loosely associated: intestinal type, tubular architecture."
        ),
        metadata={"category": "biomarker", "marker": "HER2"},
    ),
    KnowledgeDocument(
        doc_id="ebv_gastric",
        content=(
            "EBV-associated gastric carcinoma: ~10% of gastric adenocarcinomas. "
            "Dense lymphocytic stroma (lymphoepithelioma-like carcinoma), male "
            "predominance, proximal location. PIK3CA mutations common. High PD-L1 "
            "expression, excellent response to immunotherapy. EBER-ISH for detection. "
            "TCGA molecular subtype (EBV) has distinct DNA methylation signature."
        ),
        metadata={"category": "biomarker", "marker": "EBV"},
    ),
    KnowledgeDocument(
        doc_id="til_prognostic",
        content=(
            "Tumour-infiltrating lymphocytes (TILs) in gastric cancer: "
            "High TIL density correlates with better prognosis and response to "
            "immunotherapy. Brisk TILs defined as >50% stromal area occupied by "
            "lymphocytes. TIL assessment should be performed in the tumour stroma "
            "using H&E slides. High TIL density is a surrogate for MSI-H and EBV "
            "subtypes. Standardised TIL scoring recommended (Hendry et al. 2017)."
        ),
        metadata={"category": "prognostic", "feature": "TIL"},
    ),
    KnowledgeDocument(
        doc_id="nccn_gastric_treatment",
        content=(
            "NCCN Guidelines — Gastric Cancer Treatment Pathway: "
            "For locally advanced gastric adenocarcinoma: perioperative chemotherapy "
            "(FLOT regimen) or neoadjuvant chemoradiotherapy. For metastatic: "
            "platinum + fluoropyrimidine ± HER2-targeted therapy (if HER2+). "
            "Second line: ramucirumab ± paclitaxel. MSI-H or dMMR: pembrolizumab "
            "preferred for second-line or later. PD-L1 CPS ≥1: nivolumab added "
            "to first-line chemotherapy (CheckMate 649)."
        ),
        metadata={"category": "guidelines", "source": "NCCN"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_papillary",
        content="Papillary adenocarcinoma: well-differentiated exophytic growth with fibrovascular cores.",
        metadata={"category": "who_criteria", "subtype": "papillary"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_tubular",
        content="Tubular adenocarcinoma: branching, dilated, or fused tubular structures.",
        metadata={"category": "who_criteria", "subtype": "tubular"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_mucinous",
        content="Mucinous adenocarcinoma: >50% extracellular mucin pools.",
        metadata={"category": "who_criteria", "subtype": "mucinous"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_poorly_cohesive",
        content="Poorly cohesive carcinoma (including signet-ring cell): isolated cells or small aggregates.",
        metadata={"category": "who_criteria", "subtype": "poorly_cohesive"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenoca_mixed",
        content="Mixed adenocarcinoma: mixture of discrete tubular/papillary and poorly cohesive components.",
        metadata={"category": "who_criteria", "subtype": "mixed"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_squamous",
        content="Squamous cell carcinoma: rare in stomach, identical to esophageal SCC.",
        metadata={"category": "who_criteria", "subtype": "squamous"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_adenosquamous",
        content="Adenosquamous carcinoma: mixture of adenocarcinoma and squamous cell carcinoma components.",
        metadata={"category": "who_criteria", "subtype": "adenosquamous"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_undifferentiated",
        content="Undifferentiated carcinoma: lacks any glandular or squamous differentiation.",
        metadata={"category": "who_criteria", "subtype": "undifferentiated"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_medullary",
        content="Medullary carcinoma (EBV+ or MSI-H): solid syncytial growth with dense lymphoid stroma.",
        metadata={"category": "who_criteria", "subtype": "medullary"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_hepatoid",
        content="Hepatoid adenocarcinoma: morphologically resembles hepatocellular carcinoma, often AFP positive.",
        metadata={"category": "who_criteria", "subtype": "hepatoid"},
    ),
    KnowledgeDocument(
        doc_id="who_gastric_micropapillary",
        content="Micropapillary carcinoma: small clusters of cells lacking fibrovascular cores in empty spaces.",
        metadata={"category": "who_criteria", "subtype": "micropapillary"},
    ),
]

_MOCK_SIMILAR_CASES = [
    {
        "case_id": "TCGA-BR-4253",
        "diagnosis": "Gastric adenocarcinoma, intestinal type (Lauren)",
        "msi_status": "MSI-H",
        "grade": "poorly differentiated",
        "key_features": ["high TIL density", "mucinous component", "irregular glands"],
        "similarity_score": 0.87,
    },
    {
        "case_id": "TCGA-R1-A8MT",
        "diagnosis": "Gastric adenocarcinoma, diffuse type (Lauren)",
        "msi_status": "MSS",
        "grade": "poorly differentiated",
        "key_features": ["signet-ring cells", "desmoplastic stroma", "single-file infiltration"],
        "similarity_score": 0.81,
    },
    {
        "case_id": "TCGA-VQ-A91Y",
        "diagnosis": "Gastric adenocarcinoma, mixed type (Lauren)",
        "msi_status": "MSI-H",
        "grade": "moderately differentiated",
        "key_features": ["mixed glandular / poorly cohesive", "prominent lymphocytes"],
        "similarity_score": 0.75,
    },
]


# ---------------------------------------------------------------------------
# RAG Store
# ---------------------------------------------------------------------------

class RAGStore:
    """
    Simple BM25-style keyword retrieval over a document collection.

    Production replacement: implement retrieve() using a vector store and
    sentence-transformer embeddings for semantic similarity.
    """

    def __init__(self, documents: Optional[List[KnowledgeDocument]] = None, use_vector_store: bool = True):
        self.documents = documents or list(_MOCK_WHO_CRITERIA)
        self._idf: Dict[str, float] = {}
        self._build_idf()
        
        self.use_vector_store = use_vector_store
        self._faiss_index = None
        self._embedder = None
        self._doc_embeddings = []
        
        if self.use_vector_store:
            self._try_init_vector_store()

    def _try_init_vector_store(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [doc.content for doc in self.documents]
            embeddings = self._embedder.encode(texts)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)
            
            self._doc_embeddings = embeddings
        except ImportError:
            self.use_vector_store = False

    def _build_idf(self) -> None:
        """Compute inverse document frequency for all terms."""
        n = len(self.documents)
        term_doc_count: Dict[str, int] = {}
        for doc in self.documents:
            for term in set(doc.term_freq.keys()):
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        for term, df in term_doc_count.items():
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    @property
    def idf(self) -> Dict[str, float]:
        return self._idf

    def retrieve(self, query: str, top_k: int = 5) -> List[KnowledgeDocument]:
        """Return top_k documents ranked by BM25 score or FAISS vector similarity."""
        if self.use_vector_store and self._faiss_index is not None and self._embedder is not None:
            import faiss
            import numpy as np
            q_emb = self._embedder.encode([query])
            faiss.normalize_L2(q_emb)
            distances, indices = self._faiss_index.search(q_emb, min(top_k, len(self.documents)))
            
            return [self.documents[idx] for idx in indices[0] if idx != -1]
            
        # Fallback to BM25
        query_terms = _tokenize(query)
        if not query_terms:
            return self.documents[:top_k]

        k1, b = 1.5, 0.75
        avg_dl = sum(sum(d.term_freq.values()) for d in self.documents) / max(
            len(self.documents), 1
        )

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

    def retrieve_by_category(
        self, category: str, top_k: int = 5, query: Optional[str] = None
    ) -> List[KnowledgeDocument]:
        """Filter documents by metadata category, optionally sorting by relevance."""
        matches = [d for d in self.documents if d.metadata.get("category") == category]
        
        if query and matches:
            # Simple BM25 scoring just for the matches
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

    def get_similar_cases(
        self, query_embedding: Optional[List[float]] = None, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Return similar archived cases. Uses query embedding for semantic search if provided."""
        if query_embedding is not None and self.use_vector_store:
            # Here we would normally search a case database with the embedding.
            # For now, we mock by shuffling deterministically based on embedding mean
            import random
            import numpy as np
            mean_val = np.mean(query_embedding)
            rng = random.Random(int(mean_val * 10000))
            cases = list(_MOCK_SIMILAR_CASES)
            rng.shuffle(cases)
            return cases[:top_k]
            
        return _MOCK_SIMILAR_CASES[:top_k]